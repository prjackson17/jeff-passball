"""
live_stats.py

Fetches live MLB data from the Stats API to supplement RAG context on
every query. Three types of augmentation:
  1. Current division standings (cached 5 min, always injected)
  2. Season stats for any player name detected in the query
  3. League stat leaders (cached 5 min, injected for MVP/rankings queries)
"""

import re
import time
import requests
from datetime import datetime
from typing import Optional

MLB_BASE = "https://statsapi.mlb.com/api/v1"

_SEASON = datetime.today().year

# ── Simple HTTP helper ─────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> Optional[dict]:
    try:
        r = requests.get(MLB_BASE + endpoint, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Standings (5-min in-memory cache) ─────────────────────────────────────────

_standings_cache: dict = {"text": None, "ts": 0.0}
_STANDINGS_TTL = 300  # seconds


def _fetch_standings() -> str:
    now = time.time()
    if _standings_cache["text"] and now - _standings_cache["ts"] < _STANDINGS_TTL:
        return _standings_cache["text"]

    data = _get("/standings", params={
        "leagueId": "103,104",
        "season": _SEASON,
        "standingsTypes": "regularSeason",
    })
    if not data:
        return ""

    lines = [f"CURRENT MLB STANDINGS ({datetime.today().strftime('%B %d, %Y')}):"]
    for record in data.get("records", []):
        div_name = record.get("division", {}).get("nameShort", record.get("division", {}).get("name", ""))
        teams = record.get("teamRecords", [])
        if not teams:
            continue
        entries = []
        for t in teams:
            name  = t.get("team", {}).get("clubName", t.get("team", {}).get("name", "?"))
            w     = t.get("wins", 0)
            l     = t.get("losses", 0)
            gb    = t.get("gamesBack", "--")
            entries.append(f"{name} {w}-{l} ({gb} GB)")
        lines.append(f"  {div_name}: " + ", ".join(entries))

    result = "\n".join(lines)
    _standings_cache["text"] = result
    _standings_cache["ts"] = now
    return result


# ── Player name detection ──────────────────────────────────────────────────────

_NON_NAMES = {
    "AL", "NL", "MLB", "ERA", "OPS", "WAR", "HR", "RBI", "AVG", "OBP", "SLG",
    "WHIP", "The", "Who", "What", "When", "Where", "How", "Any", "Did", "Has",
    "Is", "Are", "Was", "Can", "Could", "Would", "Should", "Do", "Does",
    "East", "West", "North", "South", "Central", "American", "National",
    "World", "Series", "League", "Division", "Wild", "Card", "Spring",
    "Training", "Opening", "Day", "All", "Star", "Game",
}


def _extract_names(query: str) -> list:
    """Find consecutive pairs of capitalized words likely to be player names."""
    words = re.split(r"\s+", query)
    names = []
    i = 0
    while i < len(words) - 1:
        w1 = re.sub(r"[^A-Za-z]", "", words[i]).title()
        w2 = re.sub(r"[^A-Za-z]", "", words[i + 1]).title()
        if (w1 and w2
                and re.match(r"^[A-Z][a-z]{1,}$", w1)
                and re.match(r"^[A-Z][a-z]{1,}$", w2)
                and w1 not in _NON_NAMES
                and w2 not in _NON_NAMES):
            names.append(f"{w1} {w2}")
            i += 2
        else:
            i += 1
    return names


# ── Player stats via MLB API ───────────────────────────────────────────────────

def _search_player(name: str) -> Optional[int]:
    """Return MLB player ID for name, or None."""
    data = _get("/people/search", params={"names": name, "sportId": 1})
    if not data:
        return None
    people = data.get("people", [])
    if not people:
        return None
    # Pick closest full-name match
    name_lower = name.lower()
    for p in people:
        if p.get("fullName", "").lower() == name_lower:
            return p["id"]
    return people[0].get("id")


def _format_hitting(splits: list, name: str, team: str) -> str:
    if not splits:
        return ""
    s = splits[0].get("stat", {})
    g    = s.get("gamesPlayed", "?")
    avg  = s.get("avg", "?")
    obp  = s.get("obp", "?")
    slg  = s.get("slg", "?")
    ops  = s.get("ops", "?")
    hr   = s.get("homeRuns", "?")
    rbi  = s.get("rbi", "?")
    r    = s.get("runs", "?")
    sb   = s.get("stolenBases", "?")
    return (
        f"SEASON BATTING — {name} ({team}):  "
        f"{g} G | .{str(avg).replace('.','')[:3] if '.' in str(avg) else avg} AVG | "
        f"OBP {obp} | SLG {slg} | OPS {ops} | {hr} HR | {rbi} RBI | {r} R | {sb} SB"
    )


def _format_pitching(splits: list, name: str, team: str) -> str:
    if not splits:
        return ""
    s = splits[0].get("stat", {})
    g    = s.get("gamesPlayed", "?")
    gs   = s.get("gamesStarted", "?")
    w    = s.get("wins", "?")
    l    = s.get("losses", "?")
    era  = s.get("era", "?")
    ip   = s.get("inningsPitched", "?")
    so   = s.get("strikeOuts", "?")
    bb   = s.get("baseOnBalls", "?")
    whip = s.get("whip", "?")
    sv   = s.get("saves", "?")
    return (
        f"SEASON PITCHING — {name} ({team}):  "
        f"{g} G / {gs} GS | {w}-{l} W-L | {era} ERA | {ip} IP | "
        f"{so} K | {bb} BB | {whip} WHIP | {sv} SV"
    )


def _fetch_player_stats(player_id: int) -> str:
    data_h = _get(f"/people/{player_id}/stats", params={
        "stats": "season", "group": "hitting", "season": _SEASON, "gameType": "R",
    })
    data_p = _get(f"/people/{player_id}/stats", params={
        "stats": "season", "group": "pitching", "season": _SEASON, "gameType": "R",
    })

    # Also grab name + team from player info
    info = _get(f"/people/{player_id}")
    name = "Player"
    team = ""
    if info:
        person = (info.get("people") or [{}])[0]
        name = person.get("fullName", "Player")
        team = (person.get("currentTeam") or {}).get("name", "")

    lines = []
    if team:
        lines.append(f"[CURRENT ROSTER FACT — {_SEASON}] {name} plays for the {team}. Use this team name, not your training data.")
    if data_h:
        splits = (data_h.get("stats") or [{}])[0].get("splits", [])
        line = _format_hitting(splits, name, team)
        if line:
            lines.append(line)
    if data_p:
        splits = (data_p.get("stats") or [{}])[0].get("splits", [])
        line = _format_pitching(splits, name, team)
        if line:
            lines.append(line)
    return "\n".join(lines)


# ── League stat leaders (5-min cache) ─────────────────────────────────────────

_leaders_cache: dict = {"text": None, "ts": 0.0}

_BATTING_CATS  = ["homeRuns", "battingAverage", "rbi", "onBasePlusSlugging",
                  "stolenBases", "runs", "hits"]
_PITCHING_CATS = ["earnedRunAverage", "strikeouts", "wins", "whip", "saves"]

_CAT_LABELS = {
    "homeRuns": "HR", "battingAverage": "AVG", "rbi": "RBI",
    "onBasePlusSlugging": "OPS", "stolenBases": "SB", "runs": "R", "hits": "H",
    "earnedRunAverage": "ERA", "strikeouts": "K", "wins": "W",
    "whip": "WHIP", "saves": "SV",
}

_RANKINGS_KEYWORDS = {
    "mvp", "best", "leader", "leaders", "top", "rank", "ranking", "rankings",
    "award", "cy young", "cyyoung", "most", "leading", "who leads", "stat",
    "stats", "leaderboard", "leaderboards", "best player", "best hitter",
    "best pitcher", "slugger", "ace",
}


def _is_rankings_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _RANKINGS_KEYWORDS)


def _fetch_stat_leaders(limit: int = 5) -> str:
    now = time.time()
    if _leaders_cache["text"] and now - _leaders_cache["ts"] < _STANDINGS_TTL:
        return _leaders_cache["text"]

    categories = ",".join(_BATTING_CATS + _PITCHING_CATS)
    data = _get("/stats/leaders", params={
        "leaderCategories": categories,
        "leaderGameTypes": "R",
        "season": _SEASON,
        "sportId": 1,
        "limit": limit,
    })
    if not data:
        return ""

    sections = {"BATTING LEADERS": [], "PITCHING LEADERS": []}
    for group in data.get("leagueLeaders", []):
        cat   = group.get("leaderCategory", "")
        label = _CAT_LABELS.get(cat, cat)
        dest  = "PITCHING LEADERS" if cat in _PITCHING_CATS else "BATTING LEADERS"
        entries = []
        for entry in group.get("leaders", [])[:limit]:
            name  = entry.get("person", {}).get("fullName", "?")
            team  = entry.get("team", {}).get("clubName", "?")
            value = entry.get("value", "?")
            rank  = entry.get("rank", "?")
            entries.append(f"  {rank}. {name} ({team}) — {value}")
        if entries:
            sections[dest].append(f"{label}:\n" + "\n".join(entries))

    lines = [f"MLB STAT LEADERS — {_SEASON} Season (top {limit}):"]
    for section, groups in sections.items():
        if groups:
            lines.append(f"\n{section}:")
            lines.extend(groups)

    result = "\n".join(lines)
    _leaders_cache["text"] = result
    _leaders_cache["ts"] = now
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def augment_query_context(query: str) -> str:
    """
    Return a string of live MLB data to prepend to RAG context.
    Always includes current standings. Also includes:
      - League stat leaders for MVP/rankings/best-player queries
      - Individual player season stats for any player name detected
    """
    parts = []

    standings = _fetch_standings()
    if standings:
        parts.append(standings)

    if _is_rankings_query(query):
        leaders = _fetch_stat_leaders()
        if leaders:
            parts.append(leaders)

    for name in _extract_names(query):
        pid = _search_player(name)
        if pid:
            stats = _fetch_player_stats(pid)
            if stats:
                parts.append(stats)
            time.sleep(0.1)

    return "\n\n".join(parts)
