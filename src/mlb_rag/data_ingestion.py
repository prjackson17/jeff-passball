"""
data_ingestion.py

Fetches live MLB data from the free MLB Stats API and structures it
into text chunks ready for embedding.

MLB Stats API docs: https://statsapi.mlb.com
No API key required.

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from src.mlb_rag.historical_data import extract_game_features, GameFeatures, fetch_game_editorial


# ── Base URL ──────────────────────────────────────────────────────────────────
BASE_URL = "https://statsapi.mlb.com/api/v1"


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class MLBChunk:
    """
    A single unit of MLB content ready for embedding.

    Each chunk maps to one row in the FAISS vector store.
    The `text` field is what gets embedded; `metadata` is stored
    alongside the vector for retrieval context.
    """
    text: str
    metadata: Dict = field(default_factory=dict)
    chunk_type: str = "general"   # "game_recap", "player_stat", "standings", "trend"


# ── API Helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> Optional[dict]:
    """Simple GET wrapper with error handling."""
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[MLB API] Error fetching {endpoint}: {e}")
        return None


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_scores(date: str = None) -> List[Dict]:
    """
    Fetch all games for a given date.

    Args:
        date: "YYYY-MM-DD" string. Defaults to today.

    Returns:
        List of raw game dicts from the API.
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    data = _get("/schedule", params={
        "sportId": 1,
        "date": date,
        "hydrate": "linescore,boxscore,decisions"
    })

    if not data or "dates" not in data or not data["dates"]:
        print(f"[MLB API] No games found for {date}")
        return []

    games = data["dates"][0].get("games", [])
    for game in games:
        if game.get("status", {}).get("detailedState") == "Final":
            pk = game.get("gamePk")
            game["_direct_boxscore"] = _get(f"/game/{pk}/boxscore") or {}
            game["_editorial"] = fetch_game_editorial(pk)
    print(f"[MLB API] Found {len(games)} games on {date}")
    return games


def fetch_standings() -> List[Dict]:
    """
    Fetch current MLB standings for all divisions.

    Returns:
        List of division standing records.
    """
    data = _get("/standings", params={"leagueId": "103,104", "season": datetime.today().year})
    if not data:
        return []
    return data.get("records", [])


def fetch_player_stats(player_id: int, season: int = None) -> Optional[Dict]:
    """
    Fetch season stats for a single player.

    Args:
        player_id: MLB player ID
        season: Year. Defaults to current year.

    Returns:
        Dict of player stats or None on failure.
    """
    if season is None:
        season = datetime.today().year

    data = _get(f"/people/{player_id}/stats", params={
        "stats": "season",
        "season": season,
        "sportId": 1
    })
    return data


def fetch_team_roster(team_id: int) -> List[Dict]:
    """Fetch active roster for a team."""
    data = _get(f"/teams/{team_id}/roster", params={"rosterType": "active"})
    if not data:
        return []
    return data.get("roster", [])


def fetch_recent_games(days_back: int = 3) -> List[Dict]:
    """
    Fetch games from the last N days — useful for trend detection.

    Args:
        days_back: How many days to look back.

    Returns:
        Flat list of all game dicts across the date range.
    """
    all_games = []
    for i in range(days_back, -1, -1):
        date = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        games = fetch_scores(date)
        for g in games:
            g["_fetched_date"] = date   # tag with date for metadata
        all_games.extend(games)
    return all_games


# ── Chunk Helpers ────────────────────────────────────────────────────────────

def _extract_hr_leaders(boxscore: dict) -> List[str]:
    """Return list of 'Player (N HR)' strings for any batter with at least 1 HR."""
    teams = boxscore.get("teams", {})
    parts = []
    for side in ("home", "away"):
        team_data = teams.get(side, {})
        batters = team_data.get("batters", [])
        players = team_data.get("players", {})
        for pid in batters:
            player_data = players.get(f"ID{pid}", {})
            hrs = int(player_data.get("stats", {}).get("batting", {}).get("homeRuns", 0))
            if hrs > 0:
                name = player_data.get("person", {}).get("fullName", "")
                if name:
                    parts.append(f"{name} ({hrs} HR)")
    return parts


# ── Chunk Builders ────────────────────────────────────────────────────────────

def build_game_recap_chunk(game: Dict) -> Optional[MLBChunk]:
    """
    Convert a raw game dict into a broadcast-style recap chunk.

    This is the core text that gets embedded. We write it in natural
    language so the sentence transformer captures semantic meaning,
    not just stat tokens.
    """
    try:
        status = game.get("status", {}).get("detailedState", "Unknown")
        teams = game.get("teams", {})
        away = teams.get("away", {})
        home = teams.get("home", {})

        away_name = away.get("team", {}).get("name", "Unknown")
        home_name = home.get("team", {}).get("name", "Unknown")
        away_score = away.get("score", "?")
        home_score = home.get("score", "?")

        game_date = game.get("_fetched_date", game.get("gameDate", "")[:10])
        game_pk = game.get("gamePk", "")

        # Determine winner / loser for natural language
        if isinstance(away_score, int) and isinstance(home_score, int) and status == "Final":
            if away_score > home_score:
                winner, loser = away_name, home_name
                win_score, loss_score = away_score, home_score
            else:
                winner, loser = home_name, away_name
                win_score, loss_score = home_score, away_score
            outcome_text = (
                f"The {winner} defeated the {loser} {win_score}-{loss_score} "
                f"on {game_date}."
            )
        else:
            outcome_text = (
                f"The {away_name} and {home_name} played on {game_date}. "
                f"Score: {away_name} {away_score}, {home_name} {home_score}. "
                f"Status: {status}."
            )

        # Pull linescore details if available
        linescore = game.get("linescore", {})
        innings = linescore.get("innings", [])
        inning_summary = ""
        if innings:
            inning_summary = f" The game went {len(innings)} innings."

        # Pull winning/losing pitcher decisions if available
        decisions = game.get("decisions", {})
        pitcher_text = ""
        if decisions:
            wp = decisions.get("winner", {}).get("fullName", "")
            lp = decisions.get("loser", {}).get("fullName", "")
            sv = decisions.get("save", {}).get("fullName", "")
            if wp:
                pitcher_text += f" Winning pitcher: {wp}."
            if lp:
                pitcher_text += f" Losing pitcher: {lp}."
            if sv:
                pitcher_text += f" Save: {sv}."

        # Hits and errors (already in linescore)
        ls_teams = linescore.get("teams", {})
        total_hits = (ls_teams.get("home", {}).get("hits", 0) +
                      ls_teams.get("away", {}).get("hits", 0))
        total_errors = (ls_teams.get("home", {}).get("errors", 0) +
                        ls_teams.get("away", {}).get("errors", 0))
        hits_text = ""
        if total_hits:
            hits_text = f" The teams combined for {total_hits} hits and {total_errors} errors."

        # HR leaders and winning team SO from direct boxscore
        direct_bs = game.get("_direct_boxscore", {})
        hr_leaders = _extract_hr_leaders(direct_bs)
        hr_text = f" Home run leaders: {', '.join(hr_leaders)}." if hr_leaders else ""

        # Winning team total SO (proxy for pitching dominance)
        so_text = ""
        if status == "Final" and isinstance(away_score, int) and isinstance(home_score, int):
            winning_side = "home" if home_score > away_score else "away"
            winning_so = float(
                direct_bs.get("teams", {})
                         .get(winning_side, {})
                         .get("teamStats", {})
                         .get("pitching", {})
                         .get("strikeOuts", 0)
            )
            if winning_so >= 8:
                winner_name = home_name if winning_side == "home" else away_name
                so_text = f" The {winner_name} struck out {int(winning_so)} batters."

        full_text = outcome_text + inning_summary + pitcher_text + hits_text + hr_text + so_text

        # Prepend MLB.com editorial headline + blurb when available
        editorial = game.get("_editorial", {})
        headline = editorial.get("headline", "")
        blurb = (editorial.get("blurb", "") or "")[:300].strip()
        if headline:
            prefix = f"{headline} {blurb}".strip() if blurb else headline
            full_text = prefix + "\n\n" + full_text

        # Inject game features into metadata for classifier reranking
        game_feats = extract_game_features(game)
        feat_dict = {}
        if game_feats is not None:
            feat_dict = dict(zip(game_feats.feature_names(), game_feats.to_numpy().tolist()))

        return MLBChunk(
            text=full_text.strip(),
            metadata={
                "game_pk": game_pk,
                "status": status,
                **feat_dict,
            },
            chunk_type="game_recap"
        )

    except Exception as e:
        print(f"[Chunker] Failed to build recap chunk: {e}")
        return None


def build_standings_chunk(record: Dict) -> Optional[MLBChunk]:
    """
    Convert a division standings record into a natural language chunk.
    """
    try:
        division = record.get("division", {}).get("name", "Unknown Division")
        team_records = record.get("teamRecords", [])

        lines = [f"Current standings in the {division}:"]
        for tr in team_records:
            team_name = tr.get("team", {}).get("name", "?")
            wins = tr.get("wins", "?")
            losses = tr.get("losses", "?")
            pct = tr.get("winningPercentage", "?")
            gb = tr.get("gamesBack", "0")
            lines.append(
                f"  {team_name}: {wins}-{losses} ({pct} win pct), {gb} games back."
            )

        text = "\n".join(lines)
        return MLBChunk(
            text=text,
            metadata={"division": division, "type": "standings"},
            chunk_type="standings"
        )
    except Exception as e:
        print(f"[Chunker] Failed to build standings chunk: {e}")
        return None


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def ingest_mlb_data(days_back: int = 3) -> List[MLBChunk]:
    """
    Full ingestion pipeline: fetch → structure → chunk.

    Args:
        days_back: How many days of game history to pull.

    Returns:
        List of MLBChunk objects ready for embedding.
    """
    chunks = []

    # 1. Recent game recaps
    print(f"\n[Ingestion] Fetching games from last {days_back} days...")
    games = fetch_recent_games(days_back=days_back)
    for game in games:
        chunk = build_game_recap_chunk(game)
        if chunk:
            chunks.append(chunk)
    print(f"[Ingestion] Built {len(chunks)} game recap chunks")

    # 2. Current standings
    print("[Ingestion] Fetching standings...")
    standings = fetch_standings()
    for record in standings:
        chunk = build_standings_chunk(record)
        if chunk:
            chunks.append(chunk)
    print(f"[Ingestion] Built {len(standings)} standings chunks")

    print(f"\n[Ingestion] Total chunks ready for embedding: {len(chunks)}")
    return chunks


# ── Mock Data (for offline dev / testing) ─────────────────────────────────────

def get_mock_chunks() -> List[MLBChunk]:
    """
    Returns realistic mock MLB chunks for pipeline testing when
    the MLB API is unavailable (e.g., CI, sandboxed environments).
    Replace with ingest_mlb_data() in production.
    """
    return [
        MLBChunk(
            text="The New York Yankees defeated the Boston Red Sox 7-3 on April 18, 2026. Winning pitcher: Gerrit Cole. Losing pitcher: Brayan Bello.",
            metadata={"date": "2026-04-18", "away_team": "Boston Red Sox", "home_team": "New York Yankees", "away_score": 3, "home_score": 7, "status": "Final"},
            chunk_type="game_recap"
        ),
        MLBChunk(
            text="The Los Angeles Dodgers defeated the San Francisco Giants 5-2 on April 18, 2026. Winning pitcher: Tyler Glasnow. Losing pitcher: Logan Webb. Save: Blake Treinen.",
            metadata={"date": "2026-04-18", "away_team": "San Francisco Giants", "home_team": "Los Angeles Dodgers", "away_score": 2, "home_score": 5, "status": "Final"},
            chunk_type="game_recap"
        ),
        MLBChunk(
            text="The Atlanta Braves defeated the New York Mets 4-3 in 10 innings on April 18, 2026. The game went 10 innings. Winning pitcher: A.J. Minter. Losing pitcher: Edwin Diaz.",
            metadata={"date": "2026-04-18", "away_team": "Atlanta Braves", "home_team": "New York Mets", "away_score": 4, "home_score": 3, "status": "Final"},
            chunk_type="game_recap"
        ),
        MLBChunk(
            text="The Houston Astros and Texas Rangers played on April 18, 2026. Score: Houston Astros 6, Texas Rangers 1. Status: Final. Winning pitcher: Framber Valdez. Losing pitcher: Nathan Eovaldi.",
            metadata={"date": "2026-04-18", "away_team": "Texas Rangers", "home_team": "Houston Astros"},
            chunk_type="game_recap"
        ),
        MLBChunk(
            text="The Philadelphia Phillies defeated the Miami Marlins 9-1 on April 17, 2026. Winning pitcher: Zack Wheeler. Losing pitcher: Sandy Alcantara.",
            metadata={"date": "2026-04-17", "away_team": "Miami Marlins", "home_team": "Philadelphia Phillies"},
            chunk_type="game_recap"
        ),
        MLBChunk(
            text="Current standings in the AL East:\n  New York Yankees: 14-6 (0.700 win pct), 0.0 games back.\n  Baltimore Orioles: 11-9 (0.550 win pct), 3.0 games back.\n  Boston Red Sox: 10-10 (0.500 win pct), 4.0 games back.\n  Toronto Blue Jays: 9-11 (0.450 win pct), 5.0 games back.\n  Tampa Bay Rays: 8-12 (0.400 win pct), 6.0 games back.",
            metadata={"division": "American League East", "type": "standings"},
            chunk_type="standings"
        ),
        MLBChunk(
            text="Current standings in the NL West:\n  Los Angeles Dodgers: 15-5 (0.750 win pct), 0.0 games back.\n  San Diego Padres: 12-8 (0.600 win pct), 3.0 games back.\n  San Francisco Giants: 10-10 (0.500 win pct), 5.0 games back.\n  Arizona Diamondbacks: 9-11 (0.450 win pct), 6.0 games back.\n  Colorado Rockies: 4-16 (0.200 win pct), 11.0 games back.",
            metadata={"division": "National League West", "type": "standings"},
            chunk_type="standings"
        ),
        MLBChunk(
            text="Current standings in the NL East:\n  Philadelphia Phillies: 13-7 (0.650 win pct), 0.0 games back.\n  Atlanta Braves: 12-8 (0.600 win pct), 1.0 games back.\n  New York Mets: 11-9 (0.550 win pct), 2.0 games back.\n  Washington Nationals: 7-13 (0.350 win pct), 6.0 games back.\n  Miami Marlins: 5-15 (0.250 win pct), 8.0 games back.",
            metadata={"division": "National League East", "type": "standings"},
            chunk_type="standings"
        ),
    ]


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try live API first, fall back to mock
    chunks = ingest_mlb_data(days_back=2)
    if not chunks:
        print("\n[Dev Mode] MLB API unavailable — using mock data")
        chunks = get_mock_chunks()

    print("\n── Sample Chunks ──")
    for chunk in chunks[:5]:
        print(f"\n[{chunk.chunk_type.upper()}]")
        print(chunk.text)
        print(f"Metadata: {chunk.metadata}")
