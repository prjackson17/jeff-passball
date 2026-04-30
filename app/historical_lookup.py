"""
historical_lookup.py

Augments query context with aggregate stats and record lookups from the
full game archive (data/game_features_all.npz, ~7,800 games, 2023-present).

Two modes:
  1. Aggregates  — "how many extra innings games this season / in 2024?"
  2. Record lookup — "longest game", "most runs in a game", "most strikeouts"

Both are detected via keyword matching and injected as extra context before
the LLM responds, so the model doesn't have to rely on training-data memory.
"""

import os
import re
import numpy as np
from functools import lru_cache
from datetime import datetime
from typing import Optional

_FEATURE_NAMES = [
    "home_score", "away_score", "margin", "total_runs", "innings_played",
    "winning_pitcher_so", "losing_pitcher_so", "total_hits", "total_errors",
    "home_hrs", "away_hrs", "total_hrs", "is_extra_innings", "is_shutout",
    "had_lead_change",
]
_FI = {name: i for i, name in enumerate(_FEATURE_NAMES)}

_CURRENT_YEAR = datetime.today().year


# ── Data loading (cached for process lifetime) ─────────────────────────────────

@lru_cache(maxsize=1)
def _load_npz(path: str):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "X":           d["X"],
        "dates":       d["dates"],
        "home_teams":  d["home_teams"],
        "away_teams":  d["away_teams"],
        "recap_texts": d["recap_texts"],
        "game_pks":    d["game_pks"],
    }


def _get_data():
    path = os.environ.get("DATA_PATH", "data/game_features_all.npz")
    return _load_npz(path)


def _year_mask(data, year: int) -> np.ndarray:
    """Boolean mask for rows matching a given calendar year."""
    return np.array([d[:4] == str(year) for d in data["dates"]])


def _season_label(year: int) -> str:
    return f"{year} season" if year < _CURRENT_YEAR else f"{year} season (in progress)"


# ── Query detection ────────────────────────────────────────────────────────────

_AGGREGATE_PATTERNS = [
    ({"how many", "# of", "number of", "count", "total number", "how often"},
     {"extra inning", "extras", "extra-inning", "went extra"}),
    ({"how many", "# of", "number of", "count"},
     {"shutout", "shut out", "shutouts"}),
    ({"how many", "# of", "number of", "count"},
     {"walk-off", "walkoff", "walk off"}),
    ({"how many", "# of", "number of", "count"},
     {"blowout", "run rule", "10 or more", "10+ run"}),
    ({"how many", "# of", "number of", "count", "average", "per game"},
     {"home run", "home runs", "hr", "hrs", "homers"}),
]

_RECORD_PATTERNS = [
    ({"longest", "most innings", "most extra", "marathon"},
     "innings_played", False, "longest game (most innings)"),
    ({"highest scoring", "most runs", "most total runs", "biggest offensive"},
     "total_runs", False, "highest-scoring game"),
    ({"most home runs", "most hr", "most homers", "most long balls"},
     "total_hrs", False, "most home runs in a game"),
    ({"most strikeout", "most k ", "most ks", "most punch"},
     "winning_pitcher_so", False, "most strikeouts by winning pitcher"),
    ({"biggest blowout", "largest margin", "biggest win", "most lopsided"},
     "margin", False, "biggest margin of victory"),
    ({"fewest runs", "lowest scoring", "pitcher duel"},
     "total_runs", True, "lowest-scoring game"),
]

_YEAR_RE = re.compile(r"\b(202[3-9]|20[3-9]\d)\b")


def _detect_year(query: str) -> Optional[int]:
    """Extract an explicit year from the query, or None → current season."""
    m = _YEAR_RE.search(query)
    if m:
        return int(m.group(1))
    q = query.lower()
    if "last year" in q or "previous season" in q:
        return _CURRENT_YEAR - 1
    return None


def _is_aggregate_query(query: str) -> bool:
    q = query.lower()
    for count_kws, subject_kws in _AGGREGATE_PATTERNS:
        if any(ck in q for ck in count_kws) and any(sk in q for sk in subject_kws):
            return True
    return False


def _detect_record_query(query: str) -> Optional[tuple]:
    """Return (feature_name, ascending, label) if this is a record lookup."""
    q = query.lower()
    for keywords, feat, asc, label in _RECORD_PATTERNS:
        if any(kw in q for kw in keywords):
            return feat, asc, label
    return None


def _is_historical_query(query: str) -> bool:
    q = query.lower()
    # explicit year reference to past seasons
    year = _detect_year(query)
    if year and year < _CURRENT_YEAR:
        return True
    # aggregate questions over the season
    if _is_aggregate_query(q):
        return True
    # record questions
    if _detect_record_query(q):
        return True
    return False


# ── Aggregate builders ─────────────────────────────────────────────────────────

def _count_stat(X: np.ndarray, mask: np.ndarray, feat: str) -> int:
    return int(X[mask, _FI[feat]].sum())


def _build_aggregate_lines(data, year: int) -> list[str]:
    mask = _year_mask(data, year)
    total = int(mask.sum())
    if total == 0:
        return []
    X = data["X"]
    lines = [f"SEASON AGGREGATES — {_season_label(year)} ({total} games):"]
    ei  = _count_stat(X, mask, "is_extra_innings")
    so  = _count_stat(X, mask, "is_shutout")
    lc  = _count_stat(X, mask, "had_lead_change")
    hr  = int(X[mask, _FI["total_hrs"]].sum())
    rpg = float(X[mask, _FI["total_runs"]].mean())
    lines.append(f"  Extra-innings games: {ei} ({ei/total*100:.1f}%)")
    lines.append(f"  Shutouts: {so} ({so/total*100:.1f}%)")
    lines.append(f"  Games with lead changes: {lc} ({lc/total*100:.1f}%)")
    lines.append(f"  Total home runs: {hr} (avg {hr/total:.2f}/game)")
    lines.append(f"  Avg runs per game: {rpg:.2f}")
    return lines


# ── Record lookup builder ──────────────────────────────────────────────────────

def _top_game_recap(data, mask: np.ndarray, feat: str, ascending: bool, label: str) -> str:
    if not mask.any():
        return ""
    idx_in_mask = (data["X"][mask, _FI[feat]].argmin()
                   if ascending else data["X"][mask, _FI[feat]].argmax())
    global_idx = np.where(mask)[0][idx_in_mask]
    val = data["X"][global_idx, _FI[feat]]
    date = data["dates"][global_idx]
    away = data["away_teams"][global_idx]
    home = data["home_teams"][global_idx]
    recap = data["recap_texts"][global_idx][:400]
    return (
        f"RECORD GAME — {label}:\n"
        f"  {away} @ {home} on {date} ({feat}={int(val) if val == int(val) else round(float(val),1)})\n"
        f"  {recap}"
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def augment_historical_context(query: str) -> str:
    """
    Return historical/aggregate context to prepend before RAG chunks.
    Returns empty string if the query doesn't need historical data.
    """
    if not _is_historical_query(query):
        return ""

    data = _get_data()
    if data is None:
        return ""

    parts = []
    year = _detect_year(query) or _CURRENT_YEAR

    # Season aggregates
    if _is_aggregate_query(query):
        lines = _build_aggregate_lines(data, year)
        if lines:
            parts.append("\n".join(lines))
        # Also include prior seasons for context if asking about current year
        if year == _CURRENT_YEAR:
            for prev_year in [_CURRENT_YEAR - 1, _CURRENT_YEAR - 2]:
                prev_lines = _build_aggregate_lines(data, prev_year)
                if prev_lines:
                    parts.append("\n".join(prev_lines))

    # Record lookup
    record = _detect_record_query(query)
    if record:
        feat, asc, label = record
        mask = _year_mask(data, year)
        recap = _top_game_recap(data, mask, feat, asc, label)
        if recap:
            parts.append(recap)

    return "\n\n".join(parts)
