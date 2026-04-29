"""
novelty.py

Generates "crazy fact" novelty strings for recent MLB games.

Two tiers:
  1. Dataset facts  — percentile checks against the local 7,786-game historical
     dataset.  Pure numpy, no API calls.  E.g. "Only 9 of 7,786 games had this
     many combined runs."

  2. Season-high facts — one MLB Stats API game-log call per notable pitcher to
     check whether tonight's strikeout total is a season high.

Public API:
  generate_game_facts(chunk, X_hist, feature_names)
      → List[str]  (facts for a single game chunk)

  generate_briefing_facts(game_chunks, X_hist, feature_names, top_n=3)
      → str  (formatted block ready to inject into a briefing prompt)

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import time
import requests
import numpy as np
from datetime import datetime
from typing import List, Optional

from src.mlb_rag.data_ingestion import MLBChunk

MLB_BASE = "https://statsapi.mlb.com/api/v1"


# ── MLB API helper ─────────────────────────────────────────────────────────────

def _mlb_get(endpoint: str, params: dict = None) -> Optional[dict]:
    try:
        r = requests.get(MLB_BASE + endpoint, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Tier 1: dataset percentile facts ──────────────────────────────────────────

# Only report a dataset fact if fewer than this fraction of historical games
# matched or exceeded the value.  Keeps facts genuinely surprising.
_RARITY_THRESHOLD = 0.03   # top 3% or rarer


def _reliable_col(col: np.ndarray) -> bool:
    """Return False when a column is dominated by zeros (hydration gaps in old data)."""
    return float((col == 0).mean()) < 0.5


def _dataset_facts(meta: dict, X_hist: np.ndarray, fn: List[str]) -> List[str]:
    """Compare one game's stats against the historical dataset."""
    facts = []
    n = len(X_hist)

    def col(name: str) -> np.ndarray:
        return X_hist[:, fn.index(name)]

    total_runs         = meta.get("total_runs", 0) or 0
    total_hrs          = meta.get("total_hrs", 0) or 0
    winning_pitcher_so = meta.get("winning_pitcher_so", 0) or 0
    innings            = meta.get("innings_played", 9) or 9
    margin             = meta.get("margin", 0) or 0

    c_runs = col("total_runs")
    if total_runs > 0 and _reliable_col(c_runs):
        count = int((c_runs >= total_runs).sum())
        if count / n <= _RARITY_THRESHOLD:
            facts.append(
                f"Combined {int(total_runs)} runs — only {count} of {n:,} games since 2023 "
                f"matched or exceeded this total."
            )

    c_hrs = col("total_hrs")
    if total_hrs > 0 and _reliable_col(c_hrs):
        count = int((c_hrs >= total_hrs).sum())
        if count / n <= _RARITY_THRESHOLD:
            facts.append(
                f"{int(total_hrs)} home runs in one game — happened only {count} times "
                f"across {n:,} games since 2023."
            )

    c_so = col("winning_pitcher_so")
    if winning_pitcher_so > 0 and _reliable_col(c_so):
        count = int((c_so >= winning_pitcher_so).sum())
        if count / n <= _RARITY_THRESHOLD:
            facts.append(
                f"The winning pitcher struck out {int(winning_pitcher_so)} batters — "
                f"only {count} of {n:,} games saw a winning pitcher reach this total."
            )

    c_inn = col("innings_played")
    if innings > 9 and _reliable_col(c_inn):
        count = int((c_inn >= innings).sum())
        if count / n <= _RARITY_THRESHOLD:
            facts.append(
                f"Went {int(innings)} innings — only {count} of {n:,} games went this long."
            )

    c_margin = col("margin")
    if margin > 0 and _reliable_col(c_margin):
        count = int((c_margin >= margin).sum())
        if count / n <= _RARITY_THRESHOLD:
            facts.append(
                f"Final margin of {int(margin)} runs — only {count} games in the dataset "
                f"were this lopsided."
            )

    return facts


# ── Tier 2: season-high facts via MLB Stats API ───────────────────────────────

def _pitcher_season_gamelog_so(pitcher_id: int) -> List[int]:
    """Return list of single-game strikeout totals for this pitcher's season."""
    season = datetime.today().year
    data   = _mlb_get(f"/people/{pitcher_id}/stats", params={
        "stats": "gameLog", "group": "pitching", "season": season,
    })
    if not data:
        return []
    splits = (data.get("stats") or [{}])[0].get("splits", [])
    return [int(s.get("stat", {}).get("strikeOuts", 0)) for s in splits]


def _pitcher_season_facts(pitcher_id: int, pitcher_name: str, so_tonight: int) -> List[str]:
    """Check if tonight's SO total is a season high for this pitcher."""
    if so_tonight < 8:
        return []

    log = _pitcher_season_gamelog_so(pitcher_id)
    time.sleep(0.1)

    if not log:
        return []

    season_high = max(log)
    prev_games  = log[:-1]  # exclude tonight (last entry in log)
    prev_high   = max(prev_games) if prev_games else 0

    facts = []
    if so_tonight >= season_high and so_tonight > prev_high:
        facts.append(
            f"{pitcher_name} set a new 2026 season high with {so_tonight} strikeouts tonight "
            f"(previous best: {prev_high})."
        )
    elif so_tonight >= season_high:
        facts.append(
            f"{pitcher_name}'s {so_tonight} strikeouts matches their 2026 season high."
        )
    elif len(prev_games) > 0:
        avg = np.mean(prev_games)
        if avg > 0 and so_tonight > avg * 1.5:
            facts.append(
                f"{pitcher_name} struck out {so_tonight} tonight — "
                f"well above their 2026 average of {avg:.1f} Ks per outing."
            )

    return facts


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_game_facts(
        chunk: MLBChunk,
        X_hist: np.ndarray,
        feature_names: List[str],
        include_api_facts: bool = True,
) -> List[str]:
    """
    Generate novelty fact strings for a single game chunk.

    Args:
        chunk:            A game_recap MLBChunk with metadata populated.
        X_hist:           Historical feature matrix from load_features().
        feature_names:    GameFeatures.feature_names() list.
        include_api_facts: If True, make one MLB API call to check pitcher season highs.

    Returns:
        List of human-readable fact strings (may be empty for routine games).
    """
    if chunk.chunk_type != "game_recap":
        return []

    meta  = chunk.metadata
    facts = _dataset_facts(meta, X_hist, list(feature_names))

    if include_api_facts:
        pitcher_id   = meta.get("winning_pitcher_id")
        pitcher_name = meta.get("winning_pitcher_name", "The winning pitcher")
        so_tonight   = int(meta.get("winning_pitcher_so", 0) or 0)
        if pitcher_id and so_tonight >= 8:
            facts += _pitcher_season_facts(int(pitcher_id), pitcher_name, so_tonight)

    return facts


def generate_briefing_facts(
        game_chunks: List[MLBChunk],
        X_hist: np.ndarray,
        feature_names: List[str],
        top_n: int = 3,
) -> str:
    """
    Generate a formatted novelty-facts block for use in a briefing prompt.

    Runs dataset facts (fast, no API) on all games.  Runs season-high API
    facts only on the top_n most statistically extreme games to keep
    latency reasonable.

    Args:
        game_chunks:   List of game_recap MLBChunk objects.
        X_hist:        Historical feature matrix.
        feature_names: GameFeatures.feature_names() list.
        top_n:         How many games get the API-based season-high check.

    Returns:
        A formatted string like:
            NOTABLE FACTS:
            • Combined 23 runs — only 4 of 7,786 games since 2023 matched this.
            • Cole struck out 14 batters — new 2026 season high (previous: 12).
        Or an empty string if no facts were found.
    """
    fn = list(feature_names)

    def _extremeness(chunk: MLBChunk) -> float:
        meta = chunk.metadata
        return (
            (meta.get("total_runs", 0) or 0) * 0.4
            + (meta.get("winning_pitcher_so", 0) or 0) * 0.3
            + (meta.get("total_hrs", 0) or 0) * 0.2
            + (meta.get("innings_played", 9) or 9) * 0.1
        )

    sorted_chunks = sorted(game_chunks, key=_extremeness, reverse=True)

    all_facts: List[str] = []

    # Dataset facts on every game (fast)
    for chunk in sorted_chunks:
        all_facts += _dataset_facts(chunk.metadata, X_hist, fn)

    # API facts only for the most extreme games
    for chunk in sorted_chunks[:top_n]:
        pitcher_id   = chunk.metadata.get("winning_pitcher_id")
        pitcher_name = chunk.metadata.get("winning_pitcher_name", "The winning pitcher")
        so_tonight   = int(chunk.metadata.get("winning_pitcher_so", 0) or 0)
        if pitcher_id and so_tonight >= 8:
            all_facts += _pitcher_season_facts(int(pitcher_id), pitcher_name, so_tonight)

    if not all_facts:
        return ""

    bullet_lines = "\n".join(f"  • {f}" for f in all_facts)
    return f"NOTABLE FACTS:\n{bullet_lines}"
