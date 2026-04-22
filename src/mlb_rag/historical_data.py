"""
historical_data.py

Pulls a full season of MLB game data from the Stats API and converts
each game into a fixed-length feature vector for classifier training.

Each game → one row of features → one binary label (notable / routine)

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


BASE_URL = "https://statsapi.mlb.com/api/v1"

# ── Season date ranges ─────────────────────────────────────────────────────────
SEASON_DATES = {
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}


# ── Data Structure ─────────────────────────────────────────────────────────────

@dataclass
class GameFeatures:
    """
    Fixed-length feature vector extracted from one MLB game.
    These are the inputs to the trend classifier.

    Feature engineering choices are deliberate:
    - margin captures competitiveness
    - total_runs captures offensive environment
    - innings captures game length (extra innings = notable)
    - individual performer stats capture standout individual games
    """
    game_pk: int
    date: str

    # Score features
    home_score: float = 0.0
    away_score: float = 0.0
    margin: float = 0.0           # abs(home - away)
    total_runs: float = 0.0       # home + away

    # Game length
    innings_played: float = 9.0   # > 9 means extra innings

    # Pitching features
    winning_pitcher_so: float = 0.0    # strikeouts by winning pitcher
    losing_pitcher_so: float = 0.0
    total_hits: float = 0.0
    total_errors: float = 0.0

    # Home run features
    home_hrs: float = 0.0
    away_hrs: float = 0.0
    total_hrs: float = 0.0

    # Derived binary indicators (still stored as float for tensor compatibility)
    is_extra_innings: float = 0.0     # 1 if innings > 9
    is_shutout: float = 0.0           # 1 if loser scored 0
    had_lead_change: float = 0.0      # 1 if lead changed hands (from linescore)

    def to_numpy(self) -> np.ndarray:
        """Convert to float32 numpy array for PyTorch ingestion."""
        return np.array([
            self.home_score,
            self.away_score,
            self.margin,
            self.total_runs,
            self.innings_played,
            self.winning_pitcher_so,
            self.losing_pitcher_so,
            self.total_hits,
            self.total_errors,
            self.home_hrs,
            self.away_hrs,
            self.total_hrs,
            self.is_extra_innings,
            self.is_shutout,
            self.had_lead_change,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "home_score", "away_score", "margin", "total_runs",
            "innings_played", "winning_pitcher_so", "losing_pitcher_so",
            "total_hits", "total_errors", "home_hrs", "away_hrs",
            "total_hrs", "is_extra_innings", "is_shutout", "had_lead_change"
        ]

    @staticmethod
    def num_features() -> int:
        return 15


# ── API Helpers ────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> Optional[dict]:
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [API] Error {endpoint}: {e}")
        return None


def _date_range(start: str, end: str) -> List[str]:
    """Generate list of 'YYYY-MM-DD' strings between start and end inclusive."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    cur = start_dt
    while cur <= end_dt:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


# ── Feature Extraction ─────────────────────────────────────────────────────────

def _extract_linescore_features(linescore: dict) -> Tuple[float, float, float]:
    """
    Extract innings played, total hits, total errors from linescore.
    Returns: (innings_played, total_hits, total_errors)
    """
    innings = linescore.get("innings", [])
    innings_played = float(len(innings)) if innings else 9.0

    teams = linescore.get("teams", {})
    home_hits = teams.get("home", {}).get("hits", 0)
    away_hits = teams.get("away", {}).get("hits", 0)
    home_errors = teams.get("home", {}).get("errors", 0)
    away_errors = teams.get("away", {}).get("errors", 0)

    return innings_played, float(home_hits + away_hits), float(home_errors + away_errors)


def _detect_lead_change(linescore: dict) -> float:
    """
    Rough lead-change detection: scan inning-by-inning scores.
    Returns 1.0 if the lead changed hands at least once, 0.0 otherwise.
    """
    innings = linescore.get("innings", [])
    home_running = 0
    away_running = 0
    prev_leader = None
    changed = False

    for inning in innings:
        home_running += inning.get("home", {}).get("runs", 0) or 0
        away_running += inning.get("away", {}).get("runs", 0) or 0

        if home_running > away_running:
            leader = "home"
        elif away_running > home_running:
            leader = "away"
        else:
            leader = "tied"

        if prev_leader is not None and leader != prev_leader and leader != "tied":
            changed = True
            break
        prev_leader = leader

    return 1.0 if changed else 0.0


def _extract_boxscore_features(boxscore: dict) -> Tuple[float, float, float, float]:
    """
    Extract home runs and pitcher strikeouts from boxscore.
    Returns: (home_hrs, away_hrs, winning_so, losing_so)
    """
    teams = boxscore.get("teams", {})

    def _hrs(team_data):
        batting = team_data.get("teamStats", {}).get("batting", {})
        return float(batting.get("homeRuns", 0))

    def _so(team_data):
        pitching = team_data.get("teamStats", {}).get("pitching", {})
        return float(pitching.get("strikeOuts", 0))

    home_hrs = _hrs(teams.get("home", {}))
    away_hrs = _hrs(teams.get("away", {}))

    # We don't know which team's pitcher "won" without decisions,
    # so store home/away SO and let auto_labeler sort it out
    home_so = _so(teams.get("home", {}))
    away_so = _so(teams.get("away", {}))

    return home_hrs, away_hrs, home_so, away_so


def extract_game_features(game: dict) -> Optional[GameFeatures]:
    """
    Convert a raw MLB API game dict into a GameFeatures object.
    Returns None if the game is incomplete or not Final.
    """
    status = game.get("status", {}).get("detailedState", "")
    if status != "Final":
        return None

    teams = game.get("teams", {})
    home_score = float(teams.get("home", {}).get("score", 0) or 0)
    away_score = float(teams.get("away", {}).get("score", 0) or 0)
    margin = abs(home_score - away_score)
    total_runs = home_score + away_score

    game_pk = game.get("gamePk", 0)
    date = game.get("gameDate", "")[:10]

    # Linescore
    linescore = game.get("linescore", {})
    innings_played, total_hits, total_errors = _extract_linescore_features(linescore)
    had_lead_change = _detect_lead_change(linescore)

    # Boxscore
    boxscore = game.get("boxscore", {})
    home_hrs, away_hrs, home_so, away_so = _extract_boxscore_features(boxscore)
    total_hrs = home_hrs + away_hrs

    # Derived indicators
    is_extra_innings = 1.0 if innings_played > 9 else 0.0
    winner_is_home = home_score > away_score
    loser_score = away_score if winner_is_home else home_score
    is_shutout = 1.0 if loser_score == 0 else 0.0

    # Assign SO to winning/losing pitcher side
    winning_pitcher_so = home_so if winner_is_home else away_so
    losing_pitcher_so = away_so if winner_is_home else home_so

    return GameFeatures(
        game_pk=game_pk,
        date=date,
        home_score=home_score,
        away_score=away_score,
        margin=margin,
        total_runs=total_runs,
        innings_played=innings_played,
        winning_pitcher_so=winning_pitcher_so,
        losing_pitcher_so=losing_pitcher_so,
        total_hits=total_hits,
        total_errors=total_errors,
        home_hrs=home_hrs,
        away_hrs=away_hrs,
        total_hrs=total_hrs,
        is_extra_innings=is_extra_innings,
        is_shutout=is_shutout,
        had_lead_change=had_lead_change,
    )


# ── Season Fetcher ─────────────────────────────────────────────────────────────

def fetch_season(season: int = 2024, verbose: bool = True) -> List[GameFeatures]:
    """
    Fetch all Final games for a full MLB season and extract features.

    Args:
        season: Year to fetch (2024 or 2025).
        verbose: Print progress.

    Returns:
        List of GameFeatures, one per completed game.
    """
    if season not in SEASON_DATES:
        raise ValueError(f"Season {season} not supported. Choose from {list(SEASON_DATES.keys())}")

    start, end = SEASON_DATES[season]
    dates = _date_range(start, end)

    all_features = []
    print(f"[Historical] Fetching {season} season ({len(dates)} dates)...")

    for i, date in enumerate(dates):
        if verbose and i % 30 == 0:
            print(f"  Progress: {date} ({i}/{len(dates)} dates, {len(all_features)} games so far)")

        data = _get("/schedule", params={
            "sportId": 1,
            "date": date,
            "hydrate": "linescore,boxscore,decisions"
        })

        if not data or "dates" not in data or not data["dates"]:
            continue

        games = data["dates"][0].get("games", [])
        for game in games:
            features = extract_game_features(game)
            if features:
                all_features.append(features)

    print(f"[Historical] Done. {len(all_features)} completed games extracted.")
    return all_features


# ── DataFrame + Save ───────────────────────────────────────────────────────────

def features_to_dataframe(features: List[GameFeatures]) -> pd.DataFrame:
    """Convert list of GameFeatures to a pandas DataFrame for inspection."""
    rows = []
    for f in features:
        row = {"game_pk": f.game_pk, "date": f.date}
        for name, val in zip(GameFeatures.feature_names(), f.to_numpy()):
            row[name] = val
        rows.append(row)
    return pd.DataFrame(rows)


def save_features(features: List[GameFeatures], path: str = "./data/game_features.npz") -> None:
    """Save feature matrix and metadata to compressed numpy file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    X = np.stack([f.to_numpy() for f in features])
    game_pks = np.array([f.game_pk for f in features])
    dates = np.array([f.date for f in features])

    np.savez_compressed(path, X=X, game_pks=game_pks, dates=dates)
    print(f"[Save] Saved {len(features)} games to {path}")


def load_features(path: str = "./data/game_features.npz") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved features. Returns (X, game_pks, dates)."""
    data = np.load(path, allow_pickle=True)
    return data["X"], data["game_pks"], data["dates"]


# ── Mock Data (offline dev) ────────────────────────────────────────────────────

def get_mock_features(n: int = 500) -> List[GameFeatures]:
    """
    Generate synthetic GameFeatures for offline pipeline testing.
    Distributions are calibrated to realistic MLB game statistics.
    """
    np.random.seed(42)
    features = []
    for i in range(n):
        home = float(np.random.poisson(4.5))
        away = float(np.random.poisson(4.5))
        innings = 9.0 if np.random.random() > 0.08 else float(np.random.randint(10, 14))
        features.append(GameFeatures(
            game_pk=100000 + i,
            date=f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            home_score=home,
            away_score=away,
            margin=abs(home - away),
            total_runs=home + away,
            innings_played=innings,
            winning_pitcher_so=float(np.random.poisson(7)),
            losing_pitcher_so=float(np.random.poisson(5)),
            total_hits=float(np.random.poisson(16)),
            total_errors=float(np.random.poisson(0.4)),
            home_hrs=float(np.random.poisson(1.1)),
            away_hrs=float(np.random.poisson(1.1)),
            total_hrs=float(np.random.poisson(2.2)),
            is_extra_innings=1.0 if innings > 9 else 0.0,
            is_shutout=1.0 if min(home, away) == 0 else 0.0,
            had_lead_change=float(np.random.binomial(1, 0.35)),
        ))
    return features


if __name__ == "__main__":
    print("Fetching 2024 MLB season...")
    features = fetch_season(2024, verbose=True)
    if features:
        import os
        os.makedirs("./data", exist_ok=True)
        save_features(features, path="./data/game_features_2024.npz")
        df = features_to_dataframe(features)
        print(df.describe())
    else:
        print("No features fetched — check API connectivity")
