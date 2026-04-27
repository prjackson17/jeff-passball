"""
historical_data.py

Pulls a full season of MLB game data from the Stats API and converts
each game into a fixed-length feature vector for classifier training.

Each game → one row of features → one binary label (notable / routine)

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""
import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


BASE_URL = "https://statsapi.mlb.com/api/v1"

# ── Season date ranges ─────────────────────────────────────────────────────────
SEASON_DATES = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", "2026-09-30"),
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

    # Text fields — not part of the feature vector, stored separately
    recap_text: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None

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


def _extract_hr_leaders_text(boxscore: dict) -> str:
    """Return a comma-separated string of 'Player (N HR)' for any batter with HRs."""
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
    return ", ".join(parts)


def _build_recap_text(
    game: dict,
    direct_boxscore: dict,
    home_score: float,
    away_score: float,
    innings_played: float,
    total_hits: float,
    total_errors: float,
    home_so: float,
    away_so: float,
) -> str:
    """Build an enriched natural-language recap from already-extracted values."""
    teams = game.get("teams", {})
    home_name = teams.get("home", {}).get("team", {}).get("name", "") or "Home Team"
    away_name = teams.get("away", {}).get("team", {}).get("name", "") or "Away Team"
    date = game.get("gameDate", "")[:10]

    if home_score > away_score:
        winner, loser = home_name, away_name
        win_sc, loss_sc = int(home_score), int(away_score)
        winning_so = home_so
    else:
        winner, loser = away_name, home_name
        win_sc, loss_sc = int(away_score), int(home_score)
        winning_so = away_so

    text = f"The {winner} defeated the {loser} {win_sc}-{loss_sc} on {date}."

    if innings_played > 9:
        text += f" The game went {int(innings_played)} innings."

    decisions = game.get("decisions", {})
    if decisions:
        wp = decisions.get("winner", {}).get("fullName", "")
        lp = decisions.get("loser", {}).get("fullName", "")
        sv = decisions.get("save", {}).get("fullName", "")
        if wp:
            text += f" Winning pitcher: {wp}."
        if lp:
            text += f" Losing pitcher: {lp}."
        if sv:
            text += f" Save: {sv}."

    if total_hits > 0:
        text += f" The teams combined for {int(total_hits)} hits and {int(total_errors)} errors."

    hr_text = _extract_hr_leaders_text(direct_boxscore)
    if hr_text:
        text += f" Home run leaders: {hr_text}."

    if winning_so >= 8:
        text += f" The {winner} struck out {int(winning_so)} batters."

    return text.strip()


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

    # Boxscore — prefer the direct /game/{pk}/boxscore response (has teamStats)
    # over the schedule hydration (which omits teamStats)
    direct_boxscore = game.get("_direct_boxscore", game.get("boxscore", {}))
    home_hrs, away_hrs, home_so, away_so = _extract_boxscore_features(direct_boxscore)
    total_hrs = home_hrs + away_hrs

    # Derived indicators
    is_extra_innings = 1.0 if innings_played > 9 else 0.0
    winner_is_home = home_score > away_score
    loser_score = away_score if winner_is_home else home_score
    is_shutout = 1.0 if loser_score == 0 else 0.0

    # Assign SO to winning/losing pitcher side
    winning_pitcher_so = home_so if winner_is_home else away_so
    losing_pitcher_so = away_so if winner_is_home else home_so

    # Team names and recap text
    game_teams = game.get("teams", {})
    home_team = game_teams.get("home", {}).get("team", {}).get("name", "")
    away_team = game_teams.get("away", {}).get("team", {}).get("name", "")
    recap_text = _build_recap_text(
        game, direct_boxscore,
        home_score, away_score,
        innings_played, total_hits, total_errors,
        home_so, away_so,
    )

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
        recap_text=recap_text,
        home_team=home_team,
        away_team=away_team,
    )


# ── Season Fetcher ─────────────────────────────────────────────────────────────

def fetch_date_range(start: str, end: str, verbose: bool = True) -> List[GameFeatures]:
    """
    Fetch all Final games between two dates (inclusive) and extract features.

    Args:
        start: 'YYYY-MM-DD' start date.
        end:   'YYYY-MM-DD' end date (capped at yesterday — today's games aren't Final yet).
        verbose: Print progress.

    Returns:
        List of GameFeatures, one per completed game.
    """
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    end = min(end, yesterday)  # never ask for games that haven't finished yet

    dates = _date_range(start, end)
    all_features = []
    print(f"[Historical] Fetching {start} → {end} ({len(dates)} dates)...")

    for i, date in enumerate(dates):
        if verbose and i % 30 == 0:
            print(f"  {date}  ({i}/{len(dates)} dates, {len(all_features)} games)")

        data = _get("/schedule", params={
            "sportId": 1,
            "date": date,
            "hydrate": "linescore,boxscore,decisions"
        })

        if not data or "dates" not in data or not data["dates"]:
            continue

        for game in data["dates"][0].get("games", []):
            status = game.get("status", {}).get("detailedState", "")
            if status == "Final":
                game_pk = game.get("gamePk")
                direct_bs = _get(f"/game/{game_pk}/boxscore") or {}
                game["_direct_boxscore"] = direct_bs
                time.sleep(0.05)   # avoid rate-limiting (~2-3 hrs per full rebuild)
            features = extract_game_features(game)
            if features:
                all_features.append(features)

    print(f"[Historical] Done. {len(all_features)} completed games.")
    return all_features


def fetch_season(season: int = 2024, verbose: bool = True) -> List[GameFeatures]:
    """Fetch all Final games for a full MLB season."""
    if season not in SEASON_DATES:
        raise ValueError(f"Season {season} not in SEASON_DATES. Add it or use fetch_date_range().")
    start, end = SEASON_DATES[season]
    print(f"[Historical] Season {season}: {start} → {end}")
    return fetch_date_range(start, end, verbose=verbose)


def fetch_multiple_seasons(seasons: List[int] = [2023, 2024, 2025, 2026]) -> List[GameFeatures]:
    all_features = []
    for season in seasons:
        all_features.extend(fetch_season(season, verbose=True))
    print(f"\n[Historical] Combined total: {len(all_features)} games across {seasons}")
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
    """Save feature matrix, metadata, and recap text to compressed numpy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    X = np.stack([f.to_numpy() for f in features])
    game_pks = np.array([f.game_pk for f in features])
    dates = np.array([f.date for f in features])
    recap_texts = np.array([f.recap_text or "" for f in features])
    home_teams = np.array([f.home_team or "" for f in features])
    away_teams = np.array([f.away_team or "" for f in features])

    np.savez_compressed(
        path, X=X, game_pks=game_pks, dates=dates,
        recap_texts=recap_texts, home_teams=home_teams, away_teams=away_teams,
    )
    print(f"[Save] Saved {len(features)} games to {path}")


def append_features(
    new_features: List[GameFeatures],
    path: str = "./data/game_features_all.npz",
) -> int:
    """
    Merge new_features into an existing .npz, deduplicating by game_pk.

    Args:
        new_features: Freshly fetched GameFeatures to add.
        path:         Path to the existing .npz (created if absent).

    Returns:
        Number of new rows actually added (after deduplication).
    """
    if not new_features:
        print("[Append] No new features to add.")
        return 0

    new_X           = np.stack([f.to_numpy() for f in new_features])
    new_pks         = np.array([f.game_pk for f in new_features])
    new_dates       = np.array([f.date for f in new_features])
    new_recap_texts = np.array([f.recap_text or "" for f in new_features])
    new_home_teams  = np.array([f.home_team or "" for f in new_features])
    new_away_teams  = np.array([f.away_team or "" for f in new_features])

    if os.path.exists(path):
        old_X, old_pks, old_dates, old_recap_texts, old_home_teams, old_away_teams = load_features(path)
        existing_pks = set(old_pks.tolist())

        mask = np.array([pk not in existing_pks for pk in new_pks.tolist()])
        added = int(mask.sum())

        if added == 0:
            print(f"[Append] All {len(new_features)} games already present — nothing added.")
            return 0

        X           = np.concatenate([old_X,           new_X[mask]],           axis=0)
        pks         = np.concatenate([old_pks,         new_pks[mask]],         axis=0)
        dates       = np.concatenate([old_dates,       new_dates[mask]],       axis=0)
        recap_texts = np.concatenate([old_recap_texts, new_recap_texts[mask]], axis=0)
        home_teams  = np.concatenate([old_home_teams,  new_home_teams[mask]],  axis=0)
        away_teams  = np.concatenate([old_away_teams,  new_away_teams[mask]],  axis=0)
    else:
        X, pks, dates = new_X, new_pks, new_dates
        recap_texts, home_teams, away_teams = new_recap_texts, new_home_teams, new_away_teams
        added = len(new_features)

    np.savez_compressed(
        path, X=X, game_pks=pks, dates=dates,
        recap_texts=recap_texts, home_teams=home_teams, away_teams=away_teams,
    )
    print(f"[Append] Added {added} new games → {len(pks)} total in {path}")
    return added


def load_features(
    path: str = "./data/game_features.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load saved features. Returns (X, game_pks, dates, recap_texts, home_teams, away_teams).

    recap_texts/home_teams/away_teams fall back to empty-string arrays for
    files saved before these fields were added.
    """
    data = np.load(path, allow_pickle=True)
    n = len(data["X"])
    recap_texts = data["recap_texts"] if "recap_texts" in data else np.array([""] * n)
    home_teams  = data["home_teams"]  if "home_teams"  in data else np.array([""] * n)
    away_teams  = data["away_teams"]  if "away_teams"  in data else np.array([""] * n)
    return data["X"], data["game_pks"], data["dates"], recap_texts, home_teams, away_teams


def load_features_as_objects(path: str = "./data/game_features.npz") -> List[GameFeatures]:
    """Load saved features as a list of GameFeatures objects (including text fields)."""
    X, pks, dates, recap_texts, home_teams, away_teams = load_features(path)
    fn = GameFeatures.feature_names()
    result = []
    for i, row in enumerate(X):
        gf = GameFeatures(
            game_pk=int(pks[i]),
            date=str(dates[i]),
            home_team=str(home_teams[i]) or None,
            away_team=str(away_teams[i]) or None,
            recap_text=str(recap_texts[i]) or None,
            **dict(zip(fn, row.tolist()))
        )
        result.append(gf)
    return result


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
    import argparse

    parser = argparse.ArgumentParser(description="MLB historical game feature fetcher")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--append", type=int, metavar="SEASON",
        help="Fetch new games for SEASON and merge into game_features_all.npz. "
             "Only fetches dates not already present.",
    )
    group.add_argument(
        "--rebuild", action="store_true",
        help="Re-fetch all seasons (2023-2026) and overwrite game_features_all.npz.",
    )
    group.add_argument(
        "--range", nargs=2, metavar=("START", "END"),
        help="Fetch a custom date range (YYYY-MM-DD YYYY-MM-DD) and append.",
    )
    parser.add_argument(
        "--out", default="./data/game_features_all.npz",
        help="Output .npz path (default: ./data/game_features_all.npz)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.rebuild:
        features = fetch_multiple_seasons([2023, 2024, 2025, 2026])
        save_features(features, path=args.out)
        print(features_to_dataframe(features).groupby(
            lambda i: features[i].date[:4]
        ).size().rename("games"))

    elif args.append:
        season = args.append
        if season not in SEASON_DATES:
            raise SystemExit(f"Season {season} not in SEASON_DATES: {list(SEASON_DATES.keys())}")

        start, end = SEASON_DATES[season]

        # Fast-forward start to day after the latest date already in the file for this season.
        if os.path.exists(args.out):
            _, _, existing_dates, *_ = load_features(args.out)
            season_dates_in_file = sorted(
                d for d in existing_dates.tolist() if str(d).startswith(str(season))
            )
            if season_dates_in_file:
                latest = season_dates_in_file[-1]
                next_day = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                if next_day > end:
                    print(f"[Append] Already up to date through {latest}. Nothing to fetch.")
                    raise SystemExit(0)
                print(f"[Append] Latest {season} date on file: {latest}. Fetching {next_day} → {end}")
                start = next_day

        new_features = fetch_date_range(start, end)
        append_features(new_features, path=args.out)

    else:  # --range
        start, end = args.range
        new_features = fetch_date_range(start, end)
        append_features(new_features, path=args.out)
