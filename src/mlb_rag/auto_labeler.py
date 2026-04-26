"""
auto_labeler.py

Rule-based binary labeling of MLB games:
    1 = notable  (worth surfacing in a broadcast briefing)
    0 = routine  (just another game)

Rules are intentionally transparent and defensible — each one
maps to a real reason a broadcaster would flag the game.

This is also where you'd expand to multiclass later:
replace the binary OR with a priority-ordered label assignment.

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass

from src.mlb_rag.historical_data import GameFeatures, features_to_dataframe


# ── Labeling Thresholds (tune these!) ─────────────────────────────────────────
# These are hyperparameters of your labeling strategy.
# Document your choices and reasoning in the notebook.

THRESHOLDS = {
    "close_game_margin":     2,     # margin <= this → notable (1-run, 2-run games)
    "extra_innings":         True,  # any extra innings game → notable
    "offensive_explosion":   14,    # total runs >= this → notable (raised from 12)
    "dominant_so":           11,    # winning pitcher SO >= this → notable
    "hr_barrage":            4,     # total HRs >= this → notable
    "shutout":               True,  # shutout → notable
    "blowout_margin":        8,     # margin >= this → notable (dominant win)
    # "comeback" removed — had_lead_change fired on 77% of all games, making
    # it useless as a signal and inflating the notable rate to 92%.
}


# ── Individual Rule Functions ──────────────────────────────────────────────────

def rule_close_game(f: GameFeatures) -> bool:
    return f.margin <= THRESHOLDS["close_game_margin"]

def rule_extra_innings(f: GameFeatures) -> bool:
    return f.is_extra_innings == 1.0

def rule_offensive_explosion(f: GameFeatures) -> bool:
    return f.total_runs >= THRESHOLDS["offensive_explosion"]

def rule_dominant_pitching(f: GameFeatures) -> bool:
    return f.winning_pitcher_so >= THRESHOLDS["dominant_so"]

def rule_hr_barrage(f: GameFeatures) -> bool:
    return f.total_hrs >= THRESHOLDS["hr_barrage"]

def rule_shutout(f: GameFeatures) -> bool:
    return f.is_shutout == 1.0

def rule_blowout(f: GameFeatures) -> bool:
    return f.margin >= THRESHOLDS["blowout_margin"]


# Registry: name → function (useful for ablation studies)
RULES = {
    "close_game":          rule_close_game,
    "extra_innings":       rule_extra_innings,
    "offensive_explosion": rule_offensive_explosion,
    "dominant_pitching":   rule_dominant_pitching,
    "hr_barrage":          rule_hr_barrage,
    "shutout":             rule_shutout,
    "blowout":             rule_blowout,
}


# ── Core Labeling Functions ────────────────────────────────────────────────────

def label_game(f: GameFeatures, rules: Dict = None, min_rules: int = 2) -> int:
    """
    Apply all rules to a single game. Returns 1 (notable) or 0 (routine).

    Args:
        f:         GameFeatures instance.
        rules:     Dict of rule_name → rule_fn. Defaults to RULES.
        min_rules: Minimum number of rules that must fire to label a game notable.
                   Default 2 requires co-occurrence (e.g. shutout + blowout, close +
                   extra innings), which drops the notable rate to ~21% and prevents
                   the MLP from trivially memorizing single-threshold rules.

    Returns:
        1 if at least min_rules rules fire, 0 otherwise.
    """
    if rules is None:
        rules = RULES
    fired = sum(1 for fn in rules.values() if fn(f))
    return int(fired >= min_rules)


def label_game_with_reasons(f: GameFeatures, min_rules: int = 2) -> Tuple[int, List[str]]:
    """
    Label a game and return which rules fired.

    Returns:
        (label, list_of_fired_rule_names)
    """
    fired = [name for name, fn in RULES.items() if fn(f)]
    label = int(len(fired) >= min_rules)
    return label, fired


def label_dataset(
    features: List[GameFeatures],
    rules: Dict = None,
    min_rules: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Label a full list of GameFeatures.

    Args:
        features:  List of GameFeatures from historical_data.py
        rules:     Optional custom rule dict for ablation studies.
        min_rules: Passed through to label_game.

    Returns:
        X: np.ndarray of shape (N, num_features), float32
        y: np.ndarray of shape (N,), int64 — binary labels
    """
    X = np.stack([f.to_numpy() for f in features])
    y = np.array([label_game(f, rules, min_rules=min_rules) for f in features],
                 dtype=np.int64)
    return X, y


# ── Analysis Helpers ───────────────────────────────────────────────────────────

def label_distribution(y: np.ndarray) -> None:
    """Print class balance — important for understanding your dataset."""
    total = len(y)
    notable = y.sum()
    routine = total - notable
    print(f"Label distribution:")
    print(f"  Notable (1): {notable:4d} ({100*notable/total:.1f}%)")
    print(f"  Routine (0): {routine:4d} ({100*routine/total:.1f}%)")
    print(f"  Total:       {total:4d}")


def rule_firing_analysis(features: List[GameFeatures]) -> pd.DataFrame:
    """
    Show how often each rule fires independently.
    Helps you understand which rules are doing the most work
    and whether any are redundant.
    """
    rows = []
    for name, fn in RULES.items():
        fires = sum(1 for f in features if fn(f))
        rows.append({
            "rule": name,
            "fires": fires,
            "pct": 100 * fires / len(features)
        })
    df = pd.DataFrame(rows).sort_values("fires", ascending=False)
    return df


def ablation_study(features: List[GameFeatures]) -> pd.DataFrame:
    """
    Leave-one-out ablation: how does removing each rule change label counts?
    Shows which rules are most impactful on the dataset composition.
    Great experiment to include in your notebook.
    """
    _, y_full = label_dataset(features)
    baseline_notable = y_full.sum()

    rows = []
    for drop_rule in RULES:
        reduced_rules = {k: v for k, v in RULES.items() if k != drop_rule}
        _, y_reduced = label_dataset(features, rules=reduced_rules)
        notable_reduced = y_reduced.sum()
        rows.append({
            "dropped_rule": drop_rule,
            "notable_with_rule": baseline_notable,
            "notable_without_rule": notable_reduced,
            "games_lost": baseline_notable - notable_reduced,
            "pct_impact": 100 * (baseline_notable - notable_reduced) / baseline_notable
        })

    return pd.DataFrame(rows).sort_values("games_lost", ascending=False)


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from historical_data import get_mock_features

    print("=== Auto-Labeler Test ===\n")

    features = get_mock_features(500)
    X, y = label_dataset(features)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape:   {y.shape}\n")

    label_distribution(y)

    print("\n── Rule Firing Analysis ──")
    print(rule_firing_analysis(features).to_string(index=False))

    print("\n── Ablation Study ──")
    print(ablation_study(features).to_string(index=False))

    print("\n── Sample Notable Games ──")
    for f in features[:50]:
        label, reasons = label_game_with_reasons(f)
        if label == 1:
            print(f"  game_pk={f.game_pk} | {f.home_score:.0f}-{f.away_score:.0f} "
                  f"| inn={f.innings_played:.0f} | reasons={reasons}")
