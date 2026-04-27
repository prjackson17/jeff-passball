"""
sweep_train.py

W&B sweep agent for TrendClassifierMLP architecture search (Section 6.1).

The sweep runs a full grid over the four key hyperparameters so W&B has
the per-parameter variation it needs for "Parameter Importance" analysis:

    use_weighted_sampler : [False, True]                           (2)
    use_batch_norm       : [False, True]                           (2)
    dropout              : [0.0, 0.3]                              (2)
    hidden_units_str     : ["[32]", "[64, 32]", "[128, 64, 32]"]  (3)

    Total: 2 × 2 × 2 × 3 = 24 runs

Usage:
    # 1. Create the sweep (run once from the notebook or terminal):
    python src/mlb_rag/sweep_train.py --create-sweep

    # 2. Launch the agent (runs all 24 combinations):
    python src/mlb_rag/sweep_train.py --agent <sweep_id>

    # 3. Train one reference config without a sweep controller:
    python src/mlb_rag/sweep_train.py --single <0-5>

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import ast
import sys
import os
import argparse

import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mlb_rag.historical_data import load_features, GameFeatures
from src.mlb_rag.auto_labeler import label_game
from src.mlb_rag.trend_classifier import ClassifierConfig, TrendClassifierTrainer, GameDataset


# ── Constants ──────────────────────────────────────────────────────────────────

WANDB_ENTITY  = "bucknell-university-csci357-2026sp"
WANDB_PROJECT = "mlb-rag"
DATA_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "data", "game_features_all.npz")

# Reference configs used for documentation and --single CLI mode.
# The actual sweep explores ALL grid combinations, not just these six.
ABLATION_CONFIGS = [
    {
        "ablation_name":        "No WeightedSampler",
        "hidden_units_str":     "[64, 32]",
        "dropout":              0.3,
        "use_batch_norm":       True,
        "use_weighted_sampler": False,
        "notes":                "Predicts routine always — minority class never seen",
    },
    {
        "ablation_name":        "No BatchNorm",
        "hidden_units_str":     "[64, 32]",
        "dropout":              0.3,
        "use_batch_norm":       False,
        "use_weighted_sampler": True,
        "notes":                "Slower convergence; mixed-scale features hurt raw linear layers",
    },
    {
        "ablation_name":        "No Dropout",
        "hidden_units_str":     "[64, 32]",
        "dropout":              0.0,
        "use_batch_norm":       True,
        "use_weighted_sampler": True,
        "notes":                "Overfits early on ~6k training samples",
    },
    {
        "ablation_name":        "Hidden [32]",
        "hidden_units_str":     "[32]",
        "dropout":              0.3,
        "use_batch_norm":       True,
        "use_weighted_sampler": True,
        "notes":                "Too shallow for 15-feature interaction space",
    },
    {
        "ablation_name":        "Hidden [128, 64, 32]",
        "hidden_units_str":     "[128, 64, 32]",
        "dropout":              0.3,
        "use_batch_norm":       True,
        "use_weighted_sampler": True,
        "notes":                "Marginal gain, 3x more params than baseline",
    },
    {
        "ablation_name":        "Hidden [64, 32] + BN + WRS",
        "hidden_units_str":     "[64, 32]",
        "dropout":              0.3,
        "use_batch_norm":       True,
        "use_weighted_sampler": True,
        "notes":                "Best configuration",
    },
]

# Full grid over the four hyperparameters — 24 combinations total.
# W&B uses these parameter definitions for "Parameter Importance" analysis.
SWEEP_CONFIG = {
    "name":   "classifier-ablations-v2",
    "method": "grid",
    "metric": {"name": "best_val_macro_f1", "goal": "maximize"},
    "parameters": {
        "use_weighted_sampler": {"values": [False, True]},
        "use_batch_norm":       {"values": [False, True]},
        "dropout":              {"values": [0.0, 0.3]},
        "hidden_units_str":     {"values": ["[32]", "[64, 32]", "[128, 64, 32]"]},
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data():
    path = os.path.abspath(DATA_PATH)
    X, _, dates, *_ = load_features(path)
    feature_names = GameFeatures.feature_names()

    def _row_to_gf(row):
        fn = feature_names
        return GameFeatures(
            game_pk=0, date="",
            home_score=row[fn.index("home_score")],
            away_score=row[fn.index("away_score")],
            margin=row[fn.index("margin")],
            total_runs=row[fn.index("total_runs")],
            innings_played=row[fn.index("innings_played")],
            winning_pitcher_so=row[fn.index("winning_pitcher_so")],
            losing_pitcher_so=row[fn.index("losing_pitcher_so")],
            total_hits=row[fn.index("total_hits")],
            total_errors=row[fn.index("total_errors")],
            home_hrs=row[fn.index("home_hrs")],
            away_hrs=row[fn.index("away_hrs")],
            total_hrs=row[fn.index("total_hrs")],
            is_extra_innings=row[fn.index("is_extra_innings")],
            is_shutout=row[fn.index("is_shutout")],
            had_lead_change=row[fn.index("had_lead_change")],
        )

    # min_rules=2: a game is notable only if 2+ rules co-occur.
    # This prevents the MLP from trivially memorizing single-threshold rules.
    y = np.array([label_game(_row_to_gf(row), min_rules=2) for row in X], dtype=np.int64)
    return X, y, dates


# ── Core training logic ────────────────────────────────────────────────────────

def _train(run: "wandb.sdk.wandb_run.Run") -> None:
    """Train with hyperparameters read directly from the W&B run config."""
    hidden_units = ast.literal_eval(run.config.hidden_units_str)
    dropout      = run.config.dropout
    use_bn       = run.config.use_batch_norm
    use_wrs      = run.config.use_weighted_sampler

    run.name = (
        f"h{run.config.hidden_units_str}"
        f"_d{dropout}"
        f"{'_bn' if use_bn else '_nobn'}"
        f"{'_wrs' if use_wrs else '_nowrs'}"
    )

    X, y, dates = _load_data()

    # Temporal split: train on historical seasons, validate/test on recent seasons.
    # This is more realistic than a random split — the model must generalize forward in time.
    years = np.array([str(d)[:4] for d in dates])
    train_mask = np.isin(years, ["2023", "2024"])
    val_mask   = years == "2025"
    test_mask  = years == "2026"

    # Fall back to random split if any split is empty (e.g., 2026 data not yet fetched)
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=0.15, random_state=42, stratify=y_tv
        )
        print("[Classifier] Warning: temporal split fallback — missing season data")
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test  = X[test_mask] if test_mask.sum() > 0 else X_val
        y_test  = y[test_mask] if test_mask.sum() > 0 else y_val

    clf_config = ClassifierConfig(
        hidden_units=hidden_units,
        dropout=dropout,
        use_batch_norm=use_bn,
        use_weighted_sampler=use_wrs,
        num_epochs=60,
        early_stopping_patience=8,
        checkpoint_dir="/var/tmp/prj004/checkpoints/ablations",
        checkpoint_name=f"sweep_{run.id}.pt",
    )

    trainer = TrendClassifierTrainer(clf_config)
    trainer.fit(X_train, y_train, X_val, y_val, wandb_run=run)

    stopped_epoch = len(trainer.history["val_f1"])
    best_val_f1   = max(trainer.history["val_f1"])
    best_idx      = trainer.history["val_f1"].index(best_val_f1)
    best_val_acc  = trainer.history["val_acc"][best_idx]

    X_test_scaled = trainer.scaler.transform(X_test).astype(np.float32)
    test_ds       = GameDataset(X_test_scaled, y_test)
    test_loader   = DataLoader(test_ds, batch_size=256, shuffle=False)
    _, test_acc, test_f1 = trainer._compute_metrics(trainer.model, test_loader)

    wandb.summary["best_val_macro_f1"] = best_val_f1
    wandb.summary["best_val_acc"]      = best_val_acc
    wandb.summary["stopped_epoch"]     = stopped_epoch
    wandb.summary["test_acc"]          = test_acc
    wandb.summary["test_macro_f1"]     = test_f1


# ── Sweep agent entrypoint ─────────────────────────────────────────────────────

def _sweep_agent_fn():
    with wandb.init(tags=["ablation"]) as run:
        _train(run)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="W&B sweep for classifier architecture search")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create-sweep", action="store_true",
                       help="Register the sweep and print the sweep ID")
    group.add_argument("--agent", metavar="SWEEP_ID",
                       help="Run as a sweep agent for the given sweep ID")
    group.add_argument("--single", type=int, metavar="ABLATION_ID", choices=range(6),
                       help="Train one of the 6 reference configs (0-5) without a sweep controller")
    args = parser.parse_args()

    if args.create_sweep:
        sweep_id = wandb.sweep(SWEEP_CONFIG, entity=WANDB_ENTITY, project=WANDB_PROJECT)
        print(f"\nSweep created: {sweep_id}")
        print(f"Run agent:  python src/mlb_rag/sweep_train.py --agent {sweep_id}\n")

    elif args.agent:
        wandb.agent(
            args.agent,
            function=_sweep_agent_fn,
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
        )

    else:  # --single
        cfg = ABLATION_CONFIGS[args.single]
        with wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={
                "hidden_units_str":     cfg["hidden_units_str"],
                "dropout":              cfg["dropout"],
                "use_batch_norm":       cfg["use_batch_norm"],
                "use_weighted_sampler": cfg["use_weighted_sampler"],
            },
            tags=["ablation", "single"],
        ) as run:
            _train(run)


if __name__ == "__main__":
    main()
