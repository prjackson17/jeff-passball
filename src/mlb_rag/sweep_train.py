"""
sweep_train.py

W&B sweep agent for TrendClassifierMLP ablation study (Section 6.1).

Usage:
    # 1. Create the sweep (run once):
    python sweep_train.py --create-sweep

    # 2. Launch the agent (runs all 6 ablations):
    python sweep_train.py --agent <sweep_id>

Or from the notebook: use the provided cells to create the sweep and
run the agent via `! python src/mlb_rag/sweep_train.py --agent <sweep_id>`.

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import sys
import os
import argparse

import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Allow running from repo root or from notebooks/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.mlb_rag.historical_data import load_features, GameFeatures
from src.mlb_rag.auto_labeler import label_game
from src.mlb_rag.trend_classifier import ClassifierConfig, TrendClassifierTrainer, GameDataset


# ── Constants ──────────────────────────────────────────────────────────────────

WANDB_ENTITY  = "bucknell-university-csci357-2026sp"
WANDB_PROJECT = "mlb-rag"
DATA_PATH     = os.path.join(os.path.dirname(__file__), "..", "..", "data", "game_features_all.npz")

# Six ablation configs for Section 6.1. Index 5 is the final best config.
ABLATION_CONFIGS = [
    {
        "ablation_name":      "No WeightedSampler",
        "hidden_units":       [64, 32],
        "dropout":            0.3,
        "use_batch_norm":     True,
        "use_weighted_sampler": False,
        "notes":              "Predicts routine always — minority class never seen",
    },
    {
        "ablation_name":      "No BatchNorm",
        "hidden_units":       [64, 32],
        "dropout":            0.3,
        "use_batch_norm":     False,
        "use_weighted_sampler": True,
        "notes":              "Slower convergence; mixed-scale features hurt raw linear layers",
    },
    {
        "ablation_name":      "No Dropout",
        "hidden_units":       [64, 32],
        "dropout":            0.0,
        "use_batch_norm":     True,
        "use_weighted_sampler": True,
        "notes":              "Overfits early on ~6k training samples",
    },
    {
        "ablation_name":      "Hidden [32]",
        "hidden_units":       [32],
        "dropout":            0.3,
        "use_batch_norm":     True,
        "use_weighted_sampler": True,
        "notes":              "Too shallow for 15-feature interaction space",
    },
    {
        "ablation_name":      "Hidden [128, 64, 32]",
        "hidden_units":       [128, 64, 32],
        "dropout":            0.3,
        "use_batch_norm":     True,
        "use_weighted_sampler": True,
        "notes":              "Marginal gain, 3x more params than baseline",
    },
    {
        "ablation_name":      "Hidden [64, 32] + BN + WRS",
        "hidden_units":       [64, 32],
        "dropout":            0.3,
        "use_batch_norm":     True,
        "use_weighted_sampler": True,
        "notes":              "Best configuration",
    },
]

SWEEP_CONFIG = {
    "name":   "classifier-ablations",
    "method": "grid",
    "metric": {"name": "best_val_f1", "goal": "maximize"},
    "parameters": {
        "ablation_id": {"values": [0, 1, 2, 3, 4, 5]}
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data():
    path = os.path.abspath(DATA_PATH)
    X, _, _ = load_features(path)
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

    y = np.array([label_game(_row_to_gf(row)) for row in X], dtype=np.int64)
    return X, y


# ── Core training logic ────────────────────────────────────────────────────────

def _train(run: "wandb.sdk.wandb_run.Run") -> None:
    """Train one ablation config and log results to an active wandb run."""
    ablation_id = run.config.ablation_id
    ablation    = ABLATION_CONFIGS[ablation_id]

    # Push all ablation params into the run config so they appear in the W&B UI
    wandb.config.update(ablation, allow_val_change=True)
    run.name = ablation["ablation_name"]

    X, y = _load_data()

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15, random_state=42, stratify=y_tv
    )

    config = ClassifierConfig(
        hidden_units=ablation["hidden_units"],
        dropout=ablation["dropout"],
        use_batch_norm=ablation["use_batch_norm"],
        use_weighted_sampler=ablation["use_weighted_sampler"],
        num_epochs=60,
        early_stopping_patience=8,
        checkpoint_dir="/var/tmp/prj004/checkpoints/ablations",
        checkpoint_name=f"ablation_{ablation_id}.pt",
    )

    trainer = TrendClassifierTrainer(config)
    trainer.fit(X_train, y_train, X_val, y_val, wandb_run=run)

    # Derive summary stats from training history
    stopped_epoch = len(trainer.history["val_f1"])
    best_val_f1   = max(trainer.history["val_f1"])
    best_idx      = trainer.history["val_f1"].index(best_val_f1)
    best_val_acc  = trainer.history["val_acc"][best_idx]

    # Test-set evaluation
    X_test_scaled = trainer.scaler.transform(X_test).astype(np.float32)
    test_ds     = GameDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    _, test_acc, test_f1 = trainer._compute_metrics(trainer.model, test_loader)

    wandb.summary["best_val_f1"]   = best_val_f1
    wandb.summary["best_val_acc"]  = best_val_acc
    wandb.summary["stopped_epoch"] = stopped_epoch
    wandb.summary["test_acc"]      = test_acc
    wandb.summary["test_f1"]       = test_f1


# ── Sweep agent entrypoint ─────────────────────────────────────────────────────

def _sweep_agent_fn():
    """Called once per sweep run by wandb.agent()."""
    with wandb.init(tags=["ablation"]) as run:
        _train(run)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="W&B sweep for classifier ablation study")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create-sweep", action="store_true",
                       help="Register the sweep and print the sweep ID")
    group.add_argument("--agent", metavar="SWEEP_ID",
                       help="Run as a sweep agent for the given sweep ID")
    group.add_argument("--single", type=int, metavar="ABLATION_ID",
                       help="Train a single ablation (0-5) without a sweep controller")
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
        assert 0 <= args.single <= 5, "ablation_id must be 0-5"
        with wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={"ablation_id": args.single},
            tags=["ablation", "single"],
        ) as run:
            _train(run)


if __name__ == "__main__":
    main()
