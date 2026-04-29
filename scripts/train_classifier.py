"""Train the trend classifier on real historical game data."""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.mlb_rag.historical_data import load_features, GameFeatures
from src.mlb_rag.auto_labeler import label_game
from src.mlb_rag.trend_classifier import ClassifierConfig, TrendClassifierTrainer

DATA_PATH     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "game_features_all.npz")
CHECKPOINT_DIR = "/var/tmp/prj004/checkpoints"

def main():
    parser = argparse.ArgumentParser(description="Train trend classifier on historical game data")
    parser.add_argument("--epochs",   type=int,   default=60,   help="Max training epochs (default: 60)")
    parser.add_argument("--patience", type=int,   default=8,    help="Early stopping patience (default: 8)")
    parser.add_argument("--out",      type=str,   default=os.path.join(CHECKPOINT_DIR, "trend_classifier.pt"))
    args = parser.parse_args()

    # ── Load & label ───────────────────────────────────────────────────────────
    X, _, dates, *_ = load_features(DATA_PATH)
    fn = GameFeatures.feature_names()

    def row_to_gf(r):
        return GameFeatures(game_pk=0, date="", **dict(zip(fn, r.tolist())))

    # min_rules=2: a game must satisfy 2+ auto-labeler rules to be "notable".
    # Matches sweep_train.py and the notebook retrain cell.
    y = np.array([label_game(row_to_gf(r), min_rules=2) for r in X], dtype=np.int64)

    # ── Temporal split ─────────────────────────────────────────────────────────
    years = np.array([str(d)[:4] for d in dates])
    train = np.isin(years, ["2023", "2024"])
    val   = years == "2025"
    test  = years == "2026"

    print(f"Train: {train.sum():,}  Val: {val.sum():,}  Test: {test.sum():,}")
    print(f"Notable rate — train: {y[train].mean()*100:.1f}%  val: {y[val].mean()*100:.1f}%  test: {y[test].mean()*100:.1f}%")

    if train.sum() == 0 or val.sum() == 0:
        raise RuntimeError("Temporal split produced an empty train or val set. Check the data.")

    # ── Best config from sweep (classifier-ablations-v2, rank-0) ──────────────
    config = ClassifierConfig(
        hidden_units=[128, 64, 32],
        dropout=0.3,
        use_batch_norm=False,
        use_weighted_sampler=False,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        checkpoint_dir=os.path.dirname(args.out),
        checkpoint_name=os.path.basename(args.out),
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = TrendClassifierTrainer(config)
    trainer.fit(X[train], y[train], X[val], y[val])

    if test.sum() > 0:
        trainer.evaluate(X[test], y[test])

    trainer.save()
    print(f"\n✅ Checkpoint saved to {args.out}")

if __name__ == "__main__":
    main()
