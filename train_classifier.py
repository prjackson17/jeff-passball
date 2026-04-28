"""Train the trend classifier on real historical game data."""
from src.mlb_rag.historical_data import load_features, GameFeatures
from src.mlb_rag.auto_labeler import label_game
from src.mlb_rag.trend_classifier import ClassifierConfig, TrendClassifierTrainer
import numpy as np

X, _, dates, *_ = load_features("./data/game_features_all.npz")
fn = GameFeatures.feature_names()

def row_to_gf(r):
    return GameFeatures(game_pk=0, date="", **dict(zip(fn, r.tolist())))

y = np.array([label_game(row_to_gf(r)) for r in X])

train = np.array([str(d)[:4] in ("2023", "2024") for d in dates])
val   = np.array([str(d).startswith("2025") for d in dates])
test  = np.array([str(d).startswith("2026") for d in dates])

print(f"Train: {train.sum()}  Val: {val.sum()}  Test: {test.sum()}")
print(f"Notable rate — train: {y[train].mean()*100:.1f}%  val: {y[val].mean()*100:.1f}%  test: {y[test].mean()*100:.1f}%")

config = ClassifierConfig(
    hidden_units=[128, 64, 32],
    dropout=0.3,
    use_batch_norm=False,
    use_weighted_sampler=False,
)
trainer = TrendClassifierTrainer(config)
trainer.fit(X[train], y[train], X[val], y[val])
trainer.evaluate(X[test], y[test])
trainer.save()
