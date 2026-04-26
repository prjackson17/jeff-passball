"""
trend_classifier.py

Binary MLP classifier: given a game's feature vector,
predict whether it's a notable game worth surfacing in a briefing.

Architecture: MLP with batch norm, dropout, and configurable depth.
Training uses your existing engine patterns (TrainerConfig, etc.)
but is self-contained here so it runs without the full engine import.

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from src.mlb_rag.historical_data import GameFeatures
from src.mlb_rag.auto_labeler import label_dataset, label_distribution


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class ClassifierConfig:
    """
    Hyperparameters for the trend classifier.
    Mirrors your engine's ModelConfig/TrainerConfig pattern.
    """
    # Architecture
    input_dim: int = 15               # GameFeatures.num_features()
    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    use_batch_norm: bool = True

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 64
    early_stopping_patience: int = 8

    # Class imbalance handling
    use_weighted_sampler: bool = True  # oversample minority class

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpointing
    checkpoint_dir: str = "/var/tmp/prj004/checkpoints"
    checkpoint_name: str = "trend_classifier.pt"


# ── Dataset ────────────────────────────────────────────────────────────────────

class GameDataset(Dataset):
    """
    PyTorch Dataset wrapping normalized game feature vectors and binary labels.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ──────────────────────────────────────────────────────────────────────

class TrendClassifierMLP(nn.Module):
    """
    Multi-layer perceptron for binary game notability classification.

    Architecture per hidden layer:
        Linear → BatchNorm (optional) → ReLU → Dropout

    Final layer: Linear → (no activation, raw logits for CrossEntropyLoss)

    Why this architecture:
    - BatchNorm stabilizes training on the mixed-scale feature space
      (scores are 0-20, strikeouts are 0-20, but indicators are 0/1)
    - Dropout prevents overfitting on ~2400 training samples
    - Small network appropriate for 15 input features
    """

    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.input_dim

        for hidden_dim in config.hidden_units:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        # Output: 2 logits (routine, notable)
        layers.append(nn.Linear(in_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities. Shape: (N, 2)"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def num_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ── Training ───────────────────────────────────────────────────────────────────

class TrendClassifierTrainer:
    """
    Self-contained trainer for the trend classifier.
    Follows the same patterns as your engine's Trainer class.
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.scaler = StandardScaler()   # normalize features
        self.model: Optional[TrendClassifierMLP] = None
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    def _make_weighted_sampler(self, y: np.ndarray) -> WeightedRandomSampler:
        """
        Oversample the minority class (notable games) to handle class imbalance.
        Without this, the model learns to predict 'routine' for everything.
        """
        class_counts = np.bincount(y)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y]
        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(y),
            replacement=True
        )

    def _compute_metrics(self, model, loader) -> Tuple[float, float, float]:
        """Compute loss, accuracy, and F1 on a dataloader."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        avg_loss = total_loss / len(all_labels)
        accuracy = (all_preds == all_labels).mean()

        # Macro F1: average of per-class F1 scores.
        # Using notable-class F1 alone lets a model that predicts "notable"
        # for everything score perfectly on an imbalanced dataset.
        f1_per_class = []
        for cls in [0, 1]:
            tp = ((all_preds == cls) & (all_labels == cls)).sum()
            fp = ((all_preds == cls) & (all_labels != cls)).sum()
            fn = ((all_preds != cls) & (all_labels == cls)).sum()
            f1_per_class.append((2 * tp) / (2 * tp + fp + fn + 1e-8))
        macro_f1 = float(np.mean(f1_per_class))

        return avg_loss, float(accuracy), macro_f1

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        wandb_run=None
    ) -> TrendClassifierMLP:
        """
        Train the MLP classifier.

        Args:
            X_train, y_train: Training features and labels (raw, unscaled).
            X_val, y_val: Validation features and labels.
            wandb_run: Optional W&B run for logging.

        Returns:
            Trained TrendClassifierMLP.
        """
        # Normalize features (fit on train only — no data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val).astype(np.float32)

        # Datasets
        train_ds = GameDataset(X_train_scaled, y_train)
        val_ds = GameDataset(X_val_scaled, y_val)

        # Sampler for class imbalance
        sampler = None
        shuffle = True
        if self.config.use_weighted_sampler:
            sampler = self._make_weighted_sampler(y_train)
            shuffle = False   # mutually exclusive with sampler

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=shuffle
        )
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        # Model
        self.model = TrendClassifierMLP(self.config).to(self.device)
        total_params, train_params = self.model.num_parameters()
        print(f"[Classifier] Parameters: {total_params:,} total, {train_params:,} trainable")

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=4, factor=0.5
        )

        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        best_state = None

        print(f"\n[Classifier] Training for up to {self.config.num_epochs} epochs...")
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>10} {'Val F1':>8}")
        print("-" * 52)

        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)

            train_loss /= len(train_ds)

            # Validate
            val_loss, val_acc, val_f1 = self._compute_metrics(self.model, val_loader)
            scheduler.step(val_f1)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            if epoch % 5 == 0 or epoch == 1:
                print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
                      f"{val_acc:>10.4f} {val_f1:>8.4f}")

            if wandb_run:
                wandb_run.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                })

            # Early stopping on val F1 (not accuracy — imbalanced dataset)
            if val_f1 > best_val_f1 + 1e-4:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"\n[Classifier] Early stopping at epoch {epoch} "
                          f"(best val F1: {best_val_f1:.4f})")
                    break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        print(f"\n[Classifier] Training complete. Best val F1: {best_val_f1:.4f}")
        return self.model

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Full evaluation report on a held-out set."""
        assert self.model is not None, "Must call fit() first"
        X_scaled = self.scaler.transform(X).astype(np.float32)
        ds = GameDataset(X_scaled, y)
        loader = DataLoader(ds, batch_size=256)

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self.model(X_batch.to(self.device))
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        print("\n── Classification Report ──")
        print(classification_report(
            all_labels, all_preds,
            target_names=["Routine", "Notable"]
        ))
        print("── Confusion Matrix ──")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"           Pred Routine  Pred Notable")
        print(f"True Routine    {cm[0,0]:6d}        {cm[0,1]:6d}")
        print(f"True Notable    {cm[1,0]:6d}        {cm[1,1]:6d}")

    def save(self) -> None:
        """Save model weights and scaler."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_name)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "config": self.config,
        }, path)
        print(f"[Classifier] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TrendClassifierTrainer":
        """Load a saved classifier."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        trainer = cls(config)
        trainer.model = TrendClassifierMLP(config)
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.scaler = checkpoint["scaler"]
        trainer.model.eval()
        print(f"[Classifier] Loaded from {path}")
        return trainer


# ── Inference (used by RAG pipeline) ──────────────────────────────────────────

def score_chunks_with_classifier(
    chunks,           # List[MLBChunk] — imported lazily to avoid circular import
    trainer: TrendClassifierTrainer,
    feature_extractor=None,
) -> List[Tuple[any, float]]:
    """
    Score a list of MLBChunks for notability using the trained classifier.
    Returns list of (chunk, notability_probability) sorted descending.

    This is how the classifier plugs into the RAG pipeline as a reranker:
        retrieve top-k chunks → score each → pass highest-scoring to LLM
    """
    # For now, return chunks with placeholder scores if no feature extractor
    # In the full pipeline, feature_extractor converts chunk.metadata → GameFeatures
    if feature_extractor is None:
        return [(chunk, 0.5) for chunk in chunks]

    features = [feature_extractor(chunk) for chunk in chunks]
    X = np.stack([f.to_numpy() for f in features]).astype(np.float32)
    X_scaled = trainer.scaler.transform(X).astype(np.float32)

    trainer.model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled)
        probs = trainer.model.predict_proba(X_tensor)
        notable_probs = probs[:, 1].numpy()

    scored = list(zip(chunks, notable_probs.tolist()))
    return sorted(scored, key=lambda x: x[1], reverse=True)


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.mlb_rag.historical_data import get_mock_features
    from src.mlb_rag.auto_labeler import label_dataset, label_distribution

    print("=== Trend Classifier Training Test ===\n")

    # 1. Data
    features = get_mock_features(800)
    X, y = label_dataset(features)
    print(f"Dataset: {X.shape[0]} games, {X.shape[1]} features")
    label_distribution(y)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"\nSplit: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # 3. Train
    config = ClassifierConfig(
        hidden_units=[64, 32],
        dropout=0.3,
        num_epochs=40,
        batch_size=32,
        use_weighted_sampler=True,
    )
    trainer = TrendClassifierTrainer(config)
    model = trainer.fit(X_train, y_train, X_val, y_val)

    # 4. Evaluate
    trainer.evaluate(X_test, y_test)

    # 5. Save
    trainer.save()
