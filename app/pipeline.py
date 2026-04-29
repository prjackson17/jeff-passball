"""
pipeline.py

Singleton that owns all ML state for the web app.
Called once at startup via initialize(); refresh() re-fetches games and
regenerates the briefing on demand.

Env vars:
    ANTHROPIC_API_KEY  — required for Claude API calls
    DAYS_BACK          — completed days of games to fetch (default: 2)
    EMBEDDER_PATH      — directory of fine-tuned sentence transformer
                         (falls back to all-MiniLM-L6-v2 if not set/found)
    CLASSIFIER_PATH    — path to trend_classifier.pt
                         (reranking skipped if not set/found)
    DATA_PATH          — path to game_features_all.npz
                         (novelty facts skipped if not set/found)
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from src.mlb_rag.data_ingestion import ingest_mlb_data
from src.mlb_rag.embedder import MLBEmbedder, build_vector_store
from src.mlb_rag.commentary import load_classifier, generate_daily_briefing
from src.mlb_rag.historical_data import load_features, GameFeatures


@dataclass
class PipelineState:
    embedder: object = None
    store: object = None
    classifier: object = None
    X_hist: Optional[np.ndarray] = None
    feature_names: Optional[list] = None
    briefing: str = ""
    last_refresh: Optional[datetime] = None
    embedder_type: str = "none"    # "finetuned" | "base"
    classifier_loaded: bool = False
    novelty_enabled: bool = False
    ready: bool = False


_state = PipelineState()


def get_state() -> PipelineState:
    return _state


def initialize():
    """Load models and fetch initial game data. Called once at startup."""
    _load_embedder()
    _load_classifier()
    _load_historical()
    refresh()


def refresh():
    """Re-fetch recent games, rebuild vector store, regenerate briefing."""
    days_back = int(os.environ.get("DAYS_BACK", 2))
    print(f"[Pipeline] Fetching last {days_back} day(s) of games...")
    chunks = ingest_mlb_data(days_back=days_back)
    game_chunks = [c for c in chunks if c.chunk_type == "game_recap"]

    if not game_chunks:
        _state.briefing = "No completed games found for the requested period."
        _state.last_refresh = datetime.utcnow()
        _state.ready = False
        return

    print(f"[Pipeline] {len(game_chunks)} game recaps ingested. Building vector store...")
    _state.store = build_vector_store(chunks, embedder=_state.embedder, save=False)

    print("[Pipeline] Generating briefing...")
    _state.briefing = generate_daily_briefing(
        _state.store,
        _state.embedder,
        classifier=_state.classifier,
        X_hist=_state.X_hist,
        feature_names=_state.feature_names,
    )
    _state.last_refresh = datetime.utcnow()
    _state.ready = True
    print("[Pipeline] Ready.")


def _load_embedder():
    finetuned = os.environ.get("EMBEDDER_PATH", "").strip()
    if finetuned and os.path.isdir(finetuned):
        print(f"[Pipeline] Loading fine-tuned embedder from {finetuned}")
        _state.embedder = MLBEmbedder(model_name=finetuned)
        _state.embedder_type = "finetuned"
    else:
        print("[Pipeline] Fine-tuned embedder not found — using base all-MiniLM-L6-v2")
        _state.embedder = MLBEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _state.embedder_type = "base"


def _load_classifier():
    path = os.environ.get("CLASSIFIER_PATH", "").strip()
    if not path:
        print("[Pipeline] CLASSIFIER_PATH not set — reranker disabled")
        return
    clf = load_classifier(path)
    _state.classifier = clf
    _state.classifier_loaded = clf is not None


def _load_historical():
    path = os.environ.get("DATA_PATH", "").strip()
    if not path or not os.path.isfile(path):
        print("[Pipeline] DATA_PATH not set or not found — novelty facts disabled")
        return
    try:
        X, *_ = load_features(path)
        _state.X_hist = X
        _state.feature_names = list(GameFeatures.feature_names())
        _state.novelty_enabled = True
        print(f"[Pipeline] Loaded {len(X):,} historical games for novelty facts.")
    except Exception as e:
        print(f"[Pipeline] Could not load historical data: {e}")
