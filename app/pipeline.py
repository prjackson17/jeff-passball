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
    CACHE_PATH         — path to briefing cache JSON (default: data/briefing_cache.json)
"""

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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

    # Always fetch recent games — needed for query even if briefing is cached
    days_back = int(os.environ.get("DAYS_BACK", 2))
    print(f"[Pipeline] Fetching last {days_back} day(s) of games...")
    chunks = ingest_mlb_data(days_back=days_back)
    game_chunks = [c for c in chunks if c.chunk_type == "game_recap"]

    if not game_chunks:
        _state.briefing = "No completed games found for the requested period."
        _state.last_refresh = datetime.now(timezone.utc)
        return

    print(f"[Pipeline] {len(game_chunks)} game recaps ingested. Building vector store...")
    _state.store = build_vector_store(chunks, embedder=_state.embedder, save=False)
    _state.ready = True

    # Use cached briefing if it was generated today — skip the Claude API call
    cached = _load_cache()
    if cached:
        print("[Pipeline] Using today's cached briefing — skipping Claude API call.")
        _state.briefing = cached["briefing"]
        _state.last_refresh = datetime.fromisoformat(cached["last_refresh"])
    else:
        print("[Pipeline] No fresh cache — generating briefing...")
        _generate_and_cache(game_chunks)

    print("[Pipeline] Ready.")


def refresh():
    """Re-fetch recent games, rebuild vector store, force-regenerate briefing."""
    days_back = int(os.environ.get("DAYS_BACK", 2))
    print(f"[Pipeline] Refreshing — fetching last {days_back} day(s) of games...")
    chunks = ingest_mlb_data(days_back=days_back)
    game_chunks = [c for c in chunks if c.chunk_type == "game_recap"]

    if not game_chunks:
        _state.briefing = "No completed games found for the requested period."
        _state.last_refresh = datetime.now(timezone.utc)
        _state.ready = False
        return

    _state.store = build_vector_store(chunks, embedder=_state.embedder, save=False)
    _state.ready = True
    _generate_and_cache(game_chunks)
    print("[Pipeline] Refresh complete.")


def _generate_and_cache(game_chunks):
    """Call Claude to generate the briefing and write it to disk."""
    _state.briefing = generate_daily_briefing(
        _state.store,
        _state.embedder,
        classifier=_state.classifier,
        X_hist=_state.X_hist,
        feature_names=_state.feature_names,
    )
    _state.last_refresh = datetime.now(timezone.utc)
    _save_cache()


# ── Disk cache ─────────────────────────────────────────────────────────────────

def _cache_path() -> str:
    return os.environ.get("CACHE_PATH", "data/briefing_cache.json").strip()


def _load_cache() -> Optional[dict]:
    """Return cached briefing dict if it was generated today (UTC), else None."""
    path = _cache_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("date") == datetime.now(timezone.utc).strftime("%Y-%m-%d"):
            return data
    except Exception as e:
        print(f"[Pipeline] Cache read error: {e}")
    return None


def _save_cache():
    """Persist today's briefing to disk."""
    path = _cache_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump({
                "date": _state.last_refresh.strftime("%Y-%m-%d"),
                "briefing": _state.briefing,
                "last_refresh": _state.last_refresh.isoformat(),
            }, f)
        print(f"[Pipeline] Briefing cached → {path}")
    except Exception as e:
        print(f"[Pipeline] Cache write error: {e}")


# ── Model loaders ──────────────────────────────────────────────────────────────

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
