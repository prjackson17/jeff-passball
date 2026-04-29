"""Generate a daily MLB briefing from the command line.

Usage:
    python scripts/run_briefing.py
    python scripts/run_briefing.py --days-back 3
    python scripts/run_briefing.py --output briefing.md
    python scripts/run_briefing.py --no-novelty
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mlb_rag.data_ingestion import ingest_mlb_data
from src.mlb_rag.embedder import MLBEmbedder, build_vector_store
from src.mlb_rag.commentary import load_classifier, generate_daily_briefing
from src.mlb_rag.historical_data import load_features, GameFeatures

EMBEDDER_PATH   = "/var/tmp/prj004/checkpoints/mlb-minilm-finetuned"
CLASSIFIER_PATH = "/var/tmp/prj004/checkpoints/trend_classifier.pt"
DATA_PATH       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "game_features_all.npz")


def main():
    parser = argparse.ArgumentParser(description="Generate daily MLB briefing")
    parser.add_argument("--days-back",  type=int, default=2,
                        help="Completed days of games to pull (default: 2)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Write briefing to this file in addition to stdout")
    parser.add_argument("--no-novelty", action="store_true",
                        help="Skip novelty/crazy-facts generation")
    args = parser.parse_args()

    # ── Historical data (for novelty facts) ───────────────────────────────────
    X_hist, feature_names = None, None
    if not args.no_novelty:
        print("[Pipeline] Loading historical game data for novelty facts...")
        X_hist, _, _, *_ = load_features(DATA_PATH)
        feature_names = list(GameFeatures.feature_names())
        print(f"[Pipeline] {len(X_hist):,} historical games loaded.\n")

    # ── Ingest recent games ────────────────────────────────────────────────────
    print(f"[Pipeline] Fetching last {args.days_back} day(s) of completed games...")
    chunks      = ingest_mlb_data(days_back=args.days_back)
    game_chunks = [c for c in chunks if c.chunk_type == "game_recap"]
    print(f"[Pipeline] {len(game_chunks)} game recaps + "
          f"{len(chunks) - len(game_chunks)} standings chunks ingested.\n")

    if not game_chunks:
        print("[Pipeline] No completed games found. Exiting.")
        return

    # ── Build vector store ────────────────────────────────────────────────────
    print("[Pipeline] Embedding chunks with fine-tuned model...")
    embedder = MLBEmbedder(model_name=EMBEDDER_PATH)
    store    = build_vector_store(chunks, embedder=embedder, save=False)
    print(f"[Pipeline] Vector store: {store.index.ntotal} vectors.\n")

    # ── Load classifier reranker ───────────────────────────────────────────────
    clf = load_classifier(CLASSIFIER_PATH)

    # ── Generate briefing ──────────────────────────────────────────────────────
    print("[Pipeline] Generating briefing...\n" + "─" * 60)
    briefing = generate_daily_briefing(
        store, embedder,
        classifier=clf,
        X_hist=X_hist,
        feature_names=feature_names,
    )

    print(briefing)

    if args.output:
        with open(args.output, "w") as f:
            f.write(briefing)
        print(f"\n[Pipeline] Briefing saved → {args.output}")


if __name__ == "__main__":
    main()
