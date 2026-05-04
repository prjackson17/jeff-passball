#!/usr/bin/env bash
# Start the MLB RAG web server.
# Usage: bash scripts/start_server.sh [port]

PORT=${1:-8080}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export EMBEDDER_PATH="${EMBEDDER_PATH:-$REPO_ROOT/checkpoints/mlb-minilm-finetuned}"
export CLASSIFIER_PATH="${CLASSIFIER_PATH:-$REPO_ROOT/checkpoints/trend_classifier.pt}"
export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/game_features_all.npz}"
export QUERY_DAYS_BACK="${QUERY_DAYS_BACK:-30}"
# Optional: route queries through a local Llama model via ollama (no Claude credits for queries)
# export OLLAMA_MODEL="llama3.1:8b"
# ANTHROPIC_API_KEY must already be set in your environment (still used for the daily briefing)

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set. Export it before running this script."
    exit 1
fi

echo "[Server] Starting MLB RAG on port $PORT..."
python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
