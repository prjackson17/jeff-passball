#!/usr/bin/env bash
# Start the full jeff-passball stack: ollama daemon + web server.
# Usage: bash scripts/start_all.sh [port]
#
# Requires ANTHROPIC_API_KEY to be set (used for the daily briefing).
# Set SKIP_OLLAMA=1 to skip the local LLM and use Claude for queries too.
# Set OLLAMA_BIN to override the ollama binary path (default: ollama on PATH).
# Set OLLAMA_MODELS to override the model storage directory.

PORT=${1:-8080}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLLAMA_MODEL_NAME="llama3.1:8b"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"

export EMBEDDER_PATH="${EMBEDDER_PATH:-$REPO_ROOT/checkpoints/mlb-minilm-finetuned}"
export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/game_features_all.npz}"
export QUERY_DAYS_BACK="${QUERY_DAYS_BACK:-30}"

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set. Export it before running this script."
    exit 1
fi

# ── ollama ────────────────────────────────────────────────────────────────────

if [ "${SKIP_OLLAMA:-0}" = "1" ]; then
    echo "[ollama] Skipping local LLM (SKIP_OLLAMA=1) — queries will use Claude."
else
    if ! command -v "$OLLAMA_BIN" &> /dev/null && [ ! -x "$OLLAMA_BIN" ]; then
        echo "ERROR: ollama not found. Install from https://ollama.com or set OLLAMA_BIN."
        exit 1
    fi

    if ! curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
        echo "[ollama] Starting daemon..."
        OLLAMA_MODELS="${OLLAMA_MODELS:-}" "$OLLAMA_BIN" serve > /tmp/ollama.log 2>&1 &
        OLLAMA_PID=$!

        # Wait up to 15s for daemon to be ready
        for i in $(seq 1 15); do
            sleep 1
            if curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
                echo "[ollama] Daemon ready (pid $OLLAMA_PID)."
                break
            fi
            if [ "$i" = "15" ]; then
                echo "ERROR: ollama daemon did not start. Check /tmp/ollama.log"
                exit 1
            fi
        done
    else
        echo "[ollama] Daemon already running."
    fi

    # Pull model if not present
    if OLLAMA_MODELS="${OLLAMA_MODELS:-}" "$OLLAMA_BIN" list 2>/dev/null | grep -q "$OLLAMA_MODEL_NAME"; then
        echo "[ollama] Model $OLLAMA_MODEL_NAME already downloaded."
    else
        echo "[ollama] Pulling $OLLAMA_MODEL_NAME (~4.9GB — this may take a while)..."
        OLLAMA_MODELS="${OLLAMA_MODELS:-}" "$OLLAMA_BIN" pull "$OLLAMA_MODEL_NAME"
    fi

    export OLLAMA_MODEL="$OLLAMA_MODEL_NAME"
    echo "[ollama] Queries will use local model: $OLLAMA_MODEL_NAME"
fi

# ── Web server ─────────────────────────────────────────────────────────────────

echo "[Server] Starting MLB RAG on port $PORT..."
python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
