# MLB Broadcast Intelligence: RAG-Powered Daily Briefing

**CSCI 357 — AI and Neural Networks | Spring 2026 | Parker Jackson**  
Bucknell University | Prof. Brian King

---

## Overview

A production retrieval-augmented generation system that ingests live MLB game data, embeds it with a **domain fine-tuned sentence transformer**, reranks retrieved context using a **trained notability classifier MLP**, and prompts an LLM to generate broadcast-quality daily briefings. The system also ships as a deployed web app with live standings, player stat lookup, an interactive chat interface, and optional local LLM inference via ollama.

The two neural components solve a specific gap in standard RAG: pure semantic similarity retrieval treats a 2-0 shutout and a 10-2 blowout identically if they share vocabulary. The classifier reranker uses 15 structured game features to push genuinely newsworthy games toward the top of the context window before the LLM ever sees them.

---

## Video URL

https://www.loom.com/share/2d0cdfbbb93a49b99b81a55998f616d6

---

## Architecture

```
  MLB Stats API
       │
       ▼
 data_ingestion.py ──────────────────────────────────────────┐
       │                                                     │
  MLBChunks (text)                              GameFeatures (15 numerical)
       │                                                     │
       ▼                                                     ▼
 ┌─────────────────────────┐              ┌──────────────────────────────┐
 │  NEURAL COMPONENT 1     │              │  NEURAL COMPONENT 2          │
 │  Fine-tuned Embedder    │              │  TrendClassifier MLP         │
 │  all-MiniLM-L6-v2       │              │  [15 → 128 → 64 → 32 → 2]    │
 │  MNRL loss, lr=5e-6     │              │  Trained on 7,802 real games │
 │  Val Spearman: 0.844    │              │  Val macro F1: 0.9866        │
 └──────────┬──────────────┘              └─────────────┬────────────────┘
            │                                           │
            ▼                                           │
     FAISS Vector Store                                 │
     (top-k retrieval)  ◄──────── reranker blends ──────┘
            │             score = sim + 0.25 × P(notable)
            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  LLM  (Claude claude-sonnet-4-5  or  local Llama 3.1 8B)        │
  │  Context: retrieved chunks + live standings + player stats      │
  │  → Daily briefing (cached to disk, refreshed by cron)           │
  │  → Interactive chat Q&A (multi-turn, with conversation history) │
  └─────────────────────────────────────────────────────────────────┘ 
```

---

## Neural Network Components

### 1. Fine-tuned Sentence Embedder

Base model: `sentence-transformers/all-MiniLM-L6-v2`

Fine-tuned on baseball-specific sentence pairs (game recap sentences ↔ query-style paraphrases) using **MultipleNegativesRankingLoss**. CosineSimilarityLoss was tried first and collapsed to a degenerate solution (loss → 0.012, Spearman degraded) — MNRL treats all other examples in the batch as negatives, preventing this.

| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| Val Spearman | — | **0.844** |
| Val Pearson | — | **0.908** |
| Train loss | — | 0.2785 (10 epochs) |

### 2. Rule-Based Notability Scorer

Seven domain rules determine how "notable" a game is for reranking. Each rule that fires contributes 1/7 to a continuous notability score (0.0–1.0), which is blended with the FAISS similarity score to push genuinely newsworthy games to the top of the context window:

| Rule | Threshold |
|------|-----------|
| Close game | margin ≤ 1 run |
| Extra innings | any extra-inning game |
| Offensive explosion | ≥ 14 total runs |
| Blowout | margin ≥ 8 runs |
| Shutout | winning team held opponent scoreless |
| HR barrage | ≥ 4 total home runs |
| Dominant pitching | winning pitcher ≥ 11 strikeouts |

**Blended score:** `sim_score + 0.25 × notability_score`

An MLP trained on the same features was explored (W&B grid search, 24 configs, val macro F1 = 0.9866) but replaced — it was learning to approximate these same rules from their own outputs, adding no new signal. The rule-based scorer is simpler, interpretable, and equivalent for reranking.

---

## Results

The reranker consistently surfaces shutouts, extra-inning games, and high-margin results over low-scoring but textually-similar games that naive FAISS would rank first. See the notebook (Part 5.2) for the before/after retrieval comparison.

Sample briefing output from April 29, 2026: [`example.md`](example.md)

---

## Repo File Structure

```
jeff-passball/
├── src/mlb_rag/
│   ├── data_ingestion.py      # MLB Stats API → MLBChunk objects
│   ├── historical_data.py     # GameFeatures dataclass + feature extraction
│   ├── auto_labeler.py        # 7-rule binary labeler for training data
│   ├── embedder.py            # Fine-tuned embedder + FAISS store builder
│   ├── trend_classifier.py    # MLP architecture + trainer
│   ├── sweep_train.py         # W&B grid search (24 configurations)
│   ├── commentary.py          # Full RAG pipeline: retrieve → rerank → generate
│   └── novelty.py             # Historical novelty fact generator
├── app/
│   ├── main.py                # FastAPI web server + API endpoints
│   ├── pipeline.py            # Server state, initialization, daily refresh
│   ├── live_stats.py          # Real-time standings, player stats, stat leaders
│   ├── historical_lookup.py   # Season aggregates + record lookups from NPZ
│   └── static/index.html      # Single-page web UI
├── scripts/
│   ├── start_server.sh        # Launch web server (Claude-only)
│   └── start_all.sh           # Launch ollama daemon + web server
├── data/
│   └── game_features_all.npz  # 7,802 games (2023–2026): features + recap text
├── notebooks/
│   └── mlb_rag_notebook.ipynb # Full experiment notebook (Parts 1–7)
└── example.md                 # Sample briefing output
```

Checkpoints default to `checkpoints/` in the repo root (configurable via `--out` in `train_classifier.py` and `CHECKPOINT_DIR` in `scripts/`):
- `mlb-minilm-finetuned/` — fine-tuned sentence transformer
- `trend_classifier.pt` — trained MLP

---

## Setup

```bash
pip install torch sentence-transformers faiss-cpu fastapi uvicorn \
            anthropic wandb scikit-learn numpy pandas matplotlib seaborn
```

`ANTHROPIC_API_KEY` must be set in your environment for the daily briefing. Queries can optionally run through a local Llama 3.1 8B model via ollama (see below).

---

## Running the Notebook

```bash
cd notebooks/
jupyter lab mlb_rag_notebook.ipynb
```

Run cells in order. The notebook covers:
- **Part 1** — Problem motivation
- **Part 2** — Dataset (7,802 games, EDA, auto-labeling)
- **Part 3** — Embedding fine-tuning (loss function comparison, MNRL)
- **Part 4** — Classifier MLP (architecture, training, evaluation)
- **Part 5** — Full RAG pipeline + live briefing demo
- **Part 6** — Ablation study (W&B sweep results)
- **Part 7** — Reflection and future work

The briefing demo cell (Part 5.3) requires `ANTHROPIC_API_KEY`.

---

## Running the Web App

```bash
export ANTHROPIC_API_KEY="sk-..."

# Option A: Claude for all calls (briefing + queries)
bash scripts/start_server.sh [port]

# Option B: Llama 3.1 8B for queries, Claude for daily briefing
bash scripts/start_all.sh [port]
```

The app runs at `http://localhost:8080`.

---

## Local LLM (ollama + Llama 3.1 8B)

The system supports routing per-query inference to a local **Llama 3.1 8B** model, reducing Claude API credit usage. The daily briefing stays on Claude (higher quality; runs once per day and is cached to disk).

**Requirements:** ~5 GB VRAM, [ollama](https://ollama.com) installed and on your `PATH`.

**One-time setup:**
```bash
# Install ollama from https://ollama.com, then pull the model (~4.9 GB)
ollama pull llama3.1:8b
```

**Running with local model:**
```bash
# start_all.sh handles daemon startup + model check automatically
bash scripts/start_all.sh

# Or manually:
ollama serve &
export OLLAMA_MODEL="llama3.1:8b"
bash scripts/start_server.sh
```

**Runtime toggle:** The LLM badge in the UI header is clickable — switching between Claude and Llama 3.1 8B takes effect immediately without restarting the server. The active model is also exposed at `GET /api/model` and can be changed via `POST /api/model {"use_local": true}`.

**Routing logic** (`src/mlb_rag/commentary.py`):
```python
def _call_llm_messages(system_prompt, messages, max_tokens=1000):
    if _use_local_llm and os.environ.get("OLLAMA_MODEL", "").strip():
        return _call_ollama_messages(system_prompt, messages, max_tokens)
    return _call_claude_messages(system_prompt, messages, max_tokens)
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/briefing` | Today's cached briefing |
| `POST /api/query` | Chat query with conversation history |
| `GET /api/scores` | Yesterday's game scores |
| `GET /api/health` | Pipeline status + component flags |
| `GET /api/model` | Current LLM selection |
| `POST /api/model` | Toggle Claude ↔ local LLM at runtime |
| `POST /api/refresh` | Force-regenerate the daily briefing |

---

## Web App Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS / HTML (single-page, no framework) |
| Vector store | FAISS (in-memory) |
| Primary LLM | Anthropic Claude (`claude-sonnet-4-5`) |
| Optional LLM | Llama 3.1 8B via ollama (query inference only) |
| ML runtime | PyTorch + sentence-transformers |
| Data | MLB Stats API (live) + `.npz` archive (historical) |

---

## Web App Features

Beyond the core RAG pipeline, the deployed app adds:
- **Live standings** injected into every query context (5-min cache)
- **Player season stats** auto-fetched when a name is detected in the query
- **Stat leaderboards** injected for MVP/rankings/HR-race queries
- **Yesterday's scores** grid on the home page
- **Historical queries** — "how many extra innings games in 2024?" answered directly from the 7,800-game NPZ archive without an LLM guess
- **Query expansion** — extreme-stat questions ("longest game", "biggest blowout") trigger supplementary FAISS retrieval passes before reranking
- **Local LLM toggle** — clickable badge switches between Claude and Llama 3.1 8B at runtime

---

## Issues and Discussion

Though support for directly using MLB data was added to the "query" feature, there are often times where the models screw up and do not understand the query as the user intended. The models very quickly hallicinate.

The MLP is overkill, especially for a binary problem. However, it is still a good demonstration of `wandb` best practices and model training. Most of the "neural network component" is done with fine-tuning the open source model to baseball lingo.