# AI Usage Disclosure

## What I used AI for

**Web app development (FastAPI + frontend)**  
The `app/` directory — the FastAPI server, HTML/CSS/JS single-page UI, and the live stats augmentation layer (`live_stats.py`, `historical_lookup.py`) — was built collaboratively with Claude Code. I described what I wanted (live standings injection, yesterday's scores grid, historical aggregate queries, runtime LLM toggle) and Claude Code wrote the implementation, which I reviewed and tested on ACET before accepting.

**Bug fixes and iteration**  
Several data bugs were caught and fixed with AI assistance: the MLB Stats API `/stats/leaders` endpoint returning stale 2025 data for an in-progress 2026 season (switched to `/stats?sortStat=` per-category calls), player name detection failing on lowercase queries (title-casing fix in `_extract_names()`), and the `+00:00Z` double-timezone suffix breaking the briefing timestamp display.

**Code review and debugging**  
I used Claude Code to explain error tracebacks, check function signatures against API responses, and catch logic errors in the historical lookup module before deploying.

**Writing assistance**  
The README was drafted with Claude Code based on my descriptions of each component.

---

## What I did not use AI for

**All core machine learning work:**
- Designing the two neural network components (embedder fine-tuning strategy, MLP architecture)
- Writing `src/mlb_rag/` — data ingestion, feature extraction, auto-labeler rules, embedder training loop, classifier trainer, W&B sweep configuration, reranker blending logic, novelty generator
- Choosing MNRL over CosineSimilarityLoss (I ran both and observed the collapse firsthand)
- Designing the temporal train/val/test split and interpreting results
- Selecting the final production model from the sweep (reading the W&B charts)
- Writing the notebook — the analysis, explanations, and visualizations in Parts 1–7

**System design decisions:**
- The overall RAG pipeline architecture
- The decision to blend classifier probability with FAISS similarity score (and the 0.25 boost weight)
- The 7-rule auto-labeling strategy and the min_rules=2 threshold
- The boxscore hydration fix that activated the HR/strikeout labeling rules

---

## How I verified AI output

- **All web app code was tested on ACET** before considering it done. I ran queries, checked API responses with `curl`, and verified UI behavior in the browser. Several bugs were caught this way (wrong JSON field names, API response structure mismatches).
- **The historical lookup module** was verified by running test queries locally and comparing output against known values from the NPZ file.
- **The stat leaders fix** was verified by directly querying the MLB Stats API and confirming Judge leads the 2026 HR race with 12, not 27.
- I never merged AI-written code that I didn't read and understand line-by-line.

---

## What I learned from the interaction

Using an AI coding assistant for the deployment layer let me ship a much more capable web app than I would have built alone in the time available — live player stats, historical queries, and the local LLM integration all came from being able to describe what I wanted and get working code quickly.

The more interesting lesson was about **where AI assistance helps and where it doesn't**. The ML core — picking a loss function, interpreting training curves, deciding on a labeling strategy — required me to actually run experiments and understand what the numbers meant. Claude Code couldn't tell me that CosineSimilarityLoss was going to collapse before I tried it; it just helped me implement both options cleanly so I could find out. The value was in reducing implementation friction, not in replacing experimental judgment.
