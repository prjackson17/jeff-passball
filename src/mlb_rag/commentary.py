"""
commentary.py

RAG-powered MLB commentary generator.

Retrieves relevant context from the vector store, reranks chunks using
a rule-based notability scorer (notable games bubble up), then prompts
the LLM to generate broadcast-style analysis grounded in the retrieved data.

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import os
import requests
import json
from datetime import datetime
from typing import List, Tuple, Optional
from src.mlb_rag.data_ingestion import MLBChunk
from src.mlb_rag.embedder import MLBVectorStore, MLBEmbedder, query_store


# ── Claude API ─────────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Runtime flag — toggled via /api/model without a server restart.
# Initialised from OLLAMA_MODEL env var so start_server.sh / start_all.sh
# still work as before.
_use_local_llm: bool = bool(os.environ.get("OLLAMA_MODEL", "").strip())


def set_llm_mode(use_local: bool) -> None:
    global _use_local_llm
    _use_local_llm = use_local


def get_llm_mode() -> str:
    return "local" if _use_local_llm else "claude"


def _call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
    """Call Claude with a single user turn."""
    return _call_claude_messages(
        system_prompt,
        [{"role": "user", "content": user_prompt}],
        max_tokens,
    )


def _call_claude_messages(system_prompt: str, messages: list, max_tokens: int = 1000) -> str:
    """Call Claude with a full conversation history (multi-turn support)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_ollama_messages(system_prompt: str, messages: list, max_tokens: int = 1000) -> str:
    """Call a local ollama model. OLLAMA_MODEL env var selects the model."""
    model    = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_llm_messages(system_prompt: str, messages: list, max_tokens: int = 1000) -> str:
    """Route to local ollama or Claude based on the runtime _use_local_llm flag."""
    if _use_local_llm and os.environ.get("OLLAMA_MODEL", "").strip():
        return _call_ollama_messages(system_prompt, messages, max_tokens)
    return _call_claude_messages(system_prompt, messages, max_tokens)


# ── Query expansion for extreme-stat questions ────────────────────────────────

_EXTRA_QUERIES: list = [
    ({"longest", "most innings", "extra inning", "marathon", "went long"},
     "extra innings game went 10 11 12 13 innings"),
    ({"most runs", "highest scoring", "blowout", "lopsided", "biggest win"},
     "high scoring blowout many runs big margin"),
    ({"walk-off", "walkoff", "walk off", "comeback", "came back"},
     "walk-off win comeback last inning victory"),
    ({"shutout", "shut out", "no-hit", "no hitter", "perfect game"},
     "shutout no hits pitcher dominant"),
    ({"closest", "one run", "tight", "squeaker"},
     "one run game close tight final"),
]


def _extra_retrieval_queries(query: str) -> list[str]:
    q = query.lower()
    return [rq for keywords, rq in _EXTRA_QUERIES if any(kw in q for kw in keywords)]


# ── Prompt Templates ───────────────────────────────────────────────────────────

BROADCASTER_SYSTEM = """You are a knowledgeable MLB broadcaster giving a daily briefing.
Your job is to synthesize the provided game data and stats into engaging,
insightful commentary — the kind a TV analyst gives viewers before a game or
at the top of a highlight show. 

RULES:
- Only reference information from the provided context. Do not invent stats or scores.
- CRITICAL: Player team affiliations must come ONLY from the context. Never use your training data for current rosters — players change teams frequently and your training data is outdated. If the context says a player is on a team, that is correct.
- Be specific: name teams, players, and scores when available.
- Write in a natural, confident broadcast voice. Not robotic.
- If the context doesn't contain enough info to answer, say so honestly.
- Keep responses concise: 150-250 words unless asked for more.
"""


def build_context_string(results: List[Tuple[MLBChunk, float]]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    if not results:
        return "No relevant data found."
    lines = []
    for i, (chunk, score) in enumerate(results, 1):
        lines.append(f"[Source {i} | type={chunk.chunk_type} | similarity={score:.3f}]")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


# ── Rule-Based Reranker ────────────────────────────────────────────────────────

def _chunk_to_gamefeatures(chunk: MLBChunk):
    """Convert chunk metadata to GameFeatures. Returns None for non-game chunks."""
    meta = getattr(chunk, "metadata", None)
    if meta is None:
        return None
    from src.mlb_rag.historical_data import GameFeatures
    keys = GameFeatures.feature_names()
    if not any(k in meta for k in keys):
        return None
    try:
        kwargs = {k: float(meta.get(k, 0.0)) for k in keys}
        return GameFeatures(game_pk=0, date="", **kwargs)
    except (TypeError, ValueError):
        return None


def rerank_results(
        results: List[Tuple[MLBChunk, float]],
        notable_boost: float = 0.25,
) -> List[Tuple[MLBChunk, float]]:
    """
    Rerank retrieved chunks by blending semantic similarity with a rule-based
    notability score (fraction of auto-labeler rules that fire, 0.0–1.0).

    Notable game chunks get a `notable_boost` added to their similarity score,
    bubbling them toward the top. Chunks without game features (standings) are
    passed through unchanged.
    """
    from src.mlb_rag.auto_labeler import notability_score
    reranked = []
    for chunk, sim_score in results:
        gf = _chunk_to_gamefeatures(chunk)
        if gf is not None:
            blended = sim_score + notable_boost * notability_score(gf)
            reranked.append((chunk, blended))
        else:
            reranked.append((chunk, sim_score))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ── RAG Commentary Functions ───────────────────────────────────────────────────

def answer_query(
        query: str,
        store: MLBVectorStore,
        embedder: MLBEmbedder,
        top_k: int = 8,
        verbose: bool = False,
        history: "list | None" = None,
        extra_context: str = "",
) -> str:
    """
    Answer a natural language MLB question using RAG.

    Retrieves relevant chunks, reranks with the rule-based notability scorer,
    injects context into the system prompt, and calls the LLM. Supports
    multi-turn conversation via the history parameter.

    Args:
        query:         Current user question (used for retrieval).
        store:         Populated vector store.
        embedder:      Sentence transformer embedder.
        top_k:         Number of chunks to retrieve.
        verbose:       If True, print retrieved context.
        history:       Full conversation as [{role, content}] list.
                       The current question should be the last user entry.
                       When None, falls back to single-turn behavior.
        extra_context: Additional context (e.g. live standings/stats) to
                       prepend before the RAG chunks.

    Returns:
        Generated commentary string.
    """
    results = query_store(query, store, embedder, top_k=top_k)

    # Supplementary retrieval for extreme-stat questions (longest game, etc.)
    extra_qs = _extra_retrieval_queries(query)
    if extra_qs:
        seen = {chunk.text for chunk, _ in results}
        for eq in extra_qs:
            for chunk, score in query_store(eq, store, embedder, top_k=6):
                if chunk.text not in seen:
                    results.append((chunk, score))
                    seen.add(chunk.text)

    results = rerank_results(results)

    rag_context = build_context_string(results)

    if verbose:
        print("\n── Retrieved Context ──")
        print(rag_context)
        print("── End Context ──\n")

    full_context = (extra_context + "\n\n--- RECENT GAME DATA ---\n" + rag_context
                    if extra_context else rag_context)

    system = BROADCASTER_SYSTEM + f"\n\nCONTEXT (use this data to answer):\n{full_context}"

    if history:
        messages = history
    else:
        messages = [{"role": "user", "content": query}]

    return _call_llm_messages(system, messages)


def generate_daily_briefing(
        store: MLBVectorStore,
        embedder: MLBEmbedder,
        date: str = None,
        X_hist=None,
        feature_names=None,
) -> str:
    """
    Generate a full daily MLB briefing — like an ESPN top-of-show segment.

    Runs multiple targeted retrievals to surface:
    - Yesterday's key results
    - Current standings storylines
    - Notable trends

    Args:
        store:         Populated vector store.
        embedder:      Sentence transformer embedder.
        date:          Date string for the briefing (defaults to today).
        X_hist:        Historical game feature array for novelty facts.
        feature_names: Feature name list corresponding to X_hist columns.

    Returns:
        Full briefing as a string.
    """
    if date is None:
        date = datetime.today().strftime("%B %d, %Y")

    # Multi-query retrieval to cover different angles
    queries = [
        "game results scores winners losers",
        "standings division leaders",
        "close games extra innings comeback",
    ]

    all_results = []
    seen_texts = set()
    for q in queries:
        results = query_store(q, store, embedder, top_k=4)
        for chunk, score in results:
            if chunk.text not in seen_texts:
                all_results.append((chunk, score))
                seen_texts.add(chunk.text)

    all_results = rerank_results(all_results)

    context = build_context_string(all_results[:10])   # cap at 10 chunks

    # Generate novelty facts if historical data is available
    facts_block = ""
    if X_hist is not None and feature_names is not None:
        from src.mlb_rag.novelty import generate_briefing_facts
        game_chunks = [c for c, _ in all_results if c.chunk_type == "game_recap"]
        facts_block = generate_briefing_facts(game_chunks, X_hist, feature_names)

    novelty_section = f"\n\n{facts_block}" if facts_block else ""

    user_prompt = f"""CONTEXT (MLB data as of {date}):
{context}{novelty_section}

Generate a complete daily MLB briefing for {date}. Structure it as:
1. HEADLINE RESULTS - the 2-3 most notable outcomes
2. STANDINGS WATCH - any interesting division races or movement
3. STORYLINE OF THE DAY - one compelling narrative from the data
4. CRAZY FACTS - weave in any notable facts from the NOTABLE FACTS section above in an engaging, broadcast-style way (e.g. "By the way, that was only the 4th time in three seasons we've seen..."). Skip this section if no facts were provided.

Ground everything in the provided context."""

    return _call_claude(BROADCASTER_SYSTEM, user_prompt, max_tokens=600)


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.mlb_rag.embedder import MLBVectorStore, MLBEmbedder, build_vector_store
    from src.mlb_rag.data_ingestion import ingest_mlb_data

    print("=== MLB RAG Pipeline Test ===\n")

    chunks = ingest_mlb_data(days_back=2)
    embedder = MLBEmbedder()
    store = build_vector_store(chunks, embedder=embedder, save=False)

    # Test a direct query
    print("\n── Query Test ──")
    q = "which teams won yesterday?"
    print(f"Q: {q}")
    answer = answer_query(q, store, embedder, verbose=True)
    print(f"A: {answer}")

    # Test the daily briefing
    print("\n── Daily Briefing ──")
    briefing = generate_daily_briefing(store, embedder)
    print(briefing)