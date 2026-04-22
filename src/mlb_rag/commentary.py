"""
commentary.py

RAG-powered MLB commentary generator.

Retrieves relevant context from the vector store, then prompts
the Claude API to generate broadcast-style analysis grounded
entirely in the retrieved data.

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

CLAUDE_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def _call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
    """
    Call the Claude API and return the response text.
    Reads ANTHROPIC_API_KEY from environment.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ── Prompt Templates ───────────────────────────────────────────────────────────

BROADCASTER_SYSTEM = """You are a knowledgeable MLB broadcaster giving a daily briefing.
Your job is to synthesize the provided game data and stats into engaging,
insightful commentary — the kind a TV analyst gives viewers before a game or
at the top of a highlight show. 

RULES:
- Only reference information from the provided context. Do not invent stats or scores.
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


# ── RAG Commentary Functions ───────────────────────────────────────────────────

def answer_query(
    query: str,
    store: MLBVectorStore,
    embedder: MLBEmbedder,
    top_k: int = 6,
    verbose: bool = False
) -> str:
    """
    Answer a natural language MLB question using RAG.

    1. Embed the query
    2. Retrieve top-k semantically similar chunks
    3. Pass context + query to Claude for grounded response

    Args:
        query: Any MLB question ("who won last night?", "AL East standings?")
        store: Populated vector store.
        embedder: Sentence transformer embedder.
        top_k: Number of chunks to retrieve.
        verbose: If True, print retrieved context.

    Returns:
        Generated commentary string.
    """
    # Step 1 & 2: Retrieve
    results = query_store(query, store, embedder, top_k=top_k)
    context = build_context_string(results)

    if verbose:
        print("\n── Retrieved Context ──")
        print(context)
        print("── End Context ──\n")

    # Step 3: Generate
    user_prompt = f"""CONTEXT (retrieved MLB data):
{context}

QUESTION: {query}

Please answer based only on the context above."""

    return _call_claude(BROADCASTER_SYSTEM, user_prompt)


def generate_daily_briefing(
    store: MLBVectorStore,
    embedder: MLBEmbedder,
    date: str = None
) -> str:
    """
    Generate a full daily MLB briefing — like an ESPN top-of-show segment.

    Runs multiple targeted retrievals to surface:
    - Yesterday's key results
    - Current standings storylines
    - Notable trends

    Args:
        store: Populated vector store.
        embedder: Sentence transformer embedder.
        date: Date string for the briefing (defaults to today).

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

    # Sort by similarity score
    all_results.sort(key=lambda x: x[1], reverse=True)
    context = build_context_string(all_results[:10])   # cap at 10 chunks

    user_prompt = f"""CONTEXT (MLB data as of {date}):
{context}

Generate a complete daily MLB briefing for {date}. Structure it as:
1. HEADLINE RESULTS - the 2-3 most notable outcomes
2. STANDINGS WATCH - any interesting division races or movement
3. STORYLINE OF THE DAY - one compelling narrative from the data

Ground everything in the provided context."""

    return _call_claude(BROADCASTER_SYSTEM, user_prompt, max_tokens=600)


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from embedder import MLBVectorStore, MLBEmbedder, build_vector_store
    from data_ingestion import ingest_mlb_data

    # Build the pipeline end-to-end
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
