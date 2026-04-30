"""
main.py

FastAPI web app for the MLB RAG briefing pipeline.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.pipeline as pipeline
from app.live_stats import augment_query_context, fetch_yesterday_scores
from src.mlb_rag.commentary import answer_query
import src.mlb_rag.commentary as commentary


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.initialize()
    yield


app = FastAPI(title="MLB RAG Briefing", lifespan=lifespan)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(_static_dir, "index.html")) as f:
        return f.read()


@app.get("/api/briefing")
def get_briefing():
    s = pipeline.get_state()
    return {
        "briefing": s.briefing,
        "last_refresh": s.last_refresh.isoformat() if s.last_refresh else None,
    }


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    history: List[Message] = []


@app.post("/api/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    s = pipeline.get_state()
    if not s.ready:
        raise HTTPException(status_code=503, detail="Pipeline is still initializing — try again in a moment")

    # Prefer the larger query store; fall back to briefing store
    store = s.query_store or s.store

    extra = augment_query_context(req.question)

    # Build history for Claude — include current question as last entry
    history = [{"role": m.role, "content": m.content} for m in req.history]
    if not history or history[-1]["content"] != req.question:
        history.append({"role": "user", "content": req.question})

    answer = answer_query(
        req.question,
        store,
        s.embedder,
        classifier=s.classifier,
        history=history,
        extra_context=extra,
    )
    return {"answer": answer}


@app.get("/api/scores")
def get_scores():
    from datetime import datetime, timedelta
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%B %d, %Y")
    return {"date": yesterday, "games": fetch_yesterday_scores()}


@app.post("/api/refresh")
def refresh():
    """Called by cron job — not exposed in UI."""
    pipeline.refresh()
    s = pipeline.get_state()
    return {
        "status": "ok",
        "last_refresh": s.last_refresh.isoformat() if s.last_refresh else None,
    }


class ModelRequest(BaseModel):
    use_local: bool


@app.get("/api/model")
def get_model():
    local_available = bool(os.environ.get("OLLAMA_MODEL", "").strip())
    return {
        "active": commentary.get_llm_mode(),
        "local_available": local_available,
        "local_model": os.environ.get("OLLAMA_MODEL", ""),
    }


@app.post("/api/model")
def set_model(req: ModelRequest):
    if req.use_local and not os.environ.get("OLLAMA_MODEL", "").strip():
        raise HTTPException(status_code=400, detail="OLLAMA_MODEL is not configured on this server")
    commentary.set_llm_mode(req.use_local)
    return {"active": commentary.get_llm_mode()}


@app.get("/api/health")
def health():
    s = pipeline.get_state()
    return {
        "embedder": s.embedder_type,
        "classifier": s.classifier_loaded,
        "novelty_enabled": s.novelty_enabled,
        "query_store_ready": s.query_store is not None,
        "query_days": int(os.environ.get("QUERY_DAYS_BACK", 14)),
        "local_model": os.environ.get("OLLAMA_MODEL", ""),
        "last_refresh": s.last_refresh.isoformat() if s.last_refresh else None,
        "ready": s.ready,
    }
