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

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.pipeline as pipeline
from src.mlb_rag.commentary import answer_query


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


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    s = pipeline.get_state()
    if not s.ready or s.store is None:
        raise HTTPException(status_code=503, detail="Pipeline is still initializing — try again in a moment")
    answer = answer_query(req.question, s.store, s.embedder, classifier=s.classifier)
    return {"answer": answer}


@app.post("/api/refresh")
def refresh():
    pipeline.refresh()
    s = pipeline.get_state()
    return {
        "status": "ok",
        "last_refresh": s.last_refresh.isoformat() if s.last_refresh else None,
    }


@app.get("/api/health")
def health():
    s = pipeline.get_state()
    return {
        "embedder": s.embedder_type,
        "classifier": s.classifier_loaded,
        "novelty_enabled": s.novelty_enabled,
        "last_refresh": s.last_refresh.isoformat() if s.last_refresh else None,
        "ready": s.ready,
    }
