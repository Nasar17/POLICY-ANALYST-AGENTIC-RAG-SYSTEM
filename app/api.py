"""
FastAPI backend for the Florida Statute Research Assistant.

Endpoints:
  GET  /health       liveness check
  POST /ask          run the agentic RAG pipeline

Models are loaded once at startup via the lifespan context and shared
across all requests through module-level state.

Run locally:
    uvicorn app.api:app --reload --port 8000
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retrieval.retriever import HybridRetriever
from retrieval.pipeline import RAGPipeline
from agent.graph import AgenticRAG

ABSTAIN_MARKER = "not found in the source material"

_pipeline: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models…")
    retriever = HybridRetriever()
    _pipeline["plain"]   = RAGPipeline(retriever=retriever)
    _pipeline["agentic"] = AgenticRAG(retriever=retriever)
    print("Ready.")
    yield


app = FastAPI(
    title="Florida Statute Research Assistant",
    description="Agentic RAG over Florida Statute Chapter 119 (Public Records Act).",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    mode: str = "agentic"   # "agentic" | "plain"


class Source(BaseModel):
    section_num:   str
    section_title: str
    source_url:    str | None = None


class AskResponse(BaseModel):
    question:  str
    answer:    str
    sources:   list[Source]
    is_abstain: bool
    mode:      str
    time_s:    float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dedup_sources(chunks) -> list[Source]:
    seen: set[str] = set()
    out: list[Source] = []
    for c in chunks:
        key = c.meta["section_num"]
        if key not in seen:
            seen.add(key)
            out.append(Source(
                section_num=c.meta["section_num"],
                section_title=c.meta["section_title"],
                source_url=c.meta.get("source_url"),
            ))
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "pipelines": list(_pipeline.keys())}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.mode not in _pipeline:
        raise HTTPException(status_code=400,
                            detail=f"mode must be 'agentic' or 'plain', got '{req.mode}'")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    t0 = time.time()
    result = _pipeline[req.mode].ask(req.question.strip())
    elapsed = round(time.time() - t0, 2)

    answer = result["answer"]
    return AskResponse(
        question=req.question,
        answer=answer,
        sources=_dedup_sources(result.get("chunks", [])),
        is_abstain=ABSTAIN_MARKER in answer.lower(),
        mode=req.mode,
        time_s=elapsed,
    )
