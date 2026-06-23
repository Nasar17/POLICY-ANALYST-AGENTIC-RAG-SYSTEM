"""
Shared graph state for the agentic RAG loop.

Every node receives the full state and returns a dict of fields to update.
LangGraph merges updates into the existing state (last-write-wins for scalars,
we reset list fields explicitly on retry so replacement semantics are fine).
"""

from typing import TypedDict
from retrieval.retriever import Chunk


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    question: str              # original user question (never mutated)

    # ── Analyze ───────────────────────────────────────────────────────────────
    sub_questions: list[str]   # decomposed search queries (1 for simple, N for complex)
    metadata_filters: dict     # e.g. {"chapter_num": "119"} — applied to retrieval

    # ── Retrieve ──────────────────────────────────────────────────────────────
    all_chunks: list[Chunk]    # pooled chunks from all sub_questions this pass

    # ── Grade ─────────────────────────────────────────────────────────────────
    grade: str                 # "sufficient" | "insufficient"
    grade_reasoning: str

    # ── Loop control ──────────────────────────────────────────────────────────
    retry_count: int           # number of retrieve-grade retries so far
    searched_queries: list[str]  # accumulated for the abstain message

    # ── Output ────────────────────────────────────────────────────────────────
    answer: str                # final answer (from generate or abstain node)
