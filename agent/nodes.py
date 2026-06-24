"""
Node functions for the agentic RAG loop.

Nodes (each takes AgentState, returns partial-state dict):
  analyze  — decompose question into sub-questions, extract metadata filters
  retrieve — hybrid search for every sub-question; pool results
  grade    — LLM judges whether pooled chunks are sufficient to answer
  rewrite  — LLM rewrites sub-questions for a better second attempt
  generate — grounded, cited answer over the pooled chunks
  abstain  — clean "not found" response when retries are exhausted

All LLM calls use Groq (llama-3.3-70b-versatile) at temperature=0.
JSON output is enforced via response_format and validated before use.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from agent.state import AgentState
from retrieval.retriever import HybridRetriever, Chunk

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "llama-3.1-8b-instant"
TOP_K_PER_SUBQ = 5   # chunks retrieved per sub-question
MAX_CHUNKS = 8        # dedup ceiling passed to grade/generate
_LLM_RETRIES = 6
CHUNK_DISPLAY_CHARS = 1000  # keep requests under the 6K TPM limit

_groq: Groq | None = None


def _client() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq


def _parse_wait(err_msg: str) -> float:
    mins = re.search(r"(\d+)m", err_msg)
    secs = re.search(r"(\d+(?:\.\d+)?)s", err_msg)
    total = 0.0
    if mins:
        total += int(mins.group(1)) * 60
    if secs:
        total += float(secs.group(1))
    return max(total, 30.0)


def _call(system: str, user: str) -> str:
    for attempt in range(_LLM_RETRIES + 1):
        try:
            resp = _client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            if attempt == _LLM_RETRIES:
                raise
            wait = _parse_wait(str(e))
            print(f"\n[nodes] Rate limit — waiting {wait:.0f}s "
                  f"(attempt {attempt+1}/{_LLM_RETRIES})...")
            time.sleep(wait + 5)


# ── analyze ───────────────────────────────────────────────────────────────────

_ANALYZE_SYSTEM = """\
You are a query-analysis assistant for a Florida statute research system (Chapter 119, Public Records).

For the user question:
1. Split it into the smallest set of distinct sub-questions that each require a separate lookup.
   A single-part question produces exactly one sub-question.
2. Extract any metadata filters clearly implied by the question
   (chapter_num, section_num).  Omit a field if not implied.
3. Keep sub-questions as plain natural-language search queries — no legalese invented.

Respond ONLY with valid JSON:
{
  "sub_questions": ["query 1", ...],
  "metadata_filters": {"chapter_num": "119"},
  "reasoning": "one sentence"
}"""


def analyze(state: AgentState) -> dict:
    question = state["question"]
    raw = _call(_ANALYZE_SYSTEM, f"Question: {question}")
    try:
        parsed = json.loads(raw)
        sub_qs = parsed.get("sub_questions") or [question]
        filters = parsed.get("metadata_filters") or {}
    except (json.JSONDecodeError, KeyError):
        sub_qs = [question]
        filters = {}

    # Initialise loop-control fields on first entry
    return {
        "sub_questions":    sub_qs,
        "metadata_filters": filters,
        "retry_count":      0,
        "searched_queries": list(sub_qs),
        "all_chunks":       [],
        "grade":            "",
        "grade_reasoning":  "",
        "answer":           "",
    }


# ── retrieve ──────────────────────────────────────────────────────────────────

def make_retrieve(retriever: HybridRetriever):
    """
    Return a retrieve node closed over the shared HybridRetriever instance.
    Accumulates chunks across retries — existing chunks are preserved so that
    a worse rewrite round never loses evidence from a better earlier round.
    """
    def retrieve(state: AgentState) -> dict:
        # Seed with chunks already in state (empty on first pass)
        existing = {c.id: c for c in state.get("all_chunks", [])}
        seen_ids: set[str] = set(existing.keys())
        pooled: list[Chunk] = list(existing.values())

        for sub_q in state["sub_questions"]:
            chunks = retriever.retrieve(
                sub_q,
                top_k=TOP_K_PER_SUBQ,
                where=state.get("metadata_filters") or None,
            )
            for c in chunks:
                if c.id not in seen_ids and len(pooled) < MAX_CHUNKS:
                    seen_ids.add(c.id)
                    pooled.append(c)

        return {"all_chunks": pooled}

    return retrieve


# ── grade ─────────────────────────────────────────────────────────────────────

_GRADE_SYSTEM = """\
You are a relevance grader for a Florida statute research system.

Given the original question and the statute chunks that were retrieved, decide:
- "sufficient"   — the chunks together contain enough information to produce a
                   grounded, cited answer to the question.
- "insufficient" — key information is missing; the answer would require guessing
                   or the chunks are entirely about something else.

Respond ONLY with valid JSON:
{
  "verdict": "sufficient" | "insufficient",
  "reasoning": "one sentence"
}"""


def grade(state: AgentState) -> dict:
    chunks = state["all_chunks"]
    if not chunks:
        return {"grade": "insufficient", "grade_reasoning": "No chunks retrieved."}

    context = "\n\n".join(
        f"[§ {c.meta['section_num']}] {c.doc[:400]}" for c in chunks
    )
    user = (
        f"Original question: {state['question']}\n\n"
        f"Retrieved chunks:\n{context}"
    )
    raw = _call(_GRADE_SYSTEM, user)
    try:
        parsed = json.loads(raw)
        verdict   = parsed.get("verdict", "insufficient")
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        verdict, reasoning = "insufficient", "Parse error in grade response."

    if verdict not in ("sufficient", "insufficient"):
        verdict = "insufficient"

    return {"grade": verdict, "grade_reasoning": reasoning}


# ── rewrite ───────────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """\
You are a query-rewriting assistant for a Florida statute research system (Chapter 119, Public Records Act).

The previous retrieval was graded insufficient. You will be given:
- The original question
- Queries already tried (do NOT simply rephrase these)
- The statute sections already retrieved (do NOT target these again)
- What the grader says is missing

Your job: write 1–3 SHORT search queries that target the MISSING information.
Each query must be meaningfully different from every query already tried.
Stay within Florida Statute Chapter 119. Use plain natural language, not legalese.

Respond ONLY with valid JSON:
{
  "rewritten_queries": ["query 1", ...],
  "missing_info": "one sentence describing what is being targeted"
}"""


def rewrite(state: AgentState) -> dict:
    found_sections = sorted({c.meta["section_num"] for c in state["all_chunks"]})
    user = (
        f"Original question: {state['question']}\n"
        f"Queries already tried: {state['searched_queries']}\n"
        f"Statute sections already retrieved: {found_sections}\n"
        f"What the grader says is missing: {state['grade_reasoning']}\n"
    )
    raw = _call(_REWRITE_SYSTEM, user)
    try:
        parsed = json.loads(raw)
        new_qs = parsed.get("rewritten_queries") or state["sub_questions"]
    except (json.JSONDecodeError, KeyError):
        new_qs = state["sub_questions"]

    return {
        "sub_questions":    new_qs,
        "retry_count":      state["retry_count"] + 1,
        "searched_queries": state["searched_queries"] + new_qs,
        # all_chunks intentionally NOT cleared — retrieve accumulates on top
        "grade":            "",
        "grade_reasoning":  "",
    }


# ── generate ──────────────────────────────────────────────────────────────────

_GENERATE_SYSTEM = """\
You are a Florida statute research assistant.

Answer the question using ONLY the statute sections provided.
For every factual claim, cite the exact section (e.g., "§ 119.07(1)(a)").
If multiple sub-questions were asked, address each one clearly.
Never invent statute content not present in the provided text."""


def generate(state: AgentState) -> dict:
    chunks = state["all_chunks"]
    context = "\n\n---\n\n".join(
        f"[Florida Statute § {c.meta['section_num']} — {c.meta['section_title']}]\n"
        + (c.doc.split("\n\n", 1)[-1] if "\n\n" in c.doc else c.doc)[:CHUNK_DISPLAY_CHARS]
        for c in chunks
    )
    user_msg = (
        f"STATUTE SECTIONS:\n\n{context}\n\n"
        f"QUESTION: {state['question']}\n\nANSWER:"
    )
    for attempt in range(_LLM_RETRIES + 1):
        try:
            resp = _client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _GENERATE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return {"answer": resp.choices[0].message.content.strip()}
        except RateLimitError as e:
            if attempt == _LLM_RETRIES:
                raise
            wait = _parse_wait(str(e))
            print(f"\n[generate] Rate limit — waiting {wait:.0f}s "
                  f"(attempt {attempt+1}/{_LLM_RETRIES})...")
            time.sleep(wait + 5)


# ── abstain ───────────────────────────────────────────────────────────────────

def abstain(state: AgentState) -> dict:
    queries = ", ".join(f'"{q}"' for q in state["searched_queries"])
    msg = (
        f"Not found in the source material. "
        f"Searched Chapter 119 using queries: {queries}. "
        f"The retrieved sections did not contain sufficient information to answer "
        f"this question."
    )
    return {"answer": msg}


# ── routing ───────────────────────────────────────────────────────────────────

MAX_RETRIES = 1


def route_after_grade(state: AgentState) -> str:
    """Conditional edge: sufficient → generate; else retry or abstain."""
    if state["grade"] == "sufficient":
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite"
    return "abstain"
