"""
RAGAS-equivalent evaluation metrics implemented directly with Groq.

Each metric is scored 0.0-1.0 and matches the RAGAS definition:

  faithfulness       LLM judge: is every claim in the answer supported by the contexts?
  answer_relevance   embedding cosine similarity between question and answer
  context_precision  LLM judge: are the retrieved contexts relevant to the question?
  context_recall     LLM judge: do the contexts cover the information in ground_truth?

All three LLM-based metrics are computed in a single Groq call per question to
stay well within free-tier rate limits. A short delay is added between calls.

Reference: https://docs.ragas.io/en/latest/concepts/metrics/
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "llama-3.1-8b-instant"
DELAY_BETWEEN_CALLS = 5.0   # seconds between judge calls
MAX_RETRIES = 6             # retry up to 6× on 429 (waits 10, 20, 40, 80, 160, 320s)
CTX_CHARS   = 800           # chars of each context shown to the judge

_client: Groq | None = None
_embedder: SentenceTransformer | None = None


def _groq() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _embed_model() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


# ── Combined LLM judge ────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an expert evaluator for a question-answering system over Florida statutes.
Score each metric on a scale of 0.0 to 1.0 (two decimal places).

Definitions:
  faithfulness:      Are ALL factual claims in the ANSWER supported by the CONTEXTS?
                     1.0 = every claim is directly supported; 0.0 = major claims have no support.
  context_precision: Are the CONTEXTS relevant and useful for answering the QUESTION?
                     1.0 = all contexts are highly relevant; 0.0 = contexts are entirely off-topic.
  context_recall:    Do the CONTEXTS contain the information needed to reproduce the GROUND TRUTH?
                     1.0 = ground truth is fully covered by contexts; 0.0 = not covered at all.

Respond ONLY with valid JSON:
{
  "faithfulness": 0.00,
  "context_precision": 0.00,
  "context_recall": 0.00,
  "reasoning": "one sentence per metric separated by |"
}"""


def _parse_retry_secs(msg: str) -> float:
    """Extract seconds from Groq error like 'try again in 3m49.824s'."""
    import re
    mins = re.search(r"(\d+)m", msg)
    secs = re.search(r"(\d+(?:\.\d+)?)s", msg)
    total = 0.0
    if mins:
        total += int(mins.group(1)) * 60
    if secs:
        total += float(secs.group(1))
    return max(total, 30.0)  # at least 30s


def _llm_scores(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    ctx_text = "\n\n".join(f"[Context {i+1}]\n{c[:CTX_CHARS]}" for i, c in enumerate(contexts[:5]))
    user = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer[:600]}\n\n"
        f"CONTEXTS:\n{ctx_text}\n\n"
        f"GROUND TRUTH:\n{ground_truth[:400]}"
    )
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = _groq().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content
            try:
                parsed = json.loads(raw)
                return {
                    "faithfulness":      float(parsed.get("faithfulness",      0.0)),
                    "context_precision": float(parsed.get("context_precision", 0.0)),
                    "context_recall":    float(parsed.get("context_recall",    0.0)),
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                return {"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0}
        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                print(f"\n[evaluator] Rate limit after {MAX_RETRIES} retries — scoring 0.0")
                return {"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0}
            wait = _parse_retry_secs(str(e))
            print(f"\n[evaluator] Rate limit hit, waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
            time.sleep(wait + 5)  # small buffer


# ── Embedding-based answer relevance ─────────────────────────────────────────

def _answer_relevance(question: str, answer: str) -> float:
    """
    Cosine similarity between the question and answer embeddings.
    Abstain answers ("Not found...") receive 0.0.
    """
    if not answer.strip() or "not found in the source material" in answer.lower():
        return 0.0
    model = _embed_model()
    vecs = model.encode([question, answer], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_one(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    delay: float = DELAY_BETWEEN_CALLS,
) -> dict:
    """
    Score one Q&A pair on all four RAGAS-equivalent metrics.

    Returns a dict with faithfulness, answer_relevance, context_precision, context_recall.
    """
    time.sleep(delay)
    llm = _llm_scores(question, answer, contexts, ground_truth)
    ar  = _answer_relevance(question, answer)
    return {
        "faithfulness":      llm["faithfulness"],
        "answer_relevance":  ar,
        "context_precision": llm["context_precision"],
        "context_recall":    llm["context_recall"],
    }


def aggregate(scores: list[dict]) -> dict:
    """Mean of each metric across a list of score dicts."""
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: round(sum(s[k] for s in scores) / len(scores), 3) for k in keys}
