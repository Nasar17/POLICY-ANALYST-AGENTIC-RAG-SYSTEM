"""
Grounded generation using Groq (llama-3.3-70b-versatile).

The prompt enforces three hard rules:
  1. Answer ONLY from the provided statute sections (no outside knowledge)
  2. Cite the exact Florida Statute section number for every factual claim
  3. Respond "Not found in the source material." when the context is insufficient

This grounding-and-abstain discipline is the credibility feature of the system.
"""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from retrieval.retriever import Chunk

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 512
_MAX_RETRIES = 6
CHUNK_DISPLAY_CHARS = 1000  # keep each chunk body short enough to stay under 6K TPM

_client: Groq | None = None


def _groq() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _parse_wait(msg: str) -> float:
    mins = re.search(r"(\d+)m", msg)
    secs = re.search(r"(\d+(?:\.\d+)?)s", msg)
    total = 0.0
    if mins:
        total += int(mins.group(1)) * 60
    if secs:
        total += float(secs.group(1))
    return max(total, 30.0)

SYSTEM_PROMPT = """\
You are a Florida statute research assistant. Your job is to answer questions \
about Florida law using ONLY the statute sections provided to you.

Rules you must follow without exception:
1. Base every answer solely on the statute text provided. Do not use outside knowledge.
2. For every factual claim, cite the exact Florida Statute section (e.g., "§ 119.07(1)(a)").
3. Answer from whatever relevant information IS present in the provided sections, \
even if the answer is partial. Only respond with exactly \
"Not found in the source material." when NONE of the provided sections contain \
any information relevant to the question.
4. Never invent statute numbers, definitions, or legal content not present in the provided text.\
"""


def _format_context(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        header = f"[Florida Statute § {c.meta['section_num']} — {c.meta['section_title']}]"
        body = c.doc.split("\n\n", 1)[-1] if "\n\n" in c.doc else c.doc
        parts.append(f"{header}\n{body[:CHUNK_DISPLAY_CHARS]}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: list[Chunk]) -> str:
    """
    Call Groq to produce a grounded, cited answer from the retrieved chunks.

    Returns:
        The LLM's answer string (may be the abstain phrase if context is weak).
    """
    if not chunks:
        return "Not found in the source material."

    context = _format_context(chunks)
    user_content = f"STATUTE SECTIONS:\n\n{context}\n\nQUESTION: {question}\n\nANSWER:"

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _groq().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt == _MAX_RETRIES:
                raise
            wait = _parse_wait(str(e))
            print(f"\n[generator] Rate limit — waiting {wait:.0f}s "
                  f"(attempt {attempt+1}/{_MAX_RETRIES})...")
            time.sleep(wait + 5)
