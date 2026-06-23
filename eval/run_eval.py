"""
Phase 5 evaluation: plain-RAG vs agentic-RAG on the Chapter 119 test set.

What this computes:
  - RAGAS-equivalent metrics (faithfulness, answer_relevance, context_precision,
    context_recall) on the answerable subset (single + multi-part questions)
  - Abstention rate on the unanswerable subset for both pipelines
  - Side-by-side comparison table

Results are saved to eval/results/comparison.json and printed to stdout.

Checkpointing: partial results are saved after every question so that if the
run is interrupted (e.g. by a rate limit) it can resume from where it stopped.
Delete eval/results/checkpoint.json to start fresh.

Rate-limit safety: evaluate_one() sleeps DELAY_BETWEEN_CALLS seconds between
each Groq judge call.  Pipeline calls that hit 429 are retried automatically.
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from groq import RateLimitError

from retrieval.pipeline import RAGPipeline
from retrieval.retriever import HybridRetriever
from agent.graph import AgenticRAG
from eval.evaluator import evaluate_one, aggregate

TEST_SET_FILE  = Path(__file__).parent / "test_set.json"
RESULTS_DIR    = Path(__file__).parent / "results"
CHECKPOINT     = RESULTS_DIR / "checkpoint.json"
RESULTS_DIR.mkdir(exist_ok=True)

ABSTAIN_MARKER = "not found in the source material"
MAX_PIPELINE_RETRIES = 6


# ── Rate-limit helpers ────────────────────────────────────────────────────────

def _parse_wait(err_msg: str) -> float:
    mins = re.search(r"(\d+)m", err_msg)
    secs = re.search(r"(\d+(?:\.\d+)?)s", err_msg)
    total = 0.0
    if mins:
        total += int(mins.group(1)) * 60
    if secs:
        total += float(secs.group(1))
    return max(total, 30.0)


def _call_with_retry(fn, *args, label=""):
    """Call fn(*args), retrying on RateLimitError with parsed wait time."""
    for attempt in range(MAX_PIPELINE_RETRIES + 1):
        try:
            return fn(*args)
        except RateLimitError as e:
            if attempt == MAX_PIPELINE_RETRIES:
                raise
            wait = _parse_wait(str(e))
            print(f"\n[{label}] 429 rate limit — waiting {wait:.0f}s "
                  f"(attempt {attempt+1}/{MAX_PIPELINE_RETRIES})...")
            time.sleep(wait + 5)


# ── Helpers ───────────────────────────────────────────────────────────────────

def chunks_to_contexts(chunks) -> list[str]:
    if not chunks:
        return []
    if isinstance(chunks[0], str):
        return chunks
    return [c.doc for c in chunks]


def is_abstain(answer: str) -> bool:
    return ABSTAIN_MARKER in answer.lower()


def run_plain_rag(rag: RAGPipeline, question: str) -> tuple[str, list[str]]:
    result = _call_with_retry(rag.ask, question, label="plain-rag")
    return result["answer"], chunks_to_contexts(result["chunks"])


def run_agentic(rag: AgenticRAG, question: str) -> tuple[str, list[str]]:
    result = _call_with_retry(rag.ask, question, label="agentic")
    return result["answer"], chunks_to_contexts(result.get("chunks", []))


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        print(f"[checkpoint] Resuming from checkpoint: "
              f"{len(data.get('rows', []))} questions completed.\n")
        return data
    return {"rows": [], "plain_scores": [], "agentic_scores": []}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Phase 5: Evaluation — Plain RAG vs Agentic RAG")
    print("=" * 60)

    test_items   = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))
    answerable   = [t for t in test_items if t["type"] in ("single", "multi_part")]
    unanswerable = [t for t in test_items if t["type"] == "unanswerable"]
    print(f"Test set: {len(answerable)} answerable, {len(unanswerable)} unanswerable\n")

    # Load checkpoint before loading models (fast path if already complete)
    ckpt = load_checkpoint()
    completed_ids = {r["id"] for r in ckpt["rows"]}
    rows           = ckpt["rows"]
    plain_scores   = ckpt["plain_scores"]
    agentic_scores = ckpt["agentic_scores"]

    remaining = [t for t in answerable if t["id"] not in completed_ids]

    if not remaining:
        print("All answerable questions already evaluated (checkpoint is complete).")
    else:
        print("Loading shared retriever + pipelines...")
        shared_retriever = HybridRetriever()
        plain_rag   = RAGPipeline(top_k=5, retriever=shared_retriever)
        agentic_rag = AgenticRAG(retriever=shared_retriever)
        print()

        print("Running answerable questions through both pipelines...")
        print("-" * 60)

        for i, item in enumerate(remaining):
            global_i = len(completed_ids) + i + 1
            qid   = item["id"]
            q     = item["question"]
            gt    = item["ground_truth"]
            qtype = item["type"]
            print(f"[{global_i:02d}/{len(answerable)}] {qid} ({qtype})")
            print(f"  Q: {q[:70]}...")

            # Plain RAG
            t0 = time.time()
            plain_ans, plain_ctx = run_plain_rag(plain_rag, q)
            plain_t = time.time() - t0
            plain_sc = evaluate_one(q, plain_ans, plain_ctx, gt)
            plain_scores.append(plain_sc)

            # Agentic RAG
            t0 = time.time()
            agent_ans, agent_ctx = run_agentic(agentic_rag, q)
            agent_t = time.time() - t0
            agent_sc = evaluate_one(q, agent_ans, agent_ctx, gt)
            agentic_scores.append(agent_sc)

            rows.append({
                "id":       qid,
                "type":     qtype,
                "question": q,
                "plain":    {**plain_sc,  "answer_preview": plain_ans[:120],
                             "time_s": round(plain_t, 1)},
                "agent":    {**agent_sc,  "answer_preview": agent_ans[:120],
                             "time_s": round(agent_t, 1)},
            })

            save_checkpoint({"rows": rows,
                             "plain_scores": plain_scores,
                             "agentic_scores": agentic_scores})

            print(f"  Plain  | faith={plain_sc['faithfulness']:.2f}  "
                  f"ar={plain_sc['answer_relevance']:.2f}  "
                  f"cp={plain_sc['context_precision']:.2f}  "
                  f"cr={plain_sc['context_recall']:.2f}")
            print(f"  Agent  | faith={agent_sc['faithfulness']:.2f}  "
                  f"ar={agent_sc['answer_relevance']:.2f}  "
                  f"cp={agent_sc['context_precision']:.2f}  "
                  f"cr={agent_sc['context_recall']:.2f}")
            print()

    # ── Abstention on unanswerable ────────────────────────────────────────────
    if not remaining:
        # Need to load pipelines for unanswerable (skipped the answerable block above)
        print("Loading shared retriever + pipelines for unanswerable questions...")
        shared_retriever = HybridRetriever()
        plain_rag   = RAGPipeline(top_k=5, retriever=shared_retriever)
        agentic_rag = AgenticRAG(retriever=shared_retriever)
        print()

    print("Running unanswerable questions...")
    print("-" * 60)
    plain_abstain_count = 0
    agent_abstain_count = 0

    for item in unanswerable:
        q = item["question"]
        plain_ans, _ = run_plain_rag(plain_rag, q)
        agent_ans, _ = run_agentic(agentic_rag, q)
        pa = is_abstain(plain_ans)
        aa = is_abstain(agent_ans)
        if pa: plain_abstain_count += 1
        if aa: agent_abstain_count += 1
        print(f"  {item['id']}: plain={'abstain' if pa else 'ANSWERED'}  "
              f"agent={'abstain' if aa else 'ANSWERED'}")
        time.sleep(1)

    plain_abstain_rate = plain_abstain_count  / len(unanswerable)
    agent_abstain_rate = agent_abstain_count  / len(unanswerable)

    # ── Aggregate RAGAS metrics ───────────────────────────────────────────────
    # Re-order rows to match answerable order (checkpoint may be partial reorder)
    order = {t["id"]: idx for idx, t in enumerate(answerable)}
    rows_sorted = sorted(rows, key=lambda r: order.get(r["id"], 999))

    plain_scores_ordered   = [r["plain"]  for r in rows_sorted]
    agentic_scores_ordered = [r["agent"]  for r in rows_sorted]

    # Strip non-metric keys before aggregating
    metric_keys = {"faithfulness", "answer_relevance", "context_precision", "context_recall"}
    plain_agg  = aggregate([{k: v for k, v in s.items() if k in metric_keys}
                             for s in plain_scores_ordered])
    agent_agg  = aggregate([{k: v for k, v in s.items() if k in metric_keys}
                             for s in agentic_scores_ordered])

    multi_ids  = {t["id"] for t in answerable if t["type"] == "multi_part"}
    plain_multi = aggregate([{k: v for k, v in r["plain"].items() if k in metric_keys}
                              for r in rows_sorted if r["id"] in multi_ids])
    agent_multi = aggregate([{k: v for k, v in r["agent"].items() if k in metric_keys}
                              for r in rows_sorted if r["id"] in multi_ids])

    # ── Print comparison table ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS: Plain RAG vs Agentic RAG")
    print("=" * 60)

    METRICS = [
        ("faithfulness",      "Faithfulness"),
        ("answer_relevance",  "Answer Relevance"),
        ("context_precision", "Context Precision"),
        ("context_recall",    "Context Recall"),
    ]

    header = f"{'Metric':<22} {'Plain RAG':>10} {'Agentic':>10} {'Delta':>8}"
    print(header)
    print("-" * len(header))
    for key, label in METRICS:
        p = plain_agg.get(key, 0)
        a = agent_agg.get(key, 0)
        d = a - p
        sign = "+" if d >= 0 else ""
        print(f"{label:<22} {p:>10.3f} {a:>10.3f} {sign+f'{d:.3f}':>8}")

    print()
    print(f"{'Abstention Rate':<22} {plain_abstain_rate:>10.0%} "
          f"{agent_abstain_rate:>10.0%} "
          f"{'±'+f'{agent_abstain_rate - plain_abstain_rate:.0%}':>8}")

    if multi_ids:
        print()
        print("Multi-part questions only:")
        print("-" * len(header))
        for key, label in METRICS:
            p = plain_multi.get(key, 0)
            a = agent_multi.get(key, 0)
            d = a - p
            sign = "+" if d >= 0 else ""
            print(f"{label:<22} {p:>10.3f} {a:>10.3f} {sign+f'{d:.3f}':>8}")

    # ── Save final results ────────────────────────────────────────────────────
    output = {
        "summary": {
            "plain_rag":   {**plain_agg,  "abstention_rate": plain_abstain_rate},
            "agentic_rag": {**agent_agg,  "abstention_rate": agent_abstain_rate},
            "multi_part":  {"plain_rag": plain_multi, "agentic_rag": agent_multi},
        },
        "per_question": rows_sorted,
    }
    out_path = RESULTS_DIR / "comparison.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Full results saved to {out_path}")
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint deleted.")


if __name__ == "__main__":
    main()
