"""
Phase 3 verification: run baseline RAG on representative questions and inspect output.

Automated checks:
  - Answer is non-empty
  - Answer cites at least one section number (§ N.NN pattern)
  - Unanswerable question triggers the abstain phrase

Print full answers for human review — "sensible cited answers on simple questions"
is the spec.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.pipeline import RAGPipeline

CITATION_RE = re.compile(r"§\s*\d+\.\d+|Florida Statute|119\.\d+")
ABSTAIN_PHRASE = "not found in the source material"

# (question, expect_answer: bool, description)
QUESTIONS = [
    (
        "What is the right to inspect public records under Florida law?",
        True,
        "core inspection right",
    ),
    (
        "What are the penalties for violating Chapter 119?",
        True,
        "violation penalties",
    ),
    (
        "What are the custodial requirements for maintaining public records in Florida?",
        True,
        "custodial requirements",
    ),
    (
        "What records are exempt from public disclosure for law enforcement?",
        True,
        "law enforcement exemptions",
    ),
    (
        "Can a contractor be required to comply with public records obligations?",
        True,
        "contractor obligations",
    ),
    (
        "What is the penalty for jaywalking in Florida?",   # deliberately unanswerable
        False,
        "abstain — out of scope",
    ),
    (
        "What are the public records exemptions for trade secrets?",
        True,
        "trade secret exemption",
    ),
]


def run():
    print("Phase 3 verification: Baseline RAG")
    print("=" * 70)

    rag = RAGPipeline(top_k=5)
    print()

    failures = []

    for question, expect_answer, desc in QUESTIONS:
        print(f"[{desc.upper()}]")
        print(f"Q: {question}")
        print()

        result = rag.ask_debug(question)
        answer  = result["answer"]
        sources = result["sources"]
        debug   = result["retrieval_debug"]

        # Print retrieval comparison (dense vs BM25 vs final)
        print(f"  Dense top-5:    {', '.join(debug['dense_top5'])}")
        print(f"  BM25 top-5:     {', '.join(debug['bm25_top5'])}")
        print(f"  Reranked top-5: {', '.join(debug['reranked_top5'])}")
        print()

        print(f"  Sources retrieved:")
        for s in sources:
            print(f"    {s}")
        print()

        print(f"  Answer:")
        for line in answer.split("\n"):
            print(f"    {line}")
        print()

        # Automated checks
        ok = True
        if not answer.strip():
            failures.append(f"{desc}: answer is empty")
            ok = False

        if expect_answer:
            # Should have a real answer with citations
            if ABSTAIN_PHRASE in answer.lower():
                failures.append(f"{desc}: abstained unexpectedly")
                ok = False
            if not CITATION_RE.search(answer):
                failures.append(f"{desc}: no section citation found in answer")
                ok = False
        else:
            # Should abstain
            if ABSTAIN_PHRASE not in answer.lower():
                failures.append(f"{desc}: did NOT abstain — possible hallucination")
                ok = False

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]")
        print("-" * 70)
        print()

    passed = len(QUESTIONS) - len(failures)
    print(f"Results: {passed}/{len(QUESTIONS)} checks passed")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  FAIL  {f}")
        sys.exit(1)
    else:
        print("All automated checks passed. Review answers above for quality.")


if __name__ == "__main__":
    run()
