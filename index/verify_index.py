"""
Phase 2 verification: test that plain similarity queries return the right sections.

Checks:
  1. Collection exists and has the right chunk count
  2. A set of targeted queries each return the expected section in top-3 results
  3. Metadata filtering by chapter_num works
  4. Print ranked results for human review
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "fl_statutes"
EXPECTED_CHUNKS = 96

# (query, expected_section_num, description)
TEST_QUERIES = [
    ("What is the definition of a public record?",
     "119.011",
     "definitions section"),
    ("right to inspect and copy public records",
     "119.07",
     "core right to inspect"),
    ("penalties for violating public records law",
     "119.10",
     "violation penalties"),
    ("attorney fees in public records lawsuit",
     "119.12",
     "attorney fees"),
    ("public records exemption for criminal intelligence",
     "119.071",
     "exemptions — criminal intel"),
    ("custodian duties for maintaining public records",
     "119.021",
     "custodial requirements"),
    ("accelerated hearing for public records compliance",
     "119.11",
     "accelerated hearing"),
    ("contractor duty to provide public records",
     "119.0701",
     "contractor records"),
]


def run():
    print("Phase 2 verification: similarity queries on fl_statutes collection")
    print()

    # ── Load model & collection ───────────────────────────────────────────────
    print(f"  Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    count = collection.count()
    print(f"  Collection '{COLLECTION_NAME}': {count} chunks")
    if count != EXPECTED_CHUNKS:
        print(f"  WARN: expected {EXPECTED_CHUNKS} chunks, got {count}")
    print()

    # ── Run queries ───────────────────────────────────────────────────────────
    failures = []

    for query, expected_section, desc in TEST_QUERIES:
        vec = model.encode(query, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[vec],
            n_results=5,
            include=["metadatas", "distances", "documents"],
        )

        metas = results["metadatas"][0]
        dists = results["distances"][0]
        docs  = results["documents"][0]

        # Check if expected section appears in top-3
        top3_sections = [m["section_num"] for m in metas[:3]]
        hit = expected_section in top3_sections
        status = "PASS" if hit else "FAIL"
        if not hit:
            failures.append(f"{desc}: expected {expected_section} in top-3, got {top3_sections}")

        print(f"  [{status}] {desc}")
        print(f"         query: {query}")
        for rank, (m, dist, doc) in enumerate(zip(metas[:3], dists[:3], docs[:3])):
            preview = doc[:80].replace("\n", " ")
            print(f"         #{rank+1}  [{m['section_num']}] score={1-dist:.3f}  {preview}...")
        print()

    # ── Metadata filter test ──────────────────────────────────────────────────
    print("  [Filter test] chapter_num='119' filter")
    vec = model.encode("public records", normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[vec],
        n_results=3,
        where={"chapter_num": "119"},
        include=["metadatas"],
    )
    for m in results["metadatas"][0]:
        assert m["chapter_num"] == "119", f"Filter failed: got chapter {m['chapter_num']}"
    print("         chapter_num filter works correctly\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = len(TEST_QUERIES) - len(failures)
    print(f"  Results: {passed}/{len(TEST_QUERIES)} queries hit expected section in top-3")
    if failures:
        print("\n  FAILURES:")
        for f in failures:
            print(f"    FAIL  {f}")
        sys.exit(1)
    else:
        print("  All checks PASSED")


if __name__ == "__main__":
    run()
