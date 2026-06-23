"""
Phase 2 index: chunk, embed, and store Chapter 119 sections in ChromaDB.

Chunking rules (per build plan):
  - One section = one chunk (default, sections ≤ CHUNK_LIMIT chars)
  - Long sections: split on top-level subsection boundaries (1), (2), (3)...
    detected as (N) appearing after a sentence-ending period in the flat text.
    This filters out cross-references like "s. 119.07(1)" because those (N)
    follow letters/digits, not periods.
  - If a subsection chunk is still oversized, split further on (a), (b), (c)...
    appearing after a period or em-dash.
  - Merge chunks smaller than MIN_CHUNK into adjacent ones.
  - Never fixed-size splitting; never split mid-clause.

Each chunk document text is: "{section_title}\n\n{chunk_text}"
(title prepended so the embedding captures what section this belongs to)

Embedding: BAAI/bge-small-en-v1.5 (local CPU, sentence-transformers)
Storage:   ChromaDB persisted at data/chroma_db/, collection "fl_statutes"
"""

import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

SECTIONS_FILE = DATA_DIR / "ch119_sections.json"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "fl_statutes"
CHUNK_LIMIT = 4000   # chars; sections above this get split
MIN_CHUNK = 300      # chars; chunks below this get merged with neighbour
EMBED_BATCH = 32     # documents per embedding batch


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_at(text: str, boundaries: list[int]) -> list[str]:
    """Slice text at the given character positions."""
    parts = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        part = text[start:end].strip()
        if part:
            parts.append(part)
    return parts


def _merge_small(parts: list[str]) -> list[str]:
    """Merge consecutive chunks that are shorter than MIN_CHUNK."""
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < MIN_CHUNK:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def _numeric_boundaries(text: str) -> list[int]:
    """
    Find positions where top-level (N) subsections start.

    A real subsection boundary looks like:
        "...sentence end.(2) Next subsection..."
    We find (N) that is immediately preceded by a literal '.'
    and follows the sequence 2, 3, 4, ...  (pos 0 is always the start).

    Cross-references like "s. 119.07(1)(d) or (2)(d)" don't match because
    they follow ')' or a digit, not '.'.
    """
    boundaries = [0]
    expected = 2
    # Period directly before the opening paren (whitespace allowed between)
    for m in re.finditer(r'\.\s*\((\d{1,2})\)', text):
        if int(m.group(1)) == expected:
            # Position of the '(' — everything up to here is the previous chunk
            paren_pos = m.end() - len(f"({m.group(1)})")
            boundaries.append(paren_pos)
            expected += 1
    return boundaries


def _alpha_boundaries(text: str) -> list[int]:
    """
    Find (a), (b), (c)... starts after period or em-dash in a chunk.
    Used as a second-level split when a numeric subsection is still oversized.
    """
    boundaries = [0]
    expected = ord('b')          # (a) is at pos 0; look for (b), (c)...
    for m in re.finditer(r'[.—]\s*\(([a-z])\)', text):
        if ord(m.group(1)) == expected:
            paren_pos = m.end() - len(f"({m.group(1)})")
            boundaries.append(paren_pos)
            expected += 1
    return boundaries


def chunk_section(record: dict) -> list[tuple[str, dict]]:
    """
    Returns a list of (document_text, metadata) tuples for one section record.

    document_text = "{section_title}\n\n{chunk_body}"
    metadata      = section fields + chunk_index, chunk_count
    """
    body = record["text"]
    title = record["section_title"]
    section_num = record["section_num"]

    # ── Level 0: short section — no splitting ─────────────────────────────────
    if len(body) <= CHUNK_LIMIT:
        doc = f"{title}\n\n{body}"
        meta = _make_meta(record, 0, 1)
        return [(doc, meta)]

    # ── Level 1: split at (1), (2), (3)... numeric subsection boundaries ──────
    l1_bounds = _numeric_boundaries(body)
    l1_parts = _split_at(body, l1_bounds)
    if len(l1_parts) == 1:
        # No boundaries found — keep as single chunk (will be truncated on embed)
        doc = f"{title}\n\n{body}"
        return [(doc, _make_meta(record, 0, 1))]

    # ── Level 2: further split oversized numeric subsections at (a), (b)... ───
    final_parts: list[str] = []
    for part in l1_parts:
        if len(part) <= CHUNK_LIMIT:
            final_parts.append(part)
        else:
            l2_bounds = _alpha_boundaries(part)
            l2_parts = _split_at(part, l2_bounds)
            final_parts.extend(l2_parts if len(l2_parts) > 1 else [part])

    final_parts = _merge_small(final_parts)

    total = len(final_parts)
    results = []
    for i, chunk_body in enumerate(final_parts):
        doc = f"{title}\n\n{chunk_body}"
        results.append((doc, _make_meta(record, i, total)))
    return results


def _make_meta(record: dict, chunk_index: int, chunk_count: int) -> dict:
    """Build ChromaDB metadata dict (all values must be str/int/float/bool)."""
    return {
        "jurisdiction":  record["jurisdiction"],
        "title_num":     record["title_num"],
        "title_name":    record["title_name"],
        "chapter_num":   record["chapter_num"],
        "chapter_name":  record["chapter_name"],
        "section_num":   record["section_num"],
        "section_title": record["section_title"],
        "history":       record["history"],
        "source_url":    record["source_url"],
        "department":    record.get("department") or "",
        "chunk_index":   chunk_index,
        "chunk_count":   chunk_count,
    }


def _chunk_id(section_num: str, chunk_index: int) -> str:
    safe = section_num.replace(".", "_")
    return f"fl_119_{safe}_{chunk_index}"


# ── Main ──────────────────────────────────────────────────────────────────────

def build():
    print("Phase 2: Building ChromaDB index for Chapter 119")

    records = json.loads(SECTIONS_FILE.read_text(encoding="utf-8"))
    print(f"  Loaded {len(records)} section records")

    # ── Chunking ──────────────────────────────────────────────────────────────
    all_docs: list[str] = []
    all_metas: list[dict] = []
    all_ids: list[str] = []

    print("\n  Chunking:")
    for r in records:
        chunks = chunk_section(r)
        for i, (doc, meta) in enumerate(chunks):
            cid = _chunk_id(r["section_num"], i)
            all_docs.append(doc)
            all_metas.append(meta)
            all_ids.append(cid)
        count = len(chunks)
        flag = f"  ({count} chunks)" if count > 1 else ""
        print(f"    [{r['section_num']}] {r['section_title'][:45]:<45}{flag}")

    print(f"\n  Total chunks: {len(all_docs)}")

    # ── Embedding ─────────────────────────────────────────────────────────────
    print(f"\n  Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"  Embedding {len(all_docs)} chunks (batch={EMBED_BATCH})...")
    embeddings = model.encode(
        all_docs,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    print(f"\n  Writing to ChromaDB at {CHROMA_DIR} ...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Fresh build: delete existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches (ChromaDB has a document limit per call)
    UPSERT_BATCH = 100
    for start in range(0, len(all_docs), UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, len(all_docs))
        collection.add(
            ids=all_ids[start:end],
            documents=all_docs[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=all_metas[start:end],
        )

    count = collection.count()
    print(f"  Stored {count} chunks in collection '{COLLECTION_NAME}'")
    print("\n  Done. Run index/verify_index.py to test retrieval.")


if __name__ == "__main__":
    build()
