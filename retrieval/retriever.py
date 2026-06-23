"""
Hybrid retriever: dense vector search + BM25 keyword search → RRF fusion → cross-encoder rerank.

Flow:
  1. Dense:   embed query → ChromaDB cosine similarity, top-N candidates
  2. BM25:    tokenize query → BM25Okapi over all chunk texts, top-N candidates
  3. Fuse:    Reciprocal Rank Fusion (k=60) of both ranked lists → top_k*2
  4. Rerank:  cross-encoder scores each (query, doc) pair → top_k

The BM25 stage rescues exact-term queries (section numbers, defined phrases) that
vector search may rank poorly because the embedding collapses semantics.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION = "fl_statutes"
RRF_K = 60          # standard RRF constant
DENSE_N = 20        # dense candidates before fusion
BM25_N = 20         # BM25 candidates before fusion
FUSE_KEEP = 20      # candidates passed to reranker
DEFAULT_TOP_K = 5   # final chunks returned


@dataclass
class Chunk:
    id: str
    doc: str           # "{section_title}\n\n{chunk_body}"
    meta: dict
    score: float = 0.0


class HybridRetriever:
    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection: str = COLLECTION,
        embed_model: str = EMBED_MODEL,
        rerank_model: str = RERANK_MODEL,
    ):
        print("  [retriever] Loading embedding model...")
        self._embedder = SentenceTransformer(embed_model)

        print("  [retriever] Loading cross-encoder reranker...")
        self._reranker = CrossEncoder(rerank_model, max_length=512)

        print("  [retriever] Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self._col = client.get_collection(collection)

        # Load all docs once for BM25 (tiny corpus; in-memory is fine)
        print("  [retriever] Building BM25 index...")
        all_data = self._col.get(include=["documents", "metadatas"])
        self._all_ids: list[str] = all_data["ids"]
        self._all_docs: list[str] = all_data["documents"]
        self._all_metas: list[dict] = all_data["metadatas"]

        # Map id → index for fast lookup after BM25 ranking
        self._id_to_idx: dict[str, int] = {
            cid: i for i, cid in enumerate(self._all_ids)
        }

        tokenized = [doc.lower().split() for doc in self._all_docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"  [retriever] Ready ({len(self._all_ids)} chunks indexed)")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        where: dict | None = None,
    ) -> list[Chunk]:
        """
        Retrieve top_k chunks for query using hybrid search + reranking.

        Args:
            query:  natural-language question
            top_k:  number of final chunks to return
            where:  optional ChromaDB metadata filter dict (e.g. {"chapter_num": "119"})
        """
        dense_hits = self._dense(query, n=DENSE_N, where=where)
        bm25_hits  = self._bm25_search(query, n=BM25_N, where=where)
        fused      = self._rrf(dense_hits, bm25_hits)[:FUSE_KEEP]
        reranked   = self._rerank(query, fused)[:top_k]
        return reranked

    def retrieve_debug(self, query: str, top_k: int = DEFAULT_TOP_K) -> tuple[list[Chunk], dict]:
        """
        Same as retrieve() but also returns intermediate rankings for comparison.

        Scores are captured before RRF/reranker mutate the Chunk objects so the
        dense and BM25 columns show their original scores, not overwritten ones.

        Returns (chunks, debug_dict).
        """
        dense_hits = self._dense(query, n=DENSE_N)
        bm25_hits  = self._bm25_search(query, n=BM25_N)

        # Capture original scores before _rrf mutates them
        dense_display = [
            f"[{c.meta['section_num']}#{c.meta['chunk_index']}] {c.score:.3f}"
            for c in dense_hits[:5]
        ]
        bm25_display = [
            f"[{c.meta['section_num']}#{c.meta['chunk_index']}] {c.score:.3f}"
            for c in bm25_hits[:5]
        ]

        fused    = self._rrf(dense_hits, bm25_hits)[:FUSE_KEEP]
        reranked = self._rerank(query, fused)[:top_k]

        reranked_display = [
            f"[{c.meta['section_num']}#{c.meta['chunk_index']}] {c.score:.3f}"
            for c in reranked
        ]

        debug = {
            "dense_top5":    dense_display,
            "bm25_top5":     bm25_display,
            "reranked_top5": reranked_display,
        }
        return reranked, debug

    # ── Private: dense retrieval ──────────────────────────────────────────────

    def _dense(self, query: str, n: int, where: dict | None = None) -> list[Chunk]:
        vec = self._embedder.encode(query, normalize_embeddings=True).tolist()
        kwargs: dict = dict(query_embeddings=[vec], n_results=n,
                            include=["metadatas", "documents", "distances"])
        if where:
            kwargs["where"] = where
        res = self._col.query(**kwargs)

        chunks = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0],
        ):
            chunks.append(Chunk(id=cid, doc=doc, meta=meta, score=1 - dist))
        return chunks

    # ── Private: BM25 retrieval ───────────────────────────────────────────────

    def _bm25_search(self, query: str, n: int, where: dict | None = None) -> list[Chunk]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1]

        chunks = []
        for i in top_idx:
            if len(chunks) >= n:
                break
            meta = self._all_metas[i]
            # Apply the same metadata filter as dense if provided
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            chunks.append(Chunk(
                id=self._all_ids[i],
                doc=self._all_docs[i],
                meta=meta,
                score=float(scores[i]),
            ))
        return chunks

    # ── Private: RRF fusion ───────────────────────────────────────────────────

    def _rrf(self, *ranked_lists: list[Chunk]) -> list[Chunk]:
        """Reciprocal Rank Fusion over multiple ranked lists."""
        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank + 1)
                chunk_lookup[chunk.id] = chunk

        fused = sorted(chunk_lookup.values(), key=lambda c: rrf_scores[c.id], reverse=True)
        for c in fused:
            c.score = rrf_scores[c.id]
        return fused

    # ── Private: cross-encoder reranking ─────────────────────────────────────

    def _rerank(self, query: str, candidates: list[Chunk]) -> list[Chunk]:
        if not candidates:
            return []
        pairs = [(query, c.doc) for c in candidates]
        scores = self._reranker.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk.score = float(score)
        return sorted(candidates, key=lambda c: c.score, reverse=True)
