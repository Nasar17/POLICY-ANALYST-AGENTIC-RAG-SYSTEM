"""
Baseline RAG pipeline (Phase 3): hybrid retrieval + reranking + grounded generation.

This is the plain-RAG comparison baseline that Phase 4 (agentic loop) will be
measured against. It runs a single retrieve-then-generate pass with no retries,
query decomposition, or grading step.

Usage:
    from retrieval.pipeline import RAGPipeline

    rag = RAGPipeline()          # loads models once
    result = rag.ask("What are the public records exemptions for law enforcement?")
    print(result["answer"])
    print(result["sources"])
"""

from pathlib import Path

from retrieval.retriever import HybridRetriever, Chunk
from retrieval.generator import generate_answer

ROOT = Path(__file__).parent.parent
DEFAULT_TOP_K = 5


class RAGPipeline:
    def __init__(self, top_k: int = DEFAULT_TOP_K,
                 retriever: HybridRetriever | None = None):
        self.top_k = top_k
        self.retriever = retriever if retriever is not None else HybridRetriever()

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """
        Answer a question using baseline RAG (one retrieve-then-generate pass).

        Returns:
            question:   original question
            answer:     cited answer string (or abstain phrase)
            sources:    list of "§ N.NN — Title" strings for chunks used
            chunks:     raw Chunk objects for inspection / evaluation
        """
        k = top_k or self.top_k
        chunks = self.retriever.retrieve(question, top_k=k)
        answer = generate_answer(question, chunks)

        sources = [
            f"§ {c.meta['section_num']} — {c.meta['section_title']}"
            for c in chunks
        ]
        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "chunks":   chunks,
        }

    def ask_debug(self, question: str, top_k: int | None = None) -> dict:
        """Like ask() but also returns intermediate retrieval rankings."""
        k = top_k or self.top_k
        # retrieve_debug runs the full pipeline once and returns (chunks, debug)
        chunks, debug = self.retriever.retrieve_debug(question, top_k=k)
        answer = generate_answer(question, chunks)
        sources = [
            f"§ {c.meta['section_num']} — {c.meta['section_title']}"
            for c in chunks
        ]
        return {
            "question":        question,
            "answer":          answer,
            "sources":         sources,
            "chunks":          chunks,
            "retrieval_debug": debug,
        }
