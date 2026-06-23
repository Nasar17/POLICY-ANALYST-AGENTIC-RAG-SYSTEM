"""
LangGraph state graph for the agentic RAG loop.

Graph structure:
    analyze → retrieve → grade ──┬── "generate" → generate → END
                                 ├── "rewrite"  → rewrite → retrieve → grade → (loop)
                                 └── "abstain"  → abstain → END

The grade-and-retry loop is capped at MAX_RETRIES (defined in nodes.py).
The retriever is loaded once at graph-build time and shared across all calls.
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    analyze, grade, rewrite, generate, abstain,
    make_retrieve, route_after_grade,
)
from retrieval.retriever import HybridRetriever


def build_graph(retriever: HybridRetriever | None = None) -> StateGraph:
    """
    Build and compile the agentic RAG graph.

    Args:
        retriever: a pre-loaded HybridRetriever; created fresh if None.

    Returns:
        A compiled LangGraph runnable. Call .invoke({"question": "..."}) to run.
    """
    if retriever is None:
        retriever = HybridRetriever()

    retrieve_node = make_retrieve(retriever)

    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("analyze",  analyze)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade",    grade)
    workflow.add_node("rewrite",  rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("abstain",  abstain)

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze",  "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"},
    )
    workflow.add_edge("rewrite",  "retrieve")   # retry loop
    workflow.add_edge("generate", END)
    workflow.add_edge("abstain",  END)

    return workflow.compile()


class AgenticRAG:
    """
    Convenience wrapper: loads models once, exposes a simple ask() interface.

    Usage:
        rag = AgenticRAG()
        result = rag.ask("What are the exemptions for law enforcement AND the penalties?")
        print(result["answer"])
        print(result["sub_questions"])
    """

    def __init__(self, retriever: HybridRetriever | None = None):
        print("Building agentic RAG graph...")
        self._retriever = retriever if retriever is not None else HybridRetriever()
        self._graph = build_graph(self._retriever)
        print("Graph ready.\n")

    def ask(self, question: str) -> dict:
        """
        Run the full agentic loop and return a result dict with:
            question, sub_questions, answer, grade, retry_count, searched_queries
        """
        initial_state: AgentState = {
            "question":         question,
            "sub_questions":    [],
            "metadata_filters": {},
            "all_chunks":       [],
            "grade":            "",
            "grade_reasoning":  "",
            "retry_count":      0,
            "searched_queries": [],
            "answer":           "",
        }
        final = self._graph.invoke(initial_state)
        return {
            "question":        final["question"],
            "sub_questions":   final["sub_questions"],
            "answer":          final["answer"],
            "grade":           final["grade"],
            "retry_count":     final["retry_count"],
            "searched_queries": final["searched_queries"],
            "chunks":          final.get("all_chunks", []),
        }
