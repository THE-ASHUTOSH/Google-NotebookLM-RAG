"""Build the Corrective RAG LangGraph workflow.

Topology:

    START → retrieve → grade ─┬─(relevant_docs?)──► generate → END
                              │
                              └─(no, attempts<MAX)─► rewrite ─► retrieve (loop)

The compiled graph is portable: the same object runs unchanged from this
Streamlit app, a CLI script, a FastAPI handler, or a LangGraph Cloud
deployment. Only the inputs (question + history) and the per-call store
binding differ.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .graph_state import RAGState
from .nodes import (
    decide_after_grade,
    generate_node,
    grade_node,
    make_retrieve_node,
    rewrite_node,
)
from .vectorstore import VectorStore


def build_graph(store: VectorStore, top_k: int = 4):
    """Compile a Corrective RAG graph bound to a specific document's store."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", make_retrieve_node(store, top_k=top_k))
    builder.add_node("grade", grade_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        decide_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()
