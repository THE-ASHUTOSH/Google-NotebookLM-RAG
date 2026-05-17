"""Shared state passed between LangGraph nodes.

The graph mutates a single TypedDict as it flows through retrieve → grade →
(rewrite | generate). Keeping every field here (instead of in Streamlit
session_state) is what makes the workflow portable — the same graph runs
unchanged in a script, FastAPI endpoint, or LangGraph Cloud deployment.
"""
from __future__ import annotations

from typing import List, Tuple, TypedDict

from .vectorstore import Retrieved


class RAGState(TypedDict, total=False):
    # Inputs
    question: str
    history: List[Tuple[str, str]]  # recent (role, content) pairs

    # Working state
    query_used: str           # current question being retrieved against (may be rewritten)
    documents: List[Retrieved]
    grades: List[bool]        # per-document relevance (parallel to documents)
    relevant_docs: List[Retrieved]
    attempts: int             # how many retrieve rounds we've done

    # Output
    answer: str
