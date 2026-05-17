"""LangGraph nodes for the Corrective RAG workflow.

Each node is a pure function `RAGState -> dict` that returns the fields it
wants to merge back into state. LangGraph handles the merge and edge routing.

Every node is decorated with `@traced(...)` so its latency, error state, and
token usage are recorded by the in-process tracer (see `rag/tracer.py`).
"""
from __future__ import annotations

import re
from typing import List

from . import llm
from .generator import generate_answer
from .graph_state import RAGState
from .tracer import traced
from .vectorstore import Retrieved, VectorStore


GRADE_MODEL = "gemini-2.5-flash"
REWRITE_MODEL = "gemini-2.5-flash"

# Stop looping after this many retrieval attempts. With max_attempts=2 the
# graph runs: retrieve → grade → (rewrite → retrieve → grade) → generate.
MAX_ATTEMPTS = 2

# Chunks longer than this are truncated when sent to the grader to keep the
# grading call cheap on large documents.
GRADE_CHUNK_TRUNCATE = 1200


# --- retrieve --------------------------------------------------------------


def make_retrieve_node(store: VectorStore, top_k: int):
    """Factory: binds the per-document VectorStore + top_k into a node."""

    @traced("retrieve")
    def retrieve_node(state: RAGState) -> dict:
        query = state.get("query_used") or state["question"]
        docs = store.query(query, k=top_k)
        return {
            "query_used": query,
            "documents": docs,
            "attempts": state.get("attempts", 0) + 1,
        }

    return retrieve_node


# --- grade -----------------------------------------------------------------


def _parse_grades(text: str, n: int) -> List[bool]:
    """Parse lines like '#1: Y' / '#2: N' into a list of booleans of length n."""
    grades = [False] * n
    for line in text.splitlines():
        m = re.search(r"#?\s*(\d+)\s*[:\-)]\s*([YyNn])", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            grades[idx] = m.group(2).upper() == "Y"
    return grades


@traced("grade")
def grade_node(state: RAGState) -> dict:
    """Ask the LLM which retrieved chunks are actually relevant to the question.

    Done in ONE call (all chunks numbered together) so the corrective loop
    only costs one extra LLM hop per attempt, not k of them.
    """
    docs: List[Retrieved] = state.get("documents", [])
    if not docs:
        return {"grades": [], "relevant_docs": []}

    question = state["question"]
    blocks = []
    for i, d in enumerate(docs, start=1):
        snippet = d.text[:GRADE_CHUNK_TRUNCATE]
        blocks.append(f"[#{i}]\n{snippet}")

    prompt = (
        f"You are a relevance grader. Given a question and numbered document chunks, "
        f"decide which chunks contain information that helps answer the question.\n\n"
        f"QUESTION: {question}\n\n"
        f"CHUNKS:\n" + "\n\n---\n\n".join(blocks) + "\n\n"
        f"Reply with one line per chunk in the EXACT format:\n"
        f"#1: Y\n#2: N\n(etc.)\n"
        f"Use Y if the chunk is relevant, N if not. No other commentary."
    )

    raw = llm.call(GRADE_MODEL, prompt)
    grades = _parse_grades(raw, len(docs))
    relevant = [d for d, g in zip(docs, grades) if g]
    return {"grades": grades, "relevant_docs": relevant}


# --- rewrite ---------------------------------------------------------------


@traced("rewrite")
def rewrite_node(state: RAGState) -> dict:
    """Rewrite the query when the grader rejected every retrieved chunk."""
    original = state["question"]
    prior = state.get("query_used", original)

    prompt = (
        "The following question was used to search a document, but none of the "
        "retrieved chunks were relevant. Rewrite the question to use different "
        "keywords, synonyms, or a more specific phrasing that may match the "
        "document's wording better. Return ONLY the rewritten question — no "
        "explanation, no quotes.\n\n"
        f"ORIGINAL QUESTION: {original}\n"
        f"PREVIOUS SEARCH QUERY: {prior}\n\n"
        "REWRITTEN QUESTION:"
    )

    rewritten = llm.call(REWRITE_MODEL, prompt)
    if not rewritten or len(rewritten) > 500:
        rewritten = original
    return {"query_used": rewritten}


# --- generate --------------------------------------------------------------


@traced("generate")
def generate_node(state: RAGState) -> dict:
    """Produce the final grounded answer from whatever chunks we ended up with.

    Prefers `relevant_docs` (grader-approved); falls back to all retrieved
    documents if the grader rejected everything even after rewriting.
    """
    question = state["question"]
    history = state.get("history", [])
    docs = state.get("relevant_docs") or state.get("documents", [])

    answer = generate_answer(question, docs, history=history)
    return {"answer": answer}


# --- conditional edge ------------------------------------------------------


def decide_after_grade(state: RAGState) -> str:
    """Route after grading.

    - any relevant chunk found → generate the answer.
    - no relevant chunks and we have attempts left → rewrite and retry.
    - exhausted attempts → generate anyway (with whatever we have); the system
      prompt already handles the "couldn't find that" case.
    """
    if state.get("relevant_docs"):
        return "generate"
    if state.get("attempts", 0) < MAX_ATTEMPTS:
        return "rewrite"
    return "generate"
