"""Google Gemini embeddings wrapper."""
from __future__ import annotations

from typing import List

import google.generativeai as genai


EMBED_MODEL = "models/gemini-embedding-001"


def configure(api_key: str) -> None:
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")
    genai.configure(api_key=api_key)


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a list of document chunks. Uses RETRIEVAL_DOCUMENT task type."""
    out: List[List[float]] = []
    # Gemini's embed_content API accepts batched content; loop to keep it simple
    # and predictable across SDK versions.
    for text in texts:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        out.append(result["embedding"])
    return out


def embed_query(text: str) -> List[float]:
    """Embed a user query. Uses RETRIEVAL_QUERY task type for asymmetric search."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]
