"""Google Gemini embeddings wrapper with parallel batching for large documents."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

import google.generativeai as genai


EMBED_MODEL = "models/gemini-embedding-001"

# Bounded concurrency. Gemini free tier handles ~1500 RPM comfortably at 8–16
# parallel requests; tune via embed_documents(..., max_workers=).
DEFAULT_MAX_WORKERS = 8


def configure(api_key: str) -> None:
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")
    genai.configure(api_key=api_key)


def _embed_one(text: str) -> List[float]:
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def embed_documents(
    texts: List[str],
    max_workers: int = DEFAULT_MAX_WORKERS,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[List[float]]:
    """Embed a list of document chunks in parallel.

    Order is preserved. `on_progress(done, total)` is invoked after each chunk
    completes — useful for driving a UI progress bar on large files.
    """
    if not texts:
        return []

    results: List[Optional[List[float]]] = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_embed_one, t): i for i, t in enumerate(texts)}
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            if on_progress:
                on_progress(done, len(texts))

    return [r for r in results if r is not None]


def embed_query(text: str) -> List[float]:
    """Embed a user query. Uses RETRIEVAL_QUERY task type for asymmetric search."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]
