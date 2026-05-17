"""ChromaDB-backed vector store for document chunks."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import chromadb
from chromadb.config import Settings

from . import embeddings


PERSIST_DIR = Path("chroma_db")


@dataclass
class Retrieved:
    text: str
    score: float  # similarity (higher = better)
    metadata: dict


class VectorStore:
    def __init__(self, collection_name: str, persist_dir: Path = PERSIST_DIR):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        # cosine works well with normalized text embeddings.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def make_collection_name(filename: str, content: bytes) -> str:
        """Stable per-document collection name so re-uploads reuse the index."""
        digest = hashlib.sha1(content).hexdigest()[:12]
        # Chroma collection names: 3-63 chars, alnum + _ - . only.
        safe = "".join(c if c.isalnum() else "_" for c in filename)[:40]
        return f"doc_{safe}_{digest}"

    def has_data(self) -> bool:
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def add_chunks(
        self,
        chunks: List[str],
        metadatas: Optional[List[dict]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if not chunks:
            return
        vectors = embeddings.embed_documents(chunks, on_progress=on_progress)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metas = metadatas or [{"chunk_index": i} for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=vectors,
            metadatas=metas,
        )

    def query(self, question: str, k: int = 4) -> List[Retrieved]:
        query_vec = embeddings.embed_query(question)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        # Chroma returns cosine *distance*; convert to similarity (1 - distance).
        dists = results.get("distances", [[]])[0]

        retrieved = []
        for text, meta, dist in zip(docs, metas, dists):
            retrieved.append(
                Retrieved(text=text, score=1.0 - float(dist), metadata=meta or {})
            )
        return retrieved
