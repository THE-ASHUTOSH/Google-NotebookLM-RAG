"""
Recursive character chunking.

Strategy: try to split on the strongest semantic boundary that keeps each piece
under `chunk_size`. Order of separators (strongest first):
    paragraph break -> line break -> sentence end -> clause -> word -> char

A sliding `chunk_overlap` is added between adjacent chunks so that a sentence
straddling a boundary still appears (in part) on both sides. This preserves
local context for retrieval and reduces "lost-at-the-seam" failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    text: str
    index: int
    char_start: int
    char_end: int


def _split_recursive(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """Recursively split `text` so each piece is <= chunk_size when possible."""
    if len(text) <= chunk_size:
        return [text] if text else []

    sep = separators[0]
    rest = separators[1:]

    if sep == "":
        # Last resort: hard split by character.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    pieces: List[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, rest, chunk_size))

    # Greedily merge adjacent pieces (re-attaching the separator) while staying
    # under chunk_size. This avoids ending up with hundreds of tiny fragments.
    merged: List[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + sep + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Split `text` into overlapping chunks suitable for embedding."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    raw = _split_recursive(text, DEFAULT_SEPARATORS, chunk_size)

    chunks: List[Chunk] = []
    cursor = 0  # tracks position in original text for char_start/char_end
    prev_tail = ""

    for i, piece in enumerate(raw):
        # Prepend tail of previous chunk for overlap context.
        body = (prev_tail + " " + piece).strip() if prev_tail else piece
        # Locate the piece's position in the original text from `cursor` onward
        # so char offsets remain accurate even when separators were stripped.
        found = text.find(piece, cursor)
        if found == -1:
            found = cursor
        char_start = max(0, found - len(prev_tail))
        char_end = found + len(piece)
        cursor = char_end

        chunks.append(
            Chunk(text=body, index=i, char_start=char_start, char_end=char_end)
        )

        # Build overlap tail for next iteration: last `chunk_overlap` chars of piece.
        prev_tail = piece[-chunk_overlap:] if chunk_overlap > 0 else ""

    return chunks
