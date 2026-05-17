"""Grounded answer generation using Gemini, given retrieved context."""
from __future__ import annotations

from typing import Iterable, List, Tuple

from . import llm
from .vectorstore import Retrieved


GEN_MODEL = "gemini-2.5-flash"


SYSTEM_INSTRUCTION = (
    "You are a careful research assistant answering questions strictly from the "
    "provided document excerpts. Rules:\n"
    "1. Use ONLY the information in the excerpts below. Do not use outside knowledge.\n"
    "2. If the answer is not present, say: \"I couldn't find that in the document.\"\n"
    "3. Quote or paraphrase precisely; do not invent facts, numbers, names, or dates.\n"
    "4. When helpful, cite the chunks you used as [#1], [#2], ... matching the order shown.\n"
    "5. Keep answers concise and directly responsive to the question."
)


def _format_context(chunks: List[Retrieved]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[#{i}] (similarity={c.score:.3f})\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _format_history(history: Iterable[Tuple[str, str]]) -> str:
    """Render prior turns as plain text. history is iterable of (role, content)."""
    lines = []
    for role, content in history:
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def generate_answer(
    question: str,
    retrieved: List[Retrieved],
    history: Iterable[Tuple[str, str]] = (),
) -> str:
    """Generate a grounded answer. `history` is recent (role, content) pairs."""
    if not retrieved:
        return "I couldn't find that in the document."

    context = _format_context(retrieved)
    convo = _format_history(history)

    prompt_parts = [
        "DOCUMENT EXCERPTS (the only source you may use):",
        context,
        "",
    ]
    if convo:
        prompt_parts += ["PRIOR CONVERSATION:", convo, ""]
    prompt_parts += [
        f"QUESTION: {question}",
        "",
        "Answer using only the excerpts above. Cite chunks like [#1] where relevant.",
    ]
    prompt = "\n".join(prompt_parts)

    text = llm.call(GEN_MODEL, prompt, system=SYSTEM_INSTRUCTION)
    return text or "I couldn't find that in the document."
