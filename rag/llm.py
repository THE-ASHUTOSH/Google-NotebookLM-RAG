"""Single point of entry for Gemini text-generation calls.

Every call goes through `call(...)` so token usage is automatically attributed
to the currently-running tracer node — without each node having to handle that
plumbing itself.
"""
from __future__ import annotations

from typing import Optional

import google.generativeai as genai

from . import tracer


def call(model_name: str, prompt: str, system: Optional[str] = None) -> str:
    """Run a single-turn Gemini generation. Returns the response text (stripped).

    Token counts (when the SDK reports them) are forwarded to `tracer.record_tokens`
    so the active run's node event accumulates `tokens_in` / `tokens_out`.
    """
    model = (
        genai.GenerativeModel(model_name=model_name, system_instruction=system)
        if system
        else genai.GenerativeModel(model_name=model_name)
    )
    response = model.generate_content(prompt)

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        tracer.record_tokens(
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
        )

    return (response.text or "").strip()
