"""Lightweight in-process tracer for the Corrective RAG workflow.

No external service. For each graph invocation:

    with new_run(question) as run:
        graph.invoke({...})
    run.events            # per-node latency, summary, errors, token counts
    run.duration_ms

Nodes wrapped with `@traced("name")` are recorded automatically. LLM helpers
call `record_tokens(in, out)` to attribute usage to the currently-running node
via a ContextVar.

Each completed run is also appended as one line to `traces.jsonl` so you can
inspect history (e.g. `tail -f traces.jsonl | jq`) without touching the UI.
"""
from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


TRACE_FILE = Path("traces.jsonl")


@dataclass
class NodeEvent:
    node: str
    started_at: float
    duration_ms: float
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Run:
    run_id: str
    question: str
    started_at: float
    duration_ms: float = 0.0
    events: List[NodeEvent] = field(default_factory=list)

    def __enter__(self) -> "Run":
        _current_run.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.duration_ms = (time.time() - self.started_at) * 1000.0
        _current_run.set(None)
        self._persist()

    def _persist(self) -> None:
        # Tracing must never break a real request.
        try:
            with TRACE_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(self)) + "\n")
        except Exception:
            pass

    def total_tokens(self) -> Dict[str, int]:
        ti = sum(e.summary.get("tokens_in", 0) for e in self.events)
        to = sum(e.summary.get("tokens_out", 0) for e in self.events)
        return {"tokens_in": ti, "tokens_out": to}


def new_run(question: str) -> Run:
    return Run(
        run_id=uuid.uuid4().hex[:12],
        question=question,
        started_at=time.time(),
    )


_current_run: ContextVar[Optional[Run]] = ContextVar("_current_run", default=None)
_current_event: ContextVar[Optional[NodeEvent]] = ContextVar("_current_event", default=None)


def _summarize(diff: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a node's return dict to a compact, log-friendly summary."""
    s: Dict[str, Any] = {}
    if "documents" in diff:
        s["num_documents"] = len(diff["documents"])
    if "grades" in diff:
        s["num_relevant"] = sum(1 for g in diff["grades"] if g)
        s["num_graded"] = len(diff["grades"])
    if "relevant_docs" in diff and "grades" not in diff:
        s["num_relevant"] = len(diff["relevant_docs"])
    if "query_used" in diff:
        s["query_used"] = diff["query_used"]
    if "attempts" in diff:
        s["attempts"] = diff["attempts"]
    if "answer" in diff:
        s["answer_chars"] = len(diff["answer"])
    return s


def traced(name: str) -> Callable:
    """Decorator: record a node's latency, errors, and a compact state-diff summary."""

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state):
            run = _current_run.get()
            if run is None:
                # Tracing not active — run the node as-is, no overhead.
                return fn(state)

            event = NodeEvent(node=name, started_at=time.time(), duration_ms=0.0)
            tok = _current_event.set(event)
            result: Any = None
            try:
                result = fn(state)
                return result
            except Exception as e:
                event.error = repr(e)
                raise
            finally:
                event.duration_ms = (time.time() - event.started_at) * 1000.0
                if isinstance(result, dict):
                    event.summary.update(_summarize(result))
                run.events.append(event)
                _current_event.reset(tok)

        return wrapper

    return deco


def record_tokens(tokens_in: int, tokens_out: int) -> None:
    """Attribute token usage to the currently-running node, if any."""
    event = _current_event.get()
    if event is None:
        return
    event.summary["tokens_in"] = event.summary.get("tokens_in", 0) + int(tokens_in or 0)
    event.summary["tokens_out"] = event.summary.get("tokens_out", 0) + int(tokens_out or 0)


def load_recent_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """Read the tail of traces.jsonl for a history view."""
    if not TRACE_FILE.exists():
        return []
    try:
        lines = TRACE_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception:
        return []
    runs: List[Dict[str, Any]] = []
    for line in lines:
        try:
            runs.append(json.loads(line))
        except Exception:
            continue
    return runs
