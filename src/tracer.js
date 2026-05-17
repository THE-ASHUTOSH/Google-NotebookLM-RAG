/**
 * Lightweight in-process tracer for the Corrective RAG workflow.
 *
 * No external service. Each query runs inside `withRun(question, fn)`, and each
 * tool the agent calls wraps its body in `traced(name, summarize, fn)` so its
 * latency, errors, and a compact summary land on the active run. Completed runs
 * are also appended as one JSON line to `traces.jsonl` for offline inspection.
 *
 * Because each HTTP request awaits a single agent run to completion before the
 * next begins, a module-level "current run" is sufficient — no ContextVar needed.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const TRACE_FILE = path.resolve('traces.jsonl');

let currentRun = null;

/**
 * Open a run, execute `fn`, then stamp duration and persist. Returns whatever
 * `fn` resolves to, alongside the populated run object via `getCurrentRun`.
 */
export async function withRun(question, fn) {
  const run = {
    run_id: crypto.randomUUID().replace(/-/g, '').slice(0, 12),
    question,
    started_at: Date.now(),
    duration_ms: 0,
    events: [],
  };
  const previous = currentRun;
  currentRun = run;
  try {
    return await fn(run);
  } finally {
    run.duration_ms = Date.now() - run.started_at;
    currentRun = previous;
    persist(run);
  }
}

export function getCurrentRun() {
  return currentRun;
}

/**
 * Time a single node/tool. `summarize(result)` returns a compact object merged
 * into the event summary. Token usage recorded mid-call via `recordTokens` is
 * attributed to this event.
 */
export async function traced(name, summarize, fn) {
  const run = currentRun;
  if (!run) return fn();

  const event = { node: name, started_at: Date.now(), duration_ms: 0, summary: {}, error: null };
  run.events.push(event);
  run._activeEvent = event;
  try {
    const result = await fn();
    if (summarize) Object.assign(event.summary, summarize(result) || {});
    return result;
  } catch (err) {
    event.error = String(err && err.message ? err.message : err);
    throw err;
  } finally {
    event.duration_ms = Date.now() - event.started_at;
    if (run._activeEvent === event) run._activeEvent = null;
  }
}

/** Attribute token usage to whichever node is currently executing. */
export function recordTokens(tokensIn, tokensOut) {
  const run = currentRun;
  const event = run && run._activeEvent;
  if (!event) return;
  event.summary.tokens_in = (event.summary.tokens_in || 0) + (tokensIn || 0);
  event.summary.tokens_out = (event.summary.tokens_out || 0) + (tokensOut || 0);
}

/** Strip internal fields so the trace serializes cleanly for the client/log. */
export function serializeRun(run) {
  if (!run) return null;
  const tokens_in = run.events.reduce((a, e) => a + (e.summary.tokens_in || 0), 0);
  const tokens_out = run.events.reduce((a, e) => a + (e.summary.tokens_out || 0), 0);
  return {
    run_id: run.run_id,
    question: run.question,
    duration_ms: run.duration_ms,
    tokens_in,
    tokens_out,
    events: run.events.map((e) => ({
      node: e.node,
      duration_ms: e.duration_ms,
      summary: e.summary,
      error: e.error,
    })),
  };
}

function persist(run) {
  // Tracing must never break a real request.
  try {
    fs.appendFileSync(TRACE_FILE, JSON.stringify(serializeRun(run)) + '\n', 'utf-8');
  } catch {
    /* ignore */
  }
}
