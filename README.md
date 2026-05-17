# 📓 Google NotebookLM-style RAG

Upload any document (**PDF, TXT, MD, or DOCX**) and have a grounded conversation
with it. Answers are generated **only** from the document — never from the LLM's
general knowledge.

> **What's new in this update:** the pipeline now uses **LangGraph** for a
> **Corrective RAG** workflow (retrieve → grade → rewrite → regenerate), with a
> small **in-process tracer** for observability and **parallel embedding** for
> large files.

```
Upload ─► Extract ─► Chunk ─► Embed (parallel, Gemini) ─► Index (ChromaDB)
                                                                  │
                                                                  ▼
        Question ─► retrieve ─► grade ─┬─(relevant?)──► generate ─► Answer
                                       │
                                       └─(no)──► rewrite query ─► retrieve (loop)
```

---

## ✨ What changed vs. plain RAG

| Concern | Plain RAG | This project |
|---|---|---|
| Bad retrieval | Returns "couldn't find that" | Grader detects it, rewriter retries with new phrasing |
| Orchestration | Inline Python in the UI file | LangGraph `StateGraph` — portable + traceable |
| Large-file embedding | Sequential, 1 chunk per API call | Parallel `ThreadPoolExecutor` with progress bar |
| Observability | None | Per-node latency, errors, and token usage shown inline + logged to `traces.jsonl` |

---

## 🧠 The LangGraph workflow

Lives in [`rag/graph.py`](rag/graph.py). Nodes are in
[`rag/nodes.py`](rag/nodes.py); shared state is
[`rag/graph_state.py`](rag/graph_state.py).

**Nodes**

- `retrieve` — embed the (possibly rewritten) query, fetch top-k chunks from Chroma.
- `grade` — one LLM call, asks Gemini to mark each retrieved chunk Y/N for
  relevance. Cheap: chunks are numbered and graded together.
- `rewrite` — when the grader rejects every chunk, ask Gemini to reword the
  question with different keywords / synonyms.
- `generate` — produce the final grounded answer (with citations) from the
  surviving chunks.

**Edges**

```
START ─► retrieve ─► grade ─► (conditional)
                                ├─ "generate" if any chunk passed grading
                                └─ "rewrite"  if none did and attempts < MAX
rewrite ─► retrieve   (loop back, attempts++)
generate ─► END
```

`MAX_ATTEMPTS` (default 2) caps the corrective loop so a truly off-topic
question doesn't run forever — after that, `generate` runs anyway and the
system prompt produces "I couldn't find that in the document."

---

## 🔭 In-process observability (no external service)

[`rag/tracer.py`](rag/tracer.py) gives every graph run:

- A unique `run_id`, total duration, and a list of `NodeEvent`s.
- Per-node latency and a compact state-diff summary (num docs, num relevant,
  rewritten query, attempts, answer length).
- Per-node `tokens_in` / `tokens_out`, captured by routing every Gemini call
  through [`rag/llm.py`](rag/llm.py) which pulls counts from the SDK's
  `usage_metadata`.
- Errors per node — failures are recorded but never crash the response.

How it works:

```python
with tracer.new_run(question) as run:
    final = graph.invoke({...})
# run.events, run.duration_ms, run.total_tokens() all populated
```

Each completed run is also appended as one JSON line to `traces.jsonl` so you
can grep / `jq` history without touching the UI. Nothing leaves the machine.

The Streamlit UI renders the trace inline under every answer (look for the
`🔍 Trace` expander) so you can see *why* a particular answer came out — which
nodes ran, whether the corrective loop fired, and where the time went.

---

## 📦 Portability

The compiled graph is a plain Python object. The same workflow can be:

- Driven from this Streamlit app (current entrypoint).
- Called from a CLI: `graph.invoke({"question": "...", "history": [], "attempts": 0})`.
- Wrapped in a FastAPI handler.
- Deployed to LangGraph Cloud / Platform.

The only Streamlit-specific code lives in `app.py`; everything in `rag/` is
framework-agnostic. The tracer is also framework-agnostic — `with new_run(...)`
works the same from any host.

---

## ⚡ Large-file optimizations

- **Parallel embedding** — [`rag/embeddings.py`](rag/embeddings.py) uses
  `ThreadPoolExecutor` (default 8 workers) so a 5,000-chunk PDF embeds in
  minutes instead of an hour.
- **Per-document collections + content hashing** — re-uploading the same
  file reuses the existing Chroma index; no re-embedding cost.
- **Streamed progress** — the embedding loop calls an `on_progress`
  callback after every chunk completes, driving the Streamlit progress bar.
- **Truncated grader inputs** — the grader truncates each chunk to 1,200
  chars so grading cost stays flat as `chunk_size` grows.

---

## 🛠️ Local setup

**Requirements:** Python 3.10+

```bash
cd "Google NotebookLM RAG"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:

```
GEMINI_API_KEY=...               # required, free at https://aistudio.google.com/apikey
```

Run it:

```bash
streamlit run app.py
```

---

## 📁 Project structure

```
.
├── app.py                  # Streamlit UI — drives the LangGraph workflow
├── rag/
│   ├── loader.py           # PDF/TXT/MD/DOCX → text
│   ├── chunker.py          # Recursive character chunking
│   ├── embeddings.py       # Parallel Gemini embeddings (ThreadPoolExecutor)
│   ├── vectorstore.py      # ChromaDB persistent store
│   ├── llm.py              # Single entry-point for Gemini text gen + token capture
│   ├── generator.py        # Grounded answer generation
│   ├── graph_state.py      # TypedDict state shared between nodes
│   ├── nodes.py            # retrieve / grade / rewrite / generate + decision
│   ├── graph.py            # StateGraph wiring + compile
│   └── tracer.py           # In-process tracing (run / node events / JSONL log)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📜 License

MIT
