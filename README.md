# 📓 NotebookLM-style Corrective RAG (JavaScript)

Upload any document (**PDF, TXT, MD, or DOCX**) and have a grounded conversation
with it. Answers are generated **only** from the document — never from the
model's general knowledge.

This is a JavaScript rewrite of the original Python/Streamlit project. The
corrective retrieval loop is now driven by the **[OpenAI Agents SDK]
(https://openai.github.io/openai-agents-js/)** — but the model behind it is
**Gemini**, reached through Gemini's OpenAI-compatible endpoint. The UI is plain
HTML/CSS/JS served by a small Express server (no build step, no framework).

```
Upload ─► Extract ─► Chunk ─► Embed (Gemini, parallel) ─► Index (ChromaDB)
                                                                  │
                                                                  ▼
Question ─► decompose ─► [sub-q1, sub-q2, …]
                            │  (for each sub-question)
                            ▼
                 search_document ─► grade_chunks ─┬─(relevant?)─► pool chunks
                       ▲                          │
                       └──── rewrite_query ◄──────┘ (≤ 2× per sub-q)
                            │
                            ▼
                 generate (draft from pooled chunks) ─► verify_answer ─┬─(grounded?)─► answer
                                                            ▲          │
                                                            └──────────┘ (revise, ≤ 1×)
```

---

## 🧠 How the Agents SDK drives Corrective RAG

Instead of a hardcoded state machine, the corrective loop is expressed as
**five tools** the agent decides when to call ([`src/agent.js`](src/agent.js)):

| Tool | What it does |
|---|---|
| `decompose_question(question)` | Split a complex/multi-hop question into the minimal set of standalone sub-questions (simple questions pass through unchanged). |
| `search_document(query)` | Embed the (sub-)query (Gemini) and fetch the top-k chunks from ChromaDB. |
| `grade_chunks(question)` | One Gemini call marks each retrieved chunk relevant (Y/N). Approved chunks accumulate into a shared pool with stable global `[#n]` labels. |
| `rewrite_query(...)` | When grading rejects everything for a sub-question, propose new keywords and retry. |
| `verify_answer(draft)` | Groundedness self-check: confirm every claim in the draft is supported by the cited chunks. If not, the agent revises and re-verifies. |

The agent's system instructions encode the policy: *decompose → (per sub-question:
retrieve → grade → rewrite+retry ≤ 2×) → draft from the pooled chunks → verify
groundedness (revise ≤ 1×) → final answer with `[#n]` citations.* The model is a
Gemini model wired into the SDK via a custom `ModelProvider` +
`OpenAIChatCompletionsModel` pointed at Gemini's OpenAI-compatible base URL
([`src/config.js`](src/config.js)).

### Why this is more than the previous corrective RAG

The earlier version (and the original Python one) ran a single
*retrieve → grade → rewrite → generate* loop. Two failure modes it couldn't
handle, now fixed:

- **Multi-hop questions** ("How tall is X *and* when was it built?") — one query
  rarely retrieves both facts. `decompose_question` splits them, retrieves each
  independently, and fuses the pooled chunks. Citations stay consistent because
  the pool assigns stable global `[#n]` labels.
- **Hallucinated answers** — the old loop only graded *retrieval*; nothing
  checked the *output*. `verify_answer` catches claims not supported by the
  chunks and forces a revision before the answer is returned.

---

## 🔭 In-process observability (no external service)

[`src/tracer.js`](src/tracer.js) records, for every question:

- A `run_id`, total duration, and a per-step list.
- Per-node latency + a compact summary (sub-questions, chunks retrieved, chunks
  graded relevant, pool size, rewritten query, groundedness verdict, attempts,
  answer length).
- Per-node `tokens_in` / `tokens_out` (from the OpenAI-compatible `usage`
  field; the agent's own turns are summed via the SDK's aggregated usage).
- Errors per node — recorded, never crash the response.

The UI renders the trace inline under every answer (the `🔍 Trace` expander), and
each completed run is appended as one JSON line to `traces.jsonl` for offline
inspection (`tail -f traces.jsonl | jq`).

---

## 🛠️ Local setup

**Requirements:** Node 20+, and a ChromaDB server (the JS Chroma client talks to
a running server — unlike the Python embedded client).

### 1. Install

```bash
npm install
```

### 2. Start ChromaDB

The simplest options:

```bash
# Option A — Docker
docker run -d --name chroma -p 8000:8000 chromadb/chroma:0.6.3

# Option B — pip
pip install chromadb && chroma run --path ./chroma_db --port 8000
```

### 3. Configure

Copy `.env.example` to `.env` and add your free
[Google AI Studio key](https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=...
# GEMINI_BASE_URL, CHAT_MODEL, EMBED_MODEL, CHROMA_URL, PORT all have defaults
```

The same key powers both the Gemini chat model (via the Agents SDK) and Gemini
embeddings.

### 4. Run

```bash
npm start          # or: npm run dev   (auto-restart on file change)
```

Open **http://localhost:3000**, upload a document, and start asking questions.

---

## 📁 Project structure

```
.
├── server.js               # Express server: /api/upload, /api/chat, static UI
├── src/
│   ├── config.js           # env + Gemini-as-OpenAI model provider (Agents SDK)
│   ├── loader.js           # PDF (pdfjs-dist) / TXT / MD / DOCX (mammoth) → text
│   ├── chunker.js          # recursive character chunking with overlap
│   ├── embeddings.js       # Gemini embeddings, bounded parallelism
│   ├── vectorstore.js      # ChromaDB client (per-document collections)
│   ├── agent.js            # Corrective RAG agent + 5 tools (decompose, search,
│   │                       #   grade, rewrite, verify) — OpenAI Agents SDK
│   └── tracer.js           # in-process tracing → traces.jsonl
├── public/
│   ├── index.html          # NotebookLM-style UI (sidebar + chat)
│   ├── style.css
│   └── app.js              # upload + chat frontend logic
├── package.json
├── .env.example
└── README.md
```

---

## ⚙️ Notes & defaults

- **Per-document collections + content hashing** — re-uploading the same file
  reuses the existing Chroma index, so no re-embedding cost.
- **Parallel embedding** — chunks embed in bounded-concurrency batches (8 at a
  time) so large files index quickly.
- **Truncated grader input** — each chunk is truncated to 1,200 chars when sent
  to the grader, keeping grading cost flat as chunk size grows.
- **Single-user demo** — the loaded document and chat history live in-process,
  so it's meant for one user at a time.

---

## 📜 License

MIT
