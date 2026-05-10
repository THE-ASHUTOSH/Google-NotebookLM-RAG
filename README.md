# 📓 Google NotebookLM-style RAG

Upload any document (**PDF, TXT, MD, or DOCX**) and have a grounded conversation with it.
Answers are generated **only** from the document's contents — never from the LLM's general knowledge.

Built end-to-end as a full RAG pipeline:

```
Upload ─► Extract ─► Chunk ─► Embed (Gemini) ─► Index (ChromaDB)
                                                       │
                                                       ▼
        Question ─► Embed ─► Top-k retrieve ─► Gemini grounded answer
```

---

## ✨ Features

- **Multi-format ingestion** — PDF, TXT, Markdown, DOCX
- **Recursive character chunking** with overlap for context preservation
- **Gemini `text-embedding-004`** for embeddings (asymmetric: doc vs. query task types)
- **ChromaDB** persistent vector store with cosine similarity
- **Gemini 2.0 Flash** for grounded answer generation
- **Multi-turn chat** — conversation history is passed to the model
- **Source attribution** — every answer shows the chunks it was based on, with similarity scores
- **Per-document collections** — re-uploading the same file reuses the existing index

---

## 🧠 Chunking strategy

The chunker is a **recursive character splitter** in [`rag/chunker.py`](rag/chunker.py).

It tries to split on the strongest semantic boundary that keeps every piece under
`chunk_size`, falling back through this hierarchy:

```
paragraph (\n\n) → line (\n) → sentence (. ? !) → clause (; ,) → word ( ) → char
```

After splitting, adjacent fragments are greedily merged so we don't end up with
hundreds of tiny pieces, and a configurable **overlap** of trailing characters is
prepended to the next chunk so a sentence straddling a boundary still appears
(in part) on both sides. This reduces "lost-at-the-seam" retrieval failures.

Defaults: `chunk_size=1000`, `chunk_overlap=150`, both adjustable from the UI.

---

## 🛠️ Local setup

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd "Google NotebookLM RAG"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file (copy from `.env.example`):

```
GEMINI_API_KEY=your_key_here
```

Get a free key at <https://aistudio.google.com/apikey>.

Run it:

```bash
streamlit run app.py
```

Open <http://localhost:8501>, upload a file from the sidebar, and start asking questions.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to <https://share.streamlit.io> and click **New app**.
3. Pick the repo, branch, and `app.py` as the entrypoint.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Deploy. The live URL is your submission link.

---

## 📁 Project structure

```
.
├── app.py                  # Streamlit UI + orchestration
├── rag/
│   ├── loader.py           # PDF/TXT/MD/DOCX → text
│   ├── chunker.py          # Recursive character chunking
│   ├── embeddings.py       # Gemini embeddings
│   ├── vectorstore.py      # ChromaDB persistent store
│   └── generator.py        # Grounded answer generation
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔒 Grounding & anti-hallucination

The generator's system instruction strictly forbids using outside knowledge and
tells the model to respond `"I couldn't find that in the document."` when the
answer is absent. The prompt also asks for inline citations like `[#1]`
matching the order of retrieved chunks shown in the UI's **Sources** panel,
so you can verify every claim against the source text.

---

## 📜 License

MIT
