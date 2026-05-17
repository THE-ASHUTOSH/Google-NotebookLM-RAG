"""
Corrective RAG over a user-uploaded document (NotebookLM-style UI).

Ingestion pipeline (parallel-embedded for large files):
    upload -> extract text -> chunk -> embed (parallel, Gemini) -> Chroma

Query workflow (LangGraph Corrective RAG):
    retrieve -> grade -> (rewrite -> retrieve)* -> generate

Observability is built-in: each graph invocation runs inside a `Run` context
that records per-node latency, errors, and token usage. The trace is shown
inline below every answer and appended to `traces.jsonl` for later inspection.
"""
from __future__ import annotations

import os
from dataclasses import asdict

import streamlit as st
from dotenv import load_dotenv

from rag import embeddings, tracer
from rag.chunker import chunk_text
from rag.graph import build_graph
from rag.loader import SUPPORTED_EXTENSIONS, load_document
from rag.vectorstore import VectorStore


load_dotenv()


st.set_page_config(
    page_title="NotebookLM-style RAG",
    page_icon="📓",
    layout="wide",
)


def get_api_key() -> str | None:
    """Resolve the Gemini API key from Streamlit secrets or env."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def reset_chat():
    st.session_state.messages = []


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("store", None)
    st.session_state.setdefault("graph", None)
    st.session_state.setdefault("doc_name", None)
    st.session_state.setdefault("num_chunks", 0)


def ingest_file(uploaded_file, chunk_size: int, chunk_overlap: int, top_k: int):
    """Run the ingestion pipeline and build a LangGraph workflow over the store."""
    data = uploaded_file.getvalue()
    name = uploaded_file.name

    with st.status("Processing document...", expanded=True) as status:
        st.write(f"Reading **{name}**")
        try:
            text = load_document(name, data)
        except ValueError as e:
            status.update(label=str(e), state="error")
            return

        if not text.strip():
            status.update(label="No text could be extracted from the file.", state="error")
            return

        st.write(f"Extracted ~{len(text):,} characters")

        st.write(f"Chunking (size={chunk_size}, overlap={chunk_overlap})...")
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.write(f"Created **{len(chunks)}** chunks")

        collection_name = VectorStore.make_collection_name(name, data)
        store = VectorStore(collection_name=collection_name)

        if store.has_data():
            st.write("Found existing index for this document — reusing it.")
        else:
            st.write("Embedding chunks with Gemini (parallel batches)...")
            progress = st.progress(0.0, text=f"Embedded 0 / {len(chunks)}")

            def on_progress(done: int, total: int) -> None:
                progress.progress(done / total, text=f"Embedded {done} / {total}")

            metadatas = [
                {
                    "chunk_index": c.index,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "source": name,
                }
                for c in chunks
            ]
            store.add_chunks(
                [c.text for c in chunks],
                metadatas=metadatas,
                on_progress=on_progress,
            )
            st.write("Indexed in ChromaDB.")

        st.session_state.store = store
        st.session_state.graph = build_graph(store, top_k=top_k)
        st.session_state.doc_name = name
        st.session_state.num_chunks = len(chunks)
        st.session_state.top_k = top_k
        reset_chat()
        status.update(label=f"Ready — ask anything about {name}", state="complete")


def render_sidebar():
    with st.sidebar:
        st.title("📓 NotebookLM RAG")
        st.caption("LangGraph corrective RAG · Gemini · ChromaDB")

        api_key = get_api_key()
        if not api_key:
            st.error(
                "No `GEMINI_API_KEY` found.\n\n"
                "Create a `.env` file with `GEMINI_API_KEY=...` "
                "(get one free at https://aistudio.google.com/apikey)."
            )
            st.stop()
        embeddings.configure(api_key)

        st.subheader("1. Upload")
        uploaded = st.file_uploader(
            "Choose a file",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            help="PDF, TXT, MD, or DOCX",
        )

        st.subheader("2. Settings")
        chunk_size = st.slider("Chunk size (chars)", 400, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, 25)
        top_k = st.slider("Top-k retrieved chunks", 1, 10, 4, 1)

        if uploaded is not None:
            sig = (uploaded.name, len(uploaded.getvalue()))
            if st.session_state.get("loaded_sig") != sig:
                ingest_file(uploaded, chunk_size, chunk_overlap, top_k)
                st.session_state.loaded_sig = sig
            else:
                # Rebuild the graph if top_k slider changed.
                if st.session_state.get("top_k") != top_k and st.session_state.store:
                    st.session_state.graph = build_graph(st.session_state.store, top_k=top_k)
                    st.session_state.top_k = top_k

        st.divider()
        if st.session_state.doc_name:
            st.success(
                f"**Loaded:** {st.session_state.doc_name}\n\n"
                f"Chunks: {st.session_state.num_chunks}"
            )
            if st.button("Clear chat history"):
                reset_chat()
                st.rerun()

        st.divider()
        st.caption(
            "Workflow: LangGraph (retrieve → grade → rewrite? → generate)\n\n"
            "Tracing: in-process · written to `traces.jsonl`"
        )


def _render_trace(trace: dict | None) -> None:
    """Render an expandable per-question trace summary."""
    if not trace or not trace.get("events"):
        return
    total = trace.get("duration_ms", 0.0)
    tokens_in = sum(e["summary"].get("tokens_in", 0) for e in trace["events"])
    tokens_out = sum(e["summary"].get("tokens_out", 0) for e in trace["events"])
    label = (
        f"🔍 Trace · {total:.0f} ms · {len(trace['events'])} nodes · "
        f"{tokens_in} in / {tokens_out} out tokens"
    )
    with st.expander(label):
        for ev in trace["events"]:
            line = f"**{ev['node']}** · {ev['duration_ms']:.0f} ms"
            extras = []
            for k in ("attempts", "num_documents", "num_relevant", "answer_chars",
                      "tokens_in", "tokens_out"):
                if k in ev["summary"]:
                    extras.append(f"{k}={ev['summary'][k]}")
            if "query_used" in ev["summary"]:
                extras.append(f"query=\"{ev['summary']['query_used']}\"")
            if extras:
                line += " · " + " · ".join(extras)
            if ev.get("error"):
                line += f" · ❌ {ev['error']}"
            st.markdown(line)


def render_chat():
    st.header("Chat with your document")

    if not st.session_state.graph:
        st.info("👈 Upload a document in the sidebar to get started.")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("query_used") and msg["query_used"] != msg.get("question"):
                    st.caption(f"🔁 Query rewritten to: *{msg['query_used']}*")
                if msg.get("sources"):
                    with st.expander(f"Sources used ({len(msg['sources'])})"):
                        for i, s in enumerate(msg["sources"], start=1):
                            st.markdown(
                                f"**[#{i}]** similarity `{s['score']:.3f}` · "
                                f"chunk #{s['metadata'].get('chunk_index', '?')}"
                            )
                            st.text(s["text"])
                _render_trace(msg.get("trace"))

    question = st.chat_input("Ask a question about the document...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving · grading · (rewriting if needed) · answering..."):
            graph = st.session_state.graph
            history = [
                (m["role"], m["content"])
                for m in st.session_state.messages[-7:-1]
            ]

            with tracer.new_run(question) as run:
                final = graph.invoke(
                    {
                        "question": question,
                        "history": history,
                        "attempts": 0,
                    }
                )
            trace = asdict(run)

            answer = final.get("answer") or "I couldn't find that in the document."
            query_used = final.get("query_used", question)
            shown_docs = final.get("relevant_docs") or final.get("documents", [])

            st.markdown(answer)
            if query_used and query_used != question:
                st.caption(f"🔁 Query rewritten to: *{query_used}*")

            sources = [
                {"text": r.text, "score": r.score, "metadata": r.metadata}
                for r in shown_docs
            ]
            with st.expander(f"Sources used ({len(sources)})"):
                for i, s in enumerate(sources, start=1):
                    st.markdown(
                        f"**[#{i}]** similarity `{s['score']:.3f}` · "
                        f"chunk #{s['metadata'].get('chunk_index', '?')}"
                    )
                    st.text(s["text"])

            _render_trace(trace)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "question": question,
            "query_used": query_used,
            "sources": sources,
            "trace": trace,
        }
    )


def main():
    init_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
