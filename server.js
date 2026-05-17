/**
 * Express server for the NotebookLM-style Corrective RAG app.
 *
 *   POST /api/upload  — ingest a document (extract → chunk → embed → index)
 *   POST /api/chat    — ask a question (agent-driven corrective RAG)
 *   GET  /api/status  — current loaded document
 *   static /          — the plain HTML/CSS/JS UI
 *
 * The vector store + chat history live in-process. This is a single-user demo
 * app, so a module-level `current` document is plenty.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express from 'express';
import multer from 'multer';

import { PORT, GEMINI_API_KEY } from './src/config.js';
import { loadDocument, SUPPORTED_EXTENSIONS } from './src/loader.js';
import { chunkText } from './src/chunker.js';
import { VectorStore, makeCollectionName } from './src/vectorstore.js';
import { answerQuestion } from './src/agent.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB
});

// In-process state for the single loaded document.
const current = {
  store: null,
  docName: null,
  numChunks: 0,
  topK: 4,
  messages: [], // { role, content }
};

app.get('/api/status', (req, res) => {
  res.json({
    hasKey: Boolean(GEMINI_API_KEY),
    docName: current.docName,
    numChunks: current.numChunks,
    topK: current.topK,
  });
});

app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!GEMINI_API_KEY) {
    return res.status(400).json({ error: 'Server is missing GEMINI_API_KEY. See .env.example.' });
  }
  if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });

  const name = req.file.originalname;
  const data = req.file.buffer;
  const ext = path.extname(name).toLowerCase();
  if (!SUPPORTED_EXTENSIONS.has(ext)) {
    return res.status(400).json({
      error: `Unsupported file type: ${ext}. Supported: ${[...SUPPORTED_EXTENSIONS].sort().join(', ')}`,
    });
  }

  const chunkSize = Number(req.body.chunkSize) || 1000;
  const chunkOverlap = Number(req.body.chunkOverlap) || 150;
  const topK = Number(req.body.topK) || 4;

  try {
    const text = await loadDocument(name, data);
    if (!text.trim()) {
      return res.status(400).json({ error: 'No text could be extracted from the file.' });
    }

    const chunks = chunkText(text, { chunkSize, chunkOverlap });
    if (!chunks.length) {
      return res.status(400).json({ error: 'Document produced no chunks.' });
    }

    const store = await VectorStore.open(makeCollectionName(name, data));

    let reused = false;
    if (await store.hasData()) {
      reused = true; // existing index for this exact file — reuse it
    } else {
      const metadatas = chunks.map((c) => ({
        chunk_index: c.index,
        char_start: c.charStart,
        char_end: c.charEnd,
        source: name,
      }));
      await store.addChunks(
        chunks.map((c) => c.text),
        metadatas
      );
    }

    current.store = store;
    current.docName = name;
    current.numChunks = chunks.length;
    current.topK = topK;
    current.messages = [];

    res.json({
      ok: true,
      docName: name,
      numChunks: chunks.length,
      chars: text.length,
      reused,
    });
  } catch (err) {
    console.error('[upload]', err);
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.post('/api/chat', async (req, res) => {
  const question = (req.body.question || '').trim();
  if (!question) return res.status(400).json({ error: 'Empty question.' });
  if (!current.store) return res.status(400).json({ error: 'Upload a document first.' });

  try {
    const history = current.messages.slice(-6); // last 3 turns
    const result = await answerQuestion(current.store, current.topK, question, history);

    current.messages.push({ role: 'user', content: question });
    current.messages.push({ role: 'assistant', content: result.answer });

    res.json({
      answer: result.answer,
      queryUsed: result.queryUsed,
      rewritten: result.queryUsed && result.queryUsed !== question,
      subQuestions: result.subQuestions || [],
      sources: result.sources,
      trace: result.trace,
    });
  } catch (err) {
    console.error('[chat]', err);
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.post('/api/reset', (req, res) => {
  current.messages = [];
  res.json({ ok: true });
});

app.listen(PORT, () => {
  console.log(`\n  📓  NotebookLM RAG (JS) — http://localhost:${PORT}\n`);
  if (!GEMINI_API_KEY) {
    console.log('  ⚠  GEMINI_API_KEY not set. Copy .env.example to .env and add your key.\n');
  }
});
