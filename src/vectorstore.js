/**
 * ChromaDB-backed vector store for document chunks.
 *
 * The JavaScript Chroma client talks to a running Chroma server (unlike the
 * Python embedded PersistentClient). Start one first:
 *
 *     pip install chromadb && chroma run --path ./chroma_db
 *
 * then point CHROMA_URL at it (defaults to http://localhost:8000).
 *
 * We compute embeddings ourselves (Gemini) and hand Chroma precomputed vectors,
 * so no Chroma-side embedding function is involved.
 */
import crypto from 'node:crypto';
import { ChromaClient } from 'chromadb';
import { embedDocuments, embedQuery } from './embeddings.js';

const CHROMA_URL = process.env.CHROMA_URL || 'http://localhost:8000';

// A trivial embedding function so getOrCreateCollection never tries to pull a
// default one over the network; we never actually call it (we pass embeddings).
const noopEmbeddingFunction = {
  generate: async (texts) => texts.map(() => []),
};

let client = null;
function getClient() {
  if (!client) client = new ChromaClient({ path: CHROMA_URL });
  return client;
}

/** Stable per-document collection name so re-uploads reuse the same index. */
export function makeCollectionName(filename, content) {
  const digest = crypto.createHash('sha1').update(content).digest('hex').slice(0, 12);
  const safe = filename.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 40);
  return `doc_${safe}_${digest}`;
}

export class VectorStore {
  constructor(collection, name) {
    this.collection = collection;
    this.name = name;
  }

  /** Open (or create) the collection for a document. */
  static async open(collectionName) {
    const collection = await getClient().getOrCreateCollection({
      name: collectionName,
      metadata: { 'hnsw:space': 'cosine' },
      embeddingFunction: noopEmbeddingFunction,
    });
    return new VectorStore(collection, collectionName);
  }

  async hasData() {
    try {
      return (await this.collection.count()) > 0;
    } catch {
      return false;
    }
  }

  /** Embed and index a list of chunk texts with their metadatas. */
  async addChunks(chunks, metadatas, { onProgress } = {}) {
    if (!chunks.length) return;
    const vectors = await embedDocuments(chunks, { onProgress });
    const ids = chunks.map((_, i) => `chunk_${i}`);
    const metas = metadatas || chunks.map((_, i) => ({ chunk_index: i }));
    await this.collection.add({
      ids,
      documents: chunks,
      embeddings: vectors,
      metadatas: metas,
    });
  }

  /**
   * Retrieve the top-k chunks for a query. Returns
   * [{ text, score, metadata }], score = cosine similarity (higher is better).
   */
  async query(question, k = 4) {
    const queryVec = await embedQuery(question);
    const res = await this.collection.query({
      queryEmbeddings: [queryVec],
      nResults: k,
    });
    const docs = (res.documents && res.documents[0]) || [];
    const metas = (res.metadatas && res.metadatas[0]) || [];
    const dists = (res.distances && res.distances[0]) || [];

    return docs.map((text, i) => ({
      text,
      score: 1 - Number(dists[i] ?? 1), // cosine distance -> similarity
      metadata: metas[i] || {},
    }));
  }
}
