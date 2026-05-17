/**
 * Gemini embeddings via the OpenAI-compatible `/embeddings` route.
 *
 * Uses bounded concurrency so a large document embeds in parallel batches
 * instead of one slow request at a time. Order is preserved and an optional
 * `onProgress(done, total)` callback drives the upload progress bar.
 */
import { geminiClient, EMBED_MODEL } from './config.js';

const DEFAULT_MAX_WORKERS = 8;

async function embedOne(text) {
  const res = await geminiClient.embeddings.create({
    model: EMBED_MODEL,
    input: text,
  });
  return res.data[0].embedding;
}

/**
 * Embed a list of document chunks with bounded parallelism. Returns vectors in
 * the same order as `texts`.
 */
export async function embedDocuments(texts, { maxWorkers = DEFAULT_MAX_WORKERS, onProgress } = {}) {
  if (!texts.length) return [];

  const results = new Array(texts.length);
  let next = 0;
  let done = 0;

  async function worker() {
    while (true) {
      const i = next++;
      if (i >= texts.length) return;
      results[i] = await embedOne(texts[i]);
      done += 1;
      if (onProgress) onProgress(done, texts.length);
    }
  }

  const pool = Array.from({ length: Math.min(maxWorkers, texts.length) }, worker);
  await Promise.all(pool);
  return results;
}

/** Embed a single user query for retrieval. */
export async function embedQuery(text) {
  return embedOne(text);
}
