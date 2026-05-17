/**
 * Corrective RAG built with the OpenAI Agents SDK, driving a Gemini model.
 *
 * Rather than hardcode a retrieve → grade → rewrite → generate state machine,
 * we expose those steps as *tools* and let the agent orchestrate the corrective
 * loop itself:
 *
 *   - search_document(query)  -> retrieve top-k chunks for a query
 *   - grade_chunks(query)     -> mark which retrieved chunks are relevant
 *   - rewrite_query(...)      -> propose new keywords when retrieval missed
 *
 * The agent's instructions encode the policy (retrieve, grade, rewrite at most
 * twice, then answer only from relevant chunks with [#n] citations). Each tool
 * call is timed by the in-process tracer, and token usage from every model turn
 * is attributed to a synthetic `generate` node so the trace mirrors the original
 * LangGraph view.
 */
import { Agent, tool } from '@openai/agents';
import { z } from 'zod';
import { runner, geminiClient, CHAT_MODEL } from './config.js';
import { traced, recordTokens, withRun, serializeRun } from './tracer.js';

const MAX_REWRITES = 2;
const MAX_VERIFY_RETRIES = 1; // re-generate at most this many times if ungrounded
const GRADE_CHUNK_TRUNCATE = 1200;

const SYSTEM_INSTRUCTIONS = `You answer questions strictly from a single uploaded document, using a corrective retrieval loop. You cannot see the document directly — you must use tools to read it.

Follow this procedure for every question:

A. PLAN
1. Call decompose_question once with the user's question. It returns one or more sub-questions. If the question is simple, you'll get it back unchanged as a single sub-question.

B. RETRIEVE (do this for EACH sub-question, one at a time)
2. Call search_document with the sub-question to retrieve candidate chunks.
3. Call grade_chunks to find which retrieved chunks are actually relevant.
4. If NO chunk is relevant and you have rewrites left, call rewrite_query for a better phrasing, then repeat step 2 with that new query (at most ${MAX_REWRITES} rewrites per sub-question).
   Relevant chunks from every sub-question accumulate into a shared pool; the [#n] labels are stable across the whole question.

C. ANSWER
5. When all sub-questions have been retrieved+graded, write a draft answer using ONLY the pooled relevant chunks, citing them as [#n].
6. Call verify_answer with your draft. It checks that every claim is supported by the cited chunks.
   - If it returns GROUNDED, send that draft as your final answer.
   - If it returns NOT grounded, revise the answer to drop or fix the unsupported claims and call verify_answer again (at most ${MAX_VERIFY_RETRIES} extra time). If it still fails, answer only with what IS supported, or say you couldn't find it.

Rules for the final answer:
- Use ONLY information returned by the tools. Never use outside knowledge.
- If nothing relevant was found across all sub-questions, reply exactly: "I couldn't find that in the document."
- Do not invent facts, numbers, names, or dates.
- Cite the chunks you used as [#1], [#2], … matching the [#n] labels the tools returned.
- Keep answers concise and directly responsive to the question.`;

/**
 * Build a per-request agent bound to a document's vector store. A `session`
 * object carries mutable state across tool calls within one run.
 *
 * `pool` is the accumulating, deduplicated set of grader-approved chunks across
 * ALL sub-questions. Each pooled chunk keeps a stable global label (its 1-based
 * position in `pool`) so [#n] citations stay consistent for the whole answer —
 * this is what makes query decomposition work without scrambling citations.
 */
function buildAgent(store, topK) {
  const session = {
    subQuestions: [],
    lastQuery: null,
    lastDocs: [], // chunks from the most recent retrieve (not yet graded)
    pool: [], // [{ text, score, metadata }] — global relevant chunks, order = label
    rewrites: 0, // rewrites used for the current sub-question
    verifyRetries: 0,
  };

  // Map a chunk to its 1-based global label, adding it to the pool if new.
  const labelFor = (doc) => {
    const existing = session.pool.findIndex((d) => d.text === doc.text);
    if (existing !== -1) return existing + 1;
    session.pool.push(doc);
    return session.pool.length;
  };

  const decomposeQuestion = tool({
    name: 'decompose_question',
    description:
      'Break the user question into the minimal set of standalone sub-questions needed ' +
      'to answer it. Simple questions come back unchanged as a single sub-question. ' +
      'Call this FIRST, once per user question.',
    parameters: z.object({
      question: z.string().describe('The original user question.'),
    }),
    execute: async ({ question }) =>
      traced(
        'decompose',
        (out) => ({ num_subquestions: out._subs.length }),
        async () => {
          const subs = await decompose(question);
          session.subQuestions = subs;
          session.rewrites = 0;
          return {
            _subs: subs,
            toString: () =>
              subs.length > 1
                ? `This question needs ${subs.length} sub-questions:\n` +
                  subs.map((s, i) => `${i + 1}. ${s}`).join('\n') +
                  `\nRetrieve and grade each one, then answer from the pooled chunks.`
                : `Single-step question. Retrieve and grade it directly: "${subs[0]}".`,
          };
        }
      ).then((r) => r.toString()),
  });

  const searchDocument = tool({
    name: 'search_document',
    description:
      'Retrieve the most relevant chunks of the uploaded document for a query (one ' +
      'sub-question at a time). Returns chunks with stable global [#n] labels.',
    parameters: z.object({
      query: z.string().describe('The (sub-)question or search query to retrieve chunks for.'),
    }),
    execute: async ({ query }) =>
      traced(
        'retrieve',
        (out) => ({ query_used: query, num_documents: out._count, attempts: session.rewrites + 1 }),
        async () => {
          const docs = await store.query(query, topK);
          session.lastQuery = query;
          session.lastDocs = docs;
          return {
            _count: docs.length,
            toString: () =>
              docs.length
                ? `Retrieved ${docs.length} chunk(s) for "${query}". Now call grade_chunks.`
                : `No chunks retrieved for "${query}".`,
          };
        }
      ).then((r) => r.toString()),
  });

  const gradeChunks = tool({
    name: 'grade_chunks',
    description:
      'Judge which of the most recently retrieved chunks are relevant. Call right after ' +
      'search_document. Relevant chunks are added to the shared pool with global [#n] ' +
      'labels and their text is shown so you can cite them.',
    parameters: z.object({
      question: z.string().describe('The sub-question to grade relevance against.'),
    }),
    execute: async ({ question }) =>
      traced(
        'grade',
        (out) => ({ num_graded: out._graded, num_relevant: out._relevant, pool_size: session.pool.length }),
        async () => {
          const docs = session.lastDocs;
          if (!docs.length) {
            return { _graded: 0, _relevant: 0, toString: () => 'No retrieved chunks to grade.' };
          }
          const grades = await gradeRelevance(question, docs);
          const relevant = docs.filter((_, i) => grades[i]);
          if (relevant.length) session.rewrites = 0; // success: reset for next sub-question

          const shown = relevant
            .map((d) => {
              const n = labelFor(d);
              return `[#${n}] (similarity ${d.score.toFixed(3)})\n${d.text}`;
            })
            .join('\n\n---\n\n');

          return {
            _graded: docs.length,
            _relevant: relevant.length,
            toString: () =>
              relevant.length
                ? `Added ${relevant.length} relevant chunk(s) to the pool:\n\n${shown}`
                : 'None of the retrieved chunks are relevant to this sub-question.',
          };
        }
      ).then((r) => r.toString()),
  });

  const rewriteQuery = tool({
    name: 'rewrite_query',
    description:
      'When grading found no relevant chunks for a sub-question, propose a reworded query ' +
      '(different keywords / synonyms / more specific phrasing) to retry retrieval. ' +
      `Usable at most ${MAX_REWRITES} times per sub-question.`,
    parameters: z.object({
      original_question: z.string().describe('The sub-question being answered.'),
      previous_query: z.string().describe('The query that just failed to retrieve relevant chunks.'),
    }),
    execute: async ({ original_question, previous_query }) =>
      traced(
        'rewrite',
        (out) => ({ query_used: out._rewritten, attempts: session.rewrites }),
        async () => {
          if (session.rewrites >= MAX_REWRITES) {
            return {
              _rewritten: previous_query,
              toString: () =>
                'Rewrite limit reached for this sub-question. Move on, and answer with the pooled chunks (or say you could not find it).',
            };
          }
          session.rewrites += 1;
          const rewritten = await rewriteSearchQuery(original_question, previous_query);
          return {
            _rewritten: rewritten,
            toString: () => `Rewritten query: ${rewritten}\nNow call search_document with this query.`,
          };
        }
      ).then((r) => r.toString()),
  });

  const verifyAnswer = tool({
    name: 'verify_answer',
    description:
      'Self-check a draft answer for groundedness: confirm every claim is supported by the ' +
      'pooled chunks it cites. Call this on your draft BEFORE finalizing. Returns whether the ' +
      'answer is grounded and, if not, which claims are unsupported.',
    parameters: z.object({
      draft_answer: z.string().describe('Your draft answer, with [#n] citations.'),
    }),
    execute: async ({ draft_answer }) =>
      traced(
        'verify',
        (out) => ({ grounded: out._grounded, verify_retries: session.verifyRetries }),
        async () => {
          if (!session.pool.length) {
            return {
              _grounded: false,
              toString: () =>
                'No chunks in the pool, so nothing supports an answer. Reply: "I couldn\'t find that in the document."',
            };
          }
          const verdict = await verifyGroundedness(draft_answer, session.pool);
          if (!verdict.grounded) session.verifyRetries += 1;
          const canRetry = session.verifyRetries <= MAX_VERIFY_RETRIES;
          return {
            _grounded: verdict.grounded,
            toString: () =>
              verdict.grounded
                ? 'GROUNDED — every claim is supported. Send this draft as your final answer.'
                : `NOT grounded. Unsupported: ${verdict.issues || 'see above'}. ` +
                  (canRetry
                    ? 'Revise to remove/fix those claims, then call verify_answer again.'
                    : 'Verification budget exhausted — answer only with what IS supported by the chunks, or say you could not find it.'),
          };
        }
      ).then((r) => r.toString()),
  });

  const agent = new Agent({
    name: 'Corrective RAG',
    instructions: SYSTEM_INSTRUCTIONS,
    tools: [decomposeQuestion, searchDocument, gradeChunks, rewriteQuery, verifyAnswer],
  });

  return { agent, session };
}

/**
 * One Gemini call to split a question into standalone sub-questions. Returns the
 * original (as a single-element array) when the question is already atomic.
 */
async function decompose(question) {
  const prompt =
    `Break the following question into the MINIMAL set of standalone sub-questions ` +
    `needed to fully answer it. Rules:\n` +
    `- If the question is already a single, simple ask, return it unchanged (one line).\n` +
    `- Each sub-question must be self-contained (resolve pronouns/references).\n` +
    `- Use at most 4 sub-questions. No commentary.\n` +
    `- Output one sub-question per line, no numbering, no bullets.\n\n` +
    `QUESTION: ${question}\n\nSUB-QUESTIONS:`;
  const raw = await rawGeminiCall(prompt);
  const subs = raw
    .split('\n')
    .map((l) => l.replace(/^\s*[-*\d.)\]]+\s*/, '').trim())
    .filter(Boolean)
    .slice(0, 4);
  return subs.length ? subs : [question];
}

/**
 * One Gemini call to verify a draft answer is grounded in the pooled chunks.
 * Returns { grounded: boolean, issues: string }. Defaults to grounded=false on
 * an unparseable reply so an unverified answer never slips through as "grounded".
 */
async function verifyGroundedness(draft, pool) {
  const context = pool
    .map((d, i) => `[#${i + 1}]\n${d.text.slice(0, GRADE_CHUNK_TRUNCATE)}`)
    .join('\n\n---\n\n');
  const prompt =
    `You are a strict groundedness checker. Decide whether EVERY factual claim in the ANSWER ` +
    `is directly supported by the CHUNKS. Ignore the citation labels themselves; judge the facts.\n\n` +
    `CHUNKS:\n${context}\n\nANSWER:\n${draft}\n\n` +
    `Reply on the FIRST line exactly "GROUNDED" or "NOT_GROUNDED". If NOT_GROUNDED, on the next ` +
    `line briefly list the unsupported claim(s). No other commentary.`;
  const raw = (await rawGeminiCall(prompt)).trim();
  const firstLine = raw.split('\n')[0].toUpperCase();
  const grounded = firstLine.includes('GROUNDED') && !firstLine.includes('NOT');
  const issues = raw.split('\n').slice(1).join(' ').trim();
  return { grounded, issues };
}

/** One Gemini call that grades all retrieved chunks at once. */
async function gradeRelevance(question, docs) {
  const blocks = docs
    .map((d, i) => `[#${i + 1}]\n${d.text.slice(0, GRADE_CHUNK_TRUNCATE)}`)
    .join('\n\n---\n\n');
  const prompt =
    `You are a relevance grader. Given a question and numbered document chunks, ` +
    `decide which chunks contain information that helps answer the question.\n\n` +
    `QUESTION: ${question}\n\nCHUNKS:\n${blocks}\n\n` +
    `Reply with one line per chunk in the EXACT format:\n#1: Y\n#2: N\n(etc.)\n` +
    `Use Y if the chunk is relevant, N if not. No other commentary.`;
  const raw = await rawGeminiCall(prompt);
  return parseGrades(raw, docs.length);
}

/** Parse lines like "#1: Y" / "#2: N" into an array of booleans of length n. */
function parseGrades(text, n) {
  const grades = new Array(n).fill(false);
  for (const line of text.split('\n')) {
    const m = line.match(/#?\s*(\d+)\s*[:\-)]\s*([YyNn])/);
    if (!m) continue;
    const idx = Number(m[1]) - 1;
    if (idx >= 0 && idx < n) grades[idx] = m[2].toUpperCase() === 'Y';
  }
  return grades;
}

/** One Gemini call to reword a query that retrieved nothing relevant. */
async function rewriteSearchQuery(original, previous) {
  const prompt =
    `The following question was used to search a document, but none of the retrieved ` +
    `chunks were relevant. Rewrite the question to use different keywords, synonyms, or ` +
    `a more specific phrasing that may match the document's wording better. Return ONLY ` +
    `the rewritten question — no explanation, no quotes.\n\n` +
    `ORIGINAL QUESTION: ${original}\nPREVIOUS SEARCH QUERY: ${previous}\n\nREWRITTEN QUESTION:`;
  const rewritten = (await rawGeminiCall(prompt)).trim();
  return !rewritten || rewritten.length > 500 ? original : rewritten;
}

/**
 * Direct single-turn Gemini call used by grade/rewrite (which are deterministic
 * sub-steps, not agent turns). Goes through the same OpenAI-compatible client
 * and records token usage on the active tracer node.
 */
async function rawGeminiCall(prompt) {
  const res = await geminiClient.chat.completions.create({
    model: CHAT_MODEL,
    messages: [{ role: 'user', content: prompt }],
  });
  const usage = res.usage;
  if (usage) recordTokens(usage.prompt_tokens || 0, usage.completion_tokens || 0);
  return res.choices?.[0]?.message?.content || '';
}

/**
 * Answer one question. Runs the agent (which drives the corrective loop via
 * tools) inside a tracer run, then returns the answer, the chunks actually used
 * as sources, the final query, and the full trace.
 */
export async function answerQuestion(store, topK, question, history = []) {
  const { agent, session } = buildAgent(store, topK);

  const input = [
    ...history.map((m) => ({ role: m.role, content: m.content })),
    { role: 'user', content: question },
  ];

  return withRun(question, async (run) => {
    // The agent run drives the decompose/retrieve/grade/rewrite/verify tools
    // (each its own traced node). We then record a `generate` node for the final
    // answer synthesis so the trace reads decompose → (retrieve → grade → rewrite?)*
    // → verify → generate, and attribute the agent's own model-turn tokens to it.
    const result = await runner.run(agent, input);
    const answer = (result.finalOutput || '').trim() || "I couldn't find that in the document.";

    await traced(
      'generate',
      () => ({ answer_chars: answer.length, sub_questions: session.subQuestions.length, pool_size: session.pool.length }),
      async () => {
        const { inputTokens, outputTokens } = agentRunUsage(result);
        recordTokens(inputTokens, outputTokens);
      }
    );

    // Sources are the pooled, grader-approved chunks (across all sub-questions),
    // in the same [#n] order the agent saw them. Fall back to the last raw
    // retrieval if nothing was ever pooled.
    const shown = session.pool.length ? session.pool : session.lastDocs;
    const sources = shown.map((d) => ({ text: d.text, score: d.score, metadata: d.metadata }));

    return {
      answer,
      queryUsed: session.lastQuery || question,
      subQuestions: session.subQuestions,
      sources,
      trace: serializeRun(run),
    };
  });
}

/**
 * Total token usage across every model turn the agent made this run. Prefers
 * the SDK's aggregated `Usage` (state._context.usage); falls back to summing the
 * raw per-turn responses if the internal shape ever changes.
 */
function agentRunUsage(result) {
  const agg = result.state?._context?.usage;
  if (agg && (agg.inputTokens || agg.outputTokens)) {
    return { inputTokens: agg.inputTokens || 0, outputTokens: agg.outputTokens || 0 };
  }
  let inputTokens = 0;
  let outputTokens = 0;
  for (const r of result.rawResponses || []) {
    const u = r.usage || {};
    inputTokens += u.inputTokens ?? u.prompt_tokens ?? u.input_tokens ?? 0;
    outputTokens += u.outputTokens ?? u.completion_tokens ?? u.output_tokens ?? 0;
  }
  return { inputTokens, outputTokens };
}
