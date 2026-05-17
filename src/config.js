/**
 * Central configuration + provider wiring.
 *
 * The agent itself is Gemini, but it is driven through the OpenAI Agents SDK.
 * Gemini ships an OpenAI-compatible endpoint, so we point a plain `OpenAI`
 * client at that base URL and wrap it in the SDK's `OpenAIChatCompletionsModel`
 * via a custom `ModelProvider`. Embeddings reuse the same OpenAI-compatible
 * client (the `/embeddings` route), so the whole app needs a single Gemini key.
 */
import 'dotenv/config';
import OpenAI from 'openai';
import { Runner, OpenAIChatCompletionsModel, setTracingDisabled } from '@openai/agents';

export const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
export const GEMINI_BASE_URL =
  process.env.GEMINI_BASE_URL ||
  'https://generativelanguage.googleapis.com/v1beta/openai/';
export const CHAT_MODEL = process.env.CHAT_MODEL || 'gemini-2.5-flash';
export const EMBED_MODEL = process.env.EMBED_MODEL || 'gemini-embedding-001';
export const PORT = Number(process.env.PORT || 3000);

if (!GEMINI_API_KEY) {
  console.error(
    '\n[config] Missing GEMINI_API_KEY. Copy .env.example to .env and fill it in.\n' +
      'Get a free key at https://aistudio.google.com/apikey\n'
  );
}

// A single OpenAI-compatible client pointed at Gemini, shared by the agent
// (chat completions) and the embeddings module.
export const geminiClient = new OpenAI({
  apiKey: GEMINI_API_KEY || 'missing-key',
  baseURL: GEMINI_BASE_URL,
});

// The SDK's tracing uploads to OpenAI's platform; we're not on OpenAI, so off.
setTracingDisabled(true);

/**
 * Hands the SDK a Gemini-backed model for every agent run. The Runner calls
 * `getModel`; we ignore the requested name and always serve our chat model
 * (the Agent's `model` field is left unset so this provider decides).
 *
 * `ModelProvider` is a TypeScript-only interface in the SDK, so we satisfy it
 * structurally with a plain object exposing `getModel`.
 */
const geminiModelProvider = {
  async getModel(modelName) {
    return new OpenAIChatCompletionsModel(geminiClient, modelName || CHAT_MODEL);
  },
};

// One Runner reused across requests, bound to the Gemini provider.
export const runner = new Runner({ modelProvider: geminiModelProvider });
