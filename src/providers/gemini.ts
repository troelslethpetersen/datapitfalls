// Gemini provider — uses `@google/genai`'s `models.generateContent`. PDFs and
// images go in as `inlineData` parts (native, no local extraction), and the
// model is forced to call `report_findings` via `toolConfig.functionCallingConfig`
// in `ANY` mode with `allowedFunctionNames`.
//
// Caching: the catalog block is large and stable across requests, which is
// exactly what Gemini's explicit context caching is for. We create a
// `cachedContent` for the cacheable system text on first use (keyed by
// req.cacheKey), keep an in-process LRU of cache resource names, and reference
// it via `config.cachedContent` on subsequent calls. If creation or reuse
// fails (e.g. cache expired, key not yet usable), we fall back to sending the
// system text inline so the request still succeeds.

import { GoogleGenAI } from '@google/genai';
import type {
  NeutralContentPart,
  NeutralRequest,
  Provider,
  ProviderResponse,
} from './types.js';

export interface GeminiProviderOptions {
  client?: GoogleGenAI;
  /** API key; defaults to GOOGLE_API_KEY / GEMINI_API_KEY in the SDK. */
  apiKey?: string;
  /** TTL for explicit cachedContent entries. Defaults to 300s. */
  cacheTtlSeconds?: number;
}

interface GeminiPart {
  text?: string;
  inlineData?: { mimeType: string; data: string };
  functionCall?: { name?: string; args?: Record<string, unknown> };
}

interface GeminiContent {
  role?: string;
  parts: GeminiPart[];
}

interface GeminiUsage {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  cachedContentTokenCount?: number;
}

interface GeminiResponse {
  candidates?: { content?: GeminiContent }[];
  functionCalls?: { name?: string; args?: Record<string, unknown> }[];
  usageMetadata?: GeminiUsage;
}

interface CachedEntry {
  name: string;
  expiresAtMs: number;
}

// Process-local LRU of cached-content resource names. Keyed by the engine's
// req.cacheKey, which already folds in the catalog text, tool schema, model,
// and variant — so different requests get different cache entries.
const CACHE_REGISTRY = new Map<string, CachedEntry>();
const CACHE_MAX_ENTRIES = 16;
const DEFAULT_CACHE_TTL_S = 300;

function touchCacheEntry(key: string, entry: CachedEntry): void {
  CACHE_REGISTRY.delete(key);
  CACHE_REGISTRY.set(key, entry);
  while (CACHE_REGISTRY.size > CACHE_MAX_ENTRIES) {
    const oldest = CACHE_REGISTRY.keys().next().value;
    if (oldest === undefined) break;
    CACHE_REGISTRY.delete(oldest);
  }
}

function getCacheEntry(key: string): string | undefined {
  const entry = CACHE_REGISTRY.get(key);
  if (!entry) return undefined;
  if (Date.now() >= entry.expiresAtMs) {
    CACHE_REGISTRY.delete(key);
    return undefined;
  }
  // Refresh LRU position.
  CACHE_REGISTRY.delete(key);
  CACHE_REGISTRY.set(key, entry);
  return entry.name;
}

function toGeminiPart(part: NeutralContentPart): GeminiPart {
  if (part.type === 'text') return { text: part.text };
  if (part.type === 'image') {
    return { inlineData: { mimeType: part.mediaType, data: part.base64 } };
  }
  return { inlineData: { mimeType: 'application/pdf', data: part.base64 } };
}

export class GeminiProvider implements Provider {
  readonly name = 'gemini' as const;
  private readonly client: GoogleGenAI;
  private readonly cacheTtlSeconds: number;

  constructor(opts: GeminiProviderOptions = {}) {
    this.client =
      opts.client ??
      new GoogleGenAI(
        opts.apiKey
          ? { apiKey: opts.apiKey }
          : {
              apiKey:
                process.env.GOOGLE_API_KEY ??
                process.env.GEMINI_API_KEY ??
                process.env.GOOGLE_GENAI_API_KEY,
            }
      );
    this.cacheTtlSeconds = opts.cacheTtlSeconds ?? DEFAULT_CACHE_TTL_S;
  }

  /** Create (or reuse) an explicit cachedContent for the catalog block, so the
   *  same catalog isn't billed and re-tokenized on every audit. Returns the
   *  resource name, or undefined on any failure (caller falls back to inline). */
  private async ensureCachedSystem(
    cacheableText: string,
    model: string,
    cacheKey: string
  ): Promise<string | undefined> {
    const existing = getCacheEntry(cacheKey);
    if (existing) return existing;

    const caches = (this.client as unknown as {
      caches?: {
        create?: (params: Record<string, unknown>) => Promise<{ name?: string }>;
      };
    }).caches;
    if (!caches?.create) return undefined;

    try {
      const created = await caches.create({
        model,
        config: {
          contents: [{ role: 'user', parts: [{ text: cacheableText }] }],
          ttl: `${this.cacheTtlSeconds}s`,
        },
      });
      const name = created?.name;
      if (!name) return undefined;
      touchCacheEntry(cacheKey, {
        name,
        // Expire our local entry slightly before the server's, so we don't try
        // to reuse one that has just timed out.
        expiresAtMs: Date.now() + (this.cacheTtlSeconds - 5) * 1000,
      });
      return name;
    } catch {
      return undefined;
    }
  }

  async detect(req: NeutralRequest): Promise<ProviderResponse> {
    const cacheableText = req.system
      .filter((b) => b.cacheable)
      .map((b) => b.text)
      .join('\n\n');
    const inlineSystemText = req.system
      .filter((b) => !b.cacheable)
      .map((b) => b.text)
      .join('\n\n');

    let cachedName: string | undefined;
    if (cacheableText && req.cacheKey) {
      cachedName = await this.ensureCachedSystem(cacheableText, req.model, req.cacheKey);
    }

    // If explicit caching didn't take, fold the catalog back into the inline
    // system text so the request still has the grounding it needs.
    const systemInstruction = cachedName
      ? inlineSystemText || undefined
      : [inlineSystemText, cacheableText].filter(Boolean).join('\n\n') || undefined;

    const tool = {
      functionDeclarations: [
        {
          name: req.tool.name,
          description: req.tool.description,
          parameters: req.tool.jsonSchema,
        },
      ],
    };

    const config: Record<string, unknown> = {
      maxOutputTokens: req.maxTokens,
      tools: [tool],
      toolConfig: {
        functionCallingConfig: {
          mode: 'ANY',
          allowedFunctionNames: [req.tool.name],
        },
      },
    };
    if (systemInstruction) config.systemInstruction = systemInstruction;
    if (cachedName) config.cachedContent = cachedName;

    const generateContent = (this.client as unknown as {
      models: { generateContent: (params: Record<string, unknown>) => Promise<GeminiResponse> };
    }).models.generateContent;

    let response: GeminiResponse;
    try {
      response = await generateContent({
        model: req.model,
        contents: [{ role: 'user', parts: req.userContent.map(toGeminiPart) }],
        config,
      });
    } catch (err) {
      // If the cached resource was the problem (expired between use and now,
      // permission, etc.), retry once with the catalog inlined.
      if (cachedName) {
        CACHE_REGISTRY.delete(req.cacheKey ?? '');
        const fallbackConfig = { ...config };
        delete fallbackConfig.cachedContent;
        fallbackConfig.systemInstruction =
          [inlineSystemText, cacheableText].filter(Boolean).join('\n\n') || undefined;
        response = await generateContent({
          model: req.model,
          contents: [{ role: 'user', parts: req.userContent.map(toGeminiPart) }],
          config: fallbackConfig,
        });
      } else {
        throw err;
      }
    }

    // Prefer the SDK's flattened `functionCalls`, but fall back to scanning
    // the candidate parts for a `functionCall` so we don't break if the SDK
    // changes that convenience accessor.
    let args: Record<string, unknown> | undefined;
    const flat = response.functionCalls?.find((c) => c.name === req.tool.name);
    if (flat?.args && typeof flat.args === 'object') {
      args = flat.args;
    } else {
      const parts = response.candidates?.[0]?.content?.parts ?? [];
      for (const part of parts) {
        if (part.functionCall?.name === req.tool.name && part.functionCall.args) {
          args = part.functionCall.args;
          break;
        }
      }
    }

    const usage = response.usageMetadata;
    return {
      toolInput: args,
      model: req.model,
      usage: usage
        ? {
            inputTokens: usage.promptTokenCount ?? 0,
            outputTokens: usage.candidatesTokenCount ?? 0,
            cacheReadInputTokens: usage.cachedContentTokenCount ?? 0,
            cacheCreationInputTokens: 0,
          }
        : undefined,
    };
  }
}
