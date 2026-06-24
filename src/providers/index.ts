// Provider selection and barrel exports.
//
// The engine in `analyze.ts` calls `pickProvider()` with the user's
// `DetectionOptions`. If a provider is explicitly named or a client is
// injected, that wins; otherwise we infer the provider from the model id
// prefix (`claude-*` → anthropic, `gpt-*`/`o*` → openai, `gemini-*` → gemini).

import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { GoogleGenAI } from '@google/genai';

import { AnthropicProvider } from './anthropic.js';
import { OpenAIProvider } from './openai.js';
import { GeminiProvider } from './gemini.js';
import type { Provider, ProviderName } from './types.js';

export * from './types.js';
export { AnthropicProvider, OpenAIProvider, GeminiProvider };

/** Per-provider model defaults. `default` is the everyday model; `fast` and
 *  `thorough` are what `--fast` and `--thorough` map to on the CLI. */
export interface ProviderModelDefaults {
  default: string;
  fast: string;
  thorough: string;
}

export const PROVIDER_DEFAULTS: Record<ProviderName, ProviderModelDefaults> = {
  anthropic: {
    default: 'claude-sonnet-4-6',
    fast: 'claude-haiku-4-5',
    thorough: 'claude-opus-4-7',
  },
  openai: {
    default: 'gpt-5',
    fast: 'gpt-5-mini',
    thorough: 'gpt-5',
  },
  gemini: {
    default: 'gemini-2.5-pro',
    fast: 'gemini-2.5-flash',
    thorough: 'gemini-2.5-pro',
  },
};

/** Infer a provider from a model id when the caller didn't say which to use. */
export function inferProviderFromModel(model: string | undefined): ProviderName | undefined {
  if (!model) return undefined;
  const m = model.toLowerCase();
  if (m.startsWith('claude')) return 'anthropic';
  if (m.startsWith('gemini')) return 'gemini';
  if (m.startsWith('gpt') || m.startsWith('o1') || m.startsWith('o3') || m.startsWith('o4')) {
    return 'openai';
  }
  return undefined;
}

/** Detect which provider an injected SDK client belongs to. */
function inferProviderFromClient(client: unknown): ProviderName | undefined {
  if (!client || typeof client !== 'object') return undefined;
  if (client instanceof Anthropic) return 'anthropic';
  if (client instanceof OpenAI) return 'openai';
  if (client instanceof GoogleGenAI) return 'gemini';
  // Duck-typed fallback for test fakes that don't extend the real class.
  const c = client as Record<string, unknown>;
  if (c.messages && typeof (c.messages as { create?: unknown }).create === 'function') {
    return 'anthropic';
  }
  if (c.responses && typeof (c.responses as { create?: unknown }).create === 'function') {
    return 'openai';
  }
  if (c.models && typeof (c.models as { generateContent?: unknown }).generateContent === 'function') {
    return 'gemini';
  }
  return undefined;
}

export interface PickProviderOptions {
  provider?: ProviderName;
  client?: unknown;
  apiKey?: string;
  model?: string;
}

/** Pick the provider implementation for a detection call. Order of precedence:
 *  explicit `provider`, then injected `client`'s type, then the model id
 *  prefix, then 'anthropic' (the historical default). */
export function pickProvider(opts: PickProviderOptions): Provider {
  const explicit = opts.provider;
  const fromClient = inferProviderFromClient(opts.client);
  const fromModel = inferProviderFromModel(opts.model);
  const name: ProviderName = explicit ?? fromClient ?? fromModel ?? 'anthropic';

  switch (name) {
    case 'anthropic':
      return new AnthropicProvider({
        client: opts.client as Anthropic | undefined,
        apiKey: opts.apiKey,
      });
    case 'openai':
      return new OpenAIProvider({
        client: opts.client as OpenAI | undefined,
        apiKey: opts.apiKey,
      });
    case 'gemini':
      return new GeminiProvider({
        client: opts.client as GoogleGenAI | undefined,
        apiKey: opts.apiKey,
      });
  }
}

/** Resolve the model id for a (provider, override, env) triple. Used by both
 *  the engine default and the CLI's --fast / --thorough flags. */
export function resolveDefaultModel(provider: ProviderName): string {
  return PROVIDER_DEFAULTS[provider].default;
}
