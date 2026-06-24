// Provider abstraction — a small, provider-neutral request/response shape that
// the engine builds once and each LLM provider translates into its own SDK calls.
//
// The engine in `analyze.ts` is the single chokepoint; everything provider-
// specific (request shape, forced tool/function call, caching markers, response
// parsing) lives in one of the provider files in this directory.

import type { DetectionUsage, ImageMediaType } from '../analyze.js';

/** A system-instructions block. `cacheable: true` marks the stable prefix
 *  (the pitfall catalog) so each provider can apply its caching mechanism:
 *  Anthropic adds `cache_control: ephemeral`, Gemini creates an explicit
 *  `cachedContent` resource, OpenAI relies on automatic prefix caching and
 *  uses the prefix as a stable cache key. */
export interface NeutralSystemBlock {
  text: string;
  cacheable?: boolean;
}

/** A single content part for the user turn. Providers translate these to
 *  their own content-block shapes; document_pdf goes natively to each
 *  provider's PDF / file input channel (no local text extraction). */
export type NeutralContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; mediaType: ImageMediaType; base64: string; filename?: string }
  | { type: 'document_pdf'; base64: string; filename?: string };

/** A JSON-Schema-described tool the model must call to return its findings. */
export interface NeutralTool {
  name: string;
  description: string;
  jsonSchema: Record<string, unknown>;
}

/** A provider-neutral detection request. */
export interface NeutralRequest {
  model: string;
  maxTokens: number;
  system: NeutralSystemBlock[];
  userContent: NeutralContentPart[];
  tool: NeutralTool;
  /** Stable key (e.g. hash of cacheable system blocks + tool schema + model)
   *  that providers may use for explicit caching. */
  cacheKey?: string;
}

/** A provider's response, normalized so the engine can validate findings. */
export interface ProviderResponse {
  /** The forced tool/function call's argument object, or undefined if the
   *  provider returned no tool call (in which case findings will be empty). */
  toolInput: Record<string, unknown> | undefined;
  /** Token usage, normalized across providers. */
  usage?: DetectionUsage;
  /** The model id actually used (echoed back unchanged in most providers). */
  model: string;
}

/** A provider implementation translates a neutral request to its SDK and
 *  parses the forced tool call back into a neutral response. */
export interface Provider {
  readonly name: 'anthropic' | 'openai' | 'gemini';
  detect(req: NeutralRequest): Promise<ProviderResponse>;
}

/** Provider identifiers exposed via `DetectionOptions.provider` and the CLI. */
export type ProviderName = 'anthropic' | 'openai' | 'gemini';
