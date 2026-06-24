// OpenAI provider — uses the Responses API so PDFs and images can be sent in
// natively as `input_file` / `input_image` content parts, matching the shape
// Anthropic and Gemini use for native documents (no local text extraction).
//
// The model is forced to call `report_findings` via `tools: [{type:'function',
// strict: true, ...}]` and `tool_choice: {type:'function', name}`. Caching
// relies on OpenAI's automatic prefix caching, which kicks in for shared
// prefixes ≥ 1024 tokens; passing a stable `prompt_cache_key` makes the
// routing predictable and surfaces `cached_tokens` in usage.

import OpenAI from 'openai';
import type {
  NeutralContentPart,
  NeutralRequest,
  Provider,
  ProviderResponse,
} from './types.js';

export interface OpenAIProviderOptions {
  client?: OpenAI;
  apiKey?: string;
}

/** Inline-data URL for an image, which Responses accepts as `input_image`. */
function imageDataUrl(mediaType: string, base64: string): string {
  return `data:${mediaType};base64,${base64}`;
}

/** Synthetic filename when a PDF was uploaded without one. The Responses API
 *  requires `filename` alongside inline `file_data`. */
function pdfFilename(part: { filename?: string }): string {
  return part.filename && part.filename.trim() !== '' ? part.filename : 'document.pdf';
}

function toOpenAIContent(part: NeutralContentPart): Record<string, unknown> {
  if (part.type === 'text') {
    return { type: 'input_text', text: part.text };
  }
  if (part.type === 'image') {
    return { type: 'input_image', image_url: imageDataUrl(part.mediaType, part.base64) };
  }
  // document_pdf — Responses API takes inline PDFs as base64 `file_data` with a
  // `filename`. Files API uploads would also work but inline keeps the engine
  // stateless and matches how Anthropic/Gemini receive PDFs.
  return {
    type: 'input_file',
    filename: pdfFilename(part),
    file_data: `data:application/pdf;base64,${part.base64}`,
  };
}

interface ResponsesOutputItem {
  type?: string;
  name?: string;
  arguments?: string;
}

interface ResponsesUsage {
  input_tokens?: number;
  output_tokens?: number;
  input_tokens_details?: { cached_tokens?: number };
}

interface ResponsesResult {
  output?: ResponsesOutputItem[];
  usage?: ResponsesUsage;
}

function findFunctionCallArguments(
  output: ResponsesOutputItem[] | undefined,
  name: string
): Record<string, unknown> | undefined {
  if (!output) return undefined;
  for (const item of output) {
    if (item.type === 'function_call' && item.name === name && typeof item.arguments === 'string') {
      try {
        const parsed = JSON.parse(item.arguments) as unknown;
        if (parsed && typeof parsed === 'object') return parsed as Record<string, unknown>;
      } catch {
        return undefined;
      }
    }
  }
  return undefined;
}

export class OpenAIProvider implements Provider {
  readonly name = 'openai' as const;
  private readonly client: OpenAI;

  constructor(opts: OpenAIProviderOptions = {}) {
    this.client = opts.client ?? new OpenAI(opts.apiKey ? { apiKey: opts.apiKey } : {});
  }

  async detect(req: NeutralRequest): Promise<ProviderResponse> {
    // Build a single Responses `input` array: one developer/system message
    // carrying every system block (cacheable prefix first), then the user
    // message with the neutral content parts.
    const systemText = req.system.map((b) => b.text).join('\n\n');

    const input = [
      { role: 'developer', content: [{ type: 'input_text', text: systemText }] },
      { role: 'user', content: req.userContent.map(toOpenAIContent) },
    ];

    // Responses tool shape: a single function tool whose parameters are the
    // engine's JSON Schema. `strict: true` enforces the schema in routing.
    const tool = {
      type: 'function' as const,
      name: req.tool.name,
      description: req.tool.description,
      parameters: req.tool.jsonSchema,
      strict: false,
    };

    // The OpenAI SDK's TS types lag the API surface in places (input_file,
    // function tools on Responses), so we cast at the boundary.
    const create = this.client.responses.create.bind(this.client.responses) as (
      params: Record<string, unknown>
    ) => Promise<ResponsesResult>;

    const response = await create({
      model: req.model,
      max_output_tokens: req.maxTokens,
      input,
      tools: [tool],
      tool_choice: { type: 'function', name: req.tool.name },
      ...(req.cacheKey ? { prompt_cache_key: req.cacheKey } : {}),
    });

    const toolInput = findFunctionCallArguments(response.output, req.tool.name);

    const usage = response.usage;
    return {
      toolInput,
      model: req.model,
      usage: usage
        ? {
            inputTokens: usage.input_tokens ?? 0,
            outputTokens: usage.output_tokens ?? 0,
            // OpenAI surfaces cache hits as `input_tokens_details.cached_tokens`.
            // There is no separate cache-creation counter (writes are implicit
            // and free), so we map that to 0.
            cacheReadInputTokens: usage.input_tokens_details?.cached_tokens ?? 0,
            cacheCreationInputTokens: 0,
          }
        : undefined,
    };
  }
}
