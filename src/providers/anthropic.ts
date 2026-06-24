// Anthropic provider — the original implementation, factored out behind the
// neutral Provider interface. The request shape sent to the Anthropic SDK is
// byte-identical to the pre-refactor engine, so existing fixtures and the
// taxonomy prompt-cache continue to behave the same.

import Anthropic from '@anthropic-ai/sdk';
import type { NeutralContentPart, NeutralRequest, Provider, ProviderResponse } from './types.js';

export interface AnthropicProviderOptions {
  /** Pre-constructed Anthropic client (used by tests; overrides apiKey). */
  client?: Anthropic;
  /** API key; defaults to ANTHROPIC_API_KEY in the SDK. */
  apiKey?: string;
}

function toAnthropicBlock(part: NeutralContentPart): Anthropic.ContentBlockParam {
  if (part.type === 'text') {
    return { type: 'text', text: part.text };
  }
  if (part.type === 'image') {
    return {
      type: 'image',
      source: { type: 'base64', media_type: part.mediaType, data: part.base64 },
    };
  }
  // document_pdf — Anthropic accepts a native `document` block for PDFs.
  return {
    type: 'document',
    source: { type: 'base64', media_type: 'application/pdf', data: part.base64 },
  };
}

export class AnthropicProvider implements Provider {
  readonly name = 'anthropic' as const;
  private readonly client: Anthropic;

  constructor(opts: AnthropicProviderOptions = {}) {
    this.client = opts.client ?? new Anthropic(opts.apiKey ? { apiKey: opts.apiKey } : {});
  }

  async detect(req: NeutralRequest): Promise<ProviderResponse> {
    // The catalog block is marked cacheable; Anthropic implements that as a
    // per-block `cache_control: { type: 'ephemeral' }` marker.
    const system = req.system.map((block) =>
      block.cacheable
        ? ({
            type: 'text' as const,
            text: block.text,
            cache_control: { type: 'ephemeral' as const },
          } satisfies Anthropic.TextBlockParam)
        : ({ type: 'text' as const, text: block.text } satisfies Anthropic.TextBlockParam)
    );

    const tool: Anthropic.Tool = {
      name: req.tool.name,
      description: req.tool.description,
      // The neutral JSON Schema is the same shape Anthropic expects for
      // `input_schema`, so we pass it through.
      input_schema: req.tool.jsonSchema as unknown as Anthropic.Tool['input_schema'],
    };

    const message = await this.client.messages.create({
      model: req.model,
      max_tokens: req.maxTokens,
      system,
      tools: [tool],
      tool_choice: { type: 'tool', name: tool.name },
      messages: [{ role: 'user', content: req.userContent.map(toAnthropicBlock) }],
    });

    const toolUse = message.content.find(
      (block): block is Anthropic.ToolUseBlock =>
        block.type === 'tool_use' && block.name === tool.name
    );
    const toolInput =
      toolUse && typeof toolUse.input === 'object' && toolUse.input !== null
        ? (toolUse.input as Record<string, unknown>)
        : undefined;

    return {
      toolInput,
      model: req.model,
      usage: message.usage
        ? {
            inputTokens: message.usage.input_tokens,
            outputTokens: message.usage.output_tokens,
            cacheReadInputTokens: message.usage.cache_read_input_tokens ?? 0,
            cacheCreationInputTokens: message.usage.cache_creation_input_tokens ?? 0,
          }
        : undefined,
    };
  }
}
