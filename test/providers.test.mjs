// Tests for the OpenAI and Gemini provider adapters. They inject fake SDK
// clients (matching the surface each provider uses), so they run fully
// offline with no API key and no network.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  detectPitfalls,
  getAllRules,
  ruleCount,
} from '../dist/index.js';

const realRule = getAllRules()[0];

// ----- OpenAI -----

function fakeOpenAIClient(toolArgs, opts = {}) {
  const calls = [];
  return {
    calls,
    responses: {
      create: async (params) => {
        calls.push(params);
        return {
          output: toolArgs === null
            ? []
            : [
                {
                  type: 'function_call',
                  name: 'report_findings',
                  arguments: JSON.stringify({ findings: toolArgs, ...(opts.extra ?? {}) }),
                },
              ],
          usage: {
            input_tokens: 200,
            output_tokens: 30,
            input_tokens_details: { cached_tokens: 150 },
          },
        };
      },
    },
  };
}

test('openai provider sends a function tool, forces a tool_choice, and a prompt_cache_key', async () => {
  const client = fakeOpenAIClient([
    { rule_id: realRule.id, confidence: 'high', nature: 'active', evidence: 'x', explanation: 'y' },
  ]);
  const report = await detectPitfalls(
    { kind: 'code', content: 'SELECT 1', language: 'SQL' },
    { client, provider: 'openai', model: 'gpt-5' }
  );

  assert.equal(report.findings.length, 1);
  assert.equal(report.findings[0].ruleId, realRule.id);
  assert.equal(report.usage.cacheReadInputTokens, 150);
  assert.equal(report.model, 'gpt-5');
  assert.equal(report.rulesConsidered, ruleCount());

  const params = client.calls[0];
  assert.equal(params.model, 'gpt-5');
  assert.equal(params.tool_choice.type, 'function');
  assert.equal(params.tool_choice.name, 'report_findings');
  assert.equal(params.tools[0].type, 'function');
  assert.equal(params.tools[0].name, 'report_findings');
  assert.ok(params.tools[0].parameters.properties.findings, 'parameters carries the JSON schema');
  assert.ok(params.prompt_cache_key, 'prompt_cache_key is set for stable prefix routing');
  assert.ok(params.prompt_cache_key.startsWith('datapitfalls-'));

  // Developer message gathers the system text; user message has input_text.
  assert.equal(params.input[0].role, 'developer');
  assert.equal(params.input[0].content[0].type, 'input_text');
  assert.match(params.input[0].content[0].text, /Pitfall catalog/);
  assert.equal(params.input[1].role, 'user');
});

test('openai provider sends PDFs as native input_file parts', async () => {
  const client = fakeOpenAIClient([]);
  await detectPitfalls(
    {
      kind: 'document',
      content: 'JVBERi0xLjQK',
      mediaType: 'application/pdf',
      filename: 'report.pdf',
    },
    { client, provider: 'openai' }
  );
  const userParts = client.calls[0].input[1].content;
  const fileParts = userParts.filter((p) => p.type === 'input_file');
  assert.equal(fileParts.length, 1);
  assert.equal(fileParts[0].filename, 'report.pdf');
  assert.match(fileParts[0].file_data, /^data:application\/pdf;base64,/);
});

test('openai provider sends images as input_image data URLs', async () => {
  const client = fakeOpenAIClient([]);
  await detectPitfalls(
    {
      kind: 'image',
      images: [
        { content: 'AAAA', mediaType: 'image/png', filename: 'a.png' },
        { content: 'BBBB', mediaType: 'image/jpeg', filename: 'b.jpg' },
      ],
    },
    { client, provider: 'openai' }
  );
  const imageParts = client.calls[0].input[1].content.filter((p) => p.type === 'input_image');
  assert.equal(imageParts.length, 2);
  assert.equal(imageParts[0].image_url, 'data:image/png;base64,AAAA');
  assert.equal(imageParts[1].image_url, 'data:image/jpeg;base64,BBBB');
});

test('openai provider validates findings against the catalog like the engine does', async () => {
  const client = fakeOpenAIClient([
    { rule_id: realRule.id, confidence: 'high', nature: 'active', evidence: 'x', explanation: 'y' },
    { rule_id: '__not_a_real_rule__', confidence: 'high', nature: 'active', evidence: 'x', explanation: 'y' },
  ]);
  const report = await detectPitfalls(
    { kind: 'code', content: 'SELECT 1' },
    { client, provider: 'openai' }
  );
  assert.equal(report.findings.length, 1);
  assert.equal(report.findings[0].ruleId, realRule.id);
});

test('openai provider is auto-selected from a gpt-* model id', async () => {
  const client = fakeOpenAIClient([]);
  await detectPitfalls(
    { kind: 'code', content: 'SELECT 1' },
    { client, model: 'gpt-5-mini' }
  );
  // The fake's responses.create being called is itself the proof — Anthropic
  // would have thrown on client.messages.create being undefined.
  assert.equal(client.calls.length, 1);
});

// ----- Gemini -----

function fakeGeminiClient(toolArgs, opts = {}) {
  const calls = [];
  const cacheCalls = [];
  return {
    calls,
    cacheCalls,
    models: {
      generateContent: async (params) => {
        calls.push(params);
        return {
          functionCalls:
            toolArgs === null
              ? []
              : [{ name: 'report_findings', args: { findings: toolArgs, ...(opts.extra ?? {}) } }],
          usageMetadata: {
            promptTokenCount: 300,
            candidatesTokenCount: 40,
            cachedContentTokenCount: 250,
          },
        };
      },
    },
    caches: {
      create: async (params) => {
        cacheCalls.push(params);
        return { name: `cachedContents/${cacheCalls.length}` };
      },
    },
  };
}

test('gemini provider forces a function call via toolConfig and creates an explicit cache', async () => {
  const client = fakeGeminiClient([
    { rule_id: realRule.id, confidence: 'high', nature: 'active', evidence: 'x', explanation: 'y' },
  ]);
  const report = await detectPitfalls(
    { kind: 'code', content: 'SELECT 1', language: 'SQL' },
    { client, provider: 'gemini', model: 'gemini-2.5-pro' }
  );

  assert.equal(report.findings.length, 1);
  assert.equal(report.findings[0].ruleId, realRule.id);
  assert.equal(report.usage.cacheReadInputTokens, 250);
  assert.equal(report.model, 'gemini-2.5-pro');

  // Cache was created for the catalog block.
  assert.equal(client.cacheCalls.length, 1);
  assert.match(client.cacheCalls[0].config.contents[0].parts[0].text, /Pitfall catalog/);
  assert.equal(client.cacheCalls[0].model, 'gemini-2.5-pro');

  const params = client.calls[0];
  assert.equal(params.model, 'gemini-2.5-pro');
  assert.equal(params.config.cachedContent, 'cachedContents/1');
  assert.equal(params.config.toolConfig.functionCallingConfig.mode, 'ANY');
  assert.deepEqual(params.config.toolConfig.functionCallingConfig.allowedFunctionNames, [
    'report_findings',
  ]);
  assert.equal(params.config.tools[0].functionDeclarations[0].name, 'report_findings');
});

test('gemini provider sends PDFs as inlineData with application/pdf', async () => {
  const client = fakeGeminiClient([]);
  await detectPitfalls(
    {
      kind: 'document',
      content: 'JVBERi0xLjQK',
      mediaType: 'application/pdf',
      filename: 'report.pdf',
    },
    { client, provider: 'gemini' }
  );
  const parts = client.calls[0].contents[0].parts;
  const pdfParts = parts.filter((p) => p.inlineData?.mimeType === 'application/pdf');
  assert.equal(pdfParts.length, 1);
  assert.equal(pdfParts[0].inlineData.data, 'JVBERi0xLjQK');
});

test('gemini provider sends images as inlineData parts', async () => {
  const client = fakeGeminiClient([]);
  await detectPitfalls(
    {
      kind: 'image',
      images: [
        { content: 'AAAA', mediaType: 'image/png', filename: 'a.png' },
        { content: 'BBBB', mediaType: 'image/jpeg', filename: 'b.jpg' },
      ],
    },
    { client, provider: 'gemini' }
  );
  const parts = client.calls[0].contents[0].parts;
  const imageParts = parts.filter((p) => p.inlineData?.mimeType?.startsWith('image/'));
  assert.equal(imageParts.length, 2);
  assert.equal(imageParts[0].inlineData.data, 'AAAA');
  assert.equal(imageParts[1].inlineData.data, 'BBBB');
});

test('gemini provider falls back to inline catalog if cache creation fails', async () => {
  const client = {
    calls: [],
    models: {
      generateContent: async (params) => {
        client.calls.push(params);
        return { functionCalls: [{ name: 'report_findings', args: { findings: [] } }] };
      },
    },
    caches: {
      // Use a unique key by including a different model name so we don't hit
      // the LRU entry from the previous test.
      create: async () => {
        throw new Error('cache service unavailable');
      },
    },
  };
  await detectPitfalls(
    { kind: 'code', content: 'SELECT 2' },
    { client, provider: 'gemini', model: 'gemini-2.5-pro-fallbacktest' }
  );
  const params = client.calls[0];
  // No cachedContent set, catalog folded into systemInstruction instead.
  assert.equal(params.config.cachedContent, undefined);
  assert.match(params.config.systemInstruction, /Pitfall catalog/);
});

test('gemini provider is auto-selected from a gemini-* model id', async () => {
  const client = fakeGeminiClient([]);
  await detectPitfalls(
    { kind: 'code', content: 'SELECT 1' },
    { client, model: 'gemini-2.5-flash' }
  );
  assert.equal(client.calls.length, 1);
});
