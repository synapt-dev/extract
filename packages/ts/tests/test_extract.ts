import { describe, expect, test } from "vitest";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  createExtractionBuilder,
  extract,
  type EmbeddingRequest,
  type LlmRequest,
} from "../src/index.js";
import { extractOpenAI, type OpenAICompatibleClient } from "../src/openai.js";

const SAMPLE_TEXT = [
  "On May 10, 2026, Layne told Mark that Synapt should ship the extraction builder by Friday.",
  "Mark asked whether embeddings should cover the source and summary.",
  "Layne said the first version might use local entity IDs, and if validation passes, they will publish the gist.",
].join(" ");

const STAGE1_FULL = {
  extracted_at: "2026-05-12T14:00:00Z",
  entities: [
    {
      id: "e1",
      name: "Layne",
      type: "person",
      aliases: ["Layne Penney"],
      state: "coordinating Synapt extraction work",
      context: "Asked Mark to review builder and embedding behavior.",
      date_hint: "2026-05-10",
      relations: [
        {
          target: "e2",
          type: "collaborates_with",
          origin: "explicit",
          signals: { confidence: 0.91 },
        },
      ],
      source: { snippet: "Layne told Mark", sentence_index: 0 },
      signals: { confidence: 0.93 },
    },
    {
      id: "e2",
      name: "Mark",
      type: "person",
      aliases: ["Mark Hendrickson"],
      state: "reviewing Synapt extraction ideas",
      context: "Asked about embedding coverage.",
      date_hint: "2026-05-10",
      relations: [],
      source: { snippet: "Mark asked", sentence_index: 1 },
      signals: { confidence: 0.9 },
    },
  ],
  goals: [
    {
      text: "Ship the extraction builder by Friday.",
      status: "open",
      entity_refs: ["e1", "e2"],
      stated_at: "2026-05-10T00:00:00Z",
      source: { snippet: "ship the extraction builder by Friday", sentence_index: 0 },
      signals: { confidence: 0.88 },
    },
  ],
  themes: ["extraction pipeline", "embeddings", "schema validation"],
  keywords: ["Synapt", "extraction builder", "embeddings", "gist"],
  summary: "Layne and Mark discussed shipping a Synapt extraction builder with embedding coverage and validation.",
  sentiment: { valence: "positive", intensity: 0.55, confidence: 0.72 },
  facts: [
    {
      text: "Mark asked whether embeddings should cover the source and summary.",
      category: "technical_question",
      source: { snippet: "embeddings should cover the source and summary", sentence_index: 1 },
      signals: { confidence: 0.95 },
    },
  ],
  questions: [
    {
      text: "Should embeddings cover the source and summary?",
      directed_to: "Layne",
      source: { snippet: "whether embeddings should cover the source and summary", sentence_index: 1 },
      signals: { confidence: 0.95 },
    },
  ],
  actions: [
    {
      text: "Publish the gist if validation passes.",
      origin: "extracted",
      entity_refs: ["e1"],
      due: "2026-05-15T00:00:00Z",
      source: { snippet: "if validation passes, they will publish the gist", sentence_index: 2 },
      signals: { confidence: 0.82, condition: "validation passes" },
    },
  ],
  decisions: [
    {
      text: "The first version may use local entity IDs.",
      entity_refs: ["e1"],
      decided_at: "2026-05-10T00:00:00Z",
      source: { snippet: "first version might use local entity IDs", sentence_index: 2 },
      signals: { confidence: 0.7, hedged: true },
    },
  ],
  temporal_refs: [
    {
      raw: "Friday",
      type: "point",
      resolved: "2026-05-15T00:00:00Z",
      context: "ship the extraction builder by Friday",
    },
  ],
  language: "en-US",
  source_metadata: {
    token_count: 47,
    character_count: SAMPLE_TEXT.length,
    modality: "text",
    format: "plain",
  },
  confidence: 0.86,
};

describe("extract", () => {
  test("plans fluent full/minus/embed capability UX", () => {
    const plan = createExtractionBuilder(SAMPLE_TEXT)
      .full({ embed: true })
      .minus("questions")
      .embed("summary", false)
      .plan();

    expect(plan.capabilities).toContain("entities");
    expect(plan.capabilities).not.toContain("questions");
    expect(plan.excluded).toContain("questions");
    expect(plan.embeddedInputs).toContain("entities");
    expect(plan.embeddedInputs).not.toContain("questions");
    expect(plan.embeddedInputs).not.toContain("summary");
    expect(plan.requiredCallbacks).toEqual({ callLlm: true, getEmbedding: true });
    expect(plan.promptCharacters).toBeGreaterThan(0);
  });

  test("runs full profile LLM extraction and embedding callbacks", async () => {
    const llmRequests: LlmRequest[] = [];
    const embeddingRequests: EmbeddingRequest[] = [];

    const result = await extract(
      SAMPLE_TEXT,
      {
        callLlm: (request) => {
          llmRequests.push(request);
          return {
            output: STAGE1_FULL,
            produced_by: {
              model: "openai://gpt-5.5",
              model_version: "gpt-5.5-2026-04-23",
              configuration: { reasoning_effort: "medium" },
              operator: "test",
            },
            usage: { input_tokens: 100, output_tokens: 50, total_tokens: 150 },
          };
        },
        getEmbedding: (request) => {
          embeddingRequests.push(request);
          return {
            vector: [request.text.length / 1000, embeddingRequests.length],
            model: "openai://text-embedding-3-small",
            space: "cosine",
            computed_at: "2026-05-12T14:00:01Z",
          };
        },
      },
      {
        profile: "full",
        source_id: "fixture-full-1",
        source_type: "message",
        user_id: "user-1",
        kind: "synapt/test",
        embeddingInputs: "all",
      },
    );

    expect(llmRequests).toHaveLength(1);
    expect(llmRequests[0].responseFormat.type).toBe("json_schema");
    expect(llmRequests[0].capabilities).toContain("relation_origin");
    expect(embeddingRequests.map((request) => request.input)).toEqual([
      "source",
      "summary",
      "entities",
      "goals",
      "themes",
      "keywords",
      "facts",
      "questions",
      "actions",
      "decisions",
      "temporal_refs",
      "sentiment",
    ]);
    expect(result.validation.valid).toBe(true);
    expect(result.extraction.embeddings).toHaveLength(12);
    expect(result.extraction.embeddings?.[0]).toMatchObject({
      version: "1",
      input: "source",
      model: "openai://text-embedding-3-small",
      dimensions: 2,
    });
    expect(result.extraction.capabilities).toEqual(expect.arrayContaining(llmRequests[0].capabilities));
    expect(result.extraction.produced_by).toMatchObject({
      version: "1",
      model: "openai://gpt-5.5",
      model_version: "gpt-5.5-2026-04-23",
    });
    expect(result.usage).toEqual({
      llm_calls: 1,
      embedding_calls: 12,
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
    });
    expect(result.warnings).toEqual([]);
  });

  test("requires an embedding callback when embedding inputs are requested", async () => {
    await expect(extract(
      SAMPLE_TEXT,
      { callLlm: () => ({ output: { ...STAGE1_FULL } }) },
      {
        profile: "full",
        produced_by: "openai://gpt-5.5",
        embeddingInputs: ["source"],
      },
    )).rejects.toThrow(/getEmbedding/);
  });

  test("derives embedding inputs from capability specs", async () => {
    const embeddingInputs: string[] = [];

    const result = await extract(
      SAMPLE_TEXT,
      {
        callLlm: () => ({
          output: STAGE1_FULL,
          produced_by: "openai://gpt-5.5",
        }),
        getEmbedding: (request) => {
          embeddingInputs.push(request.input);
          return {
            vector: [0.1, 0.2],
            model: "openai://text-embedding-3-small",
          };
        },
      },
      {
        capabilities: [
          { name: "entity_context", embed: true },
          { name: "summary", embed: true },
          "goals",
        ],
        embeddingInputs: ["source"],
      },
    );

    expect(embeddingInputs).toEqual(["source", "entities", "summary"]);
    expect(result.extraction.embeddings?.map((embedding) => embedding.input)).toEqual(["source", "entities", "summary"]);
    expect(result.validation.valid).toBe(true);
  });

  test("lets dynamic extensions use normalized response context", async () => {
    const result = await extract(
      SAMPLE_TEXT,
      {
        callLlm: () => ({
          output: STAGE1_FULL,
          produced_by: "openai://gpt-5.5",
          id: "resp_test_123",
          status: "completed",
          model: "gpt-5.5-2026-04-23",
        }),
      },
      {
        profile: "full",
        extend: ({ response, stage1, embeddings }) => ({
          "synapt/response_binding": {
            response_id: response.id,
            response_status: response.status,
            response_model: response.model,
            stage1_fields: Object.keys(stage1).length,
            embedding_count: embeddings.length,
          },
        }),
      },
    );

    expect(result.validation.valid).toBe(true);
    expect(result.extraction.extensions?.["synapt/response_binding"]).toMatchObject({
      version: "1",
      response_id: "resp_test_123",
      response_status: "completed",
      response_model: "gpt-5.5-2026-04-23",
      embedding_count: 0,
    });
  });

  test("translates OpenAI raw responses into normalized context", async () => {
    const result = await extract(
      SAMPLE_TEXT,
      {
        callLlm: () => ({
          output: STAGE1_FULL,
          raw: {
            object: "response",
            id: "resp_raw_123",
            status: "completed",
            model: "gpt-5.5-2026-04-23",
            usage: { input_tokens: 42, output_tokens: 18 },
          },
        }),
      },
      {
        profile: "full",
        extend: ({ response }) => ({
          "synapt/response_binding": {
            provider: response.provider,
            response_id: response.id,
            response_status: response.status,
            response_model: response.model,
          },
        }),
      },
    );

    expect(result.validation.valid).toBe(true);
    expect(result.extraction.produced_by).toMatchObject({
      version: "1",
      model: "openai://gpt-5.5-2026-04-23",
      model_version: "gpt-5.5-2026-04-23",
    });
    expect(result.usage).toMatchObject({
      llm_calls: 1,
      input_tokens: 42,
      output_tokens: 18,
      total_tokens: 60,
    });
    expect(result.extraction.extensions?.["synapt/response_binding"]).toMatchObject({
      version: "1",
      provider: "openai",
      response_id: "resp_raw_123",
      response_status: "completed",
      response_model: "gpt-5.5-2026-04-23",
    });
  });

  test("translates Anthropic raw responses into normalized context", async () => {
    const result = await extract(
      SAMPLE_TEXT,
      {
        callLlm: () => ({
          output: STAGE1_FULL,
          raw: {
            type: "message",
            id: "msg_raw_123",
            model: "claude-sonnet-4-20250514",
            stop_reason: "end_turn",
            usage: { input_tokens: 25, output_tokens: 15 },
            content: [{ type: "text", text: "{}" }],
          },
        }),
      },
      {
        profile: "full",
        extend: ({ response }) => ({
          "synapt/response_binding": {
            provider: response.provider,
            response_id: response.id,
            response_status: response.status,
            response_model: response.model,
            stop_reason: response.stop_reason,
          },
        }),
      },
    );

    expect(result.validation.valid).toBe(true);
    expect(result.extraction.produced_by).toMatchObject({
      version: "1",
      model: "anthropic://claude-sonnet-4-20250514",
      model_version: "claude-sonnet-4-20250514",
    });
    expect(result.usage).toMatchObject({
      llm_calls: 1,
      input_tokens: 25,
      output_tokens: 15,
      total_tokens: 40,
    });
    expect(result.extraction.extensions?.["synapt/response_binding"]).toMatchObject({
      version: "1",
      provider: "anthropic",
      response_id: "msg_raw_123",
      response_status: "completed",
      response_model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
    });
  });

  test("uses custom response translators for provider-specific raw shapes", async () => {
    const result = await extract(
      SAMPLE_TEXT,
      {
        callLlm: () => ({
          output: STAGE1_FULL,
          raw: {
            request_id: "local_raw_123",
            engine: "fixture-engine",
            tokens: { input_tokens: 7, output_tokens: 5 },
          },
        }),
      },
      {
        profile: "full",
        responseTranslator: ({ raw }) => ({
          provider: "local",
          id: String(raw?.request_id),
          status: "ok",
          model: String(raw?.engine),
          usage: raw?.tokens as { input_tokens: number; output_tokens: number },
        }),
        extend: ({ response }) => ({
          "synapt/response_binding": {
            provider: response.provider,
            response_id: response.id,
            response_status: response.status,
            response_model: response.model,
          },
        }),
      },
    );

    expect(result.validation.valid).toBe(true);
    expect(result.extraction.produced_by).toMatchObject({
      version: "1",
      model: "local://fixture-engine",
      model_version: "fixture-engine",
    });
    expect(result.usage).toMatchObject({
      input_tokens: 7,
      output_tokens: 5,
      total_tokens: 12,
    });
    expect(result.extraction.extensions?.["synapt/response_binding"]).toMatchObject({
      version: "1",
      provider: "local",
      response_id: "local_raw_123",
      response_status: "ok",
      response_model: "fixture-engine",
    });
  });

  test("runs through the OpenAI-compatible adapter and writes artifacts", async () => {
    const responseBodies: Record<string, unknown>[] = [];
    const embeddingBodies: Record<string, unknown>[] = [];
    const artifactDirectory = mkdtempSync(join(tmpdir(), "synapt-extract-openai-"));

    const client: OpenAICompatibleClient = {
      responses: {
        create: async (body) => {
          responseBodies.push(body);
          return {
            object: "response",
            id: "resp_adapter_123",
            status: "completed",
            model: "gpt-5.5-2026-04-23",
            output_text: JSON.stringify(STAGE1_FULL),
            usage: { input_tokens: 101, output_tokens: 55 },
          };
        },
      },
      embeddings: {
        create: async (body) => {
          embeddingBodies.push(body);
          return {
            object: "list",
            model: "text-embedding-3-small",
            data: [{ embedding: [0.1, 0.2, 0.3] }],
          };
        },
      },
    };

    const result = await extractOpenAI(SAMPLE_TEXT, client, {
      profile: "full",
      source_id: "fixture-openai-adapter",
      source_type: "message",
      kind: "synapt/test",
      model: "gpt-5.5",
      reasoningEffort: "medium",
      maxOutputTokens: 2048,
      textVerbosity: "low",
      embeddingModel: "text-embedding-3-small",
      embeddingDimensions: 3,
      embeddingInputs: ["source"],
      deployment: "test-suite",
      operator: "synapt-dev",
      responseFormatName: "synapt_adapter_stage1",
      artifactDirectory,
    });

    expect(responseBodies).toHaveLength(1);
    expect(responseBodies[0]).toMatchObject({
      model: "gpt-5.5",
      reasoning: { effort: "medium" },
      max_output_tokens: 2048,
    });
    expect(responseBodies[0].text).toMatchObject({
      verbosity: "low",
      format: { type: "json_schema", name: "synapt_adapter_stage1" },
    });
    expect(embeddingBodies).toEqual([
      { model: "text-embedding-3-small", input: SAMPLE_TEXT, dimensions: 3 },
    ]);
    expect(result.validation.valid).toBe(true);
    expect(result.extraction.produced_by).toMatchObject({
      model: "openai://gpt-5.5",
      model_version: "gpt-5.5-2026-04-23",
      deployment: "test-suite",
      operator: "synapt-dev",
      configuration: {
        reasoning_effort: "medium",
        max_tokens: 2048,
        response_format: "synapt_adapter_stage1",
      },
    });
    expect(result.extraction.embeddings?.[0]).toMatchObject({
      input: "source",
      model: "openai://text-embedding-3-small",
      dimensions: 3,
      space: "cosine",
    });
    expect(result.usage).toMatchObject({
      llm_calls: 1,
      embedding_calls: 1,
      input_tokens: 101,
      output_tokens: 55,
      total_tokens: 156,
    });
    expect(result.artifactBundle.source.text).toBe(SAMPLE_TEXT);
    expect(result.artifactBundle.prompt?.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(result.artifactBundle.llm?.request).toMatchObject({
      model: "gpt-5.5",
      max_output_tokens: 2048,
    });
    expect(result.artifactBundle.llm?.response).toMatchObject({
      id: "resp_adapter_123",
      model: "gpt-5.5-2026-04-23",
    });
    expect(existsSync(join(artifactDirectory, "source.txt"))).toBe(true);
    expect(JSON.parse(readFileSync(join(artifactDirectory, "extraction.json"), "utf8"))).toMatchObject({
      source_id: "fixture-openai-adapter",
      produced_by: { model: "openai://gpt-5.5" },
    });
  });
});
