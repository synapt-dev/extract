import { createHash } from "node:crypto";

import { createArtifactBundle, writeArtifactBundle, type ArtifactEmbeddingRun, type ExtractArtifactBundle } from "./artifacts.js";
import {
  extract,
  type EmbeddingRequest,
  type EmbeddingResponse,
  type ExtractOptions,
  type ExtractResult,
  type LlmRequest,
  type LlmResponse,
} from "./extract.js";

export interface OpenAICompatibleClient {
  responses: {
    create: (request: Record<string, unknown>) => Promise<unknown> | unknown;
  };
  embeddings?: {
    create: (request: Record<string, unknown>) => Promise<unknown> | unknown;
  };
}

export interface OpenAIExtractOptions extends ExtractOptions {
  model?: string;
  reasoningEffort?: string;
  maxOutputTokens?: number;
  textVerbosity?: string;
  temperature?: number;
  topP?: number;
  embeddingModel?: string;
  embeddingDimensions?: number;
  embeddingSpace?: string;
  deployment?: string;
  operator?: string;
  responseFormatName?: string;
  artifactDirectory?: string;
  includeArtifactSourceText?: boolean;
}

export interface OpenAIExtractResult extends ExtractResult {
  artifactBundle: ExtractArtifactBundle;
}

export async function extractOpenAI(
  text: string,
  client: OpenAICompatibleClient,
  options: OpenAIExtractOptions = {},
): Promise<OpenAIExtractResult> {
  const model = options.model ?? "gpt-5.5";
  let promptText: string | undefined;
  let responseFormat: Record<string, unknown> | undefined;
  let providerLlmRequest: Record<string, unknown> | undefined;
  let providerLlmResponse: Record<string, unknown> | undefined;
  const embeddingRuns: ArtifactEmbeddingRun[] = [];

  const callbacks = {
    callLlm: async (request: LlmRequest): Promise<LlmResponse> => {
      promptText = request.prompt;
      const promptHash = sha256(request.prompt);
      responseFormat = request.responseFormat;
      const maxTokens = options.maxOutputTokens ?? request.max_tokens;
      const body = pruneUndefined({
        model,
        reasoning: options.reasoningEffort ? { effort: options.reasoningEffort } : undefined,
        max_output_tokens: maxTokens,
        temperature: options.temperature ?? request.temperature,
        top_p: options.topP,
        input: [{ role: "developer", content: request.prompt }],
        text: pruneUndefined({
          format: responseFormat,
          verbosity: options.textVerbosity,
        }),
      });
      providerLlmRequest = body;

      const response = await client.responses.create(body);
      const raw = toJsonObject(response);
      providerLlmResponse = raw;
      const content = outputText(response, raw);
      if (!content || content.trim() === "") {
        throw new Error("OpenAI response did not include output text");
      }

      const llmResponse: LlmResponse = {
        content,
        provider: "openai",
        id: stringFrom(raw.id),
        status: stringFrom(raw.status),
        model: stringFrom(raw.model) ?? model,
        produced_by: {
          model: `openai://${model}`,
          model_version: stringFrom(raw.model) ?? model,
          ...(options.deployment ? { deployment: options.deployment } : {}),
          configuration: pruneUndefined({
            reasoning_effort: options.reasoningEffort,
            system_prompt_hash: promptHash,
            temperature: options.temperature ?? request.temperature,
            top_p: options.topP,
            max_tokens: maxTokens,
            response_format: stringFrom(responseFormat.name),
          }),
          ...(options.operator ? { operator: options.operator } : {}),
        },
        usage: usageFrom(raw.usage),
        raw,
      };
      return llmResponse;
    },
    ...(options.embeddingModel
      ? {
          getEmbedding: async (request: EmbeddingRequest): Promise<EmbeddingResponse> => {
            const body = pruneUndefined({
              model: options.embeddingModel,
              input: request.text,
              dimensions: options.embeddingDimensions,
            });
            const response = await client.embeddings?.create(body);
            if (response === undefined) {
              throw new Error("embeddingModel was set but client.embeddings.create is unavailable");
            }
            const raw = toJsonObject(response);
            const vector = embeddingVector(raw);
            const embeddingResponse: EmbeddingResponse = {
              vector,
              model: `openai://${stringFrom(raw.model) ?? options.embeddingModel}`,
              dimensions: vector.length,
              space: options.embeddingSpace ?? "cosine",
              computed_at: new Date().toISOString(),
              raw,
            };
            embeddingRuns.push({
              input: request.input,
              request: body,
              response: raw,
            });
            return embeddingResponse;
          },
        }
      : {}),
  };

  const runnerOptions: ExtractOptions = {
    ...options,
    ...(options.maxOutputTokens !== undefined && options.max_tokens === undefined
      ? { max_tokens: options.maxOutputTokens }
      : {}),
    ...(options.responseFormatName
      ? { responseFormat: { ...(options.responseFormat ?? {}), name: options.responseFormatName } }
      : {}),
  };

  const result = await extract(text, callbacks, runnerOptions);
  const artifactBundle = createArtifactBundle({
    sourceText: text,
    result,
    ...(promptText ? { prompt: promptText } : {}),
    ...(responseFormat ? { responseFormat } : {}),
    ...(providerLlmRequest ? { llmRequest: providerLlmRequest } : {}),
    ...(providerLlmResponse ? { llmResponse: providerLlmResponse } : {}),
    embeddingRuns,
    includeSourceText: options.includeArtifactSourceText,
  });
  if (options.artifactDirectory) {
    writeArtifactBundle(options.artifactDirectory, artifactBundle);
  }

  return {
    ...result,
    artifactBundle,
  };
}

function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function pruneUndefined<T extends Record<string, unknown>>(value: T): Record<string, unknown> {
  const pruned: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    if (child !== undefined) pruned[key] = child;
  }
  return pruned;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function toJsonObject(value: unknown): Record<string, unknown> {
  if (isRecord(value)) {
    try {
      const serialized = JSON.parse(JSON.stringify(value)) as unknown;
      if (isRecord(serialized)) return serialized;
    } catch {
      return value;
    }
    return value;
  }
  return {};
}

function stringFrom(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function outputText(response: unknown, raw: Record<string, unknown>): string | undefined {
  const direct = stringFrom(raw.output_text) ?? (isRecord(response) ? stringFrom(response.output_text) : undefined);
  if (direct !== undefined) return direct;
  const output = raw.output;
  if (!Array.isArray(output)) return undefined;
  const parts: string[] = [];
  for (const item of output) {
    if (!isRecord(item) || !Array.isArray(item.content)) continue;
    for (const content of item.content) {
      if (isRecord(content) && typeof content.text === "string") {
        parts.push(content.text);
      }
    }
  }
  const joined = parts.join("");
  return joined.length > 0 ? joined : undefined;
}

function usageFrom(value: unknown): LlmResponse["usage"] | undefined {
  if (!isRecord(value)) return undefined;
  const usage: NonNullable<LlmResponse["usage"]> = { ...value };
  const inputTokens = typeof value.input_tokens === "number" ? value.input_tokens : undefined;
  const outputTokens = typeof value.output_tokens === "number" ? value.output_tokens : undefined;
  if (inputTokens !== undefined) usage.input_tokens = inputTokens;
  if (outputTokens !== undefined) usage.output_tokens = outputTokens;
  if (typeof value.total_tokens === "number") {
    usage.total_tokens = value.total_tokens;
  } else if (inputTokens !== undefined && outputTokens !== undefined) {
    usage.total_tokens = inputTokens + outputTokens;
  }
  return usage;
}

function embeddingVector(raw: Record<string, unknown>): number[] {
  const data = raw.data;
  if (!Array.isArray(data) || data.length === 0 || !isRecord(data[0]) || !Array.isArray(data[0].embedding)) {
    throw new Error("OpenAI embedding response did not include data[0].embedding");
  }
  const vector = data[0].embedding;
  if (!vector.every((value) => typeof value === "number")) {
    throw new Error("OpenAI embedding vector must contain only numbers");
  }
  return vector;
}
