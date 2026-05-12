import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type {
  EmbeddingRequest,
  EmbeddingResponse,
  ExtractResult,
  LlmRequest,
  LlmResponse,
} from "./extract.js";

export interface ArtifactSource {
  text?: string;
  sha256: string;
  characters: number;
}

export interface ArtifactPrompt {
  text: string;
  sha256: string;
  responseFormat?: Record<string, unknown>;
}

export interface ArtifactLlmRun {
  request?: Record<string, unknown> | LlmRequest;
  response?: LlmResponse | Record<string, unknown>;
}

export interface ArtifactEmbeddingRun {
  input: string;
  request: EmbeddingRequest | Record<string, unknown>;
  response: EmbeddingResponse | Record<string, unknown>;
}

export interface ExtractArtifactBundle {
  version: "1";
  createdAt: string;
  source: ArtifactSource;
  prompt?: ArtifactPrompt;
  llm?: ArtifactLlmRun;
  embeddings: ArtifactEmbeddingRun[];
  stage1: Record<string, unknown>;
  extraction: ExtractResult["extraction"];
  validation: ExtractResult["validation"];
  warnings: string[];
  usage: ExtractResult["usage"];
}

export interface CreateArtifactBundleOptions {
  sourceText: string;
  result: ExtractResult;
  prompt?: string;
  responseFormat?: Record<string, unknown>;
  llmRequest?: Record<string, unknown> | LlmRequest;
  llmResponse?: LlmResponse | Record<string, unknown>;
  embeddingRuns?: ArtifactEmbeddingRun[];
  includeSourceText?: boolean;
  createdAt?: string;
}

export interface WriteArtifactBundleOptions {
  prefix?: string;
}

export function sha256Text(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

export function createArtifactBundle(options: CreateArtifactBundleOptions): ExtractArtifactBundle {
  const requestRecord = options.llmRequest as Record<string, unknown> | undefined;
  const responseFormat = options.responseFormat ?? (isRecord(requestRecord?.responseFormat)
    ? requestRecord.responseFormat
    : isRecord(requestRecord?.response_format)
      ? requestRecord.response_format
      : undefined);
  const promptText = options.prompt ?? (options.llmRequest && typeof options.llmRequest.prompt === "string"
    ? options.llmRequest.prompt
    : undefined);
  const prompt = promptText
    ? {
        text: promptText,
        sha256: sha256Text(promptText),
        ...(responseFormat ? { responseFormat } : {}),
      }
    : undefined;

  return {
    version: "1",
    createdAt: options.createdAt ?? new Date().toISOString(),
    source: {
      ...(options.includeSourceText === false ? {} : { text: options.sourceText }),
      sha256: sha256Text(options.sourceText),
      characters: options.sourceText.length,
    },
    ...(prompt ? { prompt } : {}),
    ...(options.llmRequest || options.llmResponse
      ? {
          llm: {
            ...(options.llmRequest ? { request: options.llmRequest } : {}),
            ...(options.llmResponse ? { response: options.llmResponse } : {}),
          },
        }
      : {}),
    embeddings: options.embeddingRuns ?? [],
    stage1: options.result.stage1,
    extraction: options.result.extraction,
    validation: options.result.validation,
    warnings: options.result.warnings,
    usage: options.result.usage,
  };
}

export function writeArtifactBundle(
  directory: string,
  bundle: ExtractArtifactBundle,
  options: WriteArtifactBundleOptions = {},
): Record<string, string> {
  mkdirSync(directory, { recursive: true });
  const written: Record<string, string> = {};
  const prefix = options.prefix ? `${options.prefix}.` : "";

  const writeJson = (name: string, value: unknown): void => {
    const path = join(directory, `${prefix}${name}`);
    writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
    written[name] = path;
  };
  const writeText = (name: string, value: string): void => {
    const path = join(directory, `${prefix}${name}`);
    writeFileSync(path, value, "utf8");
    written[name] = path;
  };

  writeJson("bundle.json", bundle);
  if (bundle.source.text !== undefined) writeText("source.txt", bundle.source.text);
  if (bundle.prompt?.text !== undefined) writeText("prompt.md", bundle.prompt.text);
  if (bundle.llm?.request !== undefined) writeJson("llm-request.json", bundle.llm.request);
  if (bundle.llm?.response !== undefined) writeJson("llm-response.json", bundle.llm.response);
  if (bundle.embeddings.length > 0) writeJson("embedding-runs.json", bundle.embeddings);
  writeJson("stage1.json", bundle.stage1);
  writeJson("extraction.json", bundle.extraction);
  writeJson("validation.json", {
    validation: bundle.validation,
    warnings: bundle.warnings,
    usage: bundle.usage,
  });

  return written;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
