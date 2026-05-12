import { ExtractionBuilder, type ExtractionBuilderOptions, type JsonSchema, type ResponseFormatOptions } from "./builder.js";
import type { FinalizeContext, FinalizeResult } from "./finalize.js";
import { capabilityEmbeddingPreference, capabilityName, type CapabilityInput } from "./prompt.js";
import type { ExtractionCapability, SynaptEmbedding } from "./schema.js";
import type { ValidationResult } from "./validate.js";

export interface LlmMessage {
  role: "system" | "user";
  content: string;
}

export interface LlmUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  [key: string]: unknown;
}

export type LlmResponseProvider = "openai" | "anthropic" | (string & {});

export interface LlmRequest {
  prompt: string;
  messages: LlmMessage[];
  capabilities: ExtractionCapability[];
  schema: JsonSchema;
  responseFormat: JsonSchema;
  temperature?: number;
  max_tokens?: number;
}

export interface LlmResponse {
  content?: string;
  json?: Record<string, unknown>;
  output?: Record<string, unknown>;
  produced_by?: FinalizeContext["produced_by"];
  provider?: LlmResponseProvider;
  id?: string;
  response_id?: string;
  status?: string;
  model?: string;
  model_version?: string;
  usage?: LlmUsage;
  raw?: unknown;
}

export interface NormalizedLlmResponse {
  provider?: LlmResponseProvider;
  id?: string;
  status?: string;
  model?: string;
  model_version?: string;
  stop_reason?: string;
  produced_by?: FinalizeContext["produced_by"];
  content?: string;
  usage?: LlmUsage;
  raw?: unknown;
}

export type NamedEmbeddingInput =
  | "source"
  | "summary"
  | "entities"
  | "goals"
  | "themes"
  | "keywords"
  | "facts"
  | "questions"
  | "actions"
  | "decisions"
  | "temporal_refs"
  | "sentiment";

export interface CustomEmbeddingInput {
  input: string;
  text: string;
}

export type EmbeddingInputSelector = NamedEmbeddingInput | CustomEmbeddingInput;
export type EmbeddingInputSelection = "all" | EmbeddingInputSelector[];

export interface EmbeddingRequest {
  text: string;
  input: string;
}

export interface EmbeddingResponse {
  vector: number[];
  model: string;
  dimensions?: number;
  space?: string;
  computed_at?: string;
  raw?: unknown;
}

export interface LogEntry {
  level: "debug" | "info" | "warn" | "error";
  stage: "prompt_build" | "llm_call" | "parse" | "embed" | "finalize";
  message: string;
  data?: Record<string, unknown>;
}

export interface ExtractCallbacks {
  callLlm: (request: LlmRequest) => Promise<LlmResponse> | LlmResponse;
  getEmbedding?: (request: EmbeddingRequest) => Promise<EmbeddingResponse> | EmbeddingResponse;
  log?: (entry: LogEntry) => void;
}

export interface UsageSummary {
  llm_calls: number;
  embedding_calls: number;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
}

export interface ExtensionResolverContext {
  sourceText: string;
  capabilities: ExtractionCapability[];
  prompt: string;
  schema: JsonSchema;
  responseFormat: JsonSchema;
  llmRequest: LlmRequest;
  response: NormalizedLlmResponse;
  llmResponse: NormalizedLlmResponse;
  stage1: Record<string, unknown>;
  embeddings: Omit<SynaptEmbedding, "version">[];
  usage: UsageSummary;
  warnings: string[];
}

export type ExtensionResolver = (
  context: ExtensionResolverContext,
) => Record<string, unknown> | Promise<Record<string, unknown>>;

export interface LlmResponseTranslatorContext {
  response: LlmResponse;
  raw?: Record<string, unknown>;
  provider?: LlmResponseProvider;
}

export type LlmResponseTranslator = (
  context: LlmResponseTranslatorContext,
) => Partial<NormalizedLlmResponse> | undefined;

export interface ExtractOptions extends ExtractionBuilderOptions {
  responseFormat?: ResponseFormatOptions;
  embeddingInputs?: EmbeddingInputSelection;
  responseTranslator?: LlmResponseTranslator;
  responseTranslators?: LlmResponseTranslator[];
  extend?: ExtensionResolver;
  extensionResolver?: ExtensionResolver;
  extensionErrors?: "throw" | "warn";
  temperature?: number;
  max_tokens?: number;
}

export interface ExtractResult {
  extraction: FinalizeResult["extraction"];
  validation: ValidationResult;
  warnings: string[];
  stage1: Record<string, unknown>;
  embeddings: Omit<SynaptEmbedding, "version">[];
  usage: UsageSummary;
}

const SYSTEM_MESSAGE = "You are a deterministic information extraction engine. Return only JSON matching the supplied schema.";
const STANDARD_EMBEDDING_INPUTS: NamedEmbeddingInput[] = [
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
];
const CAPABILITY_EMBEDDING_INPUTS: Partial<Record<ExtractionCapability, NamedEmbeddingInput>> = {
  entities: "entities",
  entity_state: "entities",
  entity_context: "entities",
  entity_ids: "entities",
  relations: "entities",
  relation_origin: "entities",
  goals: "goals",
  goal_timing: "goals",
  goal_entity_refs: "goals",
  themes: "themes",
  keywords: "keywords",
  summary: "summary",
  sentiment: "sentiment",
  structured_sentiment: "sentiment",
  facts: "facts",
  questions: "questions",
  actions: "actions",
  decisions: "decisions",
  temporal_refs: "temporal_refs",
  temporal_classes: "temporal_refs",
};

function safeLog(callbacks: ExtractCallbacks, entry: LogEntry): void {
  try {
    callbacks.log?.(entry);
  } catch {
    // Logging must not affect extraction behavior.
  }
}

function parseLlmOutput(response: LlmResponse): Record<string, unknown> {
  if (response.output !== undefined) return response.output;
  if (response.json !== undefined) return response.json;
  if (typeof response.content !== "string") {
    throw new Error("LLM response must include output, json, or JSON string content");
  }
  try {
    const parsed = JSON.parse(response.content) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("parsed JSON must be an object");
    }
    return parsed as Record<string, unknown>;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to parse LLM JSON response: ${message}`);
  }
}

function modelUriFromResponse(response: LlmResponse, normalized?: NormalizedLlmResponse): FinalizeContext["produced_by"] | undefined {
  if (response.produced_by !== undefined) return response.produced_by;
  if (normalized?.produced_by !== undefined) return normalized.produced_by;

  const model = response.model ?? normalized?.model;
  if (typeof model !== "string" || model.length === 0) return undefined;
  const modelVersion = response.model_version ?? normalized?.model_version ?? model;

  if (model.includes("://")) {
    return {
      model,
      ...(modelVersion !== undefined ? { model_version: modelVersion } : {}),
    };
  }

  const provider = response.provider ?? normalized?.provider;
  if (typeof provider === "string" && provider.length > 0) {
    return {
      model: `${provider}://${model}`,
      model_version: modelVersion,
    };
  }

  return undefined;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function normalizeUsage(value: unknown): LlmUsage | undefined {
  const obj = asRecord(value);
  if (!obj) return undefined;
  const usage: LlmUsage = { ...obj };
  const inputTokens = typeof obj.input_tokens === "number" ? obj.input_tokens : undefined;
  const outputTokens = typeof obj.output_tokens === "number" ? obj.output_tokens : undefined;
  if (inputTokens !== undefined) usage.input_tokens = inputTokens;
  if (outputTokens !== undefined) usage.output_tokens = outputTokens;
  if (typeof obj.total_tokens === "number") {
    usage.total_tokens = obj.total_tokens;
  } else if (inputTokens !== undefined && outputTokens !== undefined) {
    usage.total_tokens = inputTokens + outputTokens;
  }
  return Object.keys(usage).length > 0 ? usage : undefined;
}

function textFromAnthropicContent(raw: Record<string, unknown>): string | undefined {
  const content = raw.content;
  if (!Array.isArray(content)) return undefined;
  const text = content
    .map((item) => asRecord(item))
    .filter((item): item is Record<string, unknown> => item !== undefined)
    .filter((item) => item.type === "text" && typeof item.text === "string")
    .map((item) => item.text as string)
    .join("");
  return text.length > 0 ? text : undefined;
}

function translateOpenAiResponse({ raw, provider }: LlmResponseTranslatorContext): Partial<NormalizedLlmResponse> | undefined {
  if (!raw) return undefined;
  const object = optionalString(raw.object);
  const id = optionalString(raw.id);
  if (provider !== "openai" && object !== "response" && !id?.startsWith("resp_")) {
    return undefined;
  }
  return {
    provider: "openai",
    id,
    status: optionalString(raw.status),
    model: optionalString(raw.model),
    model_version: optionalString(raw.model),
    content: optionalString(raw.output_text),
    usage: normalizeUsage(raw.usage),
  };
}

function translateAnthropicResponse({ raw, provider }: LlmResponseTranslatorContext): Partial<NormalizedLlmResponse> | undefined {
  if (!raw) return undefined;
  const type = optionalString(raw.type);
  const id = optionalString(raw.id);
  if (provider !== "anthropic" && type !== "message" && !id?.startsWith("msg_")) {
    return undefined;
  }
  return {
    provider: "anthropic",
    id,
    status: optionalString(raw.status) ?? "completed",
    model: optionalString(raw.model),
    model_version: optionalString(raw.model),
    stop_reason: optionalString(raw.stop_reason),
    content: textFromAnthropicContent(raw),
    usage: normalizeUsage(raw.usage),
  };
}

function commonRawResponseTranslation(raw: Record<string, unknown> | undefined): Partial<NormalizedLlmResponse> {
  return {
    id: optionalString(raw?.id),
    status: optionalString(raw?.status),
    model: optionalString(raw?.model),
    usage: normalizeUsage(raw?.usage),
  };
}

function mergeNormalized(...parts: (Partial<NormalizedLlmResponse> | undefined)[]): NormalizedLlmResponse {
  const merged: NormalizedLlmResponse = {};
  for (const part of parts) {
    if (!part) continue;
    for (const [key, value] of Object.entries(part) as [keyof NormalizedLlmResponse, unknown][]) {
      if (value !== undefined) {
        (merged as Record<string, unknown>)[key] = key === "usage" ? normalizeUsage(value) : value;
      }
    }
  }
  return merged;
}

function runResponseTranslator(
  translator: LlmResponseTranslator,
  context: LlmResponseTranslatorContext,
): Partial<NormalizedLlmResponse> | undefined {
  const translated = translator(context);
  if (translated === undefined) return undefined;
  if (!translated || typeof translated !== "object" || Array.isArray(translated)) {
    throw new Error("response translator must return an object or undefined");
  }
  return translated;
}

export function normalizeLlmResponse(
  response: LlmResponse,
  translators: LlmResponseTranslator[] = [],
): NormalizedLlmResponse {
  const raw = response.raw;
  const rawObject = asRecord(raw);
  const provider = response.provider;
  const translatorContext: LlmResponseTranslatorContext = { response, raw: rawObject, provider };
  const translated = [
    translateOpenAiResponse(translatorContext),
    translateAnthropicResponse(translatorContext),
    ...translators.map((translator) => runResponseTranslator(translator, translatorContext)),
  ];

  return mergeNormalized(
    commonRawResponseTranslation(rawObject),
    ...translated,
    {
      provider: response.provider,
      id: response.id ?? response.response_id,
      status: response.status,
      model: response.model,
      usage: response.usage,
    },
    {
      model_version: response.model_version,
      produced_by: response.produced_by,
      content: response.content,
      raw,
    },
  );
}

function asArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item)) as Record<string, unknown>[] : [];
}

function compactJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function textForNamedEmbedding(input: NamedEmbeddingInput, sourceText: string, stage1: Record<string, unknown>): string | undefined {
  switch (input) {
    case "source":
      return sourceText;
    case "summary":
      return typeof stage1.summary === "string" && stage1.summary.trim() ? stage1.summary : undefined;
    case "entities": {
      const entities = asArray(stage1.entities);
      if (entities.length === 0) return undefined;
      return entities.map((entity) => {
        const id = typeof entity.id === "string" ? `${entity.id}: ` : "";
        const type = typeof entity.type === "string" ? ` (${entity.type})` : "";
        const state = typeof entity.state === "string" ? ` - ${entity.state}` : "";
        const context = typeof entity.context === "string" ? ` ${entity.context}` : "";
        return `${id}${String(entity.name ?? "unknown")}${type}${state}${context}`.trim();
      }).join("\n");
    }
    case "goals": {
      const goals = asArray(stage1.goals);
      return goals.length ? goals.map((goal) => String(goal.text ?? "")).filter(Boolean).join("\n") : undefined;
    }
    case "themes":
    case "keywords": {
      const values = stage1[input];
      return Array.isArray(values) && values.length ? values.map(String).join(", ") : undefined;
    }
    case "facts":
    case "questions":
    case "actions":
    case "decisions": {
      const items = asArray(stage1[input]);
      return items.length ? items.map((item) => String(item.text ?? "")).filter(Boolean).join("\n") : undefined;
    }
    case "temporal_refs": {
      const refs = asArray(stage1.temporal_refs);
      return refs.length ? refs.map((ref) => compactJson(ref)).join("\n") : undefined;
    }
    case "sentiment":
      return stage1.sentiment !== undefined ? compactJson(stage1.sentiment) : undefined;
  }
}

function resolveEmbeddingInputs(
  selection: EmbeddingInputSelection | undefined,
  sourceText: string,
  stage1: Record<string, unknown>,
  warnings: string[],
): EmbeddingRequest[] {
  const selectors = selection === "all" ? STANDARD_EMBEDDING_INPUTS : selection ?? [];
  const requests: EmbeddingRequest[] = [];

  for (const selector of selectors) {
    if (typeof selector === "string") {
      const text = textForNamedEmbedding(selector, sourceText, stage1);
      if (text === undefined || text.trim() === "") {
        warnings.push(`embedding input "${selector}" was requested but no text was available; skipped`);
        continue;
      }
      requests.push({ input: selector, text });
    } else if (selector.text.trim() !== "") {
      requests.push({ input: selector.input, text: selector.text });
    } else {
      warnings.push(`embedding input "${selector.input}" was empty; skipped`);
    }
  }

  return requests;
}

function deriveEmbeddingSelectionFromCapabilityInputs(capabilities: CapabilityInput[] | undefined, resolved: Set<ExtractionCapability>): NamedEmbeddingInput[] {
  if (!capabilities) return [];
  const selected: NamedEmbeddingInput[] = [];
  for (const capability of capabilities) {
    if (capabilityEmbeddingPreference(capability) !== true) continue;
    const name = capabilityName(capability);
    if (!resolved.has(name)) continue;
    const input = CAPABILITY_EMBEDDING_INPUTS[name];
    if (input !== undefined && !selected.includes(input)) {
      selected.push(input);
    }
  }
  return selected;
}

function applyEmbeddingOverrides(inputs: EmbeddingInputSelector[], overrides: Partial<Record<ExtractionCapability, boolean>> | undefined): EmbeddingInputSelector[] {
  if (!overrides) return inputs;
  const selected = [...inputs];
  for (const [capability, enabled] of Object.entries(overrides) as [ExtractionCapability, boolean][]) {
    const input = CAPABILITY_EMBEDDING_INPUTS[capability];
    if (input === undefined) continue;
    const index = selected.indexOf(input);
    if (enabled && index === -1) selected.push(input);
    if (!enabled && index !== -1) selected.splice(index, 1);
  }
  return selected;
}

function embeddingSelectionForOptions(options: ExtractOptions, capabilities: ExtractionCapability[]): EmbeddingInputSelection | undefined {
  const resolved = new Set(capabilities);
  const selected: EmbeddingInputSelector[] = [
    ...deriveEmbeddingSelectionFromCapabilityInputs(options.capabilities, resolved),
    ...deriveEmbeddingSelectionFromCapabilityInputs(options.add, resolved),
  ];

  if (options.embeddingInputs === "all") return "all";
  if (Array.isArray(options.embeddingInputs)) {
    selected.unshift(...options.embeddingInputs);
  }

  const unique = applyEmbeddingOverrides(selected, options.embed).filter((input, index, arr) => arr.indexOf(input) === index);
  return unique.length > 0 ? unique : undefined;
}

function buildFinalizeContext(
  options: ExtractOptions,
  capabilities: ExtractionCapability[],
  llmResponse: LlmResponse,
  normalizedResponse: NormalizedLlmResponse,
  embeddings: Omit<SynaptEmbedding, "version">[],
  dynamicExtensions?: Record<string, unknown>,
): FinalizeContext {
  const producedBy = options.produced_by ?? modelUriFromResponse(llmResponse, normalizedResponse);
  if (producedBy === undefined) {
    throw new Error("Cannot finalize without produced_by; pass options.produced_by or return produced_by/model URI from callLlm()");
  }

  const context: FinalizeContext = {
    produced_by: producedBy,
    capabilities_hint: options.capabilities_hint ?? capabilities,
  };
  if (options.user_id !== undefined) context.user_id = options.user_id;
  if (options.source_id !== undefined) context.source_id = options.source_id;
  if (options.source_type !== undefined) context.source_type = options.source_type;
  if (options.kind !== undefined) context.kind = options.kind;
  const extensions = { ...(options.extensions ?? {}), ...(dynamicExtensions ?? {}) };
  if (Object.keys(extensions).length > 0) context.extensions = extensions;

  const manualEmbeddings = options.embeddings ?? [];
  const allEmbeddings = [...manualEmbeddings, ...embeddings];
  if (allEmbeddings.length > 0) context.embeddings = allEmbeddings;

  return context;
}

async function resolveDynamicExtensions(
  options: ExtractOptions,
  context: ExtensionResolverContext,
): Promise<Record<string, unknown> | undefined> {
  const resolver = options.extend ?? options.extensionResolver;
  if (!resolver) return undefined;
  try {
    return await resolver(context);
  } catch (err) {
    if (options.extensionErrors === "warn") {
      const message = err instanceof Error ? err.message : String(err);
      context.warnings.push(`extension resolver failed: ${message}`);
      return undefined;
    }
    throw err;
  }
}

export async function extract(
  text: string,
  callbacks: ExtractCallbacks,
  options: ExtractOptions = {},
): Promise<ExtractResult> {
  const builder = new ExtractionBuilder(text, options);
  const responseFormatOptions = options.responseFormat ?? {};
  const built = builder.build(responseFormatOptions);
  const usage: UsageSummary = { llm_calls: 0, embedding_calls: 0 };
  const warnings: string[] = [];

  safeLog(callbacks, {
    level: "info",
    stage: "prompt_build",
    message: "Built extraction prompt and response schema",
    data: { capabilities: built.capabilities },
  });

  const llmRequest: LlmRequest = {
    prompt: built.prompt,
    messages: [
      { role: "system", content: SYSTEM_MESSAGE },
      { role: "user", content: built.prompt },
    ],
    capabilities: built.capabilities,
    schema: built.schema,
    responseFormat: built.responseFormat,
    ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
    ...(options.max_tokens !== undefined ? { max_tokens: options.max_tokens } : {}),
  };

  safeLog(callbacks, {
    level: "info",
    stage: "llm_call",
    message: "Calling LLM extraction callback",
    data: { capabilities: built.capabilities },
  });
  usage.llm_calls += 1;
  const llmResponse = await callbacks.callLlm(llmRequest);
  const responseTranslators = [
    ...(options.responseTranslator ? [options.responseTranslator] : []),
    ...(options.responseTranslators ?? []),
  ];
  const normalizedResponse = normalizeLlmResponse(llmResponse, responseTranslators);
  if (normalizedResponse.usage?.input_tokens !== undefined) usage.input_tokens = normalizedResponse.usage.input_tokens;
  if (normalizedResponse.usage?.output_tokens !== undefined) usage.output_tokens = normalizedResponse.usage.output_tokens;
  if (normalizedResponse.usage?.total_tokens !== undefined) usage.total_tokens = normalizedResponse.usage.total_tokens;

  const stage1 = parseLlmOutput(llmResponse);
  safeLog(callbacks, {
    level: "info",
    stage: "parse",
    message: "Parsed LLM extraction JSON",
    data: { fields: Object.keys(stage1) },
  });

  const embeddingRequests = resolveEmbeddingInputs(embeddingSelectionForOptions(options, built.capabilities), text, stage1, warnings);
  if (embeddingRequests.length > 0 && callbacks.getEmbedding === undefined) {
    throw new Error("embeddingInputs requested but getEmbedding callback was not provided");
  }

  const embeddings: Omit<SynaptEmbedding, "version">[] = [];
  for (const request of embeddingRequests) {
    safeLog(callbacks, {
      level: "info",
      stage: "embed",
      message: "Calling embedding callback",
      data: { input: request.input },
    });
    usage.embedding_calls += 1;
    const response = await callbacks.getEmbedding!(request);
    embeddings.push({
      vector: response.vector,
      model: response.model,
      input: request.input,
      dimensions: response.dimensions ?? response.vector.length,
      ...(response.space !== undefined ? { space: response.space } : {}),
      ...(response.computed_at !== undefined ? { computed_at: response.computed_at } : {}),
    });
  }

  const dynamicExtensions = await resolveDynamicExtensions(options, {
    sourceText: text,
    capabilities: built.capabilities,
    prompt: built.prompt,
    schema: built.schema,
    responseFormat: built.responseFormat,
    llmRequest,
    response: normalizedResponse,
    llmResponse: normalizedResponse,
    stage1,
    embeddings,
    usage,
    warnings,
  });

  const context = buildFinalizeContext(options, built.capabilities, llmResponse, normalizedResponse, embeddings, dynamicExtensions);
  const finalized = builder.withFinalizeContext(context).finalize(stage1);
  safeLog(callbacks, {
    level: finalized.validation.valid ? "info" : "warn",
    stage: "finalize",
    message: "Finalized extraction",
    data: { valid: finalized.validation.valid, errors: finalized.validation.errors.length },
  });

  return {
    extraction: finalized.extraction,
    validation: finalized.validation,
    warnings: [...warnings, ...finalized.warnings],
    stage1,
    embeddings,
    usage,
  };
}

export const runExtraction = extract;
