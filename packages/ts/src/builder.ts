import { finalizeExtraction, type FinalizeContext, type FinalizeResult } from "./finalize.js";
import type { ExtractionCapability } from "./schema.js";
import {
  buildExtractionPrompt,
  CANONICAL_ORDER,
  STANDARD_EMBEDDING_INPUTS,
  capabilityEmbeddingInput,
  capabilityEmbeddingPreference,
  capabilityName,
  expandCapabilityExclusions,
  normalizeCapabilityInputs,
  profileCapabilities,
  resolveCapabilities,
  type CapabilityInput,
  type PromptOptions,
} from "./prompt.js";

export type JsonSchema = Record<string, unknown>;
type CapabilityOptions = Pick<PromptOptions, "capabilities" | "profile" | "add" | "remove">;
export type ExtractionBuilderOptions = PromptOptions & Partial<FinalizeContext> & ExtractionBuilderRuntimeOptions;

export interface ExtractionBuilderResult {
  capabilities: ExtractionCapability[];
  prompt: string;
  schema: JsonSchema;
  responseFormat: JsonSchema;
  finalizedSchema: JsonSchema;
  finalizeContext?: FinalizeContext;
}

export interface ResponseFormatOptions {
  name?: string;
  strict?: boolean;
}

export type EmbeddableInput =
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

export type CapabilityProfileOptions = {
  embed?: boolean;
};

export interface CustomBuilderEmbeddingInput {
  input: string;
  text: string;
}

export type BuilderEmbeddingInputSelector = EmbeddableInput | CustomBuilderEmbeddingInput;
export type BuilderEmbeddingInputSelection = "all" | BuilderEmbeddingInputSelector[];

export interface ExtractionBuilderRuntimeOptions {
  embeddingInputs?: BuilderEmbeddingInputSelection;
}

export interface CapabilityPlan {
  capabilities: ExtractionCapability[];
  excluded: ExtractionCapability[];
  embeddedInputs: string[];
  notEmbedded: ExtractionCapability[];
  requiredCallbacks: {
    callLlm: true;
    getEmbedding: boolean;
  };
  promptCharacters: number;
}

const DEFAULT_RESPONSE_FORMAT_NAME = "synapt_extraction_stage1";
const ALL_CAPABILITIES: ExtractionCapability[] = [...CANONICAL_ORDER];
const STANDARD_BUILDER_EMBEDDING_INPUTS = STANDARD_EMBEDDING_INPUTS as EmbeddableInput[];

function embeddableInputForCapability(capability: ExtractionCapability): EmbeddableInput | undefined {
  return capabilityEmbeddingInput(capability) as EmbeddableInput | undefined;
}

function capabilitySpecs(capabilities: ExtractionCapability[], embed?: boolean): CapabilityInput[] {
  if (embed === undefined) return capabilities;
  return capabilities.map((name) => ({ name, embed }));
}

function embeddingInputsFromCapabilities(capabilities: CapabilityInput[] | undefined, resolved: Set<ExtractionCapability>): EmbeddableInput[] {
  if (!capabilities) return [];
  const selected: EmbeddableInput[] = [];
  for (const capability of capabilities) {
    if (capabilityEmbeddingPreference(capability) !== true) continue;
    const name = capabilityName(capability);
    if (!resolved.has(name)) continue;
    const input = embeddableInputForCapability(name);
    if (input !== undefined && !selected.includes(input)) selected.push(input);
  }
  return selected;
}

function applyEmbeddingOverrides(inputs: string[], overrides: Partial<Record<ExtractionCapability, boolean>> | undefined): string[] {
  if (!overrides) return inputs;
  const selected = [...inputs];
  for (const [capability, enabled] of Object.entries(overrides) as [ExtractionCapability, boolean][]) {
    const input = embeddableInputForCapability(capability);
    if (input === undefined) continue;
    const index = selected.indexOf(input);
    if (enabled && index === -1) selected.push(input);
    if (!enabled && index !== -1) selected.splice(index, 1);
  }
  return selected;
}

function embeddingInputsFromSelection(selection: BuilderEmbeddingInputSelection | undefined): string[] {
  if (selection === undefined) return [];
  if (selection === "all") return [...STANDARD_BUILDER_EMBEDDING_INPUTS];
  return selection.map((input) => typeof input === "string" ? input : input.input);
}

function uniqueStrings(values: string[]): string[] {
  return values.filter((value, index) => values.indexOf(value) === index);
}

function orderEmbeddingInputs(values: string[]): string[] {
  const standard = STANDARD_BUILDER_EMBEDDING_INPUTS.filter((input) => values.includes(input));
  const custom = values.filter((input) => !STANDARD_BUILDER_EMBEDDING_INPUTS.includes(input as EmbeddableInput));
  return [...standard, ...custom];
}

function sourceRefSchema(finalized = false): JsonSchema {
  const properties: JsonSchema = {
    snippet: { type: "string" },
    offset_start: { type: "integer", minimum: 0 },
    offset_end: { type: "integer", minimum: 0 },
    sentence_index: { type: "integer", minimum: 0 },
  };
  const required = finalized ? ["version"] : ["snippet"];
  if (finalized) {
    properties.version = { const: "1" };
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
    ...(finalized ? { minProperties: 2 } : {}),
  };
}

function signalsSchema(finalized = false): JsonSchema {
  const properties: JsonSchema = {
    confidence: { type: "number", minimum: 0, maximum: 1 },
    negated: { type: "boolean" },
    hedged: { type: "boolean" },
    condition: { type: "string" },
  };
  if (finalized) {
    properties.version = { const: "1" };
  }

  const schema: JsonSchema = {
    type: "object",
    additionalProperties: false,
    properties,
    required: finalized ? ["version"] : [],
  };
  if (finalized) {
    schema.minProperties = 2;
  }
  return schema;
}

function relationSchema(capabilities: Set<ExtractionCapability>, finalized = false): JsonSchema {
  const properties: JsonSchema = {
    target: { type: "string" },
    type: { type: "string" },
  };
  const required = ["target", "type"];

  if (capabilities.has("relation_origin")) {
    properties.origin = { type: "string", enum: ["explicit", "inferred", "dependent"] };
  }
  if (capabilities.has("assertion_signals")) {
    properties.signals = signalsSchema(finalized);
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function entitySchema(capabilities: Set<ExtractionCapability>, finalized = false): JsonSchema {
  const properties: JsonSchema = {
    name: { type: "string" },
    type: { type: "string" },
    aliases: {
      type: "array",
      items: { type: "string" },
    },
  };
  const required = ["name", "type"];

  if (capabilities.has("entity_ids")) {
    properties.id = { type: "string" };
    required.push("id");
  }
  if (capabilities.has("entity_state")) {
    properties.state = { type: "string" };
  }
  if (capabilities.has("entity_context")) {
    properties.context = { type: "string" };
    properties.date_hint = { type: "string" };
  }
  if (capabilities.has("relations")) {
    properties.relations = {
      type: "array",
      items: relationSchema(capabilities, finalized),
    };
  }
  if (capabilities.has("assertion_signals")) {
    properties.signals = signalsSchema(finalized);
  }
  if (capabilities.has("evidence_anchoring")) {
    properties.source = sourceRefSchema(finalized);
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function goalSchema(capabilities: Set<ExtractionCapability>, finalized = false): JsonSchema {
  const properties: JsonSchema = {
    text: { type: "string" },
    status: { type: "string", enum: ["open", "resolved", "abandoned", "in_progress"] },
    entity_refs: {
      type: "array",
      items: { type: "string" },
    },
  };
  const required = ["text", "status", "entity_refs"];

  if (capabilities.has("goal_timing")) {
    properties.stated_at = { type: "string" };
    properties.resolved_at = { type: "string" };
  }
  if (capabilities.has("assertion_signals")) {
    properties.signals = signalsSchema(finalized);
  }
  if (capabilities.has("evidence_anchoring")) {
    properties.source = sourceRefSchema(finalized);
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function factSchema(capabilities: Set<ExtractionCapability>, finalized = false): JsonSchema {
  const properties: JsonSchema = {
    text: { type: "string" },
    category: { type: "string" },
  };
  const required = ["text"];

  if (capabilities.has("assertion_signals")) {
    properties.signals = signalsSchema(finalized);
  }
  if (capabilities.has("evidence_anchoring")) {
    properties.source = sourceRefSchema(finalized);
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function questionSchema(capabilities: Set<ExtractionCapability>, finalized = false): JsonSchema {
  const properties: JsonSchema = {
    text: { type: "string" },
    directed_to: { type: "string" },
  };
  const required = ["text"];

  if (capabilities.has("assertion_signals")) {
    properties.signals = signalsSchema(finalized);
  }
  if (capabilities.has("evidence_anchoring")) {
    properties.source = sourceRefSchema(finalized);
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function actionSchema(capabilities: Set<ExtractionCapability>, finalized = false): JsonSchema {
  const properties: JsonSchema = {
    text: { type: "string" },
    origin: { type: "string", enum: ["extracted", "proposed_from_goals"] },
    entity_refs: {
      type: "array",
      items: { type: "string" },
    },
    due: { type: "string" },
  };
  const required = ["text", "origin"];

  if (capabilities.has("assertion_signals")) {
    properties.signals = signalsSchema(finalized);
  }
  if (capabilities.has("evidence_anchoring")) {
    properties.source = sourceRefSchema(finalized);
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function decisionSchema(capabilities: Set<ExtractionCapability>, finalized = false): JsonSchema {
  const properties: JsonSchema = {
    text: { type: "string" },
    entity_refs: {
      type: "array",
      items: { type: "string" },
    },
    decided_at: { type: "string" },
  };
  const required = ["text"];

  if (capabilities.has("assertion_signals")) {
    properties.signals = signalsSchema(finalized);
  }
  if (capabilities.has("evidence_anchoring")) {
    properties.source = sourceRefSchema(finalized);
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function sentimentSchema(finalized = false): JsonSchema {
  const properties: JsonSchema = {
    valence: { type: "string", enum: ["positive", "negative", "neutral", "mixed"] },
    intensity: { type: "number", minimum: 0, maximum: 1 },
    confidence: { type: "number", minimum: 0, maximum: 1 },
  };
  const required = ["valence"];
  if (finalized) {
    properties.version = { const: "1" };
    required.unshift("version");
  }
  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function sourceMetadataSchema(finalized = false): JsonSchema {
  const properties: JsonSchema = {
    token_count: { type: "integer", minimum: 0 },
    character_count: { type: "integer", minimum: 0 },
    modality: { type: "string" },
    format: { type: "string" },
  };
  const required: string[] = [];
  if (finalized) {
    properties.version = { const: "1" };
    required.push("version");
  }
  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function temporalRefSchema(capabilities: Set<ExtractionCapability>, finalized = false): JsonSchema {
  const properties: JsonSchema = {
    raw: { type: "string" },
    resolved: { type: "string" },
  };
  const required = ["raw"];

  if (finalized) {
    properties.version = { const: "1" };
    required.unshift("version");
  }

  if (capabilities.has("temporal_classes")) {
    properties.type = { type: "string", enum: ["point", "range", "duration", "unresolved"] };
    properties.resolved_end = { type: "string" };
    properties.context = { type: "string" };
    required.push("type");
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

function producerSchema(): JsonSchema {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      version: { const: "1" },
      model: { type: "string" },
      model_version: { type: "string" },
      deployment: { type: "string" },
      configuration: {
        type: "object",
        additionalProperties: true,
        properties: {
          reasoning_effort: { type: "string" },
          system_prompt_hash: { type: "string" },
          temperature: { type: "number", minimum: 0 },
          top_p: { type: "number", minimum: 0, maximum: 1 },
          max_tokens: { type: "integer", minimum: 1 },
        },
      },
      operator: { type: "string" },
      signature: { type: "string" },
    },
    required: ["version", "model"],
  };
}

function embeddingSchema(): JsonSchema {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      version: { const: "1" },
      vector: { type: "array", items: { type: "number" } },
      model: { type: "string" },
      input: { type: "string" },
      dimensions: { type: "integer", minimum: 1 },
      space: { type: "string" },
      computed_at: { type: "string" },
    },
    required: ["version", "vector", "model", "input", "dimensions"],
  };
}

function capabilitiesSchema(): JsonSchema {
  return {
    type: "array",
    items: {
      type: "string",
      enum: ALL_CAPABILITIES,
    },
  };
}

export function buildExtractionSchema(options: CapabilityOptions): JsonSchema {
  const capabilities = new Set(resolveCapabilities(options));
  const properties: JsonSchema = {
    extracted_at: { type: "string" },
  };
  const required = ["extracted_at"];

  if (capabilities.has("entities")) {
    properties.entities = {
      type: "array",
      items: entitySchema(capabilities),
    };
    required.push("entities");
  }
  if (capabilities.has("goals")) {
    properties.goals = {
      type: "array",
      items: goalSchema(capabilities),
    };
    required.push("goals");
  }
  if (capabilities.has("themes")) {
    properties.themes = {
      type: "array",
      items: { type: "string" },
    };
    required.push("themes");
  }
  if (capabilities.has("keywords")) {
    properties.keywords = {
      type: "array",
      items: { type: "string" },
    };
    required.push("keywords");
  }
  if (capabilities.has("summary")) {
    properties.summary = { type: "string" };
    required.push("summary");
  }
  if (capabilities.has("sentiment")) {
    properties.sentiment = capabilities.has("structured_sentiment") ? sentimentSchema(false) : { type: "string" };
    required.push("sentiment");
  }
  if (capabilities.has("facts")) {
    properties.facts = {
      type: "array",
      items: factSchema(capabilities),
    };
    required.push("facts");
  }
  if (capabilities.has("questions")) {
    properties.questions = {
      type: "array",
      items: questionSchema(capabilities),
    };
    required.push("questions");
  }
  if (capabilities.has("actions")) {
    properties.actions = {
      type: "array",
      items: actionSchema(capabilities),
    };
    required.push("actions");
  }
  if (capabilities.has("decisions")) {
    properties.decisions = {
      type: "array",
      items: decisionSchema(capabilities),
    };
    required.push("decisions");
  }
  if (capabilities.has("temporal_refs")) {
    properties.temporal_refs = {
      type: "array",
      items: temporalRefSchema(capabilities),
    };
    required.push("temporal_refs");
  }
  if (capabilities.has("language")) {
    properties.language = { type: "string" };
    required.push("language");
  }
  if (capabilities.has("source_metadata")) {
    properties.source_metadata = sourceMetadataSchema(false);
    required.push("source_metadata");
  }
  if (capabilities.has("confidence")) {
    properties.confidence = { type: "number", minimum: 0, maximum: 1 };
    required.push("confidence");
  }

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

export function buildFinalizedExtractionSchema(options: CapabilityOptions): JsonSchema {
  const capabilities = new Set(resolveCapabilities(options));
  const properties: JsonSchema = {
    version: { const: "1" },
    extracted_at: { type: "string" },
    source_id: { type: "string" },
    source_type: { type: "string" },
    user_id: { type: "string" },
    produced_by: {
      anyOf: [
        { type: "string" },
        producerSchema(),
      ],
    },
    kind: { type: "string" },
    entities: {
      type: "array",
      items: entitySchema(capabilities, true),
    },
    goals: {
      type: "array",
      items: goalSchema(capabilities, true),
    },
    themes: {
      type: "array",
      items: { type: "string" },
    },
    keywords: {
      type: "array",
      items: { type: "string" },
    },
    sentiment: capabilities.has("structured_sentiment") ? sentimentSchema(true) : {
      anyOf: [
        { type: "string" },
        sentimentSchema(true),
      ],
    },
    summary: { type: "string" },
    facts: {
      type: "array",
      items: factSchema(capabilities, true),
    },
    questions: {
      type: "array",
      items: questionSchema(capabilities, true),
    },
    actions: {
      type: "array",
      items: actionSchema(capabilities, true),
    },
    decisions: {
      type: "array",
      items: decisionSchema(capabilities, true),
    },
    temporal_refs: {
      type: "array",
      items: temporalRefSchema(capabilities, true),
    },
    language: { type: "string" },
    source_metadata: sourceMetadataSchema(true),
    confidence: { type: "number", minimum: 0, maximum: 1 },
    capabilities: capabilitiesSchema(),
    embeddings: {
      type: "array",
      items: embeddingSchema(),
    },
    extensions: {
      type: "object",
      additionalProperties: true,
    },
  };

  return {
    type: "object",
    additionalProperties: false,
    properties,
    required: ["version", "extracted_at", "produced_by", "entities", "goals", "themes", "capabilities"],
  };
}

function nullableSchema(schema: unknown): unknown {
  if (schema && typeof schema === "object" && !Array.isArray(schema)) {
    const obj = schema as JsonSchema;
    if (typeof obj.type === "string") {
      return { ...obj, type: [obj.type, "null"] };
    }
    if (Array.isArray(obj.type) && !obj.type.includes("null")) {
      return { ...obj, type: [...obj.type, "null"] };
    }
  }
  return { anyOf: [schema, { type: "null" }] };
}

function strictifySchema(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(strictifySchema);
  }
  if (schema && typeof schema === "object") {
    const obj = schema as JsonSchema;
    const semanticRequired = new Set(Array.isArray(obj.required) ? obj.required as string[] : []);
    const copy: JsonSchema = {};
    for (const [key, value] of Object.entries(obj)) {
      copy[key] = strictifySchema(value);
    }
    if (copy.type === "object" && copy.properties && typeof copy.properties === "object" && !Array.isArray(copy.properties)) {
      const properties = copy.properties as Record<string, unknown>;
      for (const key of Object.keys(properties)) {
        if (!semanticRequired.has(key)) {
          properties[key] = nullableSchema(properties[key]);
        }
      }
      copy.required = Object.keys(properties);
    }
    return copy;
  }
  return schema;
}

export function buildExtractionResponseFormat(
  options: CapabilityOptions,
  responseOptions: ResponseFormatOptions = {},
): JsonSchema {
  const strict = responseOptions.strict ?? true;
  const schema = buildExtractionSchema(options);
  return {
    type: "json_schema",
    name: responseOptions.name ?? DEFAULT_RESPONSE_FORMAT_NAME,
    strict,
    schema: strict ? strictifySchema(schema) : schema,
  };
}

export class ExtractionBuilder {
  private textValue: string;
  private options: ExtractionBuilderOptions;

  constructor(text = "", options: ExtractionBuilderOptions = {}) {
    this.textValue = text;
    this.options = { stage: "stage1", compact: true, ...options };
  }

  withText(text: string): this {
    this.textValue = text;
    return this;
  }

  withProfile(profile: NonNullable<PromptOptions["profile"]>): this {
    this.options.profile = profile;
    delete this.options.capabilities;
    return this;
  }

  minimal(options: CapabilityProfileOptions = {}): this {
    return this.withProfileCapabilities("minimal", options);
  }

  standard(options: CapabilityProfileOptions = {}): this {
    return this.withProfileCapabilities("standard", options);
  }

  full(options: CapabilityProfileOptions = {}): this {
    return this.withProfileCapabilities("full", options);
  }

  private withProfileCapabilities(profile: NonNullable<PromptOptions["profile"]>, options: CapabilityProfileOptions): this {
    if (options.embed === undefined) {
      return this.withProfile(profile);
    }
    const capabilities = profileCapabilities(profile);
    this.options.capabilities = capabilitySpecs(capabilities, options.embed);
    if (options.embed === true && this.options.embeddingInputs === undefined) {
      this.options.embeddingInputs = ["source"];
    }
    delete this.options.profile;
    return this;
  }

  withCapabilities(capabilities: CapabilityInput[]): this {
    this.options.capabilities = capabilities;
    delete this.options.profile;
    return this;
  }

  addCapabilities(capabilities: CapabilityInput[]): this {
    this.options.add = [...(this.options.add ?? []), ...capabilities];
    return this;
  }

  removeCapabilities(capabilities: CapabilityInput[]): this {
    this.options.remove = [...(this.options.remove ?? []), ...capabilities];
    return this;
  }

  minus(...capabilities: CapabilityInput[]): this {
    return this.removeCapabilities(capabilities);
  }

  unsupported(...capabilities: CapabilityInput[]): this {
    return this.removeCapabilities(capabilities);
  }

  embed(capability: ExtractionCapability, enabled = true): this {
    this.options.embed = { ...(this.options.embed ?? {}), [capability]: enabled };
    return this;
  }

  withEmbeddingInputs(inputs: BuilderEmbeddingInputSelection = "all"): this {
    this.options.embeddingInputs = inputs;
    return this;
  }

  withStandardEmbeddings(): this {
    return this.withEmbeddingInputs("all");
  }

  withCategories(categories: string[]): this {
    this.options.categories = categories;
    return this;
  }

  withSourceType(sourceType: string): this {
    this.options.source_type = sourceType;
    return this;
  }

  withDate(date: string): this {
    this.options.date = date;
    return this;
  }

  withExtractedAt(extractedAt: string): this {
    this.options.extracted_at = extractedAt;
    return this;
  }

  withProducedBy(producedBy: FinalizeContext["produced_by"]): this {
    this.options.produced_by = producedBy;
    return this;
  }

  withUserId(userId: string): this {
    this.options.user_id = userId;
    return this;
  }

  withSource(source: { source_id?: string; source_type?: string }): this {
    if (source.source_id !== undefined) this.options.source_id = source.source_id;
    if (source.source_type !== undefined) this.options.source_type = source.source_type;
    return this;
  }

  withKind(kind: string): this {
    this.options.kind = kind;
    return this;
  }

  withExtensions(extensions: Record<string, unknown>): this {
    this.options.extensions = extensions;
    return this;
  }

  withEmbeddings(embeddings: FinalizeContext["embeddings"]): this {
    this.options.embeddings = embeddings;
    return this;
  }

  withFinalizeContext(context: Partial<FinalizeContext>): this {
    this.options = { ...this.options, ...context };
    return this;
  }

  withCompact(compact = true): this {
    this.options.compact = compact;
    return this;
  }

  resolvedCapabilities(): ExtractionCapability[] {
    return resolveCapabilities(this.options);
  }

  capabilityPlan(): CapabilityPlan {
    const capabilities = this.resolvedCapabilities();
    const resolved = new Set(capabilities);
    const excluded = expandCapabilityExclusions(normalizeCapabilityInputs(this.options.remove ?? []));
    const embeddedInputs = orderEmbeddingInputs(applyEmbeddingOverrides(uniqueStrings([
      ...embeddingInputsFromSelection(this.options.embeddingInputs),
      ...embeddingInputsFromCapabilities(this.options.capabilities, resolved),
      ...embeddingInputsFromCapabilities(this.options.add, resolved),
    ]), this.options.embed));
    const notEmbedded = capabilities.filter((capability) => {
      const input = embeddableInputForCapability(capability);
      return input !== undefined && !embeddedInputs.includes(input);
    });

    return {
      capabilities,
      excluded,
      embeddedInputs,
      notEmbedded,
      requiredCallbacks: {
        callLlm: true,
        getEmbedding: embeddedInputs.length > 0,
      },
      promptCharacters: this.prompt().length,
    };
  }

  plan(): CapabilityPlan {
    return this.capabilityPlan();
  }

  extractOptions(): ExtractionBuilderOptions {
    const plan = this.capabilityPlan();
    const customInputs = Array.isArray(this.options.embeddingInputs)
      ? this.options.embeddingInputs.filter((input): input is CustomBuilderEmbeddingInput => typeof input === "object" && input !== null)
      : [];
    const namedInputs = plan.embeddedInputs.filter((input): input is EmbeddableInput => (
      STANDARD_BUILDER_EMBEDDING_INPUTS.includes(input as EmbeddableInput)
      && !customInputs.some((custom) => custom.input === input)
    ));
    const embeddingInputs = plan.embeddedInputs.length > 0
      ? [
        ...customInputs,
        ...namedInputs,
      ]
      : undefined;
    return {
      ...this.options,
      ...(embeddingInputs !== undefined ? { embeddingInputs } : {}),
    };
  }

  toOptions(): ExtractionBuilderOptions {
    return this.extractOptions();
  }

  prompt(): string {
    return buildExtractionPrompt(this.textValue, this.options);
  }

  schema(): JsonSchema {
    return buildExtractionSchema(this.options);
  }

  finalizedSchema(): JsonSchema {
    return buildFinalizedExtractionSchema(this.options);
  }

  responseFormat(options: ResponseFormatOptions = {}): JsonSchema {
    return buildExtractionResponseFormat(this.options, options);
  }

  finalizeContext(): FinalizeContext | undefined {
    if (this.options.produced_by === undefined) {
      return undefined;
    }

    const context: FinalizeContext = {
      produced_by: this.options.produced_by,
      capabilities_hint: this.options.capabilities_hint ?? this.resolvedCapabilities(),
    };
    if (this.options.user_id !== undefined) context.user_id = this.options.user_id;
    if (this.options.source_id !== undefined) context.source_id = this.options.source_id;
    if (this.options.source_type !== undefined) context.source_type = this.options.source_type;
    if (this.options.kind !== undefined) context.kind = this.options.kind;
    if (this.options.extensions !== undefined) context.extensions = this.options.extensions;
    if (this.options.embeddings !== undefined) context.embeddings = this.options.embeddings;
    return context;
  }

  finalize(llmOutput: Record<string, unknown>): FinalizeResult {
    const context = this.finalizeContext();
    if (!context) {
      throw new Error("Cannot finalize without produced_by; call withProducedBy() or withFinalizeContext()");
    }
    return finalizeExtraction(llmOutput, context);
  }

  build(options: ResponseFormatOptions = {}): ExtractionBuilderResult {
    const result: ExtractionBuilderResult = {
      capabilities: this.resolvedCapabilities(),
      prompt: this.prompt(),
      schema: this.schema(),
      responseFormat: this.responseFormat(options),
      finalizedSchema: this.finalizedSchema(),
    };
    const context = this.finalizeContext();
    if (context) result.finalizeContext = context;
    return result;
  }
}

export function createExtractionBuilder(text = "", options: ExtractionBuilderOptions = {}): ExtractionBuilder {
  return new ExtractionBuilder(text, options);
}
