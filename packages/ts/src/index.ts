export type {
  SynaptExtraction,
  SynaptEntity,
  SynaptGoal,
  SynaptFact,
  SynaptQuestion,
  SynaptAction,
  SynaptDecision,
  SynaptSentiment,
  SynaptSourceMetadata,
  SynaptRelation,
  SynaptSourceRef,
  SynaptEmbedding,
  SynaptAssertionSignals,
  SynaptTemporalRef,
  SynaptProducer,
  SynaptProducerConfiguration,
  ExtractionCapability,
} from "./schema.js";

export { validateExtraction } from "./validate.js";
export type { ValidationResult, ValidationError } from "./validate.js";

export { finalizeExtraction } from "./finalize.js";
export type { FinalizeContext, FinalizeResult } from "./finalize.js";

export { buildExtractionPrompt, resolveCapabilities } from "./prompt.js";
export type { CapabilityInput, CapabilitySpec, PromptOptions } from "./prompt.js";

export {
  ExtractionBuilder,
  buildFinalizedExtractionSchema,
  buildExtractionSchema,
  buildExtractionResponseFormat,
  createExtractionBuilder,
} from "./builder.js";
export type { ExtractionBuilderOptions, ExtractionBuilderResult, JsonSchema, ResponseFormatOptions } from "./builder.js";
export type { CapabilityPlan, CapabilityProfileOptions, EmbeddableInput } from "./builder.js";

export { extract, normalizeLlmResponse, runExtraction } from "./extract.js";
export type {
  CustomEmbeddingInput,
  EmbeddingInputSelection,
  EmbeddingInputSelector,
  EmbeddingRequest,
  EmbeddingResponse,
  ExtractCallbacks,
  ExtractOptions,
  ExtractResult,
  ExtensionResolver,
  ExtensionResolverContext,
  LlmMessage,
  LlmRequest,
  LlmResponse,
  LlmResponseProvider,
  LlmResponseTranslator,
  LlmResponseTranslatorContext,
  LlmUsage,
  LogEntry,
  NamedEmbeddingInput,
  NormalizedLlmResponse,
  UsageSummary,
} from "./extract.js";
