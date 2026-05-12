# Extract Callback Signature Design

**Status:** SHIPPED in v0.4.0 as provider callbacks in TypeScript and Python. The 0.5.0 universal host boundary is tracked in [universal-host-boundary.md](universal-host-boundary.md).
**Author:** Apollo
**Anchored to:**
- `config/research/conversa/2026-05-04-wasm-conditions-acceptance.md` (Condition 6: API symmetry)
- `config/design/conversa-integration-alignment.md` (CDP3: host import contract)

## Summary

The `extract()` function accepts raw text and a callbacks object. The implemented callback names are `callLlm`, optional `getEmbedding`, and optional `log` in TypeScript; Python also accepts `call_llm` and `get_embedding`. `randomUuid` remains a deferred v2/WASM host-import candidate.

## Design principle: binding substitution

The v1.2 TypeScript callback interface and the v2 WASM host-import interface MUST be structurally identical. v2 migration is a binding substitution (function calls become WASM imports), not a redesign. This means:

- Same names, same argument shapes, same return shapes
- No TypeScript-specific types that can't cross the WASM ABI (no classes, no generics, no union types beyond tagged discriminants)
- All complex types are plain objects/records that serialize to JSON
- Async operations use `Promise<T>` in v1.2 and WASI Preview 2 async in v2

## Top-level API

```typescript
interface ExtractCallbacks {
  callLlm: (request: LlmRequest) => Promise<LlmResponse>;
  getEmbedding?: (request: EmbeddingRequest) => Promise<EmbeddingResponse>;
  log?: (entry: LogEntry) => void;
}

function extract(
  text: string,
  callbacks: ExtractCallbacks,
  options?: ExtractOptions,
): Promise<ExtractResult>;
```

```typescript
interface ExtractOptions {
  capabilities?: Array<ExtractionCapability | { name: ExtractionCapability; embed?: boolean }>;
  profile?: "minimal" | "standard" | "full";
  source_type?: string;
  source_id?: string;
  user_id?: string;
  kind?: string;
  date?: string;
  categories?: string[];
  extensions?: Record<string, unknown>;
  embeddingInputs?: "all" | Array<"source" | "summary" | "entities" | "goals" | "themes" | "keywords" | "facts" | "questions" | "actions" | "decisions" | "temporal_refs" | "sentiment" | { input: string; text: string }>;
  responseTranslator?: (context: LlmResponseTranslatorContext) => Partial<NormalizedLlmResponse> | undefined;
  responseTranslators?: Array<(context: LlmResponseTranslatorContext) => Partial<NormalizedLlmResponse> | undefined>;
  extend?: (context: ExtensionResolverContext) => Record<string, unknown> | Promise<Record<string, unknown>>;
  extensionErrors?: "throw" | "warn";
}

interface ExtractResult {
  extraction: SynaptExtraction;
  validation: ValidationResult;
  warnings: string[];
  usage: UsageSummary;
  stage1: Record<string, unknown>;
  embeddings: Array<Omit<SynaptEmbedding, "version">>;
}

interface UsageSummary {
  llm_calls: number;
  embedding_calls: number;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
}
```

## Callback signatures

### `callLlm`

Sends a prompt to an LLM and returns the text response. The caller (Conversa) owns the model, API key, and routing. Synapt never sees credentials.

```typescript
interface LlmRequest {
  prompt: string;
  messages: LlmMessage[];
  capabilities: ExtractionCapability[];
  schema: Record<string, unknown>;
  responseFormat: Record<string, unknown>;
  temperature?: number;
  max_tokens?: number;
}

interface LlmMessage {
  role: "system" | "user";
  content: string;
}

interface LlmResponse {
  content?: string;
  json?: Record<string, unknown>;
  output?: Record<string, unknown>;
  produced_by?: string | Omit<SynaptProducer, "version">;
  provider?: "openai" | "anthropic" | string;
  response_id?: string;
  id?: string;
  status?: string;
  model?: string;
  model_version?: string;
  usage?: LlmUsage;
  raw?: unknown;
}

interface NormalizedLlmResponse {
  provider?: "openai" | "anthropic" | string;
  id?: string;
  status?: string;
  model?: string;
  model_version?: string;
  stop_reason?: string;
  produced_by?: string | Omit<SynaptProducer, "version">;
  content?: string;
  usage?: LlmUsage;
  raw?: unknown;
}

interface ExtensionResolverContext {
  sourceText: string;
  capabilities: ExtractionCapability[];
  prompt: string;
  schema: Record<string, unknown>;
  responseFormat: Record<string, unknown>;
  llmRequest: LlmRequest;
  response: NormalizedLlmResponse;
  llmResponse: NormalizedLlmResponse;
  stage1: Record<string, unknown>;
  embeddings: Array<Omit<SynaptEmbedding, "version">>;
  usage: UsageSummary;
  warnings: string[];
}

interface LlmUsage {
  input_tokens: number;
  output_tokens: number;
}
```

**Design notes:**

- `messages` uses a system/user role pair. The user message contains the complete builder prompt. The caller forwards these to their LLM provider.
- `responseFormat` carries the builder-generated strict `json_schema` response format. The caller should pass this to their provider's structured-output mechanism if available.
- The response may provide parsed `output`/`json` or JSON string `content`. `produced_by` is preferred. If omitted, `model` must already be a provider URI for the runner to derive a producer.
- The runner normalizes provider-specific LLM responses before passing them to extension resolvers. Built-in translators handle raw OpenAI Responses and Anthropic Messages objects. Extensions should use `context.response.provider/id/status/model/stop_reason/usage/raw` instead of depending on a callback-specific response shape.
- Custom providers can pass `responseTranslator` / `response_translator` to map raw provider output into the same `NormalizedLlmResponse` envelope.
- `usage` is optional. If the caller's LLM provider returns token counts, pass them through. Synapt aggregates these into `UsageSummary` for metering. If omitted, metering is best-effort.
- `temperature` defaults to `0` if not specified by the caller's provider. Synapt may set this based on the extraction profile.
- `max_tokens` is advisory. Synapt sets it based on expected output size for the requested capabilities.

**Retry semantics:** Synapt does NOT retry. If `callLlm` throws, extraction fails with the error propagated. The caller owns retry logic, rate limiting, and fallback providers. This keeps synapt's behavior deterministic and avoids surprising the caller with retries against their API quota.

**Error contract:** Throw a standard `Error` on failure. The experimental runner does not retry and propagates callback, parse, and missing-context errors to the caller.

### `getEmbedding`

Computes a vector embedding for a text input. Used for pre-computed embeddings on the extraction (e.g., embedding the summary or source text for downstream similarity search).

```typescript
interface EmbeddingRequest {
  text: string;
  input: "source" | "summary" | "entities" | "goals" | "themes" | "keywords" | "facts" | "questions" | "actions" | "decisions" | "temporal_refs" | "sentiment" | string;
}

interface EmbeddingResponse {
  vector: number[];
  model: string;
  dimensions: number;
}
```

**Design notes:**

- `input` tells the caller what is being embedded. This is informational; some providers optimize for different input types (e.g., passage vs query).
- `model` in the response is the model identifier the caller used. Synapt records this in the extraction's `embeddings[]` array. The caller decides which embedding model to use; synapt just records what was used.
- `dimensions` must equal `vector.length`. Synapt validates this.

**Retry semantics:** Same as `callLlm`. No retries from synapt. Caller owns retry and fallback.

**When called:** Only when the caller requests embeddings via `ExtractOptions.embeddingInputs` (`embedding_inputs` in Python). Not called by default. This keeps the common extraction path free of embedding API calls.

Capability entries may also request embeddings inline: `{ name: "entities", embed: true }`. Inline capability embeddings are merged with explicit embedding inputs, so callers can request `{ name: "summary", embed: true }` plus `embeddingInputs: ["source"]`.

### `extend`

Builds dynamic extensions from the normalized response, parsed Stage 1 output, embeddings, and usage. It runs after embeddings and before finalization, so returned extension objects receive `version: "1"` like static extensions.

```typescript
extend: ({ response, stage1, embeddings }) => ({
  "synapt/response_binding": {
    response_id: response.id,
    response_model: response.model,
    stage1_fields: Object.keys(stage1).length,
    embedding_count: embeddings.length,
  },
})
```

### `log`

Structured logging for observability. Fire-and-forget; return value is ignored. Synapt emits log entries at key extraction stages for the caller to route to their logging infrastructure.

```typescript
interface LogEntry {
  level: "debug" | "info" | "warn" | "error";
  stage: "prompt_build" | "llm_call" | "parse" | "validate" | "finalize" | "embed";
  message: string;
  data?: Record<string, unknown>;
}
```

**Design notes:**

- `stage` identifies which extraction pipeline step emitted the log. This maps 1:1 to the extract pipeline stages and is stable across versions.
- `data` carries structured context (e.g., `{ capabilities: [...], profile: "standard" }` for prompt_build, `{ input_tokens: 500 }` for llm_call, `{ error_count: 3 }` for validate).
- Synapt logs at `info` for normal stages, `warn` for recoverable issues (validation errors, missing optional capabilities), `error` for failures that stop extraction, `debug` for verbose trace (prompt content, raw LLM output).

**Error handling:** If `log` throws, synapt silently ignores the error. Logging must never break extraction.

## v2 WASM symmetry

The table below shows how each v1.2 TypeScript callback maps to v2 WASM:

| v1.2 TypeScript | v2 WASM host import | Async model |
|-----------------|---------------------|-------------|
| `callLlm(req) => Promise<LlmResponse>` | `(extern "synapt") call_llm(req: LlmRequest) -> LlmResponse` | WASI Preview 2 async |
| `getEmbedding(req) => Promise<EmbeddingResponse>` | `(extern "synapt") get_embedding(req: EmbeddingRequest) -> EmbeddingResponse` | WASI Preview 2 async |
| `log(entry) => void` | `(extern "synapt") log(entry: LogEntry)` | Sync (fire and forget) |
| deferred `randomUuid() => string` | `(extern "synapt") random_uuid() -> String` | Sync |

**Name mapping:** TypeScript uses camelCase (`callLlm`); Python and future WASM imports use snake_case (`call_llm`). The mapping is mechanical and documented here.

**Serialization:** In v1.2, arguments are TypeScript objects passed by reference. In v2, arguments cross the WASM ABI as JSON-serialized buffers (Component Model canonical ABI for records). The shapes are identical; only the transport changes.

**Capability negotiation:** In v1.2, the caller provides callbacks at call time. In v2, the WASM module declares required imports at load time, and the host fulfills them. Missing imports fail at module instantiation, not at extraction time. This is the `synapt_capabilities()` export pattern from CDP3.

## For Anchor's review

**Settled (non-negotiable):**
- Implemented callback names: `callLlm`, optional `getEmbedding`, optional `log`
- No-retry policy: synapt never retries; caller owns retry logic
- `responseFormat` carries the builder-generated JSON schema on LLM calls
- `log` is fire-and-forget; errors silently ignored
- v1.2/v2 structural symmetry remains the design constraint for callback records

**Negotiable (want Anchor's eyes on):**
- `LlmRequest.messages` shape: is system+user sufficient, or does Conversa need assistant/tool roles for multi-turn extraction? Current design is single-turn (one system + one user message).
- `LlmUsage` granularity: is input/output tokens sufficient, or does Conversa need cache-read/cache-write breakdowns for cost tracking?
- `LogEntry.stage` enum values: are these the right pipeline stages for Conversa's observability needs? Missing any?
- Whether `"all"` is the right default convenience selector for embedding coverage, or whether callers should always name embedding inputs explicitly.
- `ExtractResult.usage`: is a flat summary sufficient, or does Conversa need per-call usage breakdown?

**Questions for Anchor:**
1. Does Conversa's edge function environment need host-supplied UUID generation, or can entity IDs remain model-emitted local IDs?
2. What is Conversa's preferred error shape? Plain `Error` with message, or structured error with code/details?
3. Does Conversa want streaming LLM responses in v1.2, or is single-response sufficient for launch? (We recommend deferring streaming to v2.)
