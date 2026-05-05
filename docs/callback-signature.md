# Extract Callback Signature Design

**Status:** PROPOSED -- not shipped in v0.3.x. Target: v0.4.0. Pending Anchor review at prompt-alignment session.
**Author:** Apollo
**Anchored to:**
- `config/research/conversa/2026-05-04-wasm-conditions-acceptance.md` (Condition 6: API symmetry)
- `config/design/conversa-integration-alignment.md` (CDP3: host import contract)

## Summary

The `extract()` function accepts raw text and a callbacks object. The four callback names are publicly committed: `callLlm`, `getEmbedding`, `log`, `randomUuid`. This document defines their input/output type signatures, async patterns, error handling, and the v1.2-to-v2 symmetry guarantee.

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
  getEmbedding: (request: EmbeddingRequest) => Promise<EmbeddingResponse>;
  log: (entry: LogEntry) => void;
  randomUuid: () => string;
}

function extract(
  text: string,
  callbacks: ExtractCallbacks,
  options?: ExtractOptions,
): Promise<ExtractResult>;
```

```typescript
interface ExtractOptions {
  capabilities?: ExtractionCapability[];
  profile?: "minimal" | "standard" | "full";
  source_type?: string;
  source_id?: string;
  user_id?: string;
  kind?: string;
  date?: string;
  categories?: string[];
  extensions?: Record<string, unknown>;
}

interface ExtractResult {
  extraction: SynaptExtraction;
  validation: ValidationResult;
  warnings: string[];
  usage: UsageSummary;
}

interface UsageSummary {
  llm_calls: number;
  embedding_calls: number;
  total_input_tokens: number;
  total_output_tokens: number;
}
```

## Callback signatures

### `callLlm`

Sends a prompt to an LLM and returns the text response. The caller (Conversa) owns the model, API key, and routing. Synapt never sees credentials.

```typescript
interface LlmRequest {
  messages: LlmMessage[];
  response_format: "json";
  temperature?: number;
  max_tokens?: number;
}

interface LlmMessage {
  role: "system" | "user";
  content: string;
}

interface LlmResponse {
  content: string;
  usage?: LlmUsage;
}

interface LlmUsage {
  input_tokens: number;
  output_tokens: number;
}
```

**Design notes:**

- `messages` uses the standard system/user role pair. Synapt builds both the system prompt (from capability fragments) and the user prompt (the source text). The caller forwards these to their LLM provider.
- `response_format: "json"` is always set. Synapt expects structured JSON output from the LLM. The caller should pass this to their provider's JSON mode if available.
- `usage` is optional. If the caller's LLM provider returns token counts, pass them through. Synapt aggregates these into `UsageSummary` for metering. If omitted, metering is best-effort.
- `temperature` defaults to `0` if not specified by the caller's provider. Synapt may set this based on the extraction profile.
- `max_tokens` is advisory. Synapt sets it based on expected output size for the requested capabilities.

**Retry semantics:** Synapt does NOT retry. If `callLlm` throws, extraction fails with the error propagated. The caller owns retry logic, rate limiting, and fallback providers. This keeps synapt's behavior deterministic and avoids surprising the caller with retries against their API quota.

**Error contract:** Throw a standard `Error` on failure. Synapt catches it, attaches context (which extraction step failed), and returns it in the `ExtractResult` as a validation error. The extraction is marked invalid but the partial result is still returned for diagnostics.

### `getEmbedding`

Computes a vector embedding for a text input. Used for pre-computed embeddings on the extraction (e.g., embedding the summary or source text for downstream similarity search).

```typescript
interface EmbeddingRequest {
  text: string;
  input_type: "source" | "summary" | "entities" | string;
}

interface EmbeddingResponse {
  vector: number[];
  model: string;
  dimensions: number;
}
```

**Design notes:**

- `input_type` tells the caller what is being embedded. This is informational; some providers optimize for different input types (e.g., passage vs query).
- `model` in the response is the model identifier the caller used. Synapt records this in the extraction's `embeddings[]` array. The caller decides which embedding model to use; synapt just records what was used.
- `dimensions` must equal `vector.length`. Synapt validates this.

**Retry semantics:** Same as `callLlm`. No retries from synapt. Caller owns retry and fallback.

**When called:** Only when the caller requests embeddings via `ExtractOptions` (not yet specified; likely a `compute_embeddings?: boolean` or `embedding_inputs?: string[]` option). Not called by default. This keeps the common extraction path free of embedding API calls.

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

### `randomUuid`

Generates a UUID string. Used for entity ID assignment when the LLM doesn't produce IDs and `entity_ids` capability is requested.

```typescript
randomUuid: () => string;
```

**Design notes:**

- Must return a valid UUID string (any version). Synapt does not validate the format beyond checking it's a non-empty string.
- In v1.2 TypeScript, this is typically `() => crypto.randomUUID()`.
- In v2 WASM, this becomes a host import because WASM modules don't have access to `crypto.randomUUID()`. The host provides the randomness source.
- The caller MAY return deterministic UUIDs for reproducibility (e.g., seeded UUID v5 from a namespace + counter). This supports the reproducibility contract from Condition 4.

## v2 WASM symmetry

The table below shows how each v1.2 TypeScript callback maps to v2 WASM:

| v1.2 TypeScript | v2 WASM host import | Async model |
|-----------------|---------------------|-------------|
| `callLlm(req) => Promise<LlmResponse>` | `(extern "synapt") call_llm(req: LlmRequest) -> LlmResponse` | WASI Preview 2 async |
| `getEmbedding(req) => Promise<EmbeddingResponse>` | `(extern "synapt") get_embedding(req: EmbeddingRequest) -> EmbeddingResponse` | WASI Preview 2 async |
| `log(entry) => void` | `(extern "synapt") log(entry: LogEntry)` | Sync (fire and forget) |
| `randomUuid() => string` | `(extern "synapt") random_uuid() -> String` | Sync |

**Name mapping:** TypeScript uses camelCase (`callLlm`); WASM imports use snake_case (`call_llm`). This is the only difference. The mapping is mechanical and documented here.

**Serialization:** In v1.2, arguments are TypeScript objects passed by reference. In v2, arguments cross the WASM ABI as JSON-serialized buffers (Component Model canonical ABI for records). The shapes are identical; only the transport changes.

**Capability negotiation:** In v1.2, the caller provides callbacks at call time. In v2, the WASM module declares required imports at load time, and the host fulfills them. Missing imports fail at module instantiation, not at extraction time. This is the `synapt_capabilities()` export pattern from CDP3.

## For Anchor's review

**Settled (non-negotiable):**
- Four callback names: `callLlm`, `getEmbedding`, `log`, `randomUuid`
- No-retry policy: synapt never retries; caller owns retry logic
- `response_format: "json"` always set on LLM calls
- `log` is fire-and-forget; errors silently ignored
- v1.2/v2 structural symmetry (Condition 6)

**Negotiable (want Anchor's eyes on):**
- `LlmRequest.messages` shape: is system+user sufficient, or does Conversa need assistant/tool roles for multi-turn extraction? Current design is single-turn (one system + one user message).
- `LlmUsage` granularity: is input/output tokens sufficient, or does Conversa need cache-read/cache-write breakdowns for cost tracking?
- `LogEntry.stage` enum values: are these the right pipeline stages for Conversa's observability needs? Missing any?
- `getEmbedding` trigger: should embeddings be opt-in per call, or should the callback simply not be called if the caller doesn't want embeddings? (Current design: opt-in via ExtractOptions.)
- `ExtractResult.usage`: is a flat summary sufficient, or does Conversa need per-call usage breakdown?

**Questions for Anchor:**
1. Does Conversa's edge function environment support `crypto.randomUUID()`? If not, we need to document the fallback.
2. What is Conversa's preferred error shape? Plain `Error` with message, or structured error with code/details?
3. Does Conversa want streaming LLM responses in v1.2, or is single-response sufficient for launch? (We recommend deferring streaming to v2.)
