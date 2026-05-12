# @synapt-dev/extract

SynaptExtraction is the intermediate language (IL) for [synapt](https://synapt.dev)'s product stack. It is the universal exchange format between text extraction and intelligence operations.

```
Any text + Any LLM  ->  SynaptExtraction (IL)  ->  @synapt/memory (intelligence)
```

This repo contains the v1 schema, types, validators, finalization pipeline, and composable prompt system in both TypeScript and Python.

## Install

| Package | Registry | Install |
|---------|----------|---------|
| `@synapt-dev/extract` | npm | `npm install @synapt-dev/extract@0.3.2` |
| `synapt-extract` | PyPI | `pip install synapt-extract==0.3.2` |

**Deno:**

```typescript
import { buildExtractionPrompt } from "npm:@synapt-dev/extract@0.3.2";
```

**Version pinning:** Always pin to an exact version (`@0.3.2`, `==0.3.2`). Do not use ranges (`^0.3.2`, `~0.3.2`, `>=0.3.2`). The IL schema evolves across minor versions (v1.1 added `produced_by` object form, v1.2 added 8 new fields). Pinning prevents unexpected schema changes from affecting your extraction pipeline.

## Quick start

### TypeScript

```typescript
import {
  buildExtractionPrompt,
  finalizeExtraction,
  validateExtraction,
} from "@synapt-dev/extract";

// 1. Build a prompt for your LLM
const prompt = buildExtractionPrompt(text, {
  profile: "standard",
  categories: ["Health", "Family"],
});

// 2. Send to any LLM, parse JSON response
const llmOutput = JSON.parse(await llm.complete(prompt));

// 3. Finalize: inject client context, normalize, validate
const result = finalizeExtraction(llmOutput, {
  produced_by: "openai://gpt-4o-mini",
  user_id: userId,
  kind: "conversa/prayer",
});

console.log(result.extraction);     // Complete SynaptExtraction
console.log(result.validation);     // { valid: true, errors: [] }
```

### Python

```python
from synapt_extract import (
    build_extraction_prompt,
    finalize_extraction,
    FinalizeContext,
)

# 1. Build a prompt
prompt = build_extraction_prompt(text, profile="standard")

# 2. Send to any LLM, parse JSON response
llm_output = json.loads(llm.complete(prompt))

# 3. Finalize
result = finalize_extraction(llm_output, FinalizeContext(
    produced_by="openai://gpt-4o-mini",
    user_id=user_id,
    kind="conversa/prayer",
))

assert result.validation.valid
```

## Three-stage pipeline

SynaptExtraction documents are assembled in three stages:

1. **Stage 1 (LLM)**: The LLM extracts content fields (entities, goals, themes, etc.) from text
2. **Stage 2 (Client)**: Your application injects context the LLM can't know (produced_by, user_id, embeddings, extensions)
3. **Stage 3 (Library)**: `finalizeExtraction()` normalizes the document (version injection, capability detection, sub-schema versioning, validation)

## Prompt profiles

| Profile | Model class | Capabilities (25 total) |
|---------|------------|--------------|
| `minimal` | 3B-7B local | entities, entity_state, goals, themes, summary |
| `standard` | GPT-4o-mini, Haiku | + entity_context, goal_timing, facts, temporal_refs, sentiment, evidence_anchoring |
| `full` | GPT-4o, Sonnet, Opus | + entity_ids, goal_entity_refs, keywords, structured_sentiment, questions, actions, decisions, relations, relation_origin, assertion_signals, temporal_classes, language, source_metadata, confidence |

## Prompt and schema builder

Use the builder when the model API supports structured output. It resolves capabilities once, then builds the matching prompt, Stage 1 JSON schema, OpenAI response format, finalized packet schema, and optional finalization context.

### TypeScript

```typescript
import { createExtractionBuilder } from "@synapt-dev/extract";

const builder = createExtractionBuilder(text, { profile: "standard" })
  .addCapabilities(["entity_ids", "goal_entity_refs"])
  .withExtractedAt("2026-05-11T18:00:00Z")
  .withProducedBy({
    model: "openai://gpt-5.5",
    model_version: "gpt-5.5-2026-04-23",
    configuration: { reasoning_effort: "medium" },
    operator: "synapt-dev",
  })
  .withSource({ source_id: "note-1", source_type: "note" });

const built = builder.build({ name: "synapt_extract_stage1" });

// Send built.prompt and built.responseFormat to the model.
// Then call builder.finalize(stage1Json) or finalizeExtraction(stage1Json, built.finalizeContext).
```

### Python

```python
from synapt_extract import create_extraction_builder

builder = (
    create_extraction_builder(text, profile="standard")
    .add_capabilities(["entity_ids", "goal_entity_refs"])
    .with_extracted_at("2026-05-11T18:00:00Z")
    .with_produced_by({
        "model": "openai://gpt-5.5",
        "model_version": "gpt-5.5-2026-04-23",
        "configuration": {"reasoning_effort": "medium"},
        "operator": "synapt-dev",
    })
    .with_source(source_id="note-1", source_type="note")
)

built = builder.build(name="synapt_extract_stage1")

# Send built["prompt"] and built["response_format"] to the model.
# Then call builder.finalize(stage1_json) or finalize_extraction(stage1_json, builder.finalize_context()).
```

`buildExtractionSchema()` / `build_extraction_schema()` return the semantic Stage 1 schema. `buildExtractionResponseFormat()` / `build_extraction_response_format()` return an OpenAI-compatible `json_schema` response format; strict mode requires every object property as OpenAI expects and represents semantic optional fields as nullable. `buildFinalizedExtractionSchema()` / `build_finalized_extraction_schema()` return the finalized packet shape, including `produced_by`, source context, capabilities, embeddings, and extensions.

## Full extraction runner

Use `extract()` when you want the library to execute the full pipeline while your application owns the model and embedding API calls. The callbacks receive plain JSON requests, so callers can route to OpenAI, another provider, a local model, or a test fixture.

```typescript
import { extract } from "@synapt-dev/extract";

const result = await extract(text, {
  callLlm: async (request) => {
    // Send request.messages and request.responseFormat to your model provider.
    return {
      output: stage1Json,
      produced_by: { model: "openai://gpt-5.5" },
    };
  },
  getEmbedding: async (request) => {
    // Embed request.text and record the provider URI for the embedding model.
    return { vector, model: "openai://text-embedding-3-small" };
  },
}, {
  capabilities: [
    { name: "entities", embed: true },
    { name: "goals", embed: true },
    { name: "summary", embed: true },
    "themes",
  ],
  source_id: "note-1",
  source_type: "note",
  embeddingInputs: ["source"],
  extend: ({ response, stage1, embeddings }) => ({
    "synapt/response_binding": {
      response_id: response.id,
      response_model: response.model,
      stage1_fields: Object.keys(stage1).length,
      embedding_count: embeddings.length,
    },
  }),
});
```

```python
from synapt_extract import extract

result = await extract(
    text,
    {
        "call_llm": call_llm,
        "get_embedding": get_embedding,
    },
    capabilities=[
        {"name": "entities", "embed": True},
        {"name": "goals", "embed": True},
        {"name": "summary", "embed": True},
        "themes",
    ],
    source_id="note-1",
    source_type="note",
    embedding_inputs=["source"],
    extend=lambda ctx: {
        "synapt/response_binding": {
            "response_id": ctx["response"].get("id"),
            "response_model": ctx["response"].get("model"),
            "stage1_fields": len(ctx["stage1"]),
            "embedding_count": len(ctx["embeddings"]),
        }
    },
)
```

Capability entries can be plain strings or `{ name, embed: true }` specs. The runner derives embeddings from embedded capability specs and merges them with explicit embedding inputs such as `"source"`. `embeddingInputs: "all"` / `embedding_inputs="all"` remains available for exhaustive tests and computes embeddings for source, summary, entities, goals, themes, keywords, facts, questions, actions, decisions, temporal refs, and sentiment when those fields exist. Embeddings are opt-in; if no embedding inputs are requested and no capability has `embed: true`, no embedding API call is made.

The `extend` resolver runs after the LLM response is parsed and embeddings are computed, but before finalization. It receives a normalized response envelope (`response.id`, `response.status`, `response.model`, `response.usage`, and `response.raw`), so extensions can depend on provider output without knowing the provider's raw response shape. Returned extension objects are merged over static `extensions` and receive `version: "1"` during finalization.

For preflight UX, builders expose profile helpers and a plan:

```typescript
const plan = createExtractionBuilder(text)
  .full({ embed: true })
  .minus("questions")
  .embed("summary", false)
  .plan();
```

`plan()` reports resolved capabilities, excluded capabilities, embedding inputs, requested-but-not-embedded capabilities, required callbacks, and prompt character count.

## JSON Schema

The canonical schema is hosted at:

```
https://synapt.dev/schemas/extract/v1.json
```

Sub-schemas: `source-ref/v1.json`, `embedding/v1.json`, `assertion-signals/v1.json`, `temporal-ref/v1.json`, `producer/v1.json`, `entity/v1.json`, `goal/v1.json`, `question/v1.json`, `action/v1.json`, `decision/v1.json`, `sentiment/v1.json`, `source-metadata/v1.json`.

## Compatibility

v1.x schema updates are additive only. Breaking changes require v2.

- v1.0 documents remain valid under v1.2 validators
- `produced_by` accepts string (v1.0) or SynaptProducer object (v1.1+)
- `sentiment` accepts string (v1.0) or SynaptSentiment object (v1.2+)
- Readers MUST branch on string vs object for both fields

## Supply chain

Releases are published with Sigstore provenance via npm OIDC trusted publishing and PyPI trusted publishing. Each GitHub Release includes a CycloneDX SBOM (`sbom.cdx.json`).

## Repo structure

```
extract/
  packages/
    ts/          # @synapt-dev/extract (TypeScript, npm)
    python/      # synapt-extract (Python, PyPI)
  schemas/       # JSON Schema files (language-agnostic)
  prompts/
    v1/          # 25 capability prompt fragments + preamble/postamble
    profiles/    # Profile definitions (minimal, standard, full)
  tests/
    python/      # Python test suite
    conformance/ # Cross-language conformance fixtures
  docs/          # Design documents
```

## License

MIT
