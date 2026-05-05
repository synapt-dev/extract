# @synapt-dev/extract

SynaptExtraction is the intermediate language (IL) for [synapt](https://synapt.dev)'s product stack. It is the universal exchange format between text extraction and intelligence operations.

```
Any text + Any LLM  ->  SynaptExtraction (IL)  ->  @synapt/memory (intelligence)
```

This repo contains the v1 schema, types, validators, finalization pipeline, and composable prompt system in both TypeScript and Python.

## Install

| Package | Registry | Install |
|---------|----------|---------|
| `@synapt-dev/extract` | npm | `npm install @synapt-dev/extract@0.3.1` |
| `synapt-extract` | PyPI | `pip install synapt-extract==0.3.1` |

**Deno:**

```typescript
import { buildExtractionPrompt } from "npm:@synapt-dev/extract@0.3.1";
```

**Version pinning:** Always pin to an exact version (`@0.3.1`, `==0.3.1`). Do not use ranges (`^0.3.1`, `~0.3.1`, `>=0.3.1`). The IL schema evolves across minor versions (v1.1 added `produced_by` object form, v1.2 added 8 new fields). Pinning prevents unexpected schema changes from affecting your extraction pipeline.

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
