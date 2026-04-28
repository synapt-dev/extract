# Changelog

## v0.2.0

v1.1 spec: typed SynaptProducer schema, additive backwards-compat, in-place v1.x = additive only policy.

- `produced_by` field accepts either a string (v1.0 backwards-compat) or a structured `SynaptProducer` object (v1.1)
- New sub-schema: `schemas/producer/v1.json` with model URI, model_version, deployment, configuration (open), operator, and signature fields
- TypeScript and Python types, validators, and finalize pipeline updated for dual-shape dispatch
- Schema descriptions tightened per adversarial review: model/model_version precedence, deployment as opaque label, signature as opaque attestation slot, honest compatibility boundary documentation
- 217 Python tests + 163 TypeScript tests

### Compatibility

v1.1 is additive for schema validators and type-tolerant readers. v1.0 documents with string `produced_by` remain valid. Consumer code that performs string operations on `produced_by` without a type guard will break on v1.1 object-form documents. Readers MUST branch on string vs object.

### Policy

v1.x schema updates are additive only. Breaking changes require v2.

### Known Prompt Gaps (v0.3.0 backlog)

Discovered during dogfooding (examples/dogfood-2026-04-27.json):

1. `entity_refs` is required on Goal schema but only prompted when `goal_entity_refs` capability is active. Standard profile omits it, causing systematic empty-array fixups.
2. `temporal_refs` prompt fragment describes only `raw` and `resolved` fields. LLM hallucinates entity-like properties (`stated_at`, `resolved_at`) onto temporal refs because the full sub-schema shape is not prompted.
3. `goal_timing` prompt fragment leaks into entity output: LLM adds `stated_at`/`resolved_at` to entities, which are not in the entity schema.

## v0.1.1

Initial public release. SynaptExtraction IL v1 with schema, validation, finalization, and composable prompt system.

- 5 JSON schemas (extract, source-ref, embedding, assertion-signals, temporal-ref)
- Three-stage finalization pipeline
- 17 extraction capabilities with dependency closure
- 3 prompt profiles (minimal, standard, full)
- OIDC trusted publishing with Sigstore provenance (npm + PyPI)
