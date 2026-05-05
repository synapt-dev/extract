# Changelog

## v0.3.1

Three rounds of Atlas adversarial review. Schema/runtime parity, artifact bundling, no-network CI guard hardening, Python schema self-containment, behavioral-shift documentation, doc corrections.

### Behavioral shifts

v0.3.1 tightens both JSON Schema constraints and runtime validators. **7 of 9 changes** are schema-only parity: the runtime was already this strict in v0.3.0, and the schema now matches. Documents passing v0.3.0 runtime continue passing v0.3.1 for these 7 fields.

**2 fields are runtime tightenings** that may reject documents previously accepted by v0.3.0:

- **`action.due`**: v0.3.0 runtime accepted any string; v0.3.1 requires ISO 8601 date/datetime. Documents with free-form `due` values (e.g. `"tomorrow"`, `"next week"`) will now fail both schema and runtime validation.
- **`source_metadata.version`**: v0.3.0 runtime did not require `version` on `source_metadata`; v0.3.1 requires `version: "1"`. Documents with `source_metadata: {}` or `source_metadata` missing the `version` field will now fail.

**Upgrade advice:** consumers on v0.3.0 should audit stored extractions for non-ISO `action.due` values and missing `source_metadata.version` before upgrading to v0.3.1.

| Field | Change | v0.3.0 runtime was already this strict? |
|-------|--------|----------------------------------------|
| `sentiment.version` | Schema now **requires** `version` (was optional) | **Yes.** Both runtimes required `version: "1"` since v0.3.0. |
| `source_metadata.version` | Schema + runtime now **require** `version` | **No. Runtime tightening.** v0.3.0 runtime accepted `source_metadata` without `version`. |
| `action.due` | Schema + runtime now require **ISO 8601** pattern | **No. Runtime tightening.** v0.3.0 runtime only checked optional string, not ISO format. |
| `decision.decided_at` | Schema now requires ISO 8601 pattern | **Yes.** Both runtimes validated ISO 8601 since v0.3.0. |
| `goal.stated_at` | Schema now requires ISO 8601 pattern | **Yes.** Both runtimes validated ISO 8601 since v0.3.0. |
| `goal.resolved_at` | Schema now requires ISO 8601 pattern | **Yes.** Both runtimes validated ISO 8601 since v0.3.0. |
| `temporal_ref.resolved` | Schema now requires ISO 8601 pattern | **Yes.** Both runtimes validated ISO 8601 since v0.3.0. |
| `temporal_ref.resolved_end` | Schema now requires ISO 8601 pattern | **Yes.** Both runtimes validated ISO 8601 since v0.3.0. |
| `temporal_ref` (type=unresolved) | `resolved`/`resolved_end` now **forbidden** via if/then/not | **Yes.** Both runtimes rejected these since v0.3.0. |

### Schema bundling (critical)

- JSON schema files (`schemas/`) now bundled in both npm and PyPI packages
- npm `files` includes `schemas`; Python `package-data` includes `schemas/**/*.json`
- Schemas committed into Python package source for standalone buildability
- CI drift-detection step verifies Python package schemas match root schemas
- CI assertion verifies built wheel contains exactly 13 schema JSON files

### Schema/runtime parity (high)

- `sentiment/v1.json`: `version` now required (was optional in schema, required in runtime)
- `decision/v1.json`: `decided_at` now enforces ISO 8601 pattern (was any string in schema)
- `goal/v1.json`: `stated_at` and `resolved_at` now enforce ISO 8601 pattern
- `temporal-ref/v1.json`: `resolved_end` required when `type` is `"range"` (if/then constraint)
- `temporal-ref/v1.json`: `resolved` and `resolved_end` now enforce ISO 8601 pattern
- `temporal-ref/v1.json`: `resolved` and `resolved_end` forbidden when `type` is `"unresolved"` (if/then/not constraint)
- `action/v1.json`: `due` now enforces ISO 8601 pattern (was any string)
- `source-metadata/v1.json`: `version` now required (was optional in schema, required in TS type)

### Runtime validation tightened

- `action.due` validated as ISO 8601 in both TS and Python runtimes (was unchecked string)
- `source_metadata.version` required in TS and Python runtimes (was conditionally checked)

### No-network CI guard (high)

- Best-effort regex forbidden API scanner (`scripts/check-no-network.mjs`, `scripts/check-no-network.py`)
- Scans source, compiled dist, and packed artifact on every CI run
- Detects: direct forbidden globals, computed property access on global objects, string concatenation that assembles forbidden names, array `.join("")` assembling forbidden names, `Reflect.get` on global objects, `Function()` constructor (with or without `new`), base64 decode, dynamic imports, forbidden module imports, `importlib.import_module`
- Runtime dependency allowlist (`scripts/allowed-deps.json`) with CI enforcement
- Negative test fixtures (`tests/security-probes/`) for all 4 Atlas bypass probes: `Reflect.get(globalThis, "fetch")`, `Function("return 1")`, `["fe","tch"].join("")`, `importlib.import_module("http.client")`

### Doc corrections (moderate)

- SECURITY.md: reproducible builds section clarifies wheel byte-identity vs sdist content-equivalence
- SECURITY.md: callback architecture marked as proposed (target v0.4.0), not shipped
- SECURITY.md: forbidden API enforcement described as best-effort regex (not AST-aware)
- docs/callback-signature.md: status changed to PROPOSED, target v0.4.0
- README.md: install strings updated to 0.3.1
- Schema URL smoke gate (`scripts/check-schema-urls.sh`) verifies against GitHub Pages source (`raw.githubusercontent.com`) rather than live CDN; Cloudflare blocks GitHub Actions datacenter IPs. Live CDN verification deferred to v0.3.2 (requires Cloudflare allowlist).

### Conformance

- 22 validation conformance cases, 2 prompt cases, 2 finalize cases (26 total; was 14 validation + 4 other = 18 total in v0.3.0)

## v0.3.0

v1.2 spec: 8 new extraction fields, 5 new sub-schemas, sentiment dual-shape, entity/goal sub-schema promotion.

### New sub-schemas

- `schemas/entity/v1.json` -- promoted from inline `$defs` to standalone with `$id`
- `schemas/goal/v1.json` -- promoted from inline `$defs` to standalone with `$id`
- `schemas/question/v1.json` -- questions raised in source text
- `schemas/action/v1.json` -- concrete next-steps with origin tracking (extracted vs proposed_from_goals)
- `schemas/decision/v1.json` -- directional commitments identified in source
- `schemas/sentiment/v1.json` -- structured sentiment with valence/intensity/confidence
- `schemas/source-metadata/v1.json` -- source document metadata (token count, modality, format)

### New extraction fields

- `keywords`: surface lexical terms (sibling to themes; keywords are specific terms, themes are topical categories)
- `questions`: questions raised in the source text, with optional `directed_to` entity ref
- `actions`: action items with required `origin` field ("extracted" or "proposed_from_goals")
- `decisions`: directional commitments with optional `decided_at` timestamp
- `language`: IETF BCP 47 language tag (e.g. "en-US", "es", "pt-BR")
- `source_metadata`: source document metadata for normalization across lengths and formats
- `confidence`: extraction-level overall confidence score (0.0 to 1.0)
- `sentiment` dual-shape: accepts string (v1.0) or structured SynaptSentiment object (v1.2)

### New capabilities

`keywords`, `structured_sentiment`, `questions`, `actions`, `decisions`, `language`, `source_metadata`, `confidence` (8 new, 25 total)

### Entity enhancements

- `aliases` field on entities for per-extraction same-entity grouping

### Prompt system

- 8 new prompt fragment files for all new capabilities
- `CAPABILITY_RULES` for structured_sentiment, actions, and keywords
- Updated full.json profile with all 25 capabilities
- `CANONICAL_ORDER` updated for deterministic prompt composition

### Finalize pipeline

- `detectCapabilities` detects all 8 new capabilities from payload structure
- Sub-schema version injection for questions, actions, decisions, sentiment object, and source_metadata
- Evidence anchoring and assertion signals detection extended to questions, actions, and decisions

### Validators

- Entity ID cross-referencing extended to actions and decisions (dangling entity_ref detection)
- 5 new sub-schema validators with additionalProperties enforcement
- Sentiment dual-shape dispatch (string vs object)
- BCP 47 language tag validation
- Confidence bounds validation (0.0 to 1.0)

### Tests

- 227 TypeScript tests, 282 Python tests, 14 conformance cases
- 9 new conformance fixtures for v1.2 fields

### Compatibility

v1.2 is additive. v1.0 and v1.1 documents remain valid. The `sentiment` field now accepts either a string (v1.0) or a SynaptSentiment object (v1.2). Readers MUST branch on string vs object, same pattern as `produced_by`.

### Prompt gap fixes (from v0.2.0)

Three gaps discovered during v0.2.0 dogfooding are now fixed:

1. `goals.txt` now always mentions `entity_refs` as a required field (was only prompted with `goal_entity_refs` capability)
2. `temporal_refs.txt` now describes the full sub-schema shape (`type`, `resolved_end`, `context`) instead of just `raw` and `resolved`
3. `goal_timing.txt` now explicitly scopes `stated_at`/`resolved_at` to goals only, preventing LLM from adding these to entities

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
