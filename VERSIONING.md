# Versioning Policy

This document describes how `@synapt-dev/extract` (npm) and `synapt-extract` (PyPI) version their releases, what each version bump implies for consumers, what we commit to supporting and for how long, and how we communicate behavioral changes.

## Semver discipline

Releases follow [Semantic Versioning 2.0.0](https://semver.org/):

| Version bump   | What it means                                                              | Consumer action                                |
|----------------|----------------------------------------------------------------------------|------------------------------------------------|
| Major (`X.y.z` -> `(X+1).0.0`)   | Breaking API or schema change                                  | Migration required; read migration guide       |
| Minor (`x.Y.z` -> `x.(Y+1).0`)   | Additive functionality (new fields, new sub-schemas, new capabilities) | Optional; existing code keeps working          |
| Patch (`x.y.Z` -> `x.y.(Z+1)`)   | Bug fix or behavior-preserving improvement                     | Recommended; should be a no-op for callers     |

The package version (`@synapt-dev/extract@x.y.z`) tracks the implementation. The schema version (`extract/v1.0`, `extract/v1.1`, etc.) tracks the contract shape. The two evolve independently but the relationship is documented (see Schema-Package Version Mapping below).

## v1.x additive-only policy

Within the v1 schema family, all changes are **additive only**. Specifically:

- New fields can be added (always optional; existing extractions remain valid)
- New sub-schemas can be added (referenced from the main schema; existing extractions remain valid)
- New capabilities can be added to `VALID_CAPABILITIES`
- New prompt fragments can be added to the registry

The following are **not** allowed within v1.x and require a v2 schema bump:

- Removing existing fields
- Changing the type or shape of existing fields
- Renaming existing fields, sub-schemas, or capabilities
- Adding required fields (since this would invalidate documents that don't include them)
- Tightening validation in ways that fail previously-valid documents

A v2 schema bump is a **major** package version change and requires a migration guide.

## Schema-package version mapping

Each package version ships a specific set of schema versions. Consumers who pin a package version know exactly which schemas they receive.

| Package version | Schema versions shipped |
|-----------------|-------------------------|
| `0.1.x`         | `extract/v1.0`, `source-ref/v1.0`, `embedding/v1.0`, `assertion-signals/v1.0`, `temporal-ref/v1.0` |
| `0.2.x`         | All of `0.1.x` + `producer/v1.1` (new sub-schema for typed `produced_by`) |
| `0.3.x` (v1.2)  | All of `0.2.x` + `entity/v1.0`, `goal/v1.0`, `question/v1.0`, `action/v1.0`, `decision/v1.0`, `sentiment/v1.0`, `source-metadata/v1.0` (entity and goal promoted from inline; five new sub-schemas) |

Each release's CHANGELOG includes the updated mapping row.

## Support window

Synapt commits to maintaining bug fixes for the current minor and the previous minor for at least **90 days** after the next minor ships. We aim to extend this to **180+ days** as the team and ecosystem mature.

Concretely, when `0.4.0` ships:

- `0.4.x` is fully supported
- `0.3.x` continues to receive critical bug fixes for at least 90 days
- `0.2.x` and earlier are no longer supported (recommended migration to `0.3.x` or `0.4.x`)

Critical bug fixes include: security vulnerabilities, data-integrity issues, regressions that break previously-correct extractions, and any issue blocking a partner's production deployment.

In practice we expect supported versions to receive fixes for considerably longer than the 90-day floor. The 90-day commitment is a **minimum**; production planning around `@synapt-dev/extract` should assume 6+ months of support per minor version unless we communicate otherwise.

## Forward-port commitment

When a bug is fixed in the latest minor, the same fix is also backported to all currently-supported previous minors. Specifically:

- A fix that lands as `0.5.3` also lands as `0.4.x` patches for any `0.4.x` versions in the support window.
- The CHANGELOG entry for the patch identifies which versions received the fix.
- If a fix cannot be backported (because the surrounding code has changed substantially), we document the reason and provide a migration recommendation in the CHANGELOG.

## Behavioral-shift CHANGELOG entries

Every release CHANGELOG entry includes a section identifying behavioral shifts: changes that may produce different output for the same input, even when no schema or API has changed.

Examples of behavioral shifts that require disclosure:

- Changes to default capability sets (which fields are populated by `profile: "standard"`)
- Changes to validator strictness (a previously-accepted document now fails validation)
- Changes to finalize pipeline behavior (e.g., capability detection logic, sub-schema version injection)

Each behavioral-shift entry includes a before/after example so consumers can assess impact before upgrading. This addresses the "version stayed same but output changed" failure mode.

## Deprecation notices

When a feature is scheduled for removal in a future major version, we publish a deprecation notice **at least 90 days before** the removal release.

Deprecation notices include:

- The feature being deprecated
- The package version that introduces the deprecation warning
- The package version that will remove the feature (typically the next major)
- The recommended replacement (or rationale if no replacement exists)
- A migration guide link

Deprecation notices appear in CHANGELOG entries, in the package's TypeScript type definitions (as `@deprecated` JSDoc), and in any runtime warnings the library emits when deprecated features are used.

## Pre-release versions

Pre-release versions follow the pattern `x.y.z-alpha.N`, `x.y.z-beta.N`, `x.y.z-rc.N`. These are not considered stable and may change without notice.

- **alpha**: internal testing; published to npm with `--tag alpha` (not `latest`); breaking changes allowed without version bump
- **beta**: public testing for a specific release line; published with `--tag beta`; breaking changes between beta versions noted in CHANGELOG
- **rc** (release candidate): API frozen; only critical bug fixes between rc and final; published with `--tag rc`

Production deployments should pin to stable releases (no pre-release suffix). Consumers running pre-releases accept the risk of changes between versions.

## Major version changes

A major version bump is a significant event for consumers. When we ship a major version, we provide:

- **Migration guide**: a separate doc covering each breaking change, the rationale, and the migration path
- **Blog post on synapt.dev**: explaining the change, the timeline, and what consumers should do
- **Side-by-side support window**: the previous major continues to receive critical fixes for at least 6 months after the new major ships, giving consumers time to migrate
- **Compatibility shims where possible**: helper functions or backwards-compat readers that help bridge between versions during migration

The team will not ship a major version without exhausting reasonable additive alternatives within the current major. Major version bumps are reserved for changes that genuinely cannot be expressed additively.

## Communication channels

The following are the canonical sources for version information:

- **CHANGELOG.md** in the source repository: per-version notes, behavioral shifts, deprecations, schema-package mapping
- **GitHub Releases**: formal release artifacts with notes, SBOM, signed attestations
- **npm registry** (`https://www.npmjs.com/package/@synapt-dev/extract`): published versions and metadata
- **PyPI** (`https://pypi.org/project/synapt-extract/`): same for Python
- **synapt.dev blog**: long-form coverage of major version changes and architectural shifts

CHANGELOG.md is the source of truth. Where information appears in multiple places, the CHANGELOG governs.

## Cross-references

- `SECURITY.md` -- supply-chain commitments, vulnerability disclosure, verification commands
- `SUPPORT.md` -- bug reporting channels, SLA tiers, reproducibility expectations
