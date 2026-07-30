/**
 * The version of this package, available at runtime.
 *
 * Consumers that record which extractor produced a document should read this
 * rather than hand-copying a version string, so the recorded value is evidence
 * of what ran instead of a claim about it.
 *
 * Deliberately a literal and not a read of `package.json`: the default entry
 * has to stay importable in browser and WASM hosts, and
 * `scripts/check-ts-universal-entry.mjs` fails the build if it reaches for a
 * Node built-in. `tests/test_version.ts` ties this constant to the manifest and
 * to the Python distribution, so a one-sided edit fails there rather than
 * shipping.
 *
 * Keep in step with `packages/ts/package.json` and
 * `packages/python/pyproject.toml`; `scripts/bump-version.sh` updates all three.
 */
export const VERSION = "0.6.0";
