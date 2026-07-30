import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, test } from "vitest";

import { VERSION } from "../src/index.js";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..", "..");

function packageJsonVersion(...parts: string[]): string {
  const pkg = JSON.parse(readFileSync(resolve(...parts), "utf-8")) as { version?: string };
  if (typeof pkg.version !== "string") throw new Error(`no version in ${resolve(...parts)}`);
  return pkg.version;
}

/** The `version = "x.y.z"` line from a pyproject, without adding a TOML dependency. */
function pyprojectVersion(path: string): string {
  const match = /^version\s*=\s*"([^"]+)"/m.exec(readFileSync(path, "utf-8"));
  if (match === null) throw new Error(`no version line in ${path}`);
  return match[1];
}

/**
 * WHY THIS FILE EXISTS
 *
 * A consumer that records which extractor produced a document needs to obtain
 * the version FROM the package. Before `VERSION` existed there was no way to,
 * so the only option we offered was hand-copying a string into a constant --
 * and a hand-copied version is a claim about the runtime, not evidence of it.
 * A downstream consumer did exactly that and its copy went stale (declared
 * 0.5.0 against a 0.6.0 runtime) with nothing able to detect it.
 *
 * `VERSION` is a literal rather than a read of package.json on purpose: the
 * default entry must stay importable in browser/WASM hosts, and
 * `scripts/check-ts-universal-entry.mjs` fails the build if it reaches for a
 * Node built-in. That is the same trade the embedded prompt fragments make, so
 * it carries the same obligation -- an embedded copy needs a test that fails
 * when it drifts from its source. That is what this is.
 */
describe("VERSION", () => {
  test("matches the TypeScript package manifest", () => {
    expect(VERSION).toBe(packageJsonVersion(REPO_ROOT, "packages", "ts", "package.json"));
  });

  test("matches the Python distribution version", () => {
    // The two language surfaces ship as one product at one version. Nothing
    // enforced that before this test: they were two hand-edited numbers that
    // happened to agree.
    expect(VERSION).toBe(pyprojectVersion(resolve(REPO_ROOT, "packages", "python", "pyproject.toml")));
  });

  test("is a bare semver triple, not a range or a specifier", () => {
    // Guards the shape a consumer stamps into provenance. "^0.6.0" or
    // "@synapt-dev/extract@0.6.0" would each be a plausible thing to paste
    // here and each would corrupt the recorded value.
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
