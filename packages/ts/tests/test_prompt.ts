import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, test } from "vitest";

import { buildExtractionPrompt, resolveCapabilities } from "../src/prompt.js";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..", "..");
const CONFORMANCE_DIR = resolve(REPO_ROOT, "tests", "conformance");
const PACKAGE_PROMPTS_DIR = resolve(import.meta.dirname, "..", "prompts");
const SAMPLE_TEXT = "Please pray for my mom. She had surgery on April 20 and is recovering well.";

function loadJson<T>(...parts: string[]): T {
  return JSON.parse(readFileSync(resolve(...parts), "utf-8")) as T;
}

describe("resolveCapabilities", () => {
  test("accepts explicit capabilities", () => {
    expect(new Set(resolveCapabilities({ capabilities: ["entities", "goals", "themes"] })))
      .toEqual(new Set(["entities", "goals", "themes"]));
  });

  test.each([
    ["minimal", ["entities", "entity_state", "goals", "themes", "summary"], ["relations"]],
    ["standard", ["entities", "entity_context", "goal_timing", "facts", "temporal_refs", "sentiment", "evidence_anchoring"], ["relations"]],
    ["full", ["entity_ids", "goal_entity_refs", "relations", "relation_origin", "assertion_signals", "temporal_classes"], []],
  ])("resolves %s profile", (profile, expectedIncluded, expectedExcluded) => {
    const result = resolveCapabilities({ profile: profile as "minimal" | "standard" | "full" });
    for (const capability of expectedIncluded) {
      expect(result).toContain(capability);
    }
    for (const capability of expectedExcluded) {
      expect(result).not.toContain(capability);
    }
  });

  test.each([
    [{ profile: "minimal", add: ["relations"] }, ["entities", "relations", "entity_ids"]],
    [{ profile: "standard", remove: ["sentiment"] }, ["entities"], ["sentiment"]],
    [{ profile: "standard", add: ["relations"], remove: ["sentiment"] }, ["relations", "entity_ids"], ["sentiment"]],
  ])("supports profile modifiers %#", (options, expectedIncluded, expectedExcluded = []) => {
    const result = resolveCapabilities(options as Parameters<typeof resolveCapabilities>[0]);
    for (const capability of expectedIncluded) {
      expect(result).toContain(capability);
    }
    for (const capability of expectedExcluded) {
      expect(result).not.toContain(capability);
    }
  });

  test("rejects unknown profile", () => {
    expect(() => resolveCapabilities({ profile: "psychic" as never })).toThrow(/Unknown profile/);
  });

  test("rejects missing capabilities and profile", () => {
    expect(() => resolveCapabilities({})).toThrow();
  });

  test.each([
    [{ capabilities: ["bogus"] }, /Unknown capabilities/],
    [{ capabilities: ["entities"], add: ["psychic"] }, /Unknown capabilities/],
    [{ capabilities: ["bogus", "fake", "entities"] }, /bogus.*fake|fake.*bogus/],
    [{ capabilities: ["entities"], remove: ["entities"] }, /empty/],
    [{ capabilities: ["assertion_signals"] }, /base capability/],
    [{ capabilities: ["evidence_anchoring"] }, /base capability/],
  ])("rejects invalid capability inputs %#", (options, pattern) => {
    expect(() => resolveCapabilities(options as Parameters<typeof resolveCapabilities>[0])).toThrow(pattern);
  });

  test.each([
    [{ capabilities: ["entities", "assertion_signals"] }, ["entities", "assertion_signals"]],
    [{ capabilities: ["facts", "evidence_anchoring"] }, ["facts", "evidence_anchoring"]],
    [{ capabilities: ["entity_state"] }, ["entities", "entity_state"]],
    [{ capabilities: ["entity_context"] }, ["entities", "entity_context"]],
    [{ capabilities: ["entity_ids"] }, ["entities", "entity_ids"]],
    [{ capabilities: ["goal_timing"] }, ["goals", "goal_timing"]],
    [{ capabilities: ["goal_entity_refs"] }, ["goals", "entity_ids", "entities", "goal_entity_refs"]],
    [{ capabilities: ["temporal_classes"] }, ["temporal_refs", "temporal_classes"]],
    [{ capabilities: ["relations"] }, ["entities", "entity_ids", "relations"]],
    [{ capabilities: ["relation_origin"] }, ["entities", "entity_ids", "relations", "relation_origin"]],
  ])("applies dependency closure %#", (options, expectedIncluded) => {
    const result = resolveCapabilities(options as Parameters<typeof resolveCapabilities>[0]);
    for (const capability of expectedIncluded) {
      expect(result).toContain(capability);
    }
  });
});

describe("buildExtractionPrompt", () => {
  test("returns a non-empty string", () => {
    const result = buildExtractionPrompt(SAMPLE_TEXT, { profile: "minimal" });
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  test("contains text, extracted_at, and JSON instruction", () => {
    const result = buildExtractionPrompt(SAMPLE_TEXT, { profile: "minimal" });
    expect(result).toContain(SAMPLE_TEXT);
    expect(result).toContain("extracted_at");
    expect(result).toContain("JSON");
  });

  test.each([
    [{ capabilities: ["entities"] }, ['"entities"', '"name"', '"type"']],
    [{ capabilities: ["entities", "entity_state"] }, ['"state"']],
    [{ capabilities: ["goals"] }, ['"goals"', '"status"']],
    [{ capabilities: ["themes"] }, ['"themes"']],
    [{ capabilities: ["summary"] }, ['"summary"']],
    [{ capabilities: ["sentiment"] }, ['"sentiment"']],
    [{ capabilities: ["facts"] }, ['"facts"']],
    [{ capabilities: ["temporal_refs"] }, ['"temporal_refs"']],
    [{ capabilities: ["entities", "entity_ids", "relations"] }, ['"relations"']],
    [{ capabilities: ["entities", "assertion_signals"] }, ['signals']],
    [{ capabilities: ["entities", "evidence_anchoring"] }, ['source']],
  ])("includes expected fragments %#", (options, snippets) => {
    const result = buildExtractionPrompt(SAMPLE_TEXT, options as Parameters<typeof buildExtractionPrompt>[1]);
    for (const snippet of snippets) {
      expect(result).toContain(snippet);
    }
  });

  test("excludes absent capability fragments", () => {
    const result = buildExtractionPrompt(SAMPLE_TEXT, { capabilities: ["entities"] });
    expect(result).not.toContain('"relations"');
    expect(result).not.toContain('"sentiment"');
  });

  test("includes categories, source_type, and date", () => {
    const result = buildExtractionPrompt(SAMPLE_TEXT, {
      profile: "minimal",
      categories: ["Health", "Family"],
      source_type: "prayer",
      date: "2026-04-25",
    });
    expect(result).toContain("Health");
    expect(result).toContain("Family");
    expect(result).toContain("prayer");
    expect(result).toContain("2026-04-25");
  });

  test("uses date in temporal resolution prompt", () => {
    const result = buildExtractionPrompt(SAMPLE_TEXT, {
      capabilities: ["temporal_refs"],
      date: "2026-04-25",
    });
    expect(result).toContain("2026-04-25");
  });

  test.each([
    ["minimal", false, false],
    ["standard", false, true],
    ["full", true, true],
  ])("profile composition for %s", (profile, expectsRelations, expectsFacts) => {
    const result = buildExtractionPrompt(SAMPLE_TEXT, { profile: profile as "minimal" | "standard" | "full" });
    if (expectsRelations) {
      expect(result).toContain('"relations"');
    } else {
      expect(result).not.toContain('"relations"');
    }
    if (expectsFacts) {
      expect(result).toContain('"facts"');
    }
  });

  test("rejects specifying both profile and capabilities", () => {
    expect(() => buildExtractionPrompt(SAMPLE_TEXT, {
      profile: "minimal",
      capabilities: ["entities"],
    })).toThrow();
  });

  test("prevents template injection in categories and source_type", () => {
    const categoryResult = buildExtractionPrompt("hello", {
      capabilities: ["entities"],
      categories: ["A{{text}}B"],
    });
    expect(categoryResult).toContain("A{{text}}B");
    expect(categoryResult).not.toContain("AhelloB");

    const sourceTypeResult = buildExtractionPrompt("hello", {
      capabilities: ["entities"],
      source_type: "{{date}}",
    });
    expect(sourceTypeResult).toContain("{{date}}");
  });

  test("keeps preamble before fragments and text at the end", () => {
    const result = buildExtractionPrompt(SAMPLE_TEXT, { profile: "full" });
    expect(result.indexOf("Extract structured data")).toBeLessThan(result.indexOf('"entities"'));
    expect(result.indexOf("Rules:")).toBeLessThan(result.lastIndexOf(SAMPLE_TEXT));
    expect(result.indexOf('"entities"')).toBeLessThan(result.indexOf('"id"'));
  });
});

describe("prompt assets", () => {
  test("all fragment files exist and are non-empty", () => {
    const expected = [
      "preamble.txt", "postamble.txt",
      "entities.txt", "entity_state.txt", "entity_context.txt", "entity_ids.txt",
      "goals.txt", "goal_timing.txt", "goal_entity_refs.txt",
      "themes.txt", "summary.txt", "sentiment.txt", "facts.txt",
      "temporal_refs.txt", "temporal_classes.txt",
      "relations.txt", "relation_origin.txt",
      "assertion_signals.txt", "evidence_anchoring.txt",
    ];

    for (const name of expected) {
      const content = readFileSync(resolve(PACKAGE_PROMPTS_DIR, "v1", name), "utf-8").trim();
      expect(content.length, name).toBeGreaterThan(0);
    }
  });

  test("profile files exist and contain the expected capability sets", () => {
    for (const name of ["minimal", "standard", "full"]) {
      const data = loadJson<{ capabilities: string[] }>(PACKAGE_PROMPTS_DIR, "profiles", `${name}.json`);
      expect(Array.isArray(data.capabilities), name).toBe(true);
    }

    const minimal = new Set(loadJson<{ capabilities: string[] }>(PACKAGE_PROMPTS_DIR, "profiles", "minimal.json").capabilities);
    expect(minimal).toEqual(new Set(["entities", "entity_state", "goals", "themes", "summary"]));

    const standard = new Set(loadJson<{ capabilities: string[] }>(PACKAGE_PROMPTS_DIR, "profiles", "standard.json").capabilities);
    for (const capability of [
      "entities",
      "entity_context",
      "goal_timing",
      "facts",
      "temporal_refs",
      "sentiment",
      "evidence_anchoring",
    ]) {
      expect(standard.has(capability), capability).toBe(true);
    }

    const full = new Set(loadJson<{ capabilities: string[] }>(PACKAGE_PROMPTS_DIR, "profiles", "full.json").capabilities);
    for (const capability of standard) {
      expect(full.has(capability), capability).toBe(true);
    }
    expect(full).toEqual(new Set([
      "entities", "entity_state", "entity_context", "entity_ids",
      "goals", "goal_timing", "goal_entity_refs",
      "themes", "summary", "sentiment", "facts",
      "temporal_refs", "temporal_classes",
      "relations", "relation_origin",
      "assertion_signals", "evidence_anchoring",
    ]));
  });
});

describe("prompt conformance fixtures", () => {
  test("matches shared prompt fixtures", () => {
    const cases = loadJson<Array<{
      name: string;
      text: string;
      options: Record<string, unknown>;
      expected_capabilities: string[];
      prompt_includes: string[];
      prompt_excludes: string[];
    }>>(CONFORMANCE_DIR, "prompt_cases.json");

    for (const fixture of cases) {
      const resolveOptions = Object.fromEntries(
        Object.entries(fixture.options).filter(([key]) => ["capabilities", "profile", "add", "remove"].includes(key)),
      );
      const resolved = resolveCapabilities(resolveOptions as Parameters<typeof resolveCapabilities>[0]);
      expect(resolved, fixture.name).toEqual(fixture.expected_capabilities);

      const prompt = buildExtractionPrompt(
        fixture.text,
        fixture.options as Parameters<typeof buildExtractionPrompt>[1],
      );
      for (const snippet of fixture.prompt_includes) {
        expect(prompt.includes(snippet), `${fixture.name}:${snippet}`).toBe(true);
      }
      for (const snippet of fixture.prompt_excludes) {
        expect(prompt.includes(snippet), `${fixture.name}:${snippet}`).toBe(false);
      }
    }
  });
});
