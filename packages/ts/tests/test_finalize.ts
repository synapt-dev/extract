import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, test } from "vitest";

import { finalizeExtraction } from "../src/finalize.js";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..", "..");
const CONFORMANCE_DIR = resolve(REPO_ROOT, "tests", "conformance");

function loadJson<T>(...parts: string[]): T {
  return JSON.parse(readFileSync(resolve(...parts), "utf-8")) as T;
}

function llmOutput(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    extracted_at: "2026-04-26T00:00:00Z",
    entities: [{ name: "Mom", type: "person" }],
    goals: [{ text: "Recovery", status: "open", entity_refs: [] }],
    themes: ["Health"],
    ...overrides,
  };
}

describe("finalizeExtraction", () => {
  test.each([
    ["injects version", {}, "version", "1"],
    ["injects produced_by", {}, "produced_by", "openai://gpt-4o-mini"],
    ["injects user_id", { user_id: "u123" }, "user_id", "u123"],
    ["injects source_id", { source_id: "prayer-001" }, "source_id", "prayer-001"],
    ["injects kind", { kind: "conversa/prayer" }, "kind", "conversa/prayer"],
  ])("%s", (_name, extraContext, field, expected) => {
    const result = finalizeExtraction(
      llmOutput(),
      { produced_by: "openai://gpt-4o-mini", ...extraContext },
    );
    expect(result.extraction[field as keyof typeof result.extraction]).toBe(expected);
  });

  test("injects extensions and extension versions", () => {
    const result = finalizeExtraction(llmOutput(), {
      produced_by: "test://model",
      extensions: { "conversa/prayer": { category: "Health" } },
    });
    expect(result.extraction.extensions?.["conversa/prayer"]).toEqual({
      version: "1",
      category: "Health",
    });
  });

  test("injects multiple extension versions and preserves scalar extensions", () => {
    const result = finalizeExtraction(llmOutput(), {
      produced_by: "test://model",
      extensions: {
        "conversa/prayer": { category: "Health" },
        "conversa/sermon": { topic: "Grace" },
        "conversa/scalar": "simple_string",
      },
    });
    expect(result.extraction.extensions?.["conversa/prayer"]).toMatchObject({ version: "1" });
    expect(result.extraction.extensions?.["conversa/sermon"]).toMatchObject({ version: "1" });
    expect(result.extraction.extensions?.["conversa/scalar"]).toBe("simple_string");
  });

  test("injects embedding version and auto-populates dimensions", () => {
    const result = finalizeExtraction(llmOutput(), {
      produced_by: "test://model",
      embeddings: [{
        vector: [0.1, 0.2, 0.3, 0.4],
        model: "openai://text-embedding-3-small",
        input: "source",
      }],
    });
    expect(result.extraction.embeddings?.[0]).toMatchObject({
      version: "1",
      dimensions: 4,
    });
  });

  test.each([
    [
      "injects source ref version",
      llmOutput({ entities: [{ name: "Mom", type: "person", source: { snippet: "My mom" } }] }),
      ["entities", 0, "source", "version"],
      "1",
    ],
    [
      "injects signals version",
      llmOutput({ entities: [{ name: "Mom", type: "person", signals: { confidence: 0.9 } }] }),
      ["entities", 0, "signals", "version"],
      "1",
    ],
    [
      "injects temporal ref version",
      llmOutput({ temporal_refs: [{ raw: "next week", type: "range" }] }),
      ["temporal_refs", 0, "version"],
      "1",
    ],
    [
      "injects goal source version",
      llmOutput({ goals: [{ text: "Recovery", status: "open", entity_refs: [], source: { snippet: "I hope she recovers" } }] }),
      ["goals", 0, "source", "version"],
      "1",
    ],
    [
      "injects fact source version",
      llmOutput({ facts: [{ text: "Surgery happened", source: { snippet: "Mom had surgery" } }] }),
      ["facts", 0, "source", "version"],
      "1",
    ],
    [
      "injects relation signals version",
      llmOutput({
        entities: [
          {
            id: "e1",
            name: "Mom",
            type: "person",
            relations: [{ target: "e2", type: "parent_of", signals: { confidence: 0.8 } }],
          },
          { id: "e2", name: "Me", type: "person" },
        ],
      }),
      ["entities", 0, "relations", 0, "signals", "version"],
      "1",
    ],
  ])("%s", (_name, doc, path, expected) => {
    const result = finalizeExtraction(doc, { produced_by: "test://model" });
    let current: unknown = result.extraction;
    for (const key of path) {
      current = (current as Record<string, unknown> | unknown[])[key as never];
    }
    expect(current).toBe(expected);
  });

  test("strips empty source and version-only signals", () => {
    const result = finalizeExtraction(llmOutput({
      entities: [{ name: "Mom", type: "person", source: {}, signals: { version: "1" } }],
    }), { produced_by: "test://model" });
    expect(result.extraction.entities[0]).not.toHaveProperty("source");
    expect(result.extraction.entities[0]).not.toHaveProperty("signals");
  });

  test.each([
    ["entities", llmOutput(), "entities"],
    ["entity_state", llmOutput({ entities: [{ name: "Mom", type: "person", state: "recovering" }] }), "entity_state"],
    ["entity_ids", llmOutput({ entities: [{ id: "e1", name: "Mom", type: "person" }] }), "entity_ids"],
    [
      "relations",
      llmOutput({
        entities: [
          { id: "e1", name: "Mom", type: "person", relations: [{ target: "e2", type: "knows" }] },
          { id: "e2", name: "Dad", type: "person" },
        ],
      }),
      "relations",
    ],
    ["evidence_anchoring", llmOutput({ entities: [{ name: "Mom", type: "person", source: { snippet: "My mom" } }] }), "evidence_anchoring"],
    ["assertion_signals", llmOutput({ entities: [{ name: "Mom", type: "person", signals: { confidence: 0.9 } }] }), "assertion_signals"],
    ["goal_timing", llmOutput({ goals: [{ text: "Recovery", status: "open", entity_refs: [], stated_at: "2026-04-20" }] }), "goal_timing"],
    ["summary", llmOutput({ summary: "A prayer for healing.", sentiment: "hopeful" }), "summary"],
    ["sentiment", llmOutput({ summary: "A prayer for healing.", sentiment: "hopeful" }), "sentiment"],
    [
      "temporal_classes",
      llmOutput({ temporal_refs: [{ raw: "April 20 to May 1", type: "range", resolved: "2026-04-20", resolved_end: "2026-05-01" }] }),
      "temporal_classes",
    ],
  ])("detects capability %s", (_name, doc, capability) => {
    const result = finalizeExtraction(doc, { produced_by: "test://model" });
    expect(result.extraction.capabilities).toContain(capability);
  });

  test("warns on mismatched capabilities hint", () => {
    const result = finalizeExtraction(llmOutput(), {
      produced_by: "test://model",
      capabilities_hint: ["relations"],
    });
    expect(result.warnings.some((warning) => warning.includes("relations"))).toBe(true);
  });

  test("warns on goal entity refs without entity_ids", () => {
    const result = finalizeExtraction(
      llmOutput({
        entities: [{ name: "Mom", type: "person" }],
        goals: [{ text: "Recovery", status: "open", entity_refs: ["e1"] }],
      }),
      { produced_by: "test://model" },
    );
    expect(result.warnings.some((warning) => warning.includes("entity_ids"))).toBe(true);
  });

  test("reports dangling entity refs in validation result", () => {
    const result = finalizeExtraction(
      llmOutput({
        entities: [{ name: "Mom", type: "person" }],
        goals: [{ text: "Recovery", status: "open", entity_refs: ["e_missing"] }],
      }),
      { produced_by: "openai://gpt-4o-mini" },
    );
    expect(result.validation.valid).toBe(false);
    expect(result.validation.errors.some((error) => error.path.includes("entity_refs"))).toBe(true);
  });

  test("reports malformed embeddings in validation result", () => {
    const result = finalizeExtraction(llmOutput(), {
      produced_by: "openai://gpt-4o-mini",
      embeddings: [{
        vector: [0.1, 0.2],
        model: "not-a-uri",
        input: "source",
        dimensions: 99,
      }],
    });
    expect(result.validation.valid).toBe(false);
    expect(result.validation.errors.some((error) => error.path === "embeddings[0].model")).toBe(true);
    expect(result.validation.errors.some((error) => error.path === "embeddings[0].dimensions")).toBe(true);
  });

  test("passes end-to-end finalization", () => {
    const result = finalizeExtraction(
      llmOutput({
        entities: [{
          id: "e1",
          name: "Mom",
          type: "person",
          state: "recovering",
          source: { snippet: "My mom is recovering" },
          signals: { confidence: 0.9 },
        }],
        goals: [{
          text: "Full recovery",
          status: "open",
          entity_refs: ["e1"],
          stated_at: "2026-04-20",
        }],
        facts: [{ text: "Had surgery April 20", source: { snippet: "surgery" } }],
        summary: "Prayer for mom's recovery.",
        sentiment: "hopeful",
      }),
      {
        produced_by: "openai://gpt-4o-mini",
        user_id: "user_123",
        source_id: "prayer-001",
        kind: "conversa/prayer",
        extensions: { "conversa/prayer": { category: "Health" } },
      },
    );
    expect(result.validation.valid).toBe(true);
    expect(result.extraction.version).toBe("1");
    expect(result.extraction.produced_by).toBe("openai://gpt-4o-mini");
    expect(result.extraction.capabilities).toContain("entities");
    expect(result.extraction.capabilities).toContain("evidence_anchoring");
  });
});

describe("finalizeExtraction conformance fixtures", () => {
  test("matches shared finalization fixtures", () => {
    const cases = loadJson<Array<{
      name: string;
      llm_output: Record<string, unknown>;
      context: Record<string, unknown>;
      expected_valid: boolean;
      expected_capabilities?: string[];
      expected_error_paths?: string[];
    }>>(CONFORMANCE_DIR, "finalize_cases.json");

    for (const fixture of cases) {
      const result = finalizeExtraction(
        fixture.llm_output,
        fixture.context as Parameters<typeof finalizeExtraction>[1],
      );
      expect(result.validation.valid, fixture.name).toBe(fixture.expected_valid);
      if (fixture.expected_valid) {
        expect(result.extraction.version, fixture.name).toBe("1");
        expect(result.extraction.capabilities).toEqual(expect.arrayContaining(fixture.expected_capabilities ?? []));
      } else {
        for (const path of fixture.expected_error_paths ?? []) {
          expect(result.validation.errors.some((error) => error.path === path), `${fixture.name}:${path}`).toBe(true);
        }
      }
    }
  });
});
