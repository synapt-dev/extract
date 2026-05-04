import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { describe, expect, test } from "vitest";

import { validateExtraction } from "../src/validate.js";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..", "..");
const CONFORMANCE_DIR = resolve(REPO_ROOT, "tests", "conformance");
const SCHEMAS_DIR = resolve(REPO_ROOT, "schemas");
const ALL_CAPABILITIES = [
  "entities", "entity_state", "entity_context", "entity_ids",
  "goals", "goal_timing", "goal_entity_refs",
  "themes", "keywords", "summary", "sentiment", "structured_sentiment",
  "facts", "questions", "actions", "decisions",
  "temporal_refs", "temporal_classes",
  "relations", "relation_origin",
  "assertion_signals", "evidence_anchoring",
  "language", "source_metadata", "confidence",
];

function loadJson<T>(...parts: string[]): T {
  return JSON.parse(readFileSync(resolve(...parts), "utf-8")) as T;
}

function minimalExtraction(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    version: "1",
    extracted_at: "2026-04-26T00:00:00Z",
    produced_by: "openai://gpt-4o-mini",
    entities: [],
    goals: [],
    themes: [],
    capabilities: ["entities", "goals", "themes"],
    ...overrides,
  };
}

describe("validateExtraction", () => {
  test("accepts a minimal extraction", () => {
    const result = validateExtraction(minimalExtraction());
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual([]);
  });

  test("accepts a full extraction", () => {
    const result = validateExtraction(minimalExtraction({
      entities: [
        {
          id: "e1",
          name: "Mom",
          type: "person",
          state: "recovering",
          context: "family member",
          date_hint: "2026-04-20",
          source: { version: "1", snippet: "My mom is recovering" },
          signals: { version: "1", confidence: 0.9, negated: false },
          relations: [{ target: "e2", type: "parent_of" }],
        },
        {
          id: "e2",
          name: "Surgery",
          type: "event",
        },
      ],
      goals: [
        {
          text: "Mom's full recovery",
          status: "open",
          entity_refs: ["e1"],
          stated_at: "2026-04-20T10:00:00Z",
          source: { version: "1", snippet: "I hope mom recovers" },
          signals: { version: "1", hedged: true },
        },
      ],
      themes: ["Health", "Family"],
      summary: "Prayer for mom's recovery after surgery.",
      sentiment: "hopeful",
      facts: [
        {
          text: "Mom had surgery on April 20",
          category: "Health",
          source: { version: "1", snippet: "Mom had surgery" },
        },
      ],
      temporal_refs: [
        {
          version: "1",
          raw: "April 20",
          type: "point",
          resolved: "2026-04-20",
        },
      ],
      capabilities: [
        "entities", "entity_state", "entity_context", "entity_ids",
        "goals", "goal_timing", "goal_entity_refs",
        "themes", "summary", "sentiment", "facts",
        "temporal_refs", "temporal_classes",
        "relations", "assertion_signals", "evidence_anchoring",
      ],
    }));
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual([]);
  });

  test.each([
    ["version"],
    ["extracted_at"],
    ["produced_by"],
    ["entities"],
    ["goals"],
    ["themes"],
    ["capabilities"],
  ])("rejects missing required field %s", (field) => {
    const doc = minimalExtraction();
    delete doc[field];
    const result = validateExtraction(doc);
    expect(result.valid).toBe(false);
    expect(result.errors.some((error) => error.path === field)).toBe(true);
  });

  test("rejects non-object input", () => {
    const result = validateExtraction("not an object");
    expect(result.valid).toBe(false);
    expect(result.errors[0]?.message).toBe("must be an object");
  });

  test("rejects null input", () => {
    const result = validateExtraction(null);
    expect(result.valid).toBe(false);
  });

  test.each([
    [
      "entity missing name",
      minimalExtraction({ entities: [{ type: "person" }] }),
      "entities[0].name",
    ],
    [
      "entity missing type",
      minimalExtraction({ entities: [{ name: "Mom" }] }),
      "entities[0].type",
    ],
    [
      "entity bad source version",
      minimalExtraction({
        entities: [{ name: "Mom", type: "person", source: { version: "2", snippet: "test" } }],
      }),
      "entities[0].source.version",
    ],
    [
      "entity bad confidence",
      minimalExtraction({
        entities: [{ name: "Mom", type: "person", signals: { version: "1", confidence: 1.5 } }],
      }),
      "entities[0].signals.confidence",
    ],
    [
      "entity relation missing target",
      minimalExtraction({
        entities: [{ name: "Mom", type: "person", relations: [{ type: "knows" }] }],
      }),
      "entities[0].relations[0].target",
    ],
  ])("%s", (_name, doc, path) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(false);
    expect(result.errors.some((error) => error.path === path)).toBe(true);
  });

  test.each([
    [
      "goal missing text",
      minimalExtraction({ goals: [{ status: "open", entity_refs: [] }] }),
      "goals[0].text",
    ],
    [
      "goal invalid status",
      minimalExtraction({ goals: [{ text: "recover", status: "pending", entity_refs: [] }] }),
      "goals[0].status",
    ],
    [
      "goal missing entity_refs",
      minimalExtraction({ goals: [{ text: "recover", status: "open" }] }),
      "goals[0].entity_refs",
    ],
  ])("%s", (_name, doc, path) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(false);
    expect(result.errors.some((error) => error.path === path)).toBe(true);
  });

  test("rejects unknown capability", () => {
    const result = validateExtraction(minimalExtraction({
      capabilities: ["entities", "psychic_powers"],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((error) => error.message.includes("psychic_powers"))).toBe(true);
  });

  test("accepts all valid capabilities", () => {
    const result = validateExtraction(minimalExtraction({ capabilities: ALL_CAPABILITIES }));
    expect(result.valid).toBe(true);
  });

  test.each([
    [
      "valid embedding",
      minimalExtraction({
        embeddings: [{
          version: "1",
          vector: [0.1, 0.2, 0.3],
          model: "openai://text-embedding-3-small",
          input: "source",
          dimensions: 3,
        }],
      }),
      true,
      [],
    ],
    [
      "embedding missing vector",
      minimalExtraction({
        embeddings: [{
          version: "1",
          model: "openai://text-embedding-3-small",
          input: "source",
          dimensions: 3,
        }],
      }),
      false,
      ["embeddings[0].vector"],
    ],
    [
      "embedding dimensions mismatch",
      minimalExtraction({
        embeddings: [{
          version: "1",
          vector: [0.1, 0.2],
          model: "openai://text-embedding-3-small",
          input: "source",
          dimensions: 99,
        }],
      }),
      false,
      ["embeddings[0].dimensions"],
    ],
    [
      "embedding model requires scheme",
      minimalExtraction({
        embeddings: [{
          version: "1",
          vector: [0.1, 0.2],
          model: "text-embedding-3-small",
          input: "source",
          dimensions: 2,
        }],
      }),
      false,
      ["embeddings[0].model"],
    ],
  ])("%s", (_name, doc, expectedValid, paths) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(expectedValid);
    for (const path of paths) {
      expect(result.errors.some((error) => error.path === path)).toBe(true);
    }
  });

  test.each([
    [
      "valid temporal ref",
      minimalExtraction({
        temporal_refs: [{ version: "1", raw: "next Tuesday", type: "point", resolved: "2026-04-28" }],
      }),
      true,
      [],
    ],
    [
      "invalid temporal type",
      minimalExtraction({
        temporal_refs: [{ version: "1", raw: "sometime", type: "vague" }],
      }),
      false,
      ["temporal_refs[0].type"],
    ],
    [
      "range without resolved_end",
      minimalExtraction({
        temporal_refs: [{ version: "1", raw: "April 20 to May 1", type: "range", resolved: "2026-04-20" }],
      }),
      false,
      ["temporal_refs[0].resolved_end"],
    ],
    [
      "unresolved with resolved",
      minimalExtraction({
        temporal_refs: [{ version: "1", raw: "someday", type: "unresolved", resolved: "2026-04-20" }],
      }),
      false,
      ["temporal_refs[0].resolved"],
    ],
    [
      "unresolved with resolved_end",
      minimalExtraction({
        temporal_refs: [{ version: "1", raw: "someday", type: "unresolved", resolved_end: "2026-05-01" }],
      }),
      false,
      ["temporal_refs[0].resolved_end"],
    ],
  ])("%s", (_name, doc, expectedValid, paths) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(expectedValid);
    for (const path of paths) {
      expect(result.errors.some((error) => error.path === path)).toBe(true);
    }
  });

  test.each([
    ["gpt-4o-mini", false],
    ["openai://gpt-4o-mini", true],
    ["anthropic://claude-sonnet-4-20250514", true],
    ["", false],
  ])("validates produced_by format: %s", (producedBy, expectedValid) => {
    const result = validateExtraction(minimalExtraction({ produced_by: producedBy }));
    expect(result.valid).toBe(expectedValid);
  });

  test("accepts v1.0 string produced_by for backwards compatibility", () => {
    const result = validateExtraction(minimalExtraction({
      produced_by: "anthropic://claude-sonnet-4-6",
    }));
    expect(result.valid).toBe(true);
  });

  test("accepts minimal v1.1 structured producer", () => {
    const result = validateExtraction(minimalExtraction({
      produced_by: {
        version: "1",
        model: "anthropic://claude-sonnet-4-6",
      },
    }));
    expect(result.valid).toBe(true);
  });

  test("accepts full v1.1 structured producer", () => {
    const result = validateExtraction(minimalExtraction({
      produced_by: {
        version: "1",
        model: "anthropic://claude-sonnet-4-6",
        model_version: "claude-sonnet-4-6-20250514",
        deployment: "bedrock",
        configuration: {
          reasoning_effort: "high",
          system_prompt_hash: "abc123",
          temperature: 0.2,
          top_p: 0.95,
          max_tokens: 2048,
          vendor_flag: true,
        },
        operator: "synapt-dev",
        signature: "eyJhbGciOiJIUzI1NiJ9.payload.signature",
      },
    }));
    expect(result.valid).toBe(true);
  });

  test.each([
    [
      "missing version",
      { model: "anthropic://claude-sonnet-4-6" },
      "produced_by.version",
    ],
    [
      "missing model",
      { version: "1" },
      "produced_by.model",
    ],
    [
      "unknown root field",
      { version: "1", model: "anthropic://claude-sonnet-4-6", extra_field: "boom" },
      "produced_by.extra_field",
    ],
    [
      "malformed model uri",
      { version: "1", model: "claude-sonnet-4-6" },
      "produced_by.model",
    ],
    [
      "non-string signature",
      { version: "1", model: "anthropic://claude-sonnet-4-6", signature: { alg: "HS256" } },
      "produced_by.signature",
    ],
  ])("rejects structured producer with %s", (_name, producedBy, errorPath) => {
    const result = validateExtraction(minimalExtraction({ produced_by: producedBy }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((error) => error.path === errorPath)).toBe(true);
  });

  test("accepts open configuration object with arbitrary extra fields", () => {
    const result = validateExtraction(minimalExtraction({
      produced_by: {
        version: "1",
        model: "anthropic://claude-sonnet-4-6",
        configuration: {
          provider_sampling_mode: "adaptive",
          vendor_flag: true,
        },
      },
    }));
    expect(result.valid).toBe(true);
  });

  test("accepts known configuration fields", () => {
    const result = validateExtraction(minimalExtraction({
      produced_by: {
        version: "1",
        model: "anthropic://claude-sonnet-4-6",
        configuration: {
          reasoning_effort: "medium",
          system_prompt_hash: "f00dbabe",
          temperature: 0.1,
          top_p: 0.95,
          max_tokens: 2048,
        },
      },
    }));
    expect(result.valid).toBe(true);
  });

  test.each([
    ["entity name empty", minimalExtraction({ entities: [{ name: "", type: "person" }] }), "entities[0].name"],
    ["entity type empty", minimalExtraction({ entities: [{ name: "Mom", type: "" }] }), "entities[0].type"],
    ["goal text empty", minimalExtraction({ goals: [{ text: "", status: "open", entity_refs: [] }] }), "goals[0].text"],
    ["theme empty", minimalExtraction({ themes: ["Health", ""] }), "themes[1]"],
    ["fact text empty", minimalExtraction({ facts: [{ text: "" }] }), "facts[0].text"],
    [
      "relation target empty",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", relations: [{ target: "", type: "knows" }] }] }),
      "entities[0].relations[0].target",
    ],
    [
      "relation type empty",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", relations: [{ target: "e2", type: "" }] }] }),
      "entities[0].relations[0].type",
    ],
    [
      "temporal raw empty",
      minimalExtraction({ temporal_refs: [{ version: "1", raw: "" }] }),
      "temporal_refs[0].raw",
    ],
    ["summary empty", minimalExtraction({ summary: "" }), "summary"],
    [
      "alias empty string",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", aliases: [""] }] }),
      "entities[0].aliases[0]",
    ],
    [
      "alias non-string",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", aliases: [123] }] }),
      "entities[0].aliases[0]",
    ],
  ])("%s", (_name, doc, path) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(false);
    expect(result.errors.some((error) => error.path === path)).toBe(true);
  });

  test.each([
    ["2026-04-26", false],
    ["2026-04-26T10:30:00Z", true],
    ["not-a-date", false],
  ])("validates extracted_at shape: %s", (value, expectedValid) => {
    const result = validateExtraction(minimalExtraction({ extracted_at: value }));
    expect(result.valid).toBe(expectedValid);
  });

  test.each([
    [
      "goal stated_at bad",
      minimalExtraction({ goals: [{ text: "Recovery", status: "open", entity_refs: [], stated_at: "not-a-date" }] }),
      "goals[0].stated_at",
    ],
    [
      "goal resolved_at bad",
      minimalExtraction({ goals: [{ text: "Recovery", status: "resolved", entity_refs: [], resolved_at: "whenever" }] }),
      "goals[0].resolved_at",
    ],
    [
      "temporal resolved bad",
      minimalExtraction({ temporal_refs: [{ version: "1", raw: "next week", type: "point", resolved: "not-a-date" }] }),
      "temporal_refs[0].resolved",
    ],
  ])("%s", (_name, doc, path) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(false);
    expect(result.errors.some((error) => error.path === path)).toBe(true);
  });

  test.each([
    [
      "source version only rejected",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", source: { version: "1" } }] }),
      false,
    ],
    [
      "signals version only rejected",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", signals: { version: "1" } }] }),
      false,
    ],
    [
      "source with snippet accepted",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", source: { version: "1", snippet: "My mom" } }] }),
      true,
    ],
    [
      "signals with confidence accepted",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", signals: { version: "1", confidence: 0.9 } }] }),
      true,
    ],
    [
      "goal source version only rejected",
      minimalExtraction({ goals: [{ text: "Recovery", status: "open", entity_refs: [], source: { version: "1" } }] }),
      false,
    ],
    [
      "fact signals version only rejected",
      minimalExtraction({ facts: [{ text: "Surgery happened", signals: { version: "1" } }] }),
      false,
    ],
  ])("%s", (_name, doc, expectedValid) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(expectedValid);
  });

  test.each([
    [
      "goal refs missing id",
      minimalExtraction({
        entities: [{ name: "Mom", type: "person" }],
        goals: [{ text: "Recovery", status: "open", entity_refs: ["e1"] }],
      }),
      false,
      "goals[0].entity_refs[0]",
    ],
    [
      "goal refs valid id",
      minimalExtraction({
        entities: [{ id: "e1", name: "Mom", type: "person" }],
        goals: [{ text: "Recovery", status: "open", entity_refs: ["e1"] }],
      }),
      true,
      "",
    ],
    [
      "relation target missing entity",
      minimalExtraction({
        entities: [{ id: "e1", name: "Mom", type: "person", relations: [{ target: "e99", type: "knows" }] }],
      }),
      false,
      "entities[0].relations[0].target",
    ],
    [
      "relation target valid entity",
      minimalExtraction({
        entities: [
          { id: "e1", name: "Mom", type: "person", relations: [{ target: "e2", type: "parent_of" }] },
          { id: "e2", name: "Dad", type: "person" },
        ],
      }),
      true,
      "",
    ],
    [
      "empty entity refs accepted",
      minimalExtraction({ goals: [{ text: "Recovery", status: "open", entity_refs: [] }] }),
      true,
      "",
    ],
    [
      "entity with aliases accepted",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", aliases: ["Mother", "Mama"] }] }),
      true,
      "",
    ],
    [
      "entity with empty aliases array accepted",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", aliases: [] }] }),
      true,
      "",
    ],
    [
      "entity aliases not array rejected",
      minimalExtraction({ entities: [{ name: "Mom", type: "person", aliases: "Mother" }] }),
      false,
      "entities[0].aliases",
    ],
  ])("%s", (_name, doc, expectedValid, path) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(expectedValid);
    if (path) {
      expect(result.errors.some((error) => error.path === path)).toBe(true);
    }
  });

  test.each([
    ["bad extension key", minimalExtraction({ extensions: { badkey: { foo: "bar" } } }), false, "extensions.badkey"],
    ["good extension key", minimalExtraction({ extensions: { "conversa/prayer": { category: "Health" } } }), true, ""],
    ["bad kind", minimalExtraction({ kind: "badkind" }), false, "kind"],
    ["good kind", minimalExtraction({ kind: "conversa/prayer" }), true, ""],
  ])("%s", (_name, doc, expectedValid, path) => {
    const result = validateExtraction(doc);
    expect(result.valid).toBe(expectedValid);
    if (path) {
      expect(result.errors.some((error) => error.path === path)).toBe(true);
    }
  });
});

describe("v1.2 fields", () => {
  // --- keywords ---
  test("valid keywords accepted", () => {
    const result = validateExtraction(minimalExtraction({ keywords: ["prayer", "healing"] }));
    expect(result.valid).toBe(true);
  });

  test("empty keywords array accepted", () => {
    const result = validateExtraction(minimalExtraction({ keywords: [] }));
    expect(result.valid).toBe(true);
  });

  test("keywords with empty string rejected", () => {
    const result = validateExtraction(minimalExtraction({ keywords: ["prayer", ""] }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "keywords[1]")).toBe(true);
  });

  test("keywords not array rejected", () => {
    const result = validateExtraction(minimalExtraction({ keywords: "prayer" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "keywords")).toBe(true);
  });

  // --- questions ---
  test("valid question accepted", () => {
    const result = validateExtraction(minimalExtraction({
      questions: [{ text: "How are you feeling?" }],
    }));
    expect(result.valid).toBe(true);
  });

  test("question with directed_to accepted", () => {
    const result = validateExtraction(minimalExtraction({
      entities: [{ id: "e1", name: "Mom", type: "person" }],
      questions: [{ text: "How is Mom?", directed_to: "e1" }],
    }));
    expect(result.valid).toBe(true);
  });

  test("question missing text rejected", () => {
    const result = validateExtraction(minimalExtraction({
      questions: [{ directed_to: "someone" }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "questions[0].text")).toBe(true);
  });

  test("question with empty text rejected", () => {
    const result = validateExtraction(minimalExtraction({
      questions: [{ text: "" }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "questions[0].text")).toBe(true);
  });

  test("question unknown property rejected", () => {
    const result = validateExtraction(minimalExtraction({
      questions: [{ text: "Why?", badProp: true }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "questions[0].badProp")).toBe(true);
  });

  test("questions not array rejected", () => {
    const result = validateExtraction(minimalExtraction({ questions: "why?" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "questions")).toBe(true);
  });

  // --- actions ---
  test("valid action accepted", () => {
    const result = validateExtraction(minimalExtraction({
      actions: [{ text: "Schedule appointment", origin: "extracted" }],
    }));
    expect(result.valid).toBe(true);
  });

  test("action with proposed_from_goals origin accepted", () => {
    const result = validateExtraction(minimalExtraction({
      actions: [{ text: "Follow up with doctor", origin: "proposed_from_goals" }],
    }));
    expect(result.valid).toBe(true);
  });

  test("action missing origin rejected", () => {
    const result = validateExtraction(minimalExtraction({
      actions: [{ text: "Do something" }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "actions[0].origin")).toBe(true);
  });

  test("action invalid origin rejected", () => {
    const result = validateExtraction(minimalExtraction({
      actions: [{ text: "Do something", origin: "unknown" }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "actions[0].origin")).toBe(true);
  });

  test("action missing text rejected", () => {
    const result = validateExtraction(minimalExtraction({
      actions: [{ origin: "extracted" }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "actions[0].text")).toBe(true);
  });

  test("action with entity_refs accepted", () => {
    const result = validateExtraction(minimalExtraction({
      entities: [{ id: "e1", name: "Mom", type: "person" }],
      actions: [{ text: "Call Mom", origin: "extracted", entity_refs: ["e1"] }],
    }));
    expect(result.valid).toBe(true);
  });

  test("action with dangling entity_ref rejected", () => {
    const result = validateExtraction(minimalExtraction({
      entities: [{ id: "e1", name: "Mom", type: "person" }],
      actions: [{ text: "Call Dad", origin: "extracted", entity_refs: ["e99"] }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "actions[0].entity_refs[0]")).toBe(true);
  });

  test("action unknown property rejected", () => {
    const result = validateExtraction(minimalExtraction({
      actions: [{ text: "Go", origin: "extracted", priority: "high" }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "actions[0].priority")).toBe(true);
  });

  // --- decisions ---
  test("valid decision accepted", () => {
    const result = validateExtraction(minimalExtraction({
      decisions: [{ text: "Decided to move forward" }],
    }));
    expect(result.valid).toBe(true);
  });

  test("decision with entity_refs and decided_at accepted", () => {
    const result = validateExtraction(minimalExtraction({
      entities: [{ id: "e1", name: "Team", type: "organization" }],
      decisions: [{ text: "Team chose option A", entity_refs: ["e1"], decided_at: "2026-05-01" }],
    }));
    expect(result.valid).toBe(true);
  });

  test("decision missing text rejected", () => {
    const result = validateExtraction(minimalExtraction({
      decisions: [{ entity_refs: [] }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "decisions[0].text")).toBe(true);
  });

  test("decision with dangling entity_ref rejected", () => {
    const result = validateExtraction(minimalExtraction({
      entities: [],
      decisions: [{ text: "Go with plan B", entity_refs: ["e99"] }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "decisions[0].entity_refs[0]")).toBe(true);
  });

  test("decision unknown property rejected", () => {
    const result = validateExtraction(minimalExtraction({
      decisions: [{ text: "Yes", importance: "high" }],
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "decisions[0].importance")).toBe(true);
  });

  // --- sentiment dual-shape ---
  test("string sentiment accepted", () => {
    const result = validateExtraction(minimalExtraction({ sentiment: "hopeful" }));
    expect(result.valid).toBe(true);
  });

  test("structured sentiment accepted", () => {
    const result = validateExtraction(minimalExtraction({
      sentiment: { version: "1", valence: "positive", intensity: 0.8, confidence: 0.9 },
    }));
    expect(result.valid).toBe(true);
  });

  test("structured sentiment minimal (valence only) accepted", () => {
    const result = validateExtraction(minimalExtraction({
      sentiment: { version: "1", valence: "negative" },
    }));
    expect(result.valid).toBe(true);
  });

  test("structured sentiment invalid valence rejected", () => {
    const result = validateExtraction(minimalExtraction({
      sentiment: { version: "1", valence: "angry" },
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "sentiment.valence")).toBe(true);
  });

  test("structured sentiment missing valence rejected", () => {
    const result = validateExtraction(minimalExtraction({
      sentiment: { version: "1", intensity: 0.5 },
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "sentiment.valence")).toBe(true);
  });

  test("structured sentiment intensity out of range rejected", () => {
    const result = validateExtraction(minimalExtraction({
      sentiment: { version: "1", valence: "positive", intensity: 1.5 },
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "sentiment.intensity")).toBe(true);
  });

  test("structured sentiment unknown property rejected", () => {
    const result = validateExtraction(minimalExtraction({
      sentiment: { version: "1", valence: "neutral", mood: "calm" },
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "sentiment.mood")).toBe(true);
  });

  test("sentiment as array rejected", () => {
    const result = validateExtraction(minimalExtraction({ sentiment: ["positive"] }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "sentiment")).toBe(true);
  });

  test("sentiment as number rejected", () => {
    const result = validateExtraction(minimalExtraction({ sentiment: 0.8 }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "sentiment")).toBe(true);
  });

  // --- language ---
  test("valid language tag accepted", () => {
    const result = validateExtraction(minimalExtraction({ language: "en-US" }));
    expect(result.valid).toBe(true);
  });

  test("simple language tag accepted", () => {
    const result = validateExtraction(minimalExtraction({ language: "es" }));
    expect(result.valid).toBe(true);
  });

  test("three-letter language tag accepted", () => {
    const result = validateExtraction(minimalExtraction({ language: "por" }));
    expect(result.valid).toBe(true);
  });

  test("complex BCP 47 tag accepted", () => {
    const result = validateExtraction(minimalExtraction({ language: "zh-Hans-CN" }));
    expect(result.valid).toBe(true);
  });

  test("invalid language tag rejected", () => {
    const result = validateExtraction(minimalExtraction({ language: "english" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "language")).toBe(true);
  });

  test("language as number rejected", () => {
    const result = validateExtraction(minimalExtraction({ language: 42 }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "language")).toBe(true);
  });

  // --- source_metadata ---
  test("valid source_metadata accepted", () => {
    const result = validateExtraction(minimalExtraction({
      source_metadata: { version: "1", token_count: 500, character_count: 2000, modality: "text", format: "plain" },
    }));
    expect(result.valid).toBe(true);
  });

  test("source_metadata with only version accepted", () => {
    const result = validateExtraction(minimalExtraction({
      source_metadata: { version: "1" },
    }));
    expect(result.valid).toBe(true);
  });

  test("source_metadata negative token_count rejected", () => {
    const result = validateExtraction(minimalExtraction({
      source_metadata: { version: "1", token_count: -1 },
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "source_metadata.token_count")).toBe(true);
  });

  test("source_metadata non-integer token_count rejected", () => {
    const result = validateExtraction(minimalExtraction({
      source_metadata: { version: "1", token_count: 3.5 },
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "source_metadata.token_count")).toBe(true);
  });

  test("source_metadata unknown property rejected", () => {
    const result = validateExtraction(minimalExtraction({
      source_metadata: { version: "1", word_count: 100 },
    }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "source_metadata.word_count")).toBe(true);
  });

  test("source_metadata not object rejected", () => {
    const result = validateExtraction(minimalExtraction({ source_metadata: "text" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "source_metadata")).toBe(true);
  });

  // --- confidence ---
  test("valid confidence accepted", () => {
    const result = validateExtraction(minimalExtraction({ confidence: 0.85 }));
    expect(result.valid).toBe(true);
  });

  test("confidence 0 accepted", () => {
    const result = validateExtraction(minimalExtraction({ confidence: 0 }));
    expect(result.valid).toBe(true);
  });

  test("confidence 1 accepted", () => {
    const result = validateExtraction(minimalExtraction({ confidence: 1 }));
    expect(result.valid).toBe(true);
  });

  test("confidence above 1 rejected", () => {
    const result = validateExtraction(minimalExtraction({ confidence: 1.1 }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "confidence")).toBe(true);
  });

  test("confidence below 0 rejected", () => {
    const result = validateExtraction(minimalExtraction({ confidence: -0.1 }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "confidence")).toBe(true);
  });

  test("confidence as string rejected", () => {
    const result = validateExtraction(minimalExtraction({ confidence: "high" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.path === "confidence")).toBe(true);
  });

  // --- new capabilities recognized ---
  test("new v1.2 capabilities accepted", () => {
    const result = validateExtraction(minimalExtraction({
      capabilities: [
        "entities", "goals", "themes", "keywords", "summary",
        "sentiment", "structured_sentiment", "facts", "questions",
        "actions", "decisions", "language", "source_metadata", "confidence",
      ],
    }));
    expect(result.valid).toBe(true);
  });
});

describe("validateExtraction conformance fixtures", () => {
  test("matches shared validation fixtures", () => {
    const cases = loadJson<Array<{
      name: string;
      input: Record<string, unknown>;
      expected_valid: boolean;
      expected_error_paths: string[];
    }>>(CONFORMANCE_DIR, "validate_cases.json");

    for (const fixture of cases) {
      const result = validateExtraction(fixture.input);
      expect(result.valid, fixture.name).toBe(fixture.expected_valid);
      for (const path of fixture.expected_error_paths) {
        expect(result.errors.some((error) => error.path === path), `${fixture.name}:${path}`).toBe(true);
      }
    }
  });
});

describe("JSON Schema dereference", () => {
  test("schema files are valid JSON with ids", () => {
    const files = [
      resolve(SCHEMAS_DIR, "action", "v1.json"),
      resolve(SCHEMAS_DIR, "assertion-signals", "v1.json"),
      resolve(SCHEMAS_DIR, "decision", "v1.json"),
      resolve(SCHEMAS_DIR, "embedding", "v1.json"),
      resolve(SCHEMAS_DIR, "entity", "v1.json"),
      resolve(SCHEMAS_DIR, "extract", "v1.json"),
      resolve(SCHEMAS_DIR, "goal", "v1.json"),
      resolve(SCHEMAS_DIR, "question", "v1.json"),
      resolve(SCHEMAS_DIR, "sentiment", "v1.json"),
      resolve(SCHEMAS_DIR, "source-metadata", "v1.json"),
      resolve(SCHEMAS_DIR, "source-ref", "v1.json"),
      resolve(SCHEMAS_DIR, "temporal-ref", "v1.json"),
    ];

    for (const file of files) {
      const schema = loadJson<Record<string, unknown>>(file);
      expect(schema.$schema, file).toBe("https://json-schema.org/draft/2020-12/schema");
      expect(typeof schema.$id, file).toBe("string");
    }
  });

  test("extraction schema references the expected sub-schemas", () => {
    const schema = JSON.stringify(loadJson<Record<string, unknown>>(resolve(SCHEMAS_DIR, "extract", "v1.json")));
    expect(schema).toContain("source-ref/v1.json");
    expect(schema).toContain("embedding/v1.json");
    expect(schema).toContain("assertion-signals/v1.json");
    expect(schema).toContain("temporal-ref/v1.json");
    expect(schema).toContain("producer/v1.json");
    expect(schema).toContain("entity/v1.json");
    expect(schema).toContain("goal/v1.json");
    expect(schema).toContain("question/v1.json");
    expect(schema).toContain("action/v1.json");
    expect(schema).toContain("decision/v1.json");
    expect(schema).toContain("sentiment/v1.json");
    expect(schema).toContain("source-metadata/v1.json");
  });

  test("extraction schema carries the expected required fields", () => {
    const schema = loadJson<{ required: string[] }>(resolve(SCHEMAS_DIR, "extract", "v1.json"));
    expect(schema.required).toEqual(expect.arrayContaining([
      "version",
      "extracted_at",
      "produced_by",
      "entities",
      "goals",
      "themes",
      "capabilities",
    ]));
  });

  test("ajv resolves hosted schema refs and matches validator on aligned structural cases", () => {
    const ajv = new Ajv2020({ strict: false, allErrors: true });
    addFormats(ajv);

    const schemaFiles = [
      resolve(SCHEMAS_DIR, "action", "v1.json"),
      resolve(SCHEMAS_DIR, "assertion-signals", "v1.json"),
      resolve(SCHEMAS_DIR, "decision", "v1.json"),
      resolve(SCHEMAS_DIR, "embedding", "v1.json"),
      resolve(SCHEMAS_DIR, "entity", "v1.json"),
      resolve(SCHEMAS_DIR, "goal", "v1.json"),
      resolve(SCHEMAS_DIR, "producer", "v1.json"),
      resolve(SCHEMAS_DIR, "question", "v1.json"),
      resolve(SCHEMAS_DIR, "sentiment", "v1.json"),
      resolve(SCHEMAS_DIR, "source-metadata", "v1.json"),
      resolve(SCHEMAS_DIR, "source-ref", "v1.json"),
      resolve(SCHEMAS_DIR, "temporal-ref", "v1.json"),
      resolve(SCHEMAS_DIR, "extract", "v1.json"),
    ];

    for (const file of schemaFiles) {
      const schema = loadJson<Record<string, unknown>>(file);
      ajv.addSchema(schema, schema.$id as string);
    }

    const validateSchema = ajv.getSchema("https://synapt.dev/schemas/extract/v1.json");
    expect(validateSchema).toBeTypeOf("function");

    const cases = [
      minimalExtraction(),
      minimalExtraction({ version: "2" }),
      minimalExtraction({
        produced_by: {
          version: "1",
          model: "anthropic://claude-sonnet-4-6",
        },
      }),
      minimalExtraction({
        entities: [{ id: "e1", name: "Mom", type: "person" }],
        goals: [{ text: "Recovery", status: "open", entity_refs: ["e1"] }],
      }),
    ];

    for (const doc of cases) {
      const schemaValid = validateSchema!(doc);
      const validatorValid = validateExtraction(doc).valid;
      expect(schemaValid).toBe(validatorValid);
    }
  });

  test("produced_by URI semantics stay aligned with the hosted schema", () => {
    const ajv = new Ajv2020({ strict: false, allErrors: true });
    addFormats(ajv);
    for (const file of [
      resolve(SCHEMAS_DIR, "action", "v1.json"),
      resolve(SCHEMAS_DIR, "assertion-signals", "v1.json"),
      resolve(SCHEMAS_DIR, "decision", "v1.json"),
      resolve(SCHEMAS_DIR, "embedding", "v1.json"),
      resolve(SCHEMAS_DIR, "entity", "v1.json"),
      resolve(SCHEMAS_DIR, "goal", "v1.json"),
      resolve(SCHEMAS_DIR, "producer", "v1.json"),
      resolve(SCHEMAS_DIR, "question", "v1.json"),
      resolve(SCHEMAS_DIR, "sentiment", "v1.json"),
      resolve(SCHEMAS_DIR, "source-metadata", "v1.json"),
      resolve(SCHEMAS_DIR, "source-ref", "v1.json"),
      resolve(SCHEMAS_DIR, "temporal-ref", "v1.json"),
      resolve(SCHEMAS_DIR, "extract", "v1.json"),
    ]) {
      const schema = loadJson<Record<string, unknown>>(file);
      ajv.addSchema(schema, schema.$id as string);
    }

    const doc = minimalExtraction({ produced_by: "bad-model" });
    const schemaValid = ajv.getSchema("https://synapt.dev/schemas/extract/v1.json")!(doc);
    const validatorValid = validateExtraction(doc).valid;
    expect(schemaValid).toBe(validatorValid);
  });

  test("date-only extracted_at semantics stay aligned with the hosted schema", () => {
    const ajv = new Ajv2020({ strict: false, allErrors: true });
    addFormats(ajv);
    for (const file of [
      resolve(SCHEMAS_DIR, "action", "v1.json"),
      resolve(SCHEMAS_DIR, "assertion-signals", "v1.json"),
      resolve(SCHEMAS_DIR, "decision", "v1.json"),
      resolve(SCHEMAS_DIR, "embedding", "v1.json"),
      resolve(SCHEMAS_DIR, "entity", "v1.json"),
      resolve(SCHEMAS_DIR, "goal", "v1.json"),
      resolve(SCHEMAS_DIR, "producer", "v1.json"),
      resolve(SCHEMAS_DIR, "question", "v1.json"),
      resolve(SCHEMAS_DIR, "sentiment", "v1.json"),
      resolve(SCHEMAS_DIR, "source-metadata", "v1.json"),
      resolve(SCHEMAS_DIR, "source-ref", "v1.json"),
      resolve(SCHEMAS_DIR, "temporal-ref", "v1.json"),
      resolve(SCHEMAS_DIR, "extract", "v1.json"),
    ]) {
      const schema = loadJson<Record<string, unknown>>(file);
      ajv.addSchema(schema, schema.$id as string);
    }

    const doc = minimalExtraction({ extracted_at: "2026-04-26" });
    const schemaValid = ajv.getSchema("https://synapt.dev/schemas/extract/v1.json")!(doc);
    const validatorValid = validateExtraction(doc).valid;
    expect(schemaValid).toBe(validatorValid);
  });

  test("producer schema exists with canonical id", () => {
    const producerSchema = loadJson<Record<string, unknown>>(
      resolve(SCHEMAS_DIR, "producer", "v1.json"),
    );
    expect(producerSchema.$id).toBe("https://synapt.dev/schemas/producer/v1.json");
  });
});
