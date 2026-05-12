import { readFileSync, existsSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, test } from "vitest";

import {
  CAPABILITY_REGISTRY,
  CANONICAL_ORDER,
  EXTRACTION_CAPABILITIES,
  ExtractionBuilder,
  STANDARD_EMBEDDING_INPUTS,
  buildFinalizedExtractionSchema,
  buildExtractionPrompt,
  buildExtractionResponseFormat,
  buildExtractionSchema,
  capabilityEmbeddingInput,
  createExtractionBuilder,
  resolveCapabilities,
} from "../src/index.js";
import { EMBEDDED_PROMPT_FRAGMENTS } from "../src/prompt-data.js";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..", "..");
const CONFORMANCE_DIR = resolve(REPO_ROOT, "tests", "conformance");
const _pkgPrompts = resolve(import.meta.dirname, "..", "prompts");
const PROMPTS_DIR = existsSync(_pkgPrompts) ? _pkgPrompts : resolve(REPO_ROOT, "prompts");
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
      "themes.txt", "keywords.txt", "summary.txt", "sentiment.txt", "structured_sentiment.txt",
      "facts.txt", "questions.txt", "actions.txt", "decisions.txt",
      "temporal_refs.txt", "temporal_classes.txt",
      "relations.txt", "relation_origin.txt",
      "assertion_signals.txt", "evidence_anchoring.txt",
      "language.txt", "source_metadata.txt", "confidence.txt",
    ];

    for (const name of expected) {
      const content = readFileSync(resolve(PROMPTS_DIR, "v1", name), "utf-8").trim();
      expect(content.length, name).toBeGreaterThan(0);
    }
  });

  test("profile files exist and contain the expected capability sets", () => {
    for (const name of ["minimal", "standard", "full"]) {
      const data = loadJson<{ capabilities: string[] }>(PROMPTS_DIR, "profiles", `${name}.json`);
      expect(Array.isArray(data.capabilities), name).toBe(true);
    }

    const minimal = new Set(loadJson<{ capabilities: string[] }>(PROMPTS_DIR, "profiles", "minimal.json").capabilities);
    expect(minimal).toEqual(new Set(["entities", "entity_state", "goals", "themes", "summary"]));

    const standard = new Set(loadJson<{ capabilities: string[] }>(PROMPTS_DIR, "profiles", "standard.json").capabilities);
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

    const full = new Set(loadJson<{ capabilities: string[] }>(PROMPTS_DIR, "profiles", "full.json").capabilities);
    for (const capability of standard) {
      expect(full.has(capability), capability).toBe(true);
    }
    expect(full).toEqual(new Set([
      "entities", "entity_state", "entity_context", "entity_ids",
      "goals", "goal_timing", "goal_entity_refs",
      "themes", "keywords", "summary", "sentiment", "structured_sentiment",
      "facts", "questions", "actions", "decisions",
      "temporal_refs", "temporal_classes",
      "relations", "relation_origin",
      "assertion_signals", "evidence_anchoring",
      "language", "source_metadata", "confidence",
    ]));
  });
});

describe("registry consistency", () => {
  const fragmentDir = resolve(PROMPTS_DIR, "v1");
  const fragmentFiles = readdirSync(fragmentDir)
    .filter((f) => f.endsWith(".txt"))
    .map((f) => f.replace(".txt", ""));
  const capabilityFragments = fragmentFiles.filter((f) => f !== "preamble" && f !== "postamble");

  const fullProfile = loadJson<{ capabilities: string[] }>(PROMPTS_DIR, "profiles", "full.json").capabilities;

  test("embedded capability registry matches the prompt asset", () => {
    expect(CAPABILITY_REGISTRY).toEqual(loadJson(PROMPTS_DIR, "capabilities.json"));
  });

  test("embedded prompt fragments match the prompt assets", () => {
    const expectedNames = fragmentFiles.sort();
    expect(Object.keys(EMBEDDED_PROMPT_FRAGMENTS).sort()).toEqual(expectedNames);
    for (const name of expectedNames) {
      expect(EMBEDDED_PROMPT_FRAGMENTS[name], name).toBe(readFileSync(resolve(fragmentDir, `${name}.txt`), "utf-8"));
    }
  });

  test("capability registry covers schema capabilities in canonical order", () => {
    expect(CAPABILITY_REGISTRY.capabilities.map((definition) => definition.name)).toEqual(CANONICAL_ORDER);
    expect(new Set(CANONICAL_ORDER)).toEqual(new Set(EXTRACTION_CAPABILITIES));
  });

  test("capability registry profiles match legacy profile files", () => {
    for (const profile of ["minimal", "standard", "full"] as const) {
      const fileProfile = loadJson<{ capabilities: string[] }>(PROMPTS_DIR, "profiles", `${profile}.json`).capabilities;
      expect(CAPABILITY_REGISTRY.profiles[profile]).toEqual(fileProfile);
    }
  });

  test("capability registry exposes embedding inputs", () => {
    expect(capabilityEmbeddingInput("entities")).toBe("entities");
    expect(capabilityEmbeddingInput("entity_state")).toBe("entities");
    expect(capabilityEmbeddingInput("structured_sentiment")).toBe("sentiment");
    expect(capabilityEmbeddingInput("confidence")).toBeUndefined();
    expect(STANDARD_EMBEDDING_INPUTS).toEqual([
      "source",
      "summary",
      "entities",
      "goals",
      "themes",
      "keywords",
      "facts",
      "questions",
      "actions",
      "decisions",
      "temporal_refs",
      "sentiment",
    ]);
  });

  test("every capability in full profile has a fragment file", () => {
    for (const cap of fullProfile) {
      expect(capabilityFragments, `missing fragment for ${cap}`).toContain(cap);
    }
  });

  test("every fragment file (except preamble/postamble) is a valid capability", () => {
    for (const fragment of capabilityFragments) {
      expect(() => resolveCapabilities({ capabilities: [fragment] })).not.toThrow(/Unknown/);
    }
  });

  test("no orphan fragment files beyond declared capabilities", () => {
    const fullSet = new Set(fullProfile);
    for (const fragment of capabilityFragments) {
      expect(fullSet.has(fragment), `orphan fragment: ${fragment}`).toBe(true);
    }
  });

  test("full profile has no duplicates", () => {
    expect(new Set(fullProfile).size).toBe(fullProfile.length);
  });

  test("buildExtractionPrompt succeeds for every individual capability", () => {
    const modifierOnly = new Set(["assertion_signals", "evidence_anchoring"]);
    for (const cap of fullProfile) {
      const caps = modifierOnly.has(cap) ? ["entities", cap] : [cap];
      expect(() =>
        buildExtractionPrompt("test", { capabilities: caps }),
      ).not.toThrow();
    }
  });

  test("profiles are strict subsets: minimal < standard < full", () => {
    const minimal = new Set(loadJson<{ capabilities: string[] }>(PROMPTS_DIR, "profiles", "minimal.json").capabilities);
    const standard = new Set(loadJson<{ capabilities: string[] }>(PROMPTS_DIR, "profiles", "standard.json").capabilities);
    const full = new Set(fullProfile);
    for (const cap of minimal) {
      expect(standard.has(cap) || full.has(cap), `minimal cap ${cap} not in standard or full`).toBe(true);
    }
    for (const cap of standard) {
      expect(full.has(cap), `standard cap ${cap} not in full`).toBe(true);
    }
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

describe("buildExtractionSchema", () => {
  test("includes only fields for resolved capabilities", () => {
    const schema = buildExtractionSchema({ capabilities: ["entities"] });
    const properties = schema.properties as Record<string, unknown>;
    expect(properties.extracted_at).toBeDefined();
    expect(properties.entities).toBeDefined();
    expect(properties.goals).toBeUndefined();
    expect(properties.temporal_refs).toBeUndefined();
    expect(schema.required).toEqual(["extracted_at", "entities"]);
  });

  test("applies dependency closure to schema", () => {
    const schema = buildExtractionSchema({ capabilities: ["goal_entity_refs"] });
    const properties = schema.properties as Record<string, unknown>;
    const entities = properties.entities as { items: { properties: Record<string, unknown>; required: string[] } };
    const goals = properties.goals as { items: { properties: Record<string, unknown>; required: string[] } };

    expect(properties.entities).toBeDefined();
    expect(properties.goals).toBeDefined();
    expect(entities.items.properties.id).toBeDefined();
    expect(entities.items.required).toContain("id");
    expect(goals.items.properties.entity_refs).toBeDefined();
    expect(goals.items.required).toContain("entity_refs");
  });

  test("adds modifier fields only when capabilities request them", () => {
    const base = buildExtractionSchema({ capabilities: ["entities"] });
    const enriched = buildExtractionSchema({ capabilities: ["entities", "entity_context", "evidence_anchoring"] });
    const baseEntity = ((base.properties as Record<string, unknown>).entities as { items: { properties: Record<string, unknown> } }).items.properties;
    const enrichedEntity = ((enriched.properties as Record<string, unknown>).entities as { items: { properties: Record<string, unknown> } }).items.properties;

    expect(baseEntity.context).toBeUndefined();
    expect(baseEntity.source).toBeUndefined();
    expect(enrichedEntity.context).toBeDefined();
    expect(enrichedEntity.date_hint).toBeDefined();
    expect(enrichedEntity.source).toBeDefined();
  });

  test("includes full source-ref fields when evidence anchoring is requested", () => {
    const schema = buildExtractionSchema({ capabilities: ["entities", "evidence_anchoring"] });
    const source = (((schema.properties as Record<string, unknown>).entities as { items: { properties: Record<string, unknown> } })
      .items.properties.source as { properties: Record<string, unknown>; required: string[] });

    expect(source.required).toEqual(["snippet"]);
    expect(source.properties.snippet).toBeDefined();
    expect(source.properties.offset_start).toBeDefined();
    expect(source.properties.offset_end).toBeDefined();
    expect(source.properties.sentence_index).toBeDefined();
    expect(source.properties.version).toBeUndefined();
  });

  test("builds a finalized schema with full packet fields", () => {
    const schema = buildFinalizedExtractionSchema({
      capabilities: ["entities", "evidence_anchoring", "assertion_signals", "temporal_refs"],
    });
    const properties = schema.properties as Record<string, unknown>;
    const entity = (properties.entities as { items: { properties: Record<string, unknown> } }).items;
    const source = entity.properties.source as { properties: Record<string, unknown>; required: string[] };
    const signals = entity.properties.signals as { properties: Record<string, unknown>; required: string[] };
    const temporalRef = (properties.temporal_refs as { items: { properties: Record<string, unknown>; required: string[] } }).items;

    expect(schema.required).toEqual(["version", "extracted_at", "produced_by", "entities", "goals", "themes", "capabilities"]);
    expect(properties.source_id).toBeDefined();
    expect(properties.source_type).toBeDefined();
    expect(properties.user_id).toBeDefined();
    expect(properties.kind).toBeDefined();
    expect(properties.embeddings).toBeDefined();
    expect(properties.extensions).toBeDefined();
    expect(properties.keywords).toBeDefined();
    expect(properties.questions).toBeDefined();
    expect(properties.actions).toBeDefined();
    expect(properties.decisions).toBeDefined();
    expect(properties.language).toBeDefined();
    expect(properties.source_metadata).toBeDefined();
    expect(properties.confidence).toBeDefined();
    expect(source.required).toEqual(["version"]);
    expect(source.properties.version).toEqual({ const: "1" });
    expect(source.properties.offset_start).toBeDefined();
    expect(signals.required).toEqual(["version"]);
    expect(signals.properties.confidence).toBeDefined();
    expect(temporalRef.required).toEqual(["version", "raw"]);
  });

  test("schema covers all v1.2 capability fields", () => {
    const schema = buildExtractionSchema({ profile: "full" });
    const properties = schema.properties as Record<string, unknown>;
    const entity = (properties.entities as { items: { properties: Record<string, unknown> } }).items.properties;
    const question = (properties.questions as { items: { properties: Record<string, unknown> } }).items.properties;
    const action = (properties.actions as { items: { properties: Record<string, unknown>; required: string[] } }).items;
    const decision = (properties.decisions as { items: { properties: Record<string, unknown> } }).items.properties;
    const sentiment = properties.sentiment as { properties: Record<string, unknown>; required: string[] };
    const sourceMetadata = properties.source_metadata as { properties: Record<string, unknown>; required: string[] };

    for (const field of ["keywords", "questions", "actions", "decisions", "language", "source_metadata", "confidence"]) {
      expect(properties[field], field).toBeDefined();
    }
    expect(entity.aliases).toBeDefined();
    expect(question.directed_to).toBeDefined();
    expect(action.required).toEqual(["text", "origin"]);
    expect(action.properties.entity_refs).toBeDefined();
    expect(action.properties.due).toBeDefined();
    expect(decision.entity_refs).toBeDefined();
    expect(decision.decided_at).toBeDefined();
    expect(sentiment.required).toEqual(["valence"]);
    expect(sentiment.properties.version).toBeUndefined();
    expect(sourceMetadata.required).toEqual([]);
    expect(sourceMetadata.properties.version).toBeUndefined();
  });

  test("wraps schema in an OpenAI-compatible response format", () => {
    const format = buildExtractionResponseFormat({ capabilities: ["entities"] }, { name: "custom_stage1" });
    expect(format.type).toBe("json_schema");
    expect(format.name).toBe("custom_stage1");
    expect(format.strict).toBe(true);
    expect(format.schema).not.toEqual(buildExtractionSchema({ capabilities: ["entities"] }));
  });

  test("strict response format requires every object property for OpenAI", () => {
    const format = buildExtractionResponseFormat({ capabilities: ["entities", "entity_context"] });
    const schema = format.schema as { properties: Record<string, unknown> };
    const entity = (schema.properties.entities as { items: { required: string[] } }).items;
    expect(entity.required).toEqual(["name", "type", "aliases", "context", "date_hint"]);
  });

  test("strict response format makes semantic optional fields nullable", () => {
    const format = buildExtractionResponseFormat({ capabilities: ["entities", "evidence_anchoring"] });
    const schema = format.schema as { properties: Record<string, unknown> };
    const entity = (schema.properties.entities as { items: { properties: Record<string, unknown> } }).items;
    const source = entity.properties.source as {
      type: string[];
      properties: Record<string, { type?: string | string[] }>;
      required: string[];
    };

    expect(source.type).toEqual(["object", "null"]);
    expect(source.required).toEqual(["snippet", "offset_start", "offset_end", "sentence_index"]);
    expect(source.properties.offset_start.type).toEqual(["integer", "null"]);
  });

  test("non-strict response format preserves semantic required fields", () => {
    const format = buildExtractionResponseFormat(
      { capabilities: ["entities", "entity_context"] },
      { strict: false },
    );
    expect(format.schema).toEqual(buildExtractionSchema({ capabilities: ["entities", "entity_context"] }));
  });
});

describe("ExtractionBuilder", () => {
  test("builds prompt and schema from the same resolved capabilities", () => {
    const builder = createExtractionBuilder(SAMPLE_TEXT, { profile: "minimal" })
      .addCapabilities(["goal_entity_refs"])
      .withExtractedAt("2026-05-11T12:00:00Z");
    const result = builder.build({ name: "synapt_test" });
    const properties = result.schema.properties as Record<string, unknown>;

    expect(result.capabilities).toContain("goal_entity_refs");
    expect(result.prompt).toContain("Additional run constraints");
    expect(result.prompt).toContain("Use exactly this extracted_at value: 2026-05-11T12:00:00Z.");
    expect(result.prompt).toContain("Produce Stage 1 content only.");
    expect(result.prompt).toContain("Every goal.entity_refs entry must refer to one of the entity IDs you emit.");
    expect(result.prompt).toContain("Omit temporal_refs for this run.");
    expect(properties.entities).toBeDefined();
    expect(properties.goals).toBeDefined();
    expect(result.responseFormat.name).toBe("synapt_test");
  });

  test("supports fluent construction", () => {
    const builder = new ExtractionBuilder()
      .withText(SAMPLE_TEXT)
      .withCapabilities(["entities"])
      .withSourceType("prayer")
      .withDate("2026-05-11");

    expect(builder.resolvedCapabilities()).toEqual(["entities"]);
    expect(builder.prompt()).toContain("prayer");
    expect(builder.schema().required).toEqual(["extracted_at", "entities"]);
  });

  test("carries finalization context and can finalize without a live model call", () => {
    const builder = new ExtractionBuilder(SAMPLE_TEXT, {
      capabilities: ["entities", "goals", "themes", "evidence_anchoring"],
    })
      .withExtractedAt("2026-05-11T12:00:00Z")
      .withProducedBy({
        model: "openai://gpt-5.5",
        model_version: "gpt-5.5-2026-04-23",
        configuration: { reasoning_effort: "medium" },
        operator: "synapt-dev",
      })
      .withSource({ source_id: "fixture-1", source_type: "note" })
      .withUserId("user-1")
      .withKind("synapt/test")
      .withExtensions({ "synapt/source_binding": { source_sha256: "abc" } })
      .withEmbeddings([{
        vector: [0.1, 0.2],
        model: "openai://text-embedding-3-small",
        input: "source",
        dimensions: 2,
        computed_at: "2026-05-11T12:00:01Z",
      }]);

    const built = builder.build();
    expect(built.finalizeContext?.source_id).toBe("fixture-1");
    expect(built.finalizeContext?.capabilities_hint).toContain("evidence_anchoring");
    expect((built.finalizedSchema.properties as Record<string, unknown>).produced_by).toBeDefined();

    const result = builder.finalize({
      extracted_at: "2026-05-11T12:00:00Z",
      entities: [{ name: "Mom", type: "person", source: { snippet: "mom", offset_start: null } }],
      goals: [],
      themes: [],
    });

    expect(result.validation.valid).toBe(true);
    expect(result.extraction.source_id).toBe("fixture-1");
    expect(result.extraction.produced_by).toMatchObject({ version: "1", model: "openai://gpt-5.5" });
    expect(result.extraction.embeddings?.[0]).toMatchObject({ version: "1", dimensions: 2 });
    expect(result.extraction.entities[0].source).toEqual({ version: "1", snippet: "mom" });
  });
});
