import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, test } from "vitest";

import type { SynaptExtraction } from "@synapt-dev/extract";
import {
  DirectoryStore,
  InMemoryStore,
  entitiesRelatedTo,
  entityTimeline,
  goals,
  search,
  themesOverTime,
} from "../src/index.js";

const FIXTURE_DIR = resolve(import.meta.dirname, "fixtures", "extractions");

function loadFixture(name: string): SynaptExtraction {
  return JSON.parse(readFileSync(resolve(FIXTURE_DIR, `${name}.json`), "utf-8")) as SynaptExtraction;
}

const FIXTURES = [
  loadFixture("alpha"),
  loadFixture("bravo"),
  loadFixture("charlie"),
];

describe("stores", () => {
  test("InMemoryStore lists and gets by source_id", async () => {
    const store = new InMemoryStore(FIXTURES);
    const listed = await store.list();
    const one = await store.get("alpha");

    expect(listed).toHaveLength(3);
    expect(one?.source_id).toBe("alpha");
  });

  test("DirectoryStore reads json documents from a directory", async () => {
    const store = new DirectoryStore(FIXTURE_DIR);
    const listed = await store.list();
    const one = await store.get("bravo");

    expect(listed).toHaveLength(3);
    expect(one?.source_id).toBe("bravo");
  });
});

describe("query APIs", () => {
  test("goals returns open goals across multiple extractions", async () => {
    const store = new InMemoryStore(FIXTURES);
    const result = await goals(store, { status: ["open"] });

    expect(result.map((goal) => goal.text)).toEqual([
      "Mom's recovery",
      "Keep physical therapy on schedule",
      "Organize meal train",
    ]);
  });

  test("goals filters by entity_refs correctly", async () => {
    const store = new InMemoryStore(FIXTURES);
    const result = await goals(store, { entity_refs: ["e1"] });

    expect(result.map((goal) => goal.text)).toEqual([
      "Mom's recovery",
      "Keep physical therapy on schedule",
    ]);
  });

  test("entityTimeline reconstructs entity_state evolution chronologically", async () => {
    const store = new InMemoryStore(FIXTURES);
    const result = await entityTimeline(store, "Mom");

    expect(result).toEqual([
      { extraction_id: "alpha", state: "recovering", date_hint: "2026-04-20" },
      { extraction_id: "bravo", state: "home from hospital", date_hint: "2026-04-24" },
    ]);
  });

  test("themesOverTime aggregates theme frequencies", async () => {
    const store = new InMemoryStore(FIXTURES);
    const result = await themesOverTime(store, 14);

    expect(result).toEqual([
      { theme: "Health", count: 3 },
      { theme: "Community", count: 1 },
      { theme: "Family", count: 1 },
      { theme: "Logistics", count: 1 },
      { theme: "Support", count: 1 },
    ]);
  });

  test("entitiesRelatedTo traverses relations field", async () => {
    const store = new InMemoryStore(FIXTURES);
    const result = await entitiesRelatedTo(store, "Mom");

    expect(result.map((entity) => entity.name).sort()).toEqual(["Dad", "Me"]);
  });

  test("entitiesRelatedTo filters by relation type when via is set", async () => {
    const store = new InMemoryStore(FIXTURES);
    const result = await entitiesRelatedTo(store, "Mom", "parent_of");

    expect(result.map((entity) => entity.name)).toEqual(["Me"]);
  });

  test("search returns matching extractions for a simple query", async () => {
    const store = new InMemoryStore(FIXTURES);
    const result = await search(store, "meal train");

    expect(result.map((doc) => doc.source_id)).toEqual(["charlie"]);
  });
});
