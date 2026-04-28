import { readdir, readFile } from "node:fs/promises";
import { basename, extname, join } from "node:path";

import type { SynaptEntity, SynaptExtraction, SynaptGoal } from "@synapt-dev/extract";

export type GoalStatus = SynaptGoal["status"];

export interface ExtractionStore {
  list(): Promise<SynaptExtraction[]>;
  get(id: string): Promise<SynaptExtraction | null>;
}

export interface EntityTimelineEntry {
  extraction_id: string;
  state: string;
  date_hint?: string;
}

export interface ThemeCount {
  theme: string;
  count: number;
}

function extractionId(doc: SynaptExtraction): string {
  return doc.source_id ?? doc.extracted_at;
}

function normalize(s: string): string {
  return s.trim().toLowerCase();
}

function sameName(a: string, b: string): boolean {
  return normalize(a) === normalize(b);
}

function intersects(values: string[], filter: string[]): boolean {
  const wanted = new Set(filter.map(normalize));
  return values.some((value) => wanted.has(normalize(value)));
}

function isRecent(extractedAt: string, ageDays: number): boolean {
  const extracted = new Date(extractedAt).getTime();
  if (Number.isNaN(extracted)) {
    return false;
  }
  const windowMs = ageDays * 24 * 60 * 60 * 1000;
  return extracted >= Date.now() - windowMs;
}

function timelineSortKey(entry: EntityTimelineEntry, docTime: string): number {
  const preferred = entry.date_hint ?? docTime;
  const time = new Date(preferred).getTime();
  return Number.isNaN(time) ? Number.MAX_SAFE_INTEGER : time;
}

function docSearchText(doc: SynaptExtraction): string {
  const chunks: string[] = [];
  if (doc.summary) chunks.push(doc.summary);
  if (doc.sentiment) chunks.push(doc.sentiment);
  chunks.push(...doc.themes);
  for (const entity of doc.entities) {
    chunks.push(entity.name, entity.type);
    if (entity.state) chunks.push(entity.state);
    if (entity.context) chunks.push(entity.context);
  }
  for (const goal of doc.goals) {
    chunks.push(goal.text, goal.status);
  }
  for (const fact of doc.facts ?? []) {
    chunks.push(fact.text);
    if (fact.category) chunks.push(fact.category);
  }
  return chunks.join(" ").toLowerCase();
}

function uniqueEntities(entities: SynaptEntity[]): SynaptEntity[] {
  const seen = new Set<string>();
  const out: SynaptEntity[] = [];
  for (const entity of entities) {
    const key = `${entity.id ?? ""}:${normalize(entity.name)}:${normalize(entity.type)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(entity);
  }
  return out;
}

export class InMemoryStore implements ExtractionStore {
  constructor(private readonly docs: SynaptExtraction[]) {}

  async list(): Promise<SynaptExtraction[]> {
    return [...this.docs];
  }

  async get(id: string): Promise<SynaptExtraction | null> {
    return this.docs.find((doc) => doc.source_id === id) ?? null;
  }
}

export class DirectoryStore implements ExtractionStore {
  constructor(private readonly dir: string) {}

  async list(): Promise<SynaptExtraction[]> {
    const entries = await readdir(this.dir, { withFileTypes: true });
    const files = entries
      .filter((entry) => entry.isFile() && extname(entry.name) === ".json")
      .map((entry) => entry.name)
      .sort();

    const docs: SynaptExtraction[] = [];
    for (const file of files) {
      const content = await readFile(join(this.dir, file), "utf-8");
      docs.push(JSON.parse(content) as SynaptExtraction);
    }
    return docs;
  }

  async get(id: string): Promise<SynaptExtraction | null> {
    try {
      const content = await readFile(join(this.dir, `${id}.json`), "utf-8");
      return JSON.parse(content) as SynaptExtraction;
    } catch {
      const docs = await this.list();
      return docs.find((doc) => doc.source_id === id) ?? null;
    }
  }
}

export async function goals(
  store: ExtractionStore,
  filter?: { status?: GoalStatus[]; entity_refs?: string[]; age_days?: number },
): Promise<SynaptGoal[]> {
  const docs = await store.list();
  const result: SynaptGoal[] = [];

  for (const doc of docs) {
    if (filter?.age_days !== undefined && !isRecent(doc.extracted_at, filter.age_days)) {
      continue;
    }
    for (const goal of doc.goals) {
      if (filter?.status && !filter.status.includes(goal.status)) {
        continue;
      }
      if (filter?.entity_refs && !intersects(goal.entity_refs, filter.entity_refs)) {
        continue;
      }
      result.push(goal);
    }
  }

  return result;
}

export async function entityTimeline(store: ExtractionStore, name: string): Promise<EntityTimelineEntry[]> {
  const docs = await store.list();
  const entries: Array<EntityTimelineEntry & { _doc_time: string }> = [];

  for (const doc of docs) {
    for (const entity of doc.entities) {
      if (!sameName(entity.name, name) || !entity.state) {
        continue;
      }
      entries.push({
        extraction_id: extractionId(doc),
        state: entity.state,
        date_hint: entity.date_hint,
        _doc_time: doc.extracted_at,
      });
    }
  }

  entries.sort((a, b) => timelineSortKey(a, a._doc_time) - timelineSortKey(b, b._doc_time));
  return entries.map(({ _doc_time: _ignored, ...entry }) => entry);
}

export async function themesOverTime(store: ExtractionStore, windowDays: number): Promise<ThemeCount[]> {
  const docs = await store.list();
  const counts = new Map<string, number>();

  for (const doc of docs) {
    if (!isRecent(doc.extracted_at, windowDays)) {
      continue;
    }
    for (const theme of doc.themes) {
      counts.set(theme, (counts.get(theme) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .map(([theme, count]) => ({ theme, count }))
    .sort((a, b) => b.count - a.count || a.theme.localeCompare(b.theme));
}

export async function entitiesRelatedTo(
  store: ExtractionStore,
  name: string,
  via?: string,
): Promise<SynaptEntity[]> {
  const docs = await store.list();
  const related: SynaptEntity[] = [];

  for (const doc of docs) {
    const byId = new Map<string, SynaptEntity>();
    for (const entity of doc.entities) {
      if (entity.id) {
        byId.set(entity.id, entity);
      }
    }

    for (const entity of doc.entities) {
      const matchesEntity = sameName(entity.name, name);
      for (const relation of entity.relations ?? []) {
        if (via && relation.type !== via) {
          continue;
        }

        const target = byId.get(relation.target);
        if (matchesEntity && target) {
          related.push(target);
        }
        if (target && sameName(target.name, name)) {
          related.push(entity);
        }
      }
    }
  }

  return uniqueEntities(related);
}

export async function search(store: ExtractionStore, query: string): Promise<SynaptExtraction[]> {
  const docs = await store.list();
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .map((term) => term.trim())
    .filter(Boolean);

  if (terms.length === 0) {
    return [];
  }

  return docs.filter((doc) => {
    const haystack = docSearchText(doc);
    return terms.every((term) => haystack.includes(term));
  });
}
