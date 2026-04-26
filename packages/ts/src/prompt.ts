import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import type { ExtractionCapability } from "./schema.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const _installedPrompts = resolve(__dirname, "..", "prompts");
const _repoPrompts = resolve(__dirname, "..", "..", "..", "prompts");
const PROMPTS_DIR = existsSync(_installedPrompts) ? _installedPrompts : _repoPrompts;

const VALID_CAPABILITIES: Set<string> = new Set([
  "entities", "entity_state", "entity_context", "entity_ids",
  "goals", "goal_timing", "goal_entity_refs",
  "themes", "summary", "sentiment", "facts",
  "temporal_refs", "temporal_classes",
  "relations", "relation_origin",
  "assertion_signals", "evidence_anchoring",
]);

const BASE_CAPABILITIES: Set<string> = new Set(["entities", "goals", "facts"]);
const MODIFIER_ONLY: Set<string> = new Set(["assertion_signals", "evidence_anchoring"]);

export interface PromptOptions {
  capabilities?: ExtractionCapability[];
  profile?: "minimal" | "standard" | "full";
  add?: ExtractionCapability[];
  remove?: ExtractionCapability[];
  categories?: string[];
  source_type?: string;
  date?: string;
}

const CAPABILITY_DEPS: Partial<Record<ExtractionCapability, ExtractionCapability[]>> = {
  entity_state: ["entities"],
  entity_context: ["entities"],
  entity_ids: ["entities"],
  goal_timing: ["goals"],
  goal_entity_refs: ["goals", "entity_ids"],
  temporal_classes: ["temporal_refs"],
  relations: ["entities", "entity_ids"],
  relation_origin: ["relations"],
};

const CANONICAL_ORDER: ExtractionCapability[] = [
  "entities", "goals", "themes", "summary", "sentiment", "facts", "temporal_refs",
  "entity_state", "entity_context", "entity_ids",
  "goal_timing", "goal_entity_refs",
  "temporal_classes",
  "relations", "relation_origin",
  "assertion_signals", "evidence_anchoring",
];

const CAPABILITY_RULES: Partial<Record<ExtractionCapability, string>> = {
  entity_ids: 'Assign each entity a short local ID ("e1", "e2", etc.). Goals and relations reference entities by ID.',
  temporal_refs: "Resolve all relative dates to absolute dates.",
  relation_origin: 'Mark relation origin: "explicit" if stated in text, "inferred" if deduced from context, "dependent" if derived from another relation.',
  assertion_signals: 'Preserve negation, hedging, and conditions in signals. "I might move" → hedged=true. "No longer using Redis" → negated=true. "If we get funding" → condition="we get funding".',
};

function loadProfile(name: string): ExtractionCapability[] {
  const path = resolve(PROMPTS_DIR, "profiles", `${name}.json`);
  const data = JSON.parse(readFileSync(path, "utf-8")) as { capabilities: ExtractionCapability[] };
  return data.capabilities;
}

function loadFragment(name: string): string {
  const path = resolve(PROMPTS_DIR, "v1", `${name}.txt`);
  return readFileSync(path, "utf-8");
}

function renderTemplate(template: string, ctx: Record<string, unknown>): string {
  let result = template.replace(
    /\{\{#if (\w+)\}\}([\s\S]*?)\{\{\/if\}\}/g,
    (_match, varName: string, body: string) => {
      return ctx[varName] ? body : "";
    },
  );
  return renderVars(result, ctx);
}

function renderVars(template: string, ctx: Record<string, unknown>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_match, varName: string) => {
    const val = ctx[varName];
    if (Array.isArray(val)) return val.join(", ");
    return val != null ? String(val) : "";
  });
}

function validateCapabilityNames(caps: Iterable<string>, source: string): void {
  const unknown: string[] = [];
  for (const c of caps) {
    if (!VALID_CAPABILITIES.has(c)) unknown.push(c);
  }
  if (unknown.length > 0) {
    throw new Error(`Unknown ${source}: ${unknown.sort().join(", ")}`);
  }
}

export function resolveCapabilities(options: Pick<PromptOptions, "capabilities" | "profile" | "add" | "remove">): ExtractionCapability[] {
  let caps: Set<ExtractionCapability>;

  if (options.capabilities != null) {
    validateCapabilityNames(options.capabilities, "capabilities");
    caps = new Set(options.capabilities);
  } else if (options.profile != null) {
    caps = new Set(loadProfile(options.profile));
  } else {
    throw new Error("Either capabilities or profile must be provided");
  }

  if (options.add) {
    validateCapabilityNames(options.add, "capabilities in add");
    for (const c of options.add) caps.add(c);
  }
  if (options.remove) {
    for (const c of options.remove) caps.delete(c);
  }

  let changed = true;
  while (changed) {
    changed = false;
    for (const cap of [...caps]) {
      const deps = CAPABILITY_DEPS[cap];
      if (deps) {
        for (const dep of deps) {
          if (!caps.has(dep)) {
            caps.add(dep);
            changed = true;
          }
        }
      }
    }
  }

  if (caps.size === 0) {
    throw new Error("Resolved capability set is empty");
  }

  const modifiers = [...caps].filter((c) => MODIFIER_ONLY.has(c));
  if (modifiers.length > 0 && ![...caps].some((c) => BASE_CAPABILITIES.has(c))) {
    throw new Error(
      `Modifier capabilities [${modifiers.sort().join(", ")}] require at least one ` +
      `base capability (${[...BASE_CAPABILITIES].sort().join(", ")})`
    );
  }

  return [...caps].sort((a, b) => {
    const ai = CANONICAL_ORDER.indexOf(a);
    const bi = CANONICAL_ORDER.indexOf(b);
    return (ai === -1 ? CANONICAL_ORDER.length : ai) - (bi === -1 ? CANONICAL_ORDER.length : bi);
  });
}

export function buildExtractionPrompt(text: string, options: PromptOptions): string {
  if (options.capabilities != null && options.profile != null) {
    throw new Error("Cannot specify both capabilities and profile");
  }

  const resolved = resolveCapabilities(options);

  const ctx: Record<string, unknown> = {
    text,
    categories: options.categories,
    source_type: options.source_type,
    date: options.date,
  };

  const parts: string[] = [];

  const preamble = renderTemplate(loadFragment("preamble"), ctx);
  parts.push(preamble.trim());

  for (const cap of resolved) {
    const fragment = renderTemplate(loadFragment(cap), ctx);
    parts.push(fragment.trimEnd());
  }

  const rulesSection: string[] = [];
  for (const cap of resolved) {
    const rule = CAPABILITY_RULES[cap];
    if (rule) rulesSection.push(rule);
  }

  let postamble = renderTemplate(loadFragment("postamble"), ctx).trimEnd();
  if (rulesSection.length > 0) {
    const extraRules = rulesSection.map((r) => `- ${r}`).join("\n");
    const idx = postamble.indexOf("\nText:");
    if (idx >= 0) {
      postamble = postamble.slice(0, idx) + "\n" + extraRules + postamble.slice(idx);
    }
  }
  parts.push(postamble);

  return parts.join("\n") + "\n";
}
