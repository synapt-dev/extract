import { EMBEDDED_CAPABILITY_REGISTRY, EMBEDDED_PROMPT_FRAGMENTS } from "./prompt-data.js";
import { EXTRACTION_CAPABILITIES, type ExtractionCapability } from "./schema.js";

export type PromptProfile = "minimal" | "standard" | "full";

export interface CapabilityDefinition {
  name: ExtractionCapability;
  depends_on?: ExtractionCapability[];
  embedding_input?: string;
  base?: boolean;
  modifier_only?: boolean;
  rule?: string;
}

export interface CapabilityRegistry {
  version: "1";
  profiles: Record<PromptProfile, ExtractionCapability[]>;
  omit_when_absent: ExtractionCapability[];
  standard_embedding_inputs: string[];
  capabilities: CapabilityDefinition[];
}

function duplicateValues(values: string[]): string[] {
  return values.filter((value, index) => values.indexOf(value) !== index);
}

function validateCapabilityRegistry(registry: CapabilityRegistry, sourcePath: string): void {
  const schemaCapabilities = new Set<string>(EXTRACTION_CAPABILITIES);
  const names = registry.capabilities.map((definition) => definition.name);
  const registryCapabilities = new Set<string>(names);
  const duplicateCapabilities = duplicateValues(names);
  if (duplicateCapabilities.length > 0) {
    throw new Error(`Capability registry has duplicate capabilities: ${[...new Set(duplicateCapabilities)].sort().join(", ")}`);
  }

  const unknown = names.filter((name) => !schemaCapabilities.has(name));
  if (unknown.length > 0) {
    throw new Error(`Capability registry contains unknown capabilities: ${unknown.sort().join(", ")}`);
  }

  const missing = EXTRACTION_CAPABILITIES.filter((name) => !registryCapabilities.has(name));
  if (missing.length > 0) {
    throw new Error(`Capability registry is missing schema capabilities: ${missing.sort().join(", ")}`);
  }

  for (const definition of registry.capabilities) {
    for (const dep of definition.depends_on ?? []) {
      if (!registryCapabilities.has(dep)) {
        throw new Error(`Capability registry dependency ${definition.name}.${dep} is not a valid capability`);
      }
    }
  }

  for (const [profile, capabilities] of Object.entries(registry.profiles) as [PromptProfile, ExtractionCapability[]][]) {
    for (const capability of capabilities) {
      if (!registryCapabilities.has(capability)) {
        throw new Error(`Profile ${profile} references unknown capability ${capability}`);
      }
    }
  }

  for (const capability of registry.omit_when_absent) {
    if (!registryCapabilities.has(capability)) {
      throw new Error(`omit_when_absent references unknown capability ${capability}`);
    }
  }

  if (!Array.isArray(registry.standard_embedding_inputs)) {
    throw new Error("Capability registry standard_embedding_inputs must be a list");
  }
  const embeddingInputs = new Set<string>([
    "source",
    ...registry.capabilities
      .map((definition) => definition.embedding_input)
      .filter((input): input is string => typeof input === "string"),
  ]);
  for (const input of registry.standard_embedding_inputs) {
    if (!embeddingInputs.has(input)) {
      throw new Error(`standard_embedding_inputs references unknown input ${input}`);
    }
  }

  if (registry.version !== "1") {
    throw new Error(`Unsupported capability registry version in ${sourcePath}: ${registry.version}`);
  }
}

function loadCapabilityRegistry(): CapabilityRegistry {
  const registry = EMBEDDED_CAPABILITY_REGISTRY;
  validateCapabilityRegistry(registry, "embedded capability registry");
  return registry;
}

export const CAPABILITY_REGISTRY = loadCapabilityRegistry();
export const CANONICAL_ORDER: ExtractionCapability[] = CAPABILITY_REGISTRY.capabilities.map((definition) => definition.name);
export const VALID_CAPABILITIES: Set<string> = new Set(CANONICAL_ORDER);
export const BASE_CAPABILITIES: Set<string> = new Set(
  CAPABILITY_REGISTRY.capabilities
    .filter((definition) => definition.base === true)
    .map((definition) => definition.name),
);
export const MODIFIER_ONLY: Set<string> = new Set(
  CAPABILITY_REGISTRY.capabilities
    .filter((definition) => definition.modifier_only === true)
    .map((definition) => definition.name),
);
export const CAPABILITY_DEPS: Partial<Record<ExtractionCapability, ExtractionCapability[]>> = Object.fromEntries(
  CAPABILITY_REGISTRY.capabilities
    .filter((definition) => definition.depends_on !== undefined)
    .map((definition) => [definition.name, definition.depends_on ?? []]),
) as Partial<Record<ExtractionCapability, ExtractionCapability[]>>;
export const CAPABILITY_RULES: Partial<Record<ExtractionCapability, string>> = Object.fromEntries(
  CAPABILITY_REGISTRY.capabilities
    .filter((definition) => definition.rule !== undefined)
    .map((definition) => [definition.name, definition.rule ?? ""]),
) as Partial<Record<ExtractionCapability, string>>;
export const OMIT_WHEN_ABSENT: ExtractionCapability[] = CAPABILITY_REGISTRY.omit_when_absent;
export const STANDARD_EMBEDDING_INPUTS: string[] = CAPABILITY_REGISTRY.standard_embedding_inputs;
const CAPABILITY_EMBEDDING_INPUTS: Partial<Record<ExtractionCapability, string>> = Object.fromEntries(
  CAPABILITY_REGISTRY.capabilities
    .filter((definition) => definition.embedding_input !== undefined)
    .map((definition) => [definition.name, definition.embedding_input ?? ""]),
) as Partial<Record<ExtractionCapability, string>>;

export interface PromptOptions {
  capabilities?: CapabilityInput[];
  profile?: PromptProfile;
  add?: CapabilityInput[];
  remove?: CapabilityInput[];
  categories?: string[];
  source_type?: string;
  date?: string;
  stage?: "stage1";
  extracted_at?: string;
  compact?: boolean;
  embed?: Partial<Record<ExtractionCapability, boolean>>;
}

export interface CapabilitySpec {
  name?: ExtractionCapability;
  capability?: ExtractionCapability;
  embed?: boolean;
  embedding?: boolean;
}

export type CapabilityInput = ExtractionCapability | CapabilitySpec;

function buildRunConstraintRules(resolved: ExtractionCapability[], options: PromptOptions): string[] {
  const rules: string[] = [];
  const capabilities = new Set(resolved);

  if (options.compact) {
    rules.push("Keep this extraction compact and high signal.");
  }
  if (options.extracted_at) {
    rules.push(`Use exactly this extracted_at value: ${options.extracted_at}.`);
  }
  if (options.stage === "stage1") {
    rules.push(
      "Produce Stage 1 content only. Do not include version, produced_by, source_id, source_type, kind, capabilities, extensions, or embeddings.",
    );
    rules.push("Only include fields represented in the requested capability set and response schema.");
  }
  if (capabilities.has("entity_ids")) {
    rules.push('Entity IDs are extraction-local only. Use short IDs like "e1", "e2", "e3".');
  }
  if (capabilities.has("goal_entity_refs")) {
    rules.push("Every goal.entity_refs entry must refer to one of the entity IDs you emit.");
  }
  if (!capabilities.has("temporal_refs")) {
    rules.push("Omit temporal_refs for this run.");
  }
  if (!capabilities.has("relations")) {
    rules.push("Omit relations for this run.");
  }
  if (!capabilities.has("assertion_signals")) {
    rules.push("Omit signals fields for this run.");
  }
  for (const capability of OMIT_WHEN_ABSENT) {
    if (!capabilities.has(capability)) {
      rules.push(`Omit ${capability} for this run.`);
    }
  }

  return rules;
}

function loadProfile(name: string): ExtractionCapability[] {
  if (!(name in CAPABILITY_REGISTRY.profiles)) {
    throw new Error(`Unknown profile: ${name}`);
  }
  return CAPABILITY_REGISTRY.profiles[name as PromptProfile];
}

export function profileCapabilities(profile: NonNullable<PromptOptions["profile"]>): ExtractionCapability[] {
  return [...loadProfile(profile)];
}

export function capabilityEmbeddingInput(capability: ExtractionCapability): string | undefined {
  return CAPABILITY_EMBEDDING_INPUTS[capability];
}

function loadFragment(name: string): string {
  const fragment = EMBEDDED_PROMPT_FRAGMENTS[name];
  if (fragment === undefined) {
    throw new Error(`Missing prompt fragment: ${name}`);
  }
  return fragment;
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

export function capabilityName(capability: CapabilityInput): ExtractionCapability {
  if (typeof capability === "string") return capability;
  const name = capability.name ?? capability.capability;
  if (typeof name !== "string") {
    throw new Error("Capability object must include name or capability");
  }
  return name;
}

export function normalizeCapabilityInputs(capabilities: Iterable<CapabilityInput>): ExtractionCapability[] {
  return [...capabilities].map(capabilityName);
}

export function capabilityRequestsEmbedding(capability: CapabilityInput): boolean {
  return typeof capability === "object" && capability !== null && (capability.embed === true || capability.embedding === true);
}

export function capabilityEmbeddingPreference(capability: CapabilityInput): boolean | undefined {
  if (typeof capability !== "object" || capability === null) return undefined;
  if (typeof capability.embed === "boolean") return capability.embed;
  if (typeof capability.embedding === "boolean") return capability.embedding;
  return undefined;
}

function validateCapabilityNames(caps: Iterable<CapabilityInput>, source: string): void {
  const unknown: string[] = [];
  for (const c of caps) {
    const name = capabilityName(c);
    if (!VALID_CAPABILITIES.has(name)) unknown.push(name);
  }
  if (unknown.length > 0) {
    throw new Error(`Unknown ${source}: ${unknown.sort().join(", ")}`);
  }
}

export function resolveCapabilities(options: Pick<PromptOptions, "capabilities" | "profile" | "add" | "remove">): ExtractionCapability[] {
  let caps: Set<ExtractionCapability>;

  if (options.capabilities != null) {
    validateCapabilityNames(options.capabilities, "capabilities");
    caps = new Set(normalizeCapabilityInputs(options.capabilities));
  } else if (options.profile != null) {
    caps = new Set(loadProfile(options.profile));
  } else {
    throw new Error("Either capabilities or profile must be provided");
  }

  if (options.add) {
    validateCapabilityNames(options.add, "capabilities in add");
    for (const c of normalizeCapabilityInputs(options.add)) caps.add(c);
  }
  const removed = new Set(options.remove ? expandCapabilityExclusions(normalizeCapabilityInputs(options.remove)) : []);

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

  for (const c of removed) caps.delete(c);

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

export function expandCapabilityExclusions(capabilities: Iterable<ExtractionCapability>): ExtractionCapability[] {
  const excluded = new Set(capabilities);
  let changed = true;
  while (changed) {
    changed = false;
    for (const capability of VALID_CAPABILITIES as Set<ExtractionCapability>) {
      if (excluded.has(capability)) continue;
      const deps = CAPABILITY_DEPS[capability] ?? [];
      if (deps.some((dep) => excluded.has(dep))) {
        excluded.add(capability);
        changed = true;
      }
    }
  }
  return [...excluded].sort((a, b) => CANONICAL_ORDER.indexOf(a) - CANONICAL_ORDER.indexOf(b));
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
  const runConstraintRules = buildRunConstraintRules(resolved, options);

  let postamble = renderTemplate(loadFragment("postamble"), ctx).trimEnd();
  if (rulesSection.length > 0 || runConstraintRules.length > 0) {
    const extraBlocks: string[] = [];
    if (rulesSection.length > 0) {
      extraBlocks.push(rulesSection.map((r) => `- ${r}`).join("\n"));
    }
    if (runConstraintRules.length > 0) {
      extraBlocks.push(`Additional run constraints:\n${runConstraintRules.map((r) => `- ${r}`).join("\n")}`);
    }
    const extraRules = extraBlocks.join("\n");
    const idx = postamble.indexOf("\nText:");
    if (idx >= 0) {
      postamble = postamble.slice(0, idx) + "\n" + extraRules + postamble.slice(idx);
    }
  }
  parts.push(postamble);

  return parts.join("\n") + "\n";
}
