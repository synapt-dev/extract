import type { SynaptExtraction, ExtractionCapability } from "./schema.js";

export interface ValidationError {
  path: string;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

const VALID_CAPABILITIES: Set<string> = new Set([
  "entities", "entity_state", "entity_context", "entity_ids",
  "goals", "goal_timing", "goal_entity_refs",
  "themes", "summary", "sentiment", "facts",
  "temporal_refs", "temporal_classes",
  "relations", "relation_origin",
  "assertion_signals", "evidence_anchoring",
]);

const VALID_GOAL_STATUSES: Set<string> = new Set([
  "open", "resolved", "abandoned", "in_progress",
]);

const VALID_TEMPORAL_TYPES: Set<string> = new Set([
  "point", "range", "duration", "unresolved",
]);

const URI_RE = /^[a-zA-Z][a-zA-Z0-9+.\-]*:\/\/\S+$/;
const NAMESPACED_RE = /^[a-zA-Z0-9_\-]+\/[a-zA-Z0-9_\-]+$/;
const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?)?$/;
const ISO_DATETIME_STRICT_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?$/;

const ROOT_KEYS = new Set([
  "version", "extracted_at", "source_id", "source_type", "user_id",
  "produced_by", "kind", "entities", "goals", "themes", "sentiment",
  "summary", "facts", "temporal_refs", "capabilities", "embeddings", "extensions",
]);
const ENTITY_KEYS = new Set([
  "id", "name", "type", "state", "context", "date_hint",
  "source", "signals", "relations",
]);
const GOAL_KEYS = new Set([
  "text", "status", "entity_refs", "stated_at", "resolved_at",
  "source", "signals",
]);
const FACT_KEYS = new Set(["text", "category", "source", "signals"]);
const RELATION_KEYS = new Set(["target", "type", "origin", "signals"]);
const SOURCE_REF_KEYS = new Set(["version", "snippet", "offset_start", "offset_end", "sentence_index"]);
const SIGNALS_KEYS = new Set(["version", "confidence", "negated", "hedged", "condition"]);
const TEMPORAL_REF_KEYS = new Set(["version", "raw", "type", "resolved", "resolved_end", "context"]);
const EMBEDDING_KEYS = new Set(["version", "vector", "model", "input", "dimensions", "space", "computed_at"]);

function isUri(s: string): boolean {
  return URI_RE.test(s);
}

function isIsoDatetime(s: string): boolean {
  return ISO_DATE_RE.test(s);
}

function isIsoDatetimeStrict(s: string): boolean {
  return ISO_DATETIME_STRICT_RE.test(s);
}

function isNamespaced(s: string): boolean {
  return NAMESPACED_RE.test(s);
}

function checkExtraKeys(obj: Record<string, unknown>, allowed: Set<string>, path: string, errors: ValidationError[]): void {
  for (const key of Object.keys(obj)) {
    if (!allowed.has(key)) {
      const fullPath = path ? `${path}.${key}` : key;
      errors.push({ path: fullPath, message: "additional property not allowed" });
    }
  }
}

function checkOptionalStr(obj: Record<string, unknown>, key: string, path: string, errors: ValidationError[]): void {
  if (obj[key] !== undefined && typeof obj[key] !== "string") {
    errors.push({ path: `${path}.${key}`, message: "must be a string" });
  }
}

function checkOptionalNonNegInt(obj: Record<string, unknown>, key: string, path: string, errors: ValidationError[]): void {
  if (obj[key] !== undefined) {
    const val = obj[key];
    if (typeof val !== "number" || !Number.isInteger(val) || val < 0) {
      errors.push({ path: `${path}.${key}`, message: "must be a non-negative integer" });
    }
  }
}

function hasPayloadBeyondVersion(obj: Record<string, unknown>): boolean {
  return Object.keys(obj).some((k) => k !== "version");
}

function validateSourceRef(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const ref = obj as Record<string, unknown>;
  checkExtraKeys(ref, SOURCE_REF_KEYS, path, errors);
  if (ref.version !== "1") {
    errors.push({ path: `${path}.version`, message: "must be \"1\"" });
  }
  if (!hasPayloadBeyondVersion(ref)) {
    errors.push({ path, message: "empty sub-schema (only version); must contain at least one payload field" });
    return;
  }
  checkOptionalStr(ref, "snippet", path, errors);
  checkOptionalNonNegInt(ref, "offset_start", path, errors);
  checkOptionalNonNegInt(ref, "offset_end", path, errors);
  checkOptionalNonNegInt(ref, "sentence_index", path, errors);
}

function validateSignals(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const sig = obj as Record<string, unknown>;
  checkExtraKeys(sig, SIGNALS_KEYS, path, errors);
  if (sig.version !== "1") {
    errors.push({ path: `${path}.version`, message: "must be \"1\"" });
  }
  if (!hasPayloadBeyondVersion(sig)) {
    errors.push({ path, message: "empty sub-schema (only version); must contain at least one payload field" });
    return;
  }
  if (sig.confidence !== undefined) {
    if (typeof sig.confidence !== "number" || sig.confidence < 0 || sig.confidence > 1) {
      errors.push({ path: `${path}.confidence`, message: "must be a number between 0.0 and 1.0" });
    }
  }
  if (sig.negated !== undefined && typeof sig.negated !== "boolean") {
    errors.push({ path: `${path}.negated`, message: "must be a boolean" });
  }
  if (sig.hedged !== undefined && typeof sig.hedged !== "boolean") {
    errors.push({ path: `${path}.hedged`, message: "must be a boolean" });
  }
  if (sig.condition !== undefined && typeof sig.condition !== "string") {
    errors.push({ path: `${path}.condition`, message: "must be a string" });
  }
}

function validateEmbedding(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const emb = obj as Record<string, unknown>;
  checkExtraKeys(emb, EMBEDDING_KEYS, path, errors);
  if (emb.version !== "1") {
    errors.push({ path: `${path}.version`, message: "must be \"1\"" });
  }
  const vector = emb.vector;
  if (!Array.isArray(vector)) {
    errors.push({ path: `${path}.vector`, message: "required array" });
  }
  if (typeof emb.model !== "string") {
    errors.push({ path: `${path}.model`, message: "required string" });
  } else if (!isUri(emb.model)) {
    errors.push({ path: `${path}.model`, message: "must be a provider URI (scheme://identifier)" });
  }
  if (typeof emb.input !== "string") {
    errors.push({ path: `${path}.input`, message: "required string" });
  }
  if (typeof emb.dimensions !== "number" || !Number.isInteger(emb.dimensions) || emb.dimensions < 1) {
    errors.push({ path: `${path}.dimensions`, message: "required positive integer" });
  } else if (Array.isArray(vector) && emb.dimensions !== vector.length) {
    errors.push({ path: `${path}.dimensions`, message: `dimensions (${emb.dimensions}) must equal vector length (${vector.length})` });
  }
  checkOptionalStr(emb, "space", path, errors);
  if (emb.computed_at !== undefined) {
    if (typeof emb.computed_at !== "string" || !isIsoDatetimeStrict(emb.computed_at)) {
      errors.push({ path: `${path}.computed_at`, message: "must be a valid ISO 8601 date-time" });
    }
  }
}

function validateRelation(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const rel = obj as Record<string, unknown>;
  checkExtraKeys(rel, RELATION_KEYS, path, errors);
  if (typeof rel.target !== "string" || rel.target.length === 0) {
    errors.push({ path: `${path}.target`, message: "required non-empty string" });
  }
  if (typeof rel.type !== "string" || rel.type.length === 0) {
    errors.push({ path: `${path}.type`, message: "required non-empty string" });
  }
  checkOptionalStr(rel, "origin", path, errors);
  if (rel.signals !== undefined) {
    validateSignals(rel.signals, `${path}.signals`, errors);
  }
}

function validateEntity(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const ent = obj as Record<string, unknown>;
  checkExtraKeys(ent, ENTITY_KEYS, path, errors);
  if (typeof ent.name !== "string" || ent.name.length === 0) {
    errors.push({ path: `${path}.name`, message: "required non-empty string" });
  }
  if (typeof ent.type !== "string" || ent.type.length === 0) {
    errors.push({ path: `${path}.type`, message: "required non-empty string" });
  }
  checkOptionalStr(ent, "id", path, errors);
  checkOptionalStr(ent, "state", path, errors);
  checkOptionalStr(ent, "context", path, errors);
  checkOptionalStr(ent, "date_hint", path, errors);
  if (ent.source !== undefined) {
    validateSourceRef(ent.source, `${path}.source`, errors);
  }
  if (ent.signals !== undefined) {
    validateSignals(ent.signals, `${path}.signals`, errors);
  }
  if (ent.relations !== undefined) {
    if (!Array.isArray(ent.relations)) {
      errors.push({ path: `${path}.relations`, message: "must be an array" });
    } else {
      for (let i = 0; i < ent.relations.length; i++) {
        validateRelation(ent.relations[i], `${path}.relations[${i}]`, errors);
      }
    }
  }
}

function validateGoal(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const goal = obj as Record<string, unknown>;
  checkExtraKeys(goal, GOAL_KEYS, path, errors);
  if (typeof goal.text !== "string" || goal.text.length === 0) {
    errors.push({ path: `${path}.text`, message: "required non-empty string" });
  }
  if (typeof goal.status !== "string" || !VALID_GOAL_STATUSES.has(goal.status)) {
    errors.push({ path: `${path}.status`, message: "must be one of: open, resolved, abandoned, in_progress" });
  }
  if (!Array.isArray(goal.entity_refs)) {
    errors.push({ path: `${path}.entity_refs`, message: "required array of strings" });
  } else {
    for (let i = 0; i < goal.entity_refs.length; i++) {
      if (typeof goal.entity_refs[i] !== "string") {
        errors.push({ path: `${path}.entity_refs[${i}]`, message: "must be a string" });
      }
    }
  }
  if (goal.stated_at !== undefined) {
    if (typeof goal.stated_at !== "string" || !isIsoDatetime(goal.stated_at)) {
      errors.push({ path: `${path}.stated_at`, message: "must be a valid ISO 8601 date/datetime" });
    }
  }
  if (goal.resolved_at !== undefined) {
    if (typeof goal.resolved_at !== "string" || !isIsoDatetime(goal.resolved_at)) {
      errors.push({ path: `${path}.resolved_at`, message: "must be a valid ISO 8601 date/datetime" });
    }
  }
  if (goal.source !== undefined) {
    validateSourceRef(goal.source, `${path}.source`, errors);
  }
  if (goal.signals !== undefined) {
    validateSignals(goal.signals, `${path}.signals`, errors);
  }
}

function validateFact(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const fact = obj as Record<string, unknown>;
  checkExtraKeys(fact, FACT_KEYS, path, errors);
  if (typeof fact.text !== "string" || fact.text.length === 0) {
    errors.push({ path: `${path}.text`, message: "required non-empty string" });
  }
  checkOptionalStr(fact, "category", path, errors);
  if (fact.source !== undefined) {
    validateSourceRef(fact.source, `${path}.source`, errors);
  }
  if (fact.signals !== undefined) {
    validateSignals(fact.signals, `${path}.signals`, errors);
  }
}

function validateTemporalRef(obj: unknown, path: string, errors: ValidationError[]): void {
  if (typeof obj !== "object" || obj === null) {
    errors.push({ path, message: "must be an object" });
    return;
  }
  const ref = obj as Record<string, unknown>;
  checkExtraKeys(ref, TEMPORAL_REF_KEYS, path, errors);
  if (ref.version !== "1") {
    errors.push({ path: `${path}.version`, message: "must be \"1\"" });
  }
  if (typeof ref.raw !== "string" || ref.raw.length === 0) {
    errors.push({ path: `${path}.raw`, message: "required non-empty string" });
  }
  if (ref.type !== undefined) {
    if (typeof ref.type !== "string" || !VALID_TEMPORAL_TYPES.has(ref.type)) {
      errors.push({ path: `${path}.type`, message: "must be one of: point, range, duration, unresolved" });
    } else if (ref.type === "range" && ref.resolved_end === undefined) {
      errors.push({ path: `${path}.resolved_end`, message: "required when type is 'range'" });
    } else if (ref.type === "unresolved") {
      if (ref.resolved !== undefined) {
        errors.push({ path: `${path}.resolved`, message: "must not be present when type is 'unresolved'" });
      }
      if (ref.resolved_end !== undefined) {
        errors.push({ path: `${path}.resolved_end`, message: "must not be present when type is 'unresolved'" });
      }
    }
  }
  if (ref.resolved !== undefined) {
    if (typeof ref.resolved !== "string" || !isIsoDatetime(ref.resolved)) {
      errors.push({ path: `${path}.resolved`, message: "must be a valid ISO 8601 date/datetime" });
    }
  }
  if (ref.resolved_end !== undefined) {
    if (typeof ref.resolved_end !== "string" || !isIsoDatetime(ref.resolved_end)) {
      errors.push({ path: `${path}.resolved_end`, message: "must be a valid ISO 8601 date/datetime" });
    }
  }
  checkOptionalStr(ref, "context", path, errors);
}

export function validateExtraction(obj: unknown): ValidationResult {
  const errors: ValidationError[] = [];

  if (typeof obj !== "object" || obj === null) {
    return { valid: false, errors: [{ path: "", message: "must be an object" }] };
  }

  const doc = obj as Record<string, unknown>;

  checkExtraKeys(doc, ROOT_KEYS, "", errors);

  if (doc.version !== "1") {
    errors.push({ path: "version", message: "must be \"1\"" });
  }

  if (typeof doc.extracted_at !== "string") {
    errors.push({ path: "extracted_at", message: "required string (ISO 8601 date-time)" });
  } else if (!isIsoDatetimeStrict(doc.extracted_at)) {
    errors.push({ path: "extracted_at", message: "must be a valid ISO 8601 date-time (e.g. 2026-04-26T12:00:00Z)" });
  }

  if (typeof doc.produced_by !== "string") {
    errors.push({ path: "produced_by", message: "required string (provider URI)" });
  } else if (!isUri(doc.produced_by)) {
    errors.push({ path: "produced_by", message: "must be a provider URI (scheme://identifier)" });
  }

  if (doc.kind !== undefined) {
    if (typeof doc.kind !== "string" || !isNamespaced(doc.kind)) {
      errors.push({ path: "kind", message: "must be namespaced (e.g. 'conversa/prayer')" });
    }
  }

  checkOptionalStr(doc, "sentiment", "", errors);
  checkOptionalStr(doc, "source_id", "", errors);
  checkOptionalStr(doc, "source_type", "", errors);
  checkOptionalStr(doc, "user_id", "", errors);

  if (doc.extensions !== undefined) {
    if (typeof doc.extensions !== "object" || doc.extensions === null || Array.isArray(doc.extensions)) {
      errors.push({ path: "extensions", message: "must be an object" });
    } else {
      for (const key of Object.keys(doc.extensions as Record<string, unknown>)) {
        if (!isNamespaced(key)) {
          errors.push({ path: `extensions.${key}`, message: "extension key must be namespaced (e.g. 'conversa/prayer')" });
        }
      }
    }
  }

  const entityIds: Set<string> = new Set();
  if (!Array.isArray(doc.entities)) {
    errors.push({ path: "entities", message: "required array" });
  } else {
    for (let i = 0; i < doc.entities.length; i++) {
      validateEntity(doc.entities[i], `entities[${i}]`, errors);
      const ent = doc.entities[i] as Record<string, unknown> | undefined;
      if (ent && typeof ent.id === "string") {
        entityIds.add(ent.id);
      }
    }
  }

  if (!Array.isArray(doc.goals)) {
    errors.push({ path: "goals", message: "required array" });
  } else {
    for (let i = 0; i < doc.goals.length; i++) {
      validateGoal(doc.goals[i], `goals[${i}]`, errors);
      const goal = doc.goals[i] as Record<string, unknown> | undefined;
      if (goal && Array.isArray(goal.entity_refs)) {
        for (let j = 0; j < goal.entity_refs.length; j++) {
          const ref = goal.entity_refs[j];
          if (typeof ref === "string" && !entityIds.has(ref)) {
            errors.push({
              path: `goals[${i}].entity_refs[${j}]`,
              message: `references entity ID '${ref}' which is not declared in entities`,
            });
          }
        }
      }
    }
  }

  if (!Array.isArray(doc.themes)) {
    errors.push({ path: "themes", message: "required array" });
  } else {
    for (let i = 0; i < doc.themes.length; i++) {
      if (typeof doc.themes[i] !== "string" || (doc.themes[i] as string).length === 0) {
        errors.push({ path: `themes[${i}]`, message: "must be a non-empty string" });
      }
    }
  }

  if (doc.summary !== undefined) {
    if (typeof doc.summary !== "string" || (doc.summary as string).length === 0) {
      errors.push({ path: "summary", message: "must be a non-empty string" });
    }
  }

  if (!Array.isArray(doc.capabilities)) {
    errors.push({ path: "capabilities", message: "required array" });
  } else {
    for (let i = 0; i < doc.capabilities.length; i++) {
      if (typeof doc.capabilities[i] !== "string") {
        errors.push({ path: `capabilities[${i}]`, message: "must be a string" });
      } else if (!VALID_CAPABILITIES.has(doc.capabilities[i] as string)) {
        errors.push({ path: `capabilities[${i}]`, message: `unknown capability: "${doc.capabilities[i]}"` });
      }
    }
  }

  if (doc.facts !== undefined) {
    if (!Array.isArray(doc.facts)) {
      errors.push({ path: "facts", message: "must be an array" });
    } else {
      for (let i = 0; i < doc.facts.length; i++) {
        validateFact(doc.facts[i], `facts[${i}]`, errors);
      }
    }
  }

  if (doc.temporal_refs !== undefined) {
    if (!Array.isArray(doc.temporal_refs)) {
      errors.push({ path: "temporal_refs", message: "must be an array" });
    } else {
      for (let i = 0; i < doc.temporal_refs.length; i++) {
        validateTemporalRef(doc.temporal_refs[i], `temporal_refs[${i}]`, errors);
      }
    }
  }

  if (doc.embeddings !== undefined) {
    if (!Array.isArray(doc.embeddings)) {
      errors.push({ path: "embeddings", message: "must be an array" });
    } else {
      for (let i = 0; i < doc.embeddings.length; i++) {
        validateEmbedding(doc.embeddings[i], `embeddings[${i}]`, errors);
      }
    }
  }

  if (entityIds.size > 0 && Array.isArray(doc.entities)) {
    for (let i = 0; i < doc.entities.length; i++) {
      const ent = doc.entities[i] as Record<string, unknown>;
      if (ent && Array.isArray(ent.relations)) {
        for (let j = 0; j < ent.relations.length; j++) {
          const rel = ent.relations[j] as Record<string, unknown>;
          if (rel && typeof rel.target === "string" && rel.target.length > 0 && !entityIds.has(rel.target)) {
            errors.push({
              path: `entities[${i}].relations[${j}].target`,
              message: `references entity ID '${rel.target}' which is not declared in entities`,
            });
          }
        }
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
