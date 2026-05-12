export interface SynaptSourceRef {
  version: "1";
  snippet?: string;
  offset_start?: number;
  offset_end?: number;
  sentence_index?: number;
}

export interface SynaptEmbedding {
  version: "1";
  vector: number[];
  model: string;
  input: "source" | "summary" | "entities" | string;
  dimensions: number;
  space?: string;
  computed_at?: string;
}

export interface SynaptAssertionSignals {
  version: "1";
  confidence?: number;
  negated?: boolean;
  hedged?: boolean;
  condition?: string;
}

export interface SynaptTemporalRef {
  version: "1";
  raw: string;
  type?: "point" | "range" | "duration" | "unresolved";
  resolved?: string;
  resolved_end?: string;
  context?: string;
}

export interface SynaptProducerConfiguration {
  reasoning_effort?: string;
  system_prompt_hash?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  [key: string]: unknown;
}

export interface SynaptProducer {
  version: "1";
  model: string;
  model_version?: string;
  deployment?: string;
  configuration?: SynaptProducerConfiguration;
  operator?: string;
  signature?: string;
}

export interface SynaptRelation {
  target: string;
  type: string;
  origin?: string;
  signals?: SynaptAssertionSignals;
}

export interface SynaptEntity {
  id?: string;
  name: string;
  type: "person" | "place" | "event" | "concept" | "organization" | "object" | string;
  aliases?: string[];
  state?: string;
  context?: string;
  date_hint?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
  relations?: SynaptRelation[];
}

export interface SynaptGoal {
  text: string;
  status: "open" | "resolved" | "abandoned" | "in_progress";
  entity_refs: string[];
  stated_at?: string;
  resolved_at?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
}

export interface SynaptFact {
  text: string;
  category?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
}

export interface SynaptQuestion {
  text: string;
  directed_to?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
}

export interface SynaptAction {
  text: string;
  origin: "extracted" | "proposed_from_goals";
  entity_refs?: string[];
  due?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
}

export interface SynaptDecision {
  text: string;
  entity_refs?: string[];
  decided_at?: string;
  source?: SynaptSourceRef;
  signals?: SynaptAssertionSignals;
}

export interface SynaptSentiment {
  version: "1";
  valence: "positive" | "negative" | "neutral" | "mixed";
  intensity?: number;
  confidence?: number;
}

export interface SynaptSourceMetadata {
  version: "1";
  token_count?: number;
  character_count?: number;
  modality?: string;
  format?: string;
}

export const EXTRACTION_CAPABILITIES = [
  "entities", "entity_state", "entity_context", "entity_ids",
  "goals", "goal_timing", "goal_entity_refs",
  "themes", "keywords", "summary", "sentiment", "structured_sentiment",
  "facts", "questions", "actions", "decisions",
  "temporal_refs", "temporal_classes",
  "relations", "relation_origin",
  "assertion_signals", "evidence_anchoring",
  "language", "source_metadata", "confidence",
] as const;

export type ExtractionCapability = typeof EXTRACTION_CAPABILITIES[number];

export interface SynaptExtraction {
  version: "1";
  extracted_at: string;
  source_id?: string;
  source_type?: string;
  user_id?: string;
  produced_by: string | SynaptProducer;
  kind?: string;
  entities: SynaptEntity[];
  goals: SynaptGoal[];
  themes: string[];
  keywords?: string[];
  sentiment?: string | SynaptSentiment;
  summary?: string;
  facts?: SynaptFact[];
  questions?: SynaptQuestion[];
  actions?: SynaptAction[];
  decisions?: SynaptDecision[];
  temporal_refs?: SynaptTemporalRef[];
  language?: string;
  source_metadata?: SynaptSourceMetadata;
  confidence?: number;
  capabilities: ExtractionCapability[];
  embeddings?: SynaptEmbedding[];
  extensions?: Record<string, unknown>;
}
