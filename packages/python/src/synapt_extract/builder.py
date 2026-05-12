"""Builder helpers for coupled SynaptExtraction prompts and Stage 1 schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synapt_extract.finalize import FinalizeContext, FinalizeResult, finalize_extraction
from synapt_extract.prompt import (
    CANONICAL_ORDER,
    STANDARD_EMBEDDING_INPUTS,
    build_extraction_prompt,
    capability_embedding_input,
    capability_embedding_preference,
    capability_name,
    expand_capability_exclusions,
    normalize_capability_inputs,
    profile_capabilities,
    resolve_capabilities,
)


JsonSchema = dict[str, Any]

DEFAULT_RESPONSE_FORMAT_NAME = "synapt_extraction_stage1"
ALL_CAPABILITIES = list(CANONICAL_ORDER)


def _capability_specs(capabilities: list[str], embed: bool | None = None) -> list[Any]:
    if embed is None:
        return list(capabilities)
    return [{"name": capability, "embed": embed} for capability in capabilities]


def _embedding_inputs_from_capabilities(capabilities: list[Any] | None, resolved: set[str]) -> list[str]:
    if not capabilities:
        return []
    selected: list[str] = []
    for capability in capabilities:
        if capability_embedding_preference(capability) is not True:
            continue
        name = capability_name(capability)
        if name not in resolved:
            continue
        input_name = capability_embedding_input(name)
        if input_name is not None and input_name not in selected:
            selected.append(input_name)
    return selected


def _apply_embedding_overrides(inputs: list[str], overrides: dict[str, bool] | None) -> list[str]:
    if not overrides:
        return inputs
    selected = list(inputs)
    for capability, enabled in overrides.items():
        input_name = capability_embedding_input(capability)
        if input_name is None:
            continue
        if enabled and input_name not in selected:
            selected.append(input_name)
        if not enabled and input_name in selected:
            selected.remove(input_name)
    return selected


def _embedding_inputs_from_selection(selection: str | list[Any] | None) -> list[str]:
    if selection is None:
        return []
    if selection == "all":
        return list(STANDARD_EMBEDDING_INPUTS)
    if not isinstance(selection, list):
        return []
    values: list[str] = []
    for input_item in selection:
        if isinstance(input_item, str):
            values.append(input_item)
        elif isinstance(input_item, dict) and isinstance(input_item.get("input"), str):
            values.append(input_item["input"])
    return values


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _order_embedding_inputs(values: list[str]) -> list[str]:
    standard = [input_name for input_name in STANDARD_EMBEDDING_INPUTS if input_name in values]
    custom = [input_name for input_name in values if input_name not in STANDARD_EMBEDDING_INPUTS]
    return [*standard, *custom]


def _source_ref_schema(finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "snippet": {"type": "string"},
        "offset_start": {"type": "integer", "minimum": 0},
        "offset_end": {"type": "integer", "minimum": 0},
        "sentence_index": {"type": "integer", "minimum": 0},
    }
    required = ["version"] if finalized else ["snippet"]
    if finalized:
        properties["version"] = {"const": "1"}

    schema: JsonSchema = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }
    if finalized:
        schema["minProperties"] = 2
    return schema


def _signals_schema(finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "negated": {"type": "boolean"},
        "hedged": {"type": "boolean"},
        "condition": {"type": "string"},
    }
    if finalized:
        properties["version"] = {"const": "1"}

    schema: JsonSchema = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": ["version"] if finalized else [],
    }
    if finalized:
        schema["minProperties"] = 2
    return schema


def _relation_schema(capabilities: set[str], finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "target": {"type": "string"},
        "type": {"type": "string"},
    }
    required = ["target", "type"]

    if "relation_origin" in capabilities:
        properties["origin"] = {"type": "string", "enum": ["explicit", "inferred", "dependent"]}
    if "assertion_signals" in capabilities:
        properties["signals"] = _signals_schema(finalized)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _entity_schema(capabilities: set[str], finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "name": {"type": "string"},
        "type": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
    }
    required = ["name", "type"]

    if "entity_ids" in capabilities:
        properties["id"] = {"type": "string"}
        required.append("id")
    if "entity_state" in capabilities:
        properties["state"] = {"type": "string"}
    if "entity_context" in capabilities:
        properties["context"] = {"type": "string"}
        properties["date_hint"] = {"type": "string"}
    if "relations" in capabilities:
        properties["relations"] = {"type": "array", "items": _relation_schema(capabilities, finalized)}
    if "assertion_signals" in capabilities:
        properties["signals"] = _signals_schema(finalized)
    if "evidence_anchoring" in capabilities:
        properties["source"] = _source_ref_schema(finalized)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _goal_schema(capabilities: set[str], finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "text": {"type": "string"},
        "status": {"type": "string", "enum": ["open", "resolved", "abandoned", "in_progress"]},
        "entity_refs": {"type": "array", "items": {"type": "string"}},
    }
    required = ["text", "status", "entity_refs"]

    if "goal_timing" in capabilities:
        properties["stated_at"] = {"type": "string"}
        properties["resolved_at"] = {"type": "string"}
    if "assertion_signals" in capabilities:
        properties["signals"] = _signals_schema(finalized)
    if "evidence_anchoring" in capabilities:
        properties["source"] = _source_ref_schema(finalized)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _fact_schema(capabilities: set[str], finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "text": {"type": "string"},
        "category": {"type": "string"},
    }
    required = ["text"]

    if "assertion_signals" in capabilities:
        properties["signals"] = _signals_schema(finalized)
    if "evidence_anchoring" in capabilities:
        properties["source"] = _source_ref_schema(finalized)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _question_schema(capabilities: set[str], finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "text": {"type": "string"},
        "directed_to": {"type": "string"},
    }
    required = ["text"]

    if "assertion_signals" in capabilities:
        properties["signals"] = _signals_schema(finalized)
    if "evidence_anchoring" in capabilities:
        properties["source"] = _source_ref_schema(finalized)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _action_schema(capabilities: set[str], finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "text": {"type": "string"},
        "origin": {"type": "string", "enum": ["extracted", "proposed_from_goals"]},
        "entity_refs": {"type": "array", "items": {"type": "string"}},
        "due": {"type": "string"},
    }
    required = ["text", "origin"]

    if "assertion_signals" in capabilities:
        properties["signals"] = _signals_schema(finalized)
    if "evidence_anchoring" in capabilities:
        properties["source"] = _source_ref_schema(finalized)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _decision_schema(capabilities: set[str], finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "text": {"type": "string"},
        "entity_refs": {"type": "array", "items": {"type": "string"}},
        "decided_at": {"type": "string"},
    }
    required = ["text"]

    if "assertion_signals" in capabilities:
        properties["signals"] = _signals_schema(finalized)
    if "evidence_anchoring" in capabilities:
        properties["source"] = _source_ref_schema(finalized)

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _sentiment_schema(finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "valence": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]},
        "intensity": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    required = ["valence"]
    if finalized:
        properties["version"] = {"const": "1"}
        required.insert(0, "version")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _source_metadata_schema(finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "token_count": {"type": "integer", "minimum": 0},
        "character_count": {"type": "integer", "minimum": 0},
        "modality": {"type": "string"},
        "format": {"type": "string"},
    }
    required: list[str] = []
    if finalized:
        properties["version"] = {"const": "1"}
        required.append("version")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _temporal_ref_schema(capabilities: set[str], finalized: bool = False) -> JsonSchema:
    properties: JsonSchema = {
        "raw": {"type": "string"},
        "resolved": {"type": "string"},
    }
    required = ["raw"]

    if finalized:
        properties["version"] = {"const": "1"}
        required.insert(0, "version")

    if "temporal_classes" in capabilities:
        properties["type"] = {"type": "string", "enum": ["point", "range", "duration", "unresolved"]}
        properties["resolved_end"] = {"type": "string"}
        properties["context"] = {"type": "string"}
        required.append("type")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _producer_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"const": "1"},
            "model": {"type": "string"},
            "model_version": {"type": "string"},
            "deployment": {"type": "string"},
            "configuration": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "reasoning_effort": {"type": "string"},
                    "system_prompt_hash": {"type": "string"},
                    "temperature": {"type": "number", "minimum": 0},
                    "top_p": {"type": "number", "minimum": 0, "maximum": 1},
                    "max_tokens": {"type": "integer", "minimum": 1},
                },
            },
            "operator": {"type": "string"},
            "signature": {"type": "string"},
        },
        "required": ["version", "model"],
    }


def _embedding_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"const": "1"},
            "vector": {"type": "array", "items": {"type": "number"}},
            "model": {"type": "string"},
            "input": {"type": "string"},
            "dimensions": {"type": "integer", "minimum": 1},
            "space": {"type": "string"},
            "computed_at": {"type": "string"},
        },
        "required": ["version", "vector", "model", "input", "dimensions"],
    }


def _capabilities_schema() -> JsonSchema:
    return {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ALL_CAPABILITIES,
        },
    }


def build_extraction_schema(
    *,
    capabilities: list[str] | None = None,
    profile: str | None = None,
    add: list[str] | None = None,
    remove: list[str] | None = None,
) -> JsonSchema:
    resolved = set(resolve_capabilities(capabilities=capabilities, profile=profile, add=add, remove=remove))
    properties: JsonSchema = {
        "extracted_at": {"type": "string"},
    }
    required = ["extracted_at"]

    if "entities" in resolved:
        properties["entities"] = {"type": "array", "items": _entity_schema(resolved)}
        required.append("entities")
    if "goals" in resolved:
        properties["goals"] = {"type": "array", "items": _goal_schema(resolved)}
        required.append("goals")
    if "themes" in resolved:
        properties["themes"] = {"type": "array", "items": {"type": "string"}}
        required.append("themes")
    if "keywords" in resolved:
        properties["keywords"] = {"type": "array", "items": {"type": "string"}}
        required.append("keywords")
    if "summary" in resolved:
        properties["summary"] = {"type": "string"}
        required.append("summary")
    if "sentiment" in resolved:
        properties["sentiment"] = _sentiment_schema(False) if "structured_sentiment" in resolved else {"type": "string"}
        required.append("sentiment")
    if "facts" in resolved:
        properties["facts"] = {"type": "array", "items": _fact_schema(resolved)}
        required.append("facts")
    if "questions" in resolved:
        properties["questions"] = {"type": "array", "items": _question_schema(resolved)}
        required.append("questions")
    if "actions" in resolved:
        properties["actions"] = {"type": "array", "items": _action_schema(resolved)}
        required.append("actions")
    if "decisions" in resolved:
        properties["decisions"] = {"type": "array", "items": _decision_schema(resolved)}
        required.append("decisions")
    if "temporal_refs" in resolved:
        properties["temporal_refs"] = {"type": "array", "items": _temporal_ref_schema(resolved)}
        required.append("temporal_refs")
    if "language" in resolved:
        properties["language"] = {"type": "string"}
        required.append("language")
    if "source_metadata" in resolved:
        properties["source_metadata"] = _source_metadata_schema(False)
        required.append("source_metadata")
    if "confidence" in resolved:
        properties["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
        required.append("confidence")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def build_finalized_extraction_schema(
    *,
    capabilities: list[str] | None = None,
    profile: str | None = None,
    add: list[str] | None = None,
    remove: list[str] | None = None,
) -> JsonSchema:
    resolved = set(resolve_capabilities(capabilities=capabilities, profile=profile, add=add, remove=remove))
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"const": "1"},
            "extracted_at": {"type": "string"},
            "source_id": {"type": "string"},
            "source_type": {"type": "string"},
            "user_id": {"type": "string"},
            "produced_by": {"anyOf": [{"type": "string"}, _producer_schema()]},
            "kind": {"type": "string"},
            "entities": {"type": "array", "items": _entity_schema(resolved, finalized=True)},
            "goals": {"type": "array", "items": _goal_schema(resolved, finalized=True)},
            "themes": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "sentiment": (
                _sentiment_schema(True)
                if "structured_sentiment" in resolved
                else {"anyOf": [{"type": "string"}, _sentiment_schema(True)]}
            ),
            "summary": {"type": "string"},
            "facts": {"type": "array", "items": _fact_schema(resolved, finalized=True)},
            "questions": {"type": "array", "items": _question_schema(resolved, finalized=True)},
            "actions": {"type": "array", "items": _action_schema(resolved, finalized=True)},
            "decisions": {"type": "array", "items": _decision_schema(resolved, finalized=True)},
            "temporal_refs": {"type": "array", "items": _temporal_ref_schema(resolved, finalized=True)},
            "language": {"type": "string"},
            "source_metadata": _source_metadata_schema(True),
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "capabilities": _capabilities_schema(),
            "embeddings": {"type": "array", "items": _embedding_schema()},
            "extensions": {"type": "object", "additionalProperties": True},
        },
        "required": ["version", "extracted_at", "produced_by", "entities", "goals", "themes", "capabilities"],
    }


def build_extraction_response_format(
    *,
    capabilities: list[str] | None = None,
    profile: str | None = None,
    add: list[str] | None = None,
    remove: list[str] | None = None,
    name: str = DEFAULT_RESPONSE_FORMAT_NAME,
    strict: bool = True,
) -> JsonSchema:
    schema = build_extraction_schema(
        capabilities=capabilities,
        profile=profile,
        add=add,
        remove=remove,
    )
    return {
        "type": "json_schema",
        "name": name,
        "strict": strict,
        "schema": _strictify_schema(schema) if strict else schema,
    }


def _strictify_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_strictify_schema(item) for item in schema]
    if isinstance(schema, dict):
        semantic_required = set(schema.get("required", [])) if isinstance(schema.get("required"), list) else set()
        copy = {key: _strictify_schema(value) for key, value in schema.items()}
        properties = copy.get("properties")
        if copy.get("type") == "object" and isinstance(properties, dict):
            for key in list(properties.keys()):
                if key not in semantic_required:
                    properties[key] = _nullable_schema(properties[key])
            copy["required"] = list(properties.keys())
        return copy
    return schema


def _nullable_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        typ = schema.get("type")
        if isinstance(typ, str):
            copy = dict(schema)
            copy["type"] = [typ, "null"]
            return copy
        if isinstance(typ, list) and "null" not in typ:
            copy = dict(schema)
            copy["type"] = [*typ, "null"]
            return copy
    return {"anyOf": [schema, {"type": "null"}]}


@dataclass
class ExtractionBuilder:
    text: str = ""
    capabilities: list[Any] | None = None
    profile: str | None = None
    add: list[Any] | None = None
    remove: list[Any] | None = None
    categories: list[str] | None = None
    source_type: str | None = None
    date: str | None = None
    stage: str | None = "stage1"
    extracted_at: str | None = None
    compact: bool | None = True
    produced_by: str | dict[str, Any] | None = None
    user_id: str | None = None
    source_id: str | None = None
    kind: str | None = None
    extensions: dict[str, Any] | None = None
    embeddings: list[dict[str, Any]] | None = None
    capabilities_hint: list[str] | None = None
    embedding_overrides: dict[str, bool] | None = None
    embedding_inputs: str | list[Any] | None = None

    def with_text(self, text: str) -> "ExtractionBuilder":
        self.text = text
        return self

    def with_profile(self, profile: str) -> "ExtractionBuilder":
        self.profile = profile
        self.capabilities = None
        return self

    def minimal(self, *, embed: bool | None = None) -> "ExtractionBuilder":
        return self._with_profile_capabilities("minimal", embed=embed)

    def standard(self, *, embed: bool | None = None) -> "ExtractionBuilder":
        return self._with_profile_capabilities("standard", embed=embed)

    def full(self, *, embed: bool | None = None) -> "ExtractionBuilder":
        return self._with_profile_capabilities("full", embed=embed)

    def _with_profile_capabilities(self, profile: str, *, embed: bool | None = None) -> "ExtractionBuilder":
        if embed is None:
            return self.with_profile(profile)
        capabilities = profile_capabilities(profile)
        self.capabilities = _capability_specs(capabilities, embed)
        if embed is True and self.embedding_inputs is None:
            self.embedding_inputs = ["source"]
        self.profile = None
        return self

    def with_capabilities(self, capabilities: list[Any]) -> "ExtractionBuilder":
        self.capabilities = capabilities
        self.profile = None
        return self

    def add_capabilities(self, capabilities: list[Any]) -> "ExtractionBuilder":
        self.add = [*(self.add or []), *capabilities]
        return self

    def remove_capabilities(self, capabilities: list[Any]) -> "ExtractionBuilder":
        self.remove = [*(self.remove or []), *capabilities]
        return self

    def minus(self, *capabilities: Any) -> "ExtractionBuilder":
        return self.remove_capabilities(list(capabilities))

    def unsupported(self, *capabilities: Any) -> "ExtractionBuilder":
        return self.remove_capabilities(list(capabilities))

    def embed(self, capability: str, enabled: bool = True) -> "ExtractionBuilder":
        self.embedding_overrides = {**(self.embedding_overrides or {}), capability: enabled}
        return self

    def embed_capability(self, capability: str, enabled: bool = True) -> "ExtractionBuilder":
        return self.embed(capability, enabled)

    def embed_field(self, capability: str, enabled: bool = True) -> "ExtractionBuilder":
        return self.embed(capability, enabled)

    def with_embedding_inputs(self, inputs: str | list[Any] = "all") -> "ExtractionBuilder":
        self.embedding_inputs = inputs
        return self

    def with_standard_embeddings(self) -> "ExtractionBuilder":
        return self.with_embedding_inputs("all")

    def with_categories(self, categories: list[str]) -> "ExtractionBuilder":
        self.categories = categories
        return self

    def with_source_type(self, source_type: str) -> "ExtractionBuilder":
        self.source_type = source_type
        return self

    def with_date(self, date: str) -> "ExtractionBuilder":
        self.date = date
        return self

    def with_extracted_at(self, extracted_at: str) -> "ExtractionBuilder":
        self.extracted_at = extracted_at
        return self

    def with_produced_by(self, produced_by: str | dict[str, Any]) -> "ExtractionBuilder":
        self.produced_by = produced_by
        return self

    def with_user_id(self, user_id: str) -> "ExtractionBuilder":
        self.user_id = user_id
        return self

    def with_source(
        self,
        *,
        source_id: str | None = None,
        source_type: str | None = None,
    ) -> "ExtractionBuilder":
        if source_id is not None:
            self.source_id = source_id
        if source_type is not None:
            self.source_type = source_type
        return self

    def with_kind(self, kind: str) -> "ExtractionBuilder":
        self.kind = kind
        return self

    def with_extensions(self, extensions: dict[str, Any]) -> "ExtractionBuilder":
        self.extensions = extensions
        return self

    def with_embeddings(self, embeddings: list[dict[str, Any]]) -> "ExtractionBuilder":
        self.embeddings = embeddings
        return self

    def with_finalize_context(self, context: FinalizeContext | dict[str, Any]) -> "ExtractionBuilder":
        values = context.__dict__ if isinstance(context, FinalizeContext) else context
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def with_compact(self, compact: bool = True) -> "ExtractionBuilder":
        self.compact = compact
        return self

    def resolved_capabilities(self) -> list[str]:
        return resolve_capabilities(
            capabilities=self.capabilities,
            profile=self.profile,
            add=self.add,
            remove=self.remove,
        )

    def capability_plan(self) -> dict[str, Any]:
        capabilities = self.resolved_capabilities()
        resolved = set(capabilities)
        excluded = expand_capability_exclusions(normalize_capability_inputs(self.remove or []))
        embedded_inputs = _order_embedding_inputs(_apply_embedding_overrides(_unique([
            *(_embedding_inputs_from_selection(self.embedding_inputs)),
            *(_embedding_inputs_from_capabilities(self.capabilities, resolved)),
            *(_embedding_inputs_from_capabilities(self.add, resolved)),
        ]), self.embedding_overrides))
        not_embedded = []
        for capability in capabilities:
            input_name = capability_embedding_input(capability)
            if input_name is not None and input_name not in embedded_inputs:
                not_embedded.append(capability)
        return {
            "capabilities": capabilities,
            "excluded": excluded,
            "embedded_inputs": embedded_inputs,
            "not_embedded": not_embedded,
            "required_callbacks": {
                "call_llm": True,
                "get_embedding": len(embedded_inputs) > 0,
            },
            "prompt_characters": len(self.prompt()),
        }

    def plan(self) -> dict[str, Any]:
        return self.capability_plan()

    def extract_options(self) -> dict[str, Any]:
        plan = self.capability_plan()
        custom_inputs = [
            input_item
            for input_item in self.embedding_inputs
            if isinstance(input_item, dict) and isinstance(input_item.get("input"), str)
        ] if isinstance(self.embedding_inputs, list) else []
        embedding_inputs = [
            *custom_inputs,
            *(input_name for input_name in plan["embedded_inputs"] if all(item["input"] != input_name for item in custom_inputs)),
        ] if plan["embedded_inputs"] else None
        values = {
            "capabilities": self.capabilities,
            "profile": self.profile,
            "add": self.add,
            "remove": self.remove,
            "categories": self.categories,
            "source_type": self.source_type,
            "date": self.date,
            "stage": self.stage,
            "extracted_at": self.extracted_at,
            "compact": self.compact,
            "produced_by": self.produced_by,
            "user_id": self.user_id,
            "source_id": self.source_id,
            "kind": self.kind,
            "extensions": self.extensions,
            "embeddings": self.embeddings,
            "capabilities_hint": self.capabilities_hint,
            "embed": self.embedding_overrides,
            "embedding_inputs": embedding_inputs,
        }
        return {key: value for key, value in values.items() if value is not None}

    def to_options(self) -> dict[str, Any]:
        return self.extract_options()

    def prompt(self) -> str:
        return build_extraction_prompt(
            self.text,
            capabilities=self.capabilities,
            profile=self.profile,
            add=self.add,
            remove=self.remove,
            categories=self.categories,
            source_type=self.source_type,
            date=self.date,
            stage=self.stage,
            extracted_at=self.extracted_at,
            compact=self.compact,
        )

    def schema(self) -> JsonSchema:
        return build_extraction_schema(
            capabilities=self.capabilities,
            profile=self.profile,
            add=self.add,
            remove=self.remove,
        )

    def finalized_schema(self) -> JsonSchema:
        return build_finalized_extraction_schema(
            capabilities=self.capabilities,
            profile=self.profile,
            add=self.add,
            remove=self.remove,
        )

    def response_format(self, *, name: str = DEFAULT_RESPONSE_FORMAT_NAME, strict: bool = True) -> JsonSchema:
        return build_extraction_response_format(
            capabilities=self.capabilities,
            profile=self.profile,
            add=self.add,
            remove=self.remove,
            name=name,
            strict=strict,
        )

    def finalize_context_dict(self) -> dict[str, Any] | None:
        if self.produced_by is None:
            return None

        context: dict[str, Any] = {
            "produced_by": self.produced_by,
            "capabilities_hint": self.capabilities_hint or self.resolved_capabilities(),
        }
        if self.user_id is not None:
            context["user_id"] = self.user_id
        if self.source_id is not None:
            context["source_id"] = self.source_id
        if self.source_type is not None:
            context["source_type"] = self.source_type
        if self.kind is not None:
            context["kind"] = self.kind
        if self.extensions is not None:
            context["extensions"] = self.extensions
        if self.embeddings is not None:
            context["embeddings"] = self.embeddings
        return context

    def finalize_context(self) -> FinalizeContext | None:
        context = self.finalize_context_dict()
        return FinalizeContext(**context) if context is not None else None

    def finalize(self, llm_output: dict[str, Any]) -> FinalizeResult:
        context = self.finalize_context()
        if context is None:
            raise ValueError("Cannot finalize without produced_by; call with_produced_by() or with_finalize_context()")
        return finalize_extraction(llm_output, context)

    def build(self, *, name: str = DEFAULT_RESPONSE_FORMAT_NAME, strict: bool = True) -> dict[str, Any]:
        result = {
            "capabilities": self.resolved_capabilities(),
            "prompt": self.prompt(),
            "schema": self.schema(),
            "response_format": self.response_format(name=name, strict=strict),
            "finalized_schema": self.finalized_schema(),
        }
        finalize_context = self.finalize_context_dict()
        if finalize_context is not None:
            result["finalize_context"] = finalize_context
        return result


def create_extraction_builder(text: str = "", **options: Any) -> ExtractionBuilder:
    return ExtractionBuilder(text=text, **options)
