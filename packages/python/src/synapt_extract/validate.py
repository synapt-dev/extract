"""Structural validation for SynaptExtraction IL v1."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from synapt_extract.schema import EXTRACTION_CAPABILITIES

VALID_GOAL_STATUSES = frozenset(["open", "resolved", "abandoned", "in_progress"])
VALID_TEMPORAL_TYPES = frozenset(["point", "range", "duration", "unresolved"])

_URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://\S+$")
_NAMESPACED_RE = re.compile(r"^[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+$")
_ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?"
    r"(?:Z|[+\-]\d{2}:?\d{2})?)?$"
)
_ISO_DATETIME_STRICT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?"
    r"(?:Z|[+\-]\d{2}:?\d{2})?$"
)

_ROOT_KEYS = frozenset([
    "version", "extracted_at", "source_id", "source_type", "user_id",
    "produced_by", "kind", "entities", "goals", "themes", "sentiment",
    "summary", "facts", "temporal_refs", "capabilities", "embeddings", "extensions",
])
_ENTITY_KEYS = frozenset([
    "id", "name", "type", "state", "context", "date_hint",
    "source", "signals", "relations",
])
_GOAL_KEYS = frozenset([
    "text", "status", "entity_refs", "stated_at", "resolved_at",
    "source", "signals",
])
_FACT_KEYS = frozenset(["text", "category", "source", "signals"])
_RELATION_KEYS = frozenset(["target", "type", "origin", "signals"])
_SOURCE_REF_KEYS = frozenset(["version", "snippet", "offset_start", "offset_end", "sentence_index"])
_SIGNALS_KEYS = frozenset(["version", "confidence", "negated", "hedged", "condition"])
_TEMPORAL_REF_KEYS = frozenset(["version", "raw", "type", "resolved", "resolved_end", "context"])
_EMBEDDING_KEYS = frozenset(["version", "vector", "model", "input", "dimensions", "space", "computed_at"])
_PRODUCER_KEYS = frozenset([
    "version", "model", "model_version", "deployment", "configuration",
    "operator", "signature",
])
_PRODUCER_CONFIG_KNOWN_KEYS = frozenset([
    "reasoning_effort", "system_prompt_hash", "temperature", "top_p", "max_tokens",
])


@dataclass
class ValidationError:
    path: str
    message: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)


def _is_uri(s: str) -> bool:
    return bool(_URI_RE.match(s))


def _is_iso_datetime(s: str) -> bool:
    return bool(_ISO_DATE_RE.match(s))


def _is_iso_datetime_strict(s: str) -> bool:
    return bool(_ISO_DATETIME_STRICT_RE.match(s))


def _is_namespaced(s: str) -> bool:
    return bool(_NAMESPACED_RE.match(s))


def _check_extra_keys(obj: dict, allowed: frozenset[str], path: str, errors: list[ValidationError]) -> None:
    for key in obj:
        if key not in allowed:
            full_path = f"{path}.{key}" if path else key
            errors.append(ValidationError(full_path, "additional property not allowed"))


def _require_non_empty_str(obj: dict, key: str, path: str, errors: list[ValidationError], label: str = "required non-empty string") -> bool:
    val = obj.get(key)
    if not isinstance(val, str) or len(val) == 0:
        errors.append(ValidationError(f"{path}.{key}", label))
        return False
    return True


def _check_optional_str(obj: dict, key: str, path: str, errors: list[ValidationError]) -> None:
    if key in obj and not isinstance(obj[key], str):
        errors.append(ValidationError(f"{path}.{key}", "must be a string"))


def _check_optional_non_neg_int(obj: dict, key: str, path: str, errors: list[ValidationError]) -> None:
    if key in obj:
        val = obj[key]
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            errors.append(ValidationError(f"{path}.{key}", "must be a non-negative integer"))


def _has_payload_beyond_version(obj: dict[str, Any]) -> bool:
    return any(k != "version" for k in obj)


def _check_source_ref(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _SOURCE_REF_KEYS, path, errors)
    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))
    if not _has_payload_beyond_version(obj):
        errors.append(ValidationError(path, "empty sub-schema (only version); must contain at least one payload field"))
        return
    _check_optional_str(obj, "snippet", path, errors)
    _check_optional_non_neg_int(obj, "offset_start", path, errors)
    _check_optional_non_neg_int(obj, "offset_end", path, errors)
    _check_optional_non_neg_int(obj, "sentence_index", path, errors)


def _check_signals(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _SIGNALS_KEYS, path, errors)
    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))
    if not _has_payload_beyond_version(obj):
        errors.append(ValidationError(path, "empty sub-schema (only version); must contain at least one payload field"))
        return
    if "confidence" in obj:
        c = obj["confidence"]
        if not isinstance(c, (int, float)) or isinstance(c, bool) or c < 0 or c > 1:
            errors.append(ValidationError(f"{path}.confidence", "must be a number between 0.0 and 1.0"))
    if "negated" in obj and not isinstance(obj["negated"], bool):
        errors.append(ValidationError(f"{path}.negated", "must be a boolean"))
    if "hedged" in obj and not isinstance(obj["hedged"], bool):
        errors.append(ValidationError(f"{path}.hedged", "must be a boolean"))
    if "condition" in obj and not isinstance(obj["condition"], str):
        errors.append(ValidationError(f"{path}.condition", "must be a string"))


def _check_embedding(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _EMBEDDING_KEYS, path, errors)
    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))
    vector = obj.get("vector")
    if not isinstance(vector, list):
        errors.append(ValidationError(f"{path}.vector", "required array"))
    else:
        for i, v in enumerate(vector):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errors.append(ValidationError(f"{path}.vector[{i}]", "must be a number"))
                break
    model = obj.get("model")
    if not isinstance(model, str):
        errors.append(ValidationError(f"{path}.model", "required string"))
    elif not _is_uri(model):
        errors.append(ValidationError(f"{path}.model", "must be a provider URI (scheme://identifier)"))
    if not isinstance(obj.get("input"), str):
        errors.append(ValidationError(f"{path}.input", "required string"))
    dims = obj.get("dimensions")
    if not isinstance(dims, int) or isinstance(dims, bool) or dims < 1:
        errors.append(ValidationError(f"{path}.dimensions", "required positive integer"))
    elif isinstance(vector, list) and dims != len(vector):
        errors.append(ValidationError(f"{path}.dimensions", f"dimensions ({dims}) must equal vector length ({len(vector)})"))
    _check_optional_str(obj, "space", path, errors)
    if "computed_at" in obj:
        if not isinstance(obj["computed_at"], str) or not _is_iso_datetime_strict(obj["computed_at"]):
            errors.append(ValidationError(f"{path}.computed_at", "must be a valid ISO 8601 date-time"))


def _check_relation(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _RELATION_KEYS, path, errors)
    target = obj.get("target")
    if not isinstance(target, str) or len(target) == 0:
        errors.append(ValidationError(f"{path}.target", "required non-empty string"))
    rtype = obj.get("type")
    if not isinstance(rtype, str) or len(rtype) == 0:
        errors.append(ValidationError(f"{path}.type", "required non-empty string"))
    _check_optional_str(obj, "origin", path, errors)
    if "signals" in obj:
        _check_signals(obj["signals"], f"{path}.signals", errors)


def _check_entity(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _ENTITY_KEYS, path, errors)
    _require_non_empty_str(obj, "name", path, errors)
    _require_non_empty_str(obj, "type", path, errors)
    _check_optional_str(obj, "id", path, errors)
    _check_optional_str(obj, "state", path, errors)
    _check_optional_str(obj, "context", path, errors)
    _check_optional_str(obj, "date_hint", path, errors)
    if "source" in obj:
        _check_source_ref(obj["source"], f"{path}.source", errors)
    if "signals" in obj:
        _check_signals(obj["signals"], f"{path}.signals", errors)
    if "relations" in obj:
        if not isinstance(obj["relations"], list):
            errors.append(ValidationError(f"{path}.relations", "must be an array"))
        else:
            for i, rel in enumerate(obj["relations"]):
                _check_relation(rel, f"{path}.relations[{i}]", errors)


def _check_goal(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _GOAL_KEYS, path, errors)
    _require_non_empty_str(obj, "text", path, errors)
    status = obj.get("status")
    if not isinstance(status, str) or status not in VALID_GOAL_STATUSES:
        errors.append(ValidationError(f"{path}.status", "must be one of: open, resolved, abandoned, in_progress"))
    if not isinstance(obj.get("entity_refs"), list):
        errors.append(ValidationError(f"{path}.entity_refs", "required array of strings"))
    else:
        for i, ref in enumerate(obj["entity_refs"]):
            if not isinstance(ref, str):
                errors.append(ValidationError(f"{path}.entity_refs[{i}]", "must be a string"))
    if "stated_at" in obj:
        if not isinstance(obj["stated_at"], str) or not _is_iso_datetime(obj["stated_at"]):
            errors.append(ValidationError(f"{path}.stated_at", "must be a valid ISO 8601 date/datetime"))
    if "resolved_at" in obj:
        if not isinstance(obj["resolved_at"], str) or not _is_iso_datetime(obj["resolved_at"]):
            errors.append(ValidationError(f"{path}.resolved_at", "must be a valid ISO 8601 date/datetime"))
    if "source" in obj:
        _check_source_ref(obj["source"], f"{path}.source", errors)
    if "signals" in obj:
        _check_signals(obj["signals"], f"{path}.signals", errors)


def _check_fact(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _FACT_KEYS, path, errors)
    _require_non_empty_str(obj, "text", path, errors)
    _check_optional_str(obj, "category", path, errors)
    if "source" in obj:
        _check_source_ref(obj["source"], f"{path}.source", errors)
    if "signals" in obj:
        _check_signals(obj["signals"], f"{path}.signals", errors)


def _check_temporal_ref(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _TEMPORAL_REF_KEYS, path, errors)
    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))
    raw = obj.get("raw")
    if not isinstance(raw, str) or len(raw) == 0:
        errors.append(ValidationError(f"{path}.raw", "required non-empty string"))
    ttype = obj.get("type")
    if "type" in obj:
        if not isinstance(ttype, str) or ttype not in VALID_TEMPORAL_TYPES:
            errors.append(ValidationError(f"{path}.type", "must be one of: point, range, duration, unresolved"))
        elif ttype == "range" and "resolved_end" not in obj:
            errors.append(ValidationError(f"{path}.resolved_end", "required when type is 'range'"))
        elif ttype == "unresolved":
            if "resolved" in obj:
                errors.append(ValidationError(f"{path}.resolved", "must not be present when type is 'unresolved'"))
            if "resolved_end" in obj:
                errors.append(ValidationError(f"{path}.resolved_end", "must not be present when type is 'unresolved'"))
    if "resolved" in obj:
        if not isinstance(obj["resolved"], str) or not _is_iso_datetime(obj["resolved"]):
            errors.append(ValidationError(f"{path}.resolved", "must be a valid ISO 8601 date/datetime"))
    if "resolved_end" in obj:
        if not isinstance(obj["resolved_end"], str) or not _is_iso_datetime(obj["resolved_end"]):
            errors.append(ValidationError(f"{path}.resolved_end", "must be a valid ISO 8601 date/datetime"))
    _check_optional_str(obj, "context", path, errors)


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _check_producer(obj: Any, path: str, errors: list[ValidationError]) -> None:
    if not isinstance(obj, dict):
        errors.append(ValidationError(path, "must be an object"))
        return
    _check_extra_keys(obj, _PRODUCER_KEYS, path, errors)

    if obj.get("version") != "1":
        errors.append(ValidationError(f"{path}.version", 'must be "1"'))

    model = obj.get("model")
    if not isinstance(model, str):
        errors.append(ValidationError(f"{path}.model", "required string (provider URI)"))
    elif not _is_uri(model):
        errors.append(ValidationError(f"{path}.model", "must be a provider URI (scheme://identifier)"))

    _check_optional_str(obj, "model_version", path, errors)
    _check_optional_str(obj, "deployment", path, errors)
    _check_optional_str(obj, "operator", path, errors)
    _check_optional_str(obj, "signature", path, errors)

    if "configuration" in obj:
        config = obj["configuration"]
        if not isinstance(config, dict):
            errors.append(ValidationError(f"{path}.configuration", "must be an object"))
        else:
            _check_optional_str(config, "reasoning_effort", f"{path}.configuration", errors)
            if "system_prompt_hash" in config:
                val = config["system_prompt_hash"]
                if not isinstance(val, str) or not _HEX_RE.match(val):
                    errors.append(ValidationError(f"{path}.configuration.system_prompt_hash", "must be a hex string"))
            if "temperature" in config:
                val = config["temperature"]
                if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
                    errors.append(ValidationError(f"{path}.configuration.temperature", "must be a non-negative number"))
            if "top_p" in config:
                val = config["top_p"]
                if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0 or val > 1:
                    errors.append(ValidationError(f"{path}.configuration.top_p", "must be a number between 0 and 1"))
            if "max_tokens" in config:
                val = config["max_tokens"]
                if not isinstance(val, int) or isinstance(val, bool) or val < 1:
                    errors.append(ValidationError(f"{path}.configuration.max_tokens", "must be a positive integer"))


def validate_extraction(obj: Any) -> ValidationResult:
    errors: list[ValidationError] = []

    if not isinstance(obj, dict):
        return ValidationResult(valid=False, errors=[ValidationError("", "must be an object")])

    _check_extra_keys(obj, _ROOT_KEYS, "", errors)

    if obj.get("version") != "1":
        errors.append(ValidationError("version", 'must be "1"'))

    extracted_at = obj.get("extracted_at")
    if not isinstance(extracted_at, str):
        errors.append(ValidationError("extracted_at", "required string (ISO 8601 date-time)"))
    elif not _is_iso_datetime_strict(extracted_at):
        errors.append(ValidationError("extracted_at", "must be a valid ISO 8601 date-time (e.g. 2026-04-26T12:00:00Z)"))

    produced_by = obj.get("produced_by")
    if isinstance(produced_by, str):
        if not _is_uri(produced_by):
            errors.append(ValidationError("produced_by", "must be a provider URI (scheme://identifier)"))
    elif isinstance(produced_by, dict):
        _check_producer(produced_by, "produced_by", errors)
    else:
        errors.append(ValidationError("produced_by", "required string (provider URI) or SynaptProducer object"))

    if "kind" in obj:
        kind = obj["kind"]
        if not isinstance(kind, str) or not _is_namespaced(kind):
            errors.append(ValidationError("kind", "must be namespaced (e.g. 'conversa/prayer')"))

    _check_optional_str(obj, "sentiment", "", errors)
    _check_optional_str(obj, "source_id", "", errors)
    _check_optional_str(obj, "source_type", "", errors)
    _check_optional_str(obj, "user_id", "", errors)

    if "extensions" in obj:
        ext = obj["extensions"]
        if not isinstance(ext, dict):
            errors.append(ValidationError("extensions", "must be an object"))
        else:
            for key in ext:
                if not _is_namespaced(key):
                    errors.append(ValidationError(f"extensions.{key}", "extension key must be namespaced (e.g. 'conversa/prayer')"))

    entity_ids: set[str] = set()
    if not isinstance(obj.get("entities"), list):
        errors.append(ValidationError("entities", "required array"))
    else:
        for i, ent in enumerate(obj["entities"]):
            _check_entity(ent, f"entities[{i}]", errors)
            if isinstance(ent, dict) and isinstance(ent.get("id"), str):
                entity_ids.add(ent["id"])

    if not isinstance(obj.get("goals"), list):
        errors.append(ValidationError("goals", "required array"))
    else:
        for i, goal in enumerate(obj["goals"]):
            _check_goal(goal, f"goals[{i}]", errors)
            if isinstance(goal, dict) and isinstance(goal.get("entity_refs"), list):
                for j, ref in enumerate(goal["entity_refs"]):
                    if isinstance(ref, str) and ref not in entity_ids:
                        errors.append(ValidationError(
                            f"goals[{i}].entity_refs[{j}]",
                            f"references entity ID '{ref}' which is not declared in entities",
                        ))

    if not isinstance(obj.get("themes"), list):
        errors.append(ValidationError("themes", "required array"))
    else:
        for i, theme in enumerate(obj["themes"]):
            if not isinstance(theme, str) or len(theme) == 0:
                errors.append(ValidationError(f"themes[{i}]", "must be a non-empty string"))

    if "summary" in obj:
        if not isinstance(obj["summary"], str) or len(obj["summary"]) == 0:
            errors.append(ValidationError("summary", "must be a non-empty string"))

    if not isinstance(obj.get("capabilities"), list):
        errors.append(ValidationError("capabilities", "required array"))
    else:
        for i, cap in enumerate(obj["capabilities"]):
            if not isinstance(cap, str):
                errors.append(ValidationError(f"capabilities[{i}]", "must be a string"))
            elif cap not in EXTRACTION_CAPABILITIES:
                errors.append(ValidationError(f"capabilities[{i}]", f'unknown capability: "{cap}"'))

    if "facts" in obj:
        if not isinstance(obj["facts"], list):
            errors.append(ValidationError("facts", "must be an array"))
        else:
            for i, fact in enumerate(obj["facts"]):
                _check_fact(fact, f"facts[{i}]", errors)

    if "temporal_refs" in obj:
        if not isinstance(obj["temporal_refs"], list):
            errors.append(ValidationError("temporal_refs", "must be an array"))
        else:
            for i, ref in enumerate(obj["temporal_refs"]):
                _check_temporal_ref(ref, f"temporal_refs[{i}]", errors)

    if "embeddings" in obj:
        if not isinstance(obj["embeddings"], list):
            errors.append(ValidationError("embeddings", "must be an array"))
        else:
            for i, emb in enumerate(obj["embeddings"]):
                _check_embedding(emb, f"embeddings[{i}]", errors)

    if entity_ids and isinstance(obj.get("entities"), list):
        for i, ent in enumerate(obj["entities"]):
            if isinstance(ent, dict) and isinstance(ent.get("relations"), list):
                for j, rel in enumerate(ent["relations"]):
                    if isinstance(rel, dict):
                        target = rel.get("target")
                        if isinstance(target, str) and len(target) > 0 and target not in entity_ids:
                            errors.append(ValidationError(
                                f"entities[{i}].relations[{j}].target",
                                f"references entity ID '{target}' which is not declared in entities",
                            ))

    return ValidationResult(valid=len(errors) == 0, errors=errors)
