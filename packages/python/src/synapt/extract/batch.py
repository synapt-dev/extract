"""Batch Stage-1 extraction primitive for SynaptExtraction.

Implements the pinned contract (config/design/extract-batch-limits-characterization-
2026-07-13.md §"Contract decisions") and Sentinel's spec (extract#28,
tests/python/test_extract_batch.py). Reliability logic is per-unit: shaping +
per-item validation + fail-closed fallback, with every failure contained to its
own unit slot (count-invariant).

Why this primitive exists
-------------------------
Atlas's characterization found the generic single-text builder cannot reliably
produce a schema-valid packet even for ONE clean pre-identified unit (NO_VIABLE_N
at N=1). The failure is MALFORMATION on GROUNDED content (40/40 source-supported),
not confabulation — the model returns the right facts in the wrong shape. Fixed
with structural machinery (shaping + per-item validation + fallback), not prompt
tuning. A NEW primitive, not a wrapper/loop over the generic builder.

Contract (pinned + spec-confirmed)
----------------------------------
  • Input: list[BatchUnit(id, text, capabilities?, date?)] — explicit attribution; the
    id rides into the output as source_unit_id (boundaries stay out-of-band, never
    in model-visible text). `date` (optional) is the unit's SOURCE date, threaded into
    Stage-1 as the temporal resolution anchor (config/design/extract-temporal-role-
    2026-07-14.md) so partial/relative dates resolve against the source, not a guess.
  • Inference: an injected `infer` seam receiving a request {prompt, messages,
    capabilities} and returning a completion string. ZERO recall dependency.
  • v1 strategy: PER-UNIT (one infer call per unit) — trivially out-of-band, clean
    1:1 attribution. batch-all / safe-N are future INTERNAL ladder rungs, not v1
    contract surface. Retry: one deterministic retry per failed unit (2 attempts
    total), then a terminal marker from the last failure class; never replay a
    successful neighbor.
  • Output: COUNT-INVARIANT len(out)==len(in). Each unit → an "ok" BatchUnitResult
    (valid envelope) OR a terminal {source_unit_id, status:"failed", reason} marker.
    reason ∈ BatchFailureReason. No silent drops.
  • Shaping (folded from recall #870/#871, held/superseded):
      Class-A PRE-parse text hygiene — strip ``` fences + `//` comments, STRING-
        LITERAL-AWARE (a `//` inside a JSON string, e.g. a URL, must survive).
      Class-B POST-parse coercion — capability set is the arbiter: in-scope fields
        coerced (scalar→array; null/non-string OPTIONAL fields like category or
        decided_at are omitted; an invalid REQUIRED field is kept so strict
        validation rejects it), out-of-scope dropped; temporal_refs → schema-valid
        raw/resolved + base-tier role/resolved_end (type/context stay temporal_classes-
        gated, so they are stripped at the base tier); non-dict leaves preserved into
        strict validation.

Harvest map: scratchpad/extract_batch_craft_harvest.md. Boundary: OSS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypedDict

from synapt.extract.builder import build_extraction_schema
from synapt.extract.finalize import FinalizeContext, finalize_extraction
from synapt.extract.prompt import (
    build_extraction_prompt,
    profile_capabilities,
    resolve_capabilities,
)

# Container capabilities the finalized schema always requires, even when a caller
# did not request them (mirrors recall's backfill so validation does not fail on
# containers we deliberately did not request).
_ALWAYS_BACKFILL = ("entities", "goals", "themes")
# One deterministic retry per failed unit → 2 attempts total (Q-B, Sentinel).
_MAX_ATTEMPTS = 2

# Terminal per-unit failure reasons (Q5). A Literal (not an Enum) so the spec's
# get_args(BatchFailureReason) reads the members. "merged" is reserved for a future
# batch-all path; the per-unit v1 path never emits it.
BatchFailureReason = Literal["unparseable", "schema_invalid", "dropped", "merged"]


class BatchInferRequest(TypedDict):
    """The exact request the injected `infer` seam receives (Q-D). No unit id /
    boundary tag ever appears here — boundaries stay in extract_batch bookkeeping,
    out of model-visible text."""

    prompt: str
    messages: list[dict[str, str]]
    capabilities: list[str]


# The injected inference seam (Q4): request → completion. The caller (recall) passes
# a model-backed callable; tests pass a deterministic/recorded one. Zero recall dep.
Inferer = Callable[[BatchInferRequest], str]


@dataclass
class BatchUnit:
    """One pre-identified unit to extract (Q1). ``id`` is stable and rides into the
    output as ``source_unit_id`` so merge/split/drop is detectable. ``capabilities``
    optionally overrides the per-call default for this unit. ``date`` is the unit's SOURCE
    date (config/design/extract-temporal-role-2026-07-14.md) — the resolution anchor Stage-1
    uses to resolve partial/relative dates in ``unit.text`` (e.g. "expires April 30") against
    the ACTUAL date the source material was written, not "today" or an unanchored guess.
    Optional: a caller with no source date (or extracting non-temporal-sensitive units)
    simply omits it, degrading gracefully to unanchored resolution."""

    id: str
    text: str
    capabilities: list[str] | None = None
    date: str | None = None


@dataclass
class BatchUnitResult:
    """Per-unit outcome (Q5). ``status`` "ok" sets ``extraction``; "failed" sets
    ``reason``. ``source_unit_id`` ties the slot back to its BatchUnit."""

    source_unit_id: str
    status: str                          # "ok" | "failed"
    extraction: Any | None = None        # a finalized SynaptExtraction, or None
    reason: BatchFailureReason | None = None


async def extract_batch(
    units: list[BatchUnit],
    *,
    infer: Inferer,
    produced_by: str,
    capabilities: list[str] | None = None,
) -> list[BatchUnitResult]:
    """Shape + validate a batch of pre-identified units into per-unit envelopes.

    COUNT-INVARIANT: returns exactly one BatchUnitResult per input unit, in a 1:1
    slot mapping (Q5). extract_batch owns the reliability orchestration (v1 =
    per-unit calls with one deterministic retry per failed unit) driven through the
    injected ``infer`` seam, with zero dependency on any specific model client (Q4).
    ``capabilities`` defaults to the standard profile when omitted (Q3).
    """
    if not units:
        return []
    ids = [unit.id for unit in units]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate unit id; BatchUnit ids must be unique")

    default_capabilities = (
        capabilities if capabilities is not None else profile_capabilities("standard")
    )
    results: list[BatchUnitResult] = []
    for unit in units:
        unit_capabilities = (
            unit.capabilities if unit.capabilities is not None else default_capabilities
        )
        results.append(_extract_unit(unit, infer, produced_by, unit_capabilities))
    return results


def _extract_unit(
    unit: BatchUnit,
    infer: Inferer,
    produced_by: str,
    capabilities: list[str],
) -> BatchUnitResult:
    """Run one unit through the reliability ladder: build an out-of-band request →
    infer → Class-A hygiene + parse → Class-B coerce → finalize/validate. One
    deterministic retry on failure (2 attempts total); a persisting failure yields a
    terminal marker carrying the last failure's reason (Q-B)."""
    reason: BatchFailureReason = "dropped"
    for _attempt in range(_MAX_ATTEMPTS):
        # Out-of-band: the model sees the unit TEXT only — never its id or a boundary
        # tag (Q-D). The id lives in bookkeeping and rides into the packet post-hoc.
        # unit.date threads as the temporal RESOLUTION anchor (config/design/extract-
        # temporal-role-2026-07-14.md) — None degrades gracefully (build_extraction_prompt
        # already handles an absent date).
        prompt = build_extraction_prompt(
            unit.text, capabilities=list(capabilities), stage="stage1", date=unit.date,
        )
        request: BatchInferRequest = {
            "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "capabilities": list(capabilities),
        }
        # Contain the injected seam per-unit: an infer failure (e.g. RuntimeError)
        # must NOT escape and void the whole batch — it is this unit's failure,
        # retried once then terminal, while neighbours still produce their slots.
        # No output was produced, so the closest Q5 class is "dropped".
        try:
            completion = infer(request)
        except Exception:
            reason = "dropped"
            continue
        parsed = _parse_completion(completion)
        if parsed is None:
            reason = "unparseable"
            continue

        coerced = _coerce_shape(parsed, capabilities)
        for key in _ALWAYS_BACKFILL:
            coerced.setdefault(key, [])
        context = FinalizeContext(
            produced_by=produced_by,
            source_id=unit.id,
            capabilities_hint=list(capabilities),
        )
        try:
            finalized = finalize_extraction(coerced, context)
        except Exception:
            reason = "schema_invalid"
            continue
        if not finalized.validation.valid:
            reason = "schema_invalid"
            continue
        if _is_empty_extraction(finalized.extraction, capabilities):
            reason = "dropped"
            continue
        return BatchUnitResult(
            source_unit_id=unit.id, status="ok", extraction=finalized.extraction
        )

    return BatchUnitResult(source_unit_id=unit.id, status="failed", reason=reason)


def _parse_completion(completion: str) -> dict | None:
    """Class-A hygiene + JSON parse; None if the result is not a JSON object."""
    try:
        parsed = json.loads(_strip_output_hygiene(completion))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_output_hygiene(raw: str) -> str:
    """Class-A PRE-parse (NET-NEW): strip ``` fences + ``//`` comments so grounded-
    but-wrapped JSON parses. STRING-LITERAL-AWARE — a ``//`` inside a JSON string
    value (e.g. ``https://…``) is preserved; only real line-comments are removed."""
    text = raw.strip()
    # strip_markdown_fence: drop a leading ```/```json fence line, then the closing
    # ``` and anything trailing it (e.g. a "Reasoning:" epilogue the model appends).
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1:] if newline != -1 else ""
        close = text.rfind("```")
        if close != -1:
            text = text[:close]
    # strip_line_comments_outside_strings: remove `//` to end-of-line, but never when
    # inside a JSON string literal (so a URL's `//` survives). Tracks string + escape.
    out: list[str] = []
    in_string = False
    escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
        elif ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1  # drop the comment body; the newline (if any) is kept next loop
        else:
            out.append(ch)
            i += 1
    return "".join(out).strip()


def _coerce_shape(parsed: dict, capabilities: list[str]) -> dict:
    """Class-B POST-parse (harvest ``_sanitize_stage1_output`` whitelist backbone):
    the capability set is the arbiter (Q2). Per the Stage-1 schema for the requested
    capabilities, whitelist each item type to its fields, coerce (scalar→array,
    null/non-string optional → omit), and drop out-of-scope item types. ``entity_refs``
    is retained only when the ``entities`` capability is in scope; ``temporal_refs``
    keeps ``raw``/``resolved`` and drops schema-illegal extras (type/context/…)."""
    resolved = set(resolve_capabilities(capabilities=list(capabilities)))
    schema = build_extraction_schema(capabilities=list(capabilities))
    props = schema.get("properties", {})
    entities_in_scope = "entities" in resolved

    result: dict[str, Any] = {}
    if "extracted_at" in parsed:
        result["extracted_at"] = parsed["extracted_at"]

    for type_name, type_schema in props.items():
        if type_name == "extracted_at":
            continue
        if type_schema.get("type") != "array":
            if type_name in parsed:
                result[type_name] = parsed[type_name]
            continue
        items_schema = type_schema.get("items")
        parsed_items = parsed.get(type_name)
        if not isinstance(items_schema, dict) or "properties" not in items_schema:
            result[type_name] = parsed_items if isinstance(parsed_items, list) else []
            continue
        item_props = items_schema["properties"]
        required = set(items_schema.get("required", []))
        coerced_items: list[Any] = []
        if isinstance(parsed_items, list):
            for item in parsed_items:
                if isinstance(item, dict):
                    coerced_items.append(
                        _coerce_item(item, item_props, required, entities_in_scope)
                    )
                else:
                    # Preserve non-dict leaves (null, 42, "str") verbatim so strict
                    # validation REJECTS them (→ schema_invalid) instead of silently
                    # dropping — a null sibling must fail its whole unit, not vanish.
                    coerced_items.append(item)
        result[type_name] = coerced_items
    return result


def _coerce_item(
    item: dict,
    item_props: dict,
    required: set[str],
    entities_in_scope: bool,
) -> dict:
    """Whitelist + type-coerce one item to its schema fields. Null/non-string optional
    fields are omitted (they were grounded but wrongly shaped); a scalar for an
    array-typed field is wrapped; an invalid REQUIRED field is kept so finalize
    rejects it (→ schema_invalid) rather than silently passing."""
    new_item: dict[str, Any] = {}
    for field, field_schema in item_props.items():
        if field == "entity_refs" and not entities_in_scope:
            continue  # out-of-scope reference field → drop (Q2)
        if field not in item:
            continue
        value = item[field]
        field_type = field_schema.get("type")
        if field_type == "array" and not isinstance(value, list):
            if value is None:
                continue  # omit null optional array
            value = [value]  # scalar → array (in-scope coerce)
        elif value is None:
            if field in required:
                new_item[field] = value  # keep null required → finalize rejects
            continue
        elif field_type == "string" and not isinstance(value, str):
            if field in required:
                new_item[field] = value  # keep invalid required → finalize rejects
            continue
        new_item[field] = value
    return new_item


def _is_empty_extraction(extraction: Any, capabilities: list[str]) -> bool:
    """True when the model produced NO payload for the unit across the REQUESTED
    capabilities — every requested payload is empty. Type-aware over the Stage-1
    schema: an array payload (facts/decisions/entities/goals/…) counts when
    non-empty; a scalar payload (summary/sentiment) counts when present and
    non-empty. So an entities-only or summary-only extraction is NOT a false-drop.
    Empty-but-valid is the 10/45 "dropped" mode: caught here and retried, never
    silently absorbed."""
    schema = build_extraction_schema(capabilities=list(capabilities))
    for name, prop_schema in schema.get("properties", {}).items():
        if name == "extracted_at":
            continue
        value = (
            extraction.get(name) if isinstance(extraction, dict)
            else getattr(extraction, name, None)
        )
        if prop_schema.get("type") == "array":
            if isinstance(value, list) and value:
                return False
        elif value not in (None, "", [], {}):
            return False
    return True
