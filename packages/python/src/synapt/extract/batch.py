"""Batch Stage-1 extraction primitive for SynaptExtraction.

SKELETON (recall#868 → extract_batch). API conformed to the pinned contract
(config/design/extract-batch-limits-characterization-2026-07-13.md §"Contract
decisions") AND to Sentinel's spec (extract#28, tests/python/test_extract_batch.py).
Every body raises NotImplementedError — the implementation lands in the follow-up
impl PR (TDD: this skeleton makes the spec COLLECT and run RED, not ImportError).

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
  • Input: list[BatchUnit(id, text, capabilities?)] — explicit attribution; the
    id rides into the output as source_unit_id (boundaries stay out-of-band, never
    in model-visible text).
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
        coerced (scalar→array, decided_at null→omit, category→valid/default),
        out-of-scope dropped; temporal_refs → schema-valid raw/resolved only.

Harvest map: scratchpad/extract_batch_craft_harvest.md. Boundary: OSS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, TypedDict

from synapt.extract.finalize import finalize_extraction

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
    optionally overrides the per-call default for this unit."""

    id: str
    text: str
    capabilities: list[str] | None = None


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

    SKELETON — body is NotImplementedError; the impl lands in the follow-up PR.
    """
    raise NotImplementedError(
        "extract_batch skeleton conforms to the pinned contract + spec; the "
        "implementation lands in the impl PR (recall#868)."
    )


# --- Intended internal decomposition (stubs; bodies in the impl PR) ------------

def _strip_output_hygiene(raw: str) -> str:
    """Class-A PRE-parse (NET-NEW): strip ``` fences + ``//`` comments so grounded-
    but-wrapped JSON parses. STRING-LITERAL-AWARE — a ``//`` inside a JSON string
    value (e.g. ``https://…``) is preserved; only real line-comments are removed."""
    raise NotImplementedError


def _coerce_shape(parsed: dict, capabilities: list[str]) -> dict:
    """Class-B POST-parse (harvest ``_sanitize_stage1_output`` whitelist backbone):
    the capability set is the arbiter (Q2) — in-scope fields coerced (scalar→array,
    ``decided_at`` null→omit, ``category``→valid/default), out-of-scope dropped;
    ``temporal_refs`` coerced to schema-valid ``raw``/``resolved`` only."""
    raise NotImplementedError
