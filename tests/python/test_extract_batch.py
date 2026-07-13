"""Contract tests for reliable, source-attributed batch extraction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import get_args

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "python" / "src"))

from synapt.extract import (
    BatchFailureReason,
    BatchUnit,
    extract_batch,
    profile_capabilities,
    validate_extraction,
)
from synapt.extract.batch import _coerce_shape, _strip_output_hygiene


RECALL_CAPABILITIES = ["facts", "decisions", "temporal_refs"]
PRODUCED_BY = "mlx://mlx-community/Ministral-3-3B-Instruct-2512-4bit"
EXTRACTED_AT = "2026-07-13T10:00:00Z"
FIXTURE_SHA256 = "9b183f18ab5116cfb1f5ee67d0e99cd5af3fb7f7b99d649b1d58821f9e7489f1"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "extract-batch-real-failures-v1.json"
FIXTURE_BYTES = FIXTURE_PATH.read_bytes()
FIXTURES = json.loads(FIXTURE_BYTES)
USE_STANDARD_DEFAULT = object()


def _stage1(*, facts=None, decisions=None, temporal_refs=None, **extra):
    return {
        "extracted_at": EXTRACTED_AT,
        "facts": [] if facts is None else facts,
        "decisions": [] if decisions is None else decisions,
        "temporal_refs": [] if temporal_refs is None else temporal_refs,
        **extra,
    }


def _request_prompt(request):
    return request["prompt"]


def _model_visible_text(request):
    return "\n".join(message["content"] for message in request["messages"])


def _run_batch(units, infer, *, capabilities=RECALL_CAPABILITIES, **options):
    kwargs = {
        "infer": infer,
        "produced_by": PRODUCED_BY,
        **options,
    }
    if capabilities is not USE_STANDARD_DEFAULT:
        kwargs["capabilities"] = capabilities
    return asyncio.run(extract_batch(units, **kwargs))


def _response_for_prompt(request, responses_by_source):
    prompt = _request_prompt(request)
    matching = [source for source in responses_by_source if source in prompt]
    if len(matching) > 1:
        # A batch-first implementation may choose this path. Force its per-unit
        # fallback without prescribing the primary batching strategy.
        return "primary batch requires per-unit fallback"
    assert len(matching) == 1, f"request did not contain a known source unit: {prompt}"
    return responses_by_source[matching[0]]


def _assert_success(output, source_unit_id):
    assert output.source_unit_id == source_unit_id
    assert output.extraction is not None
    assert validate_extraction(output.extraction).valid


def _assert_failure(output, source_unit_id, reason):
    assert output.source_unit_id == source_unit_id
    assert output.status == "failed"
    assert output.reason == reason


def _fixture_case(group, fixture_id):
    return next(case for case in FIXTURES[group] if case["fixture_id"] == fixture_id)


def test_real_failure_fixture_pack_is_sha_pinned_and_complete():
    assert hashlib.sha256(FIXTURE_BYTES).hexdigest() == FIXTURE_SHA256
    assert len(FIXTURES["raw_response_cases"]) == 21
    assert len(FIXTURES["malformed_leaf_cases"]) == 25
    assert len(FIXTURES["dropped_source_occurrence_cases"]) == 10
    assert len(FIXTURES["unknown_key_leaf_cases"]) == 2
    assert len(FIXTURES["temporal_shape_cases"]) == 3
    assert len(FIXTURES["contract_derived_cases"]) == 1


@pytest.mark.parametrize(
    "case",
    FIXTURES["raw_response_cases"],
    ids=lambda case: case["fixture_id"],
)
def test_real_raw_responses_strip_hygiene_and_normalize_exactly(case):
    cleaned = _strip_output_hygiene(case["raw_model_output"])
    normalized = _coerce_shape(json.loads(cleaned), RECALL_CAPABILITIES)

    assert normalized == case["expected_normalized_stage1"]


def test_comment_hygiene_preserves_double_slashes_inside_json_strings():
    raw = '''```json
{
  "extracted_at": "2026-07-13T10:00:00Z", // remove this comment
  "facts": [{"text": "Schema: https://synapt.dev/schemas/extract/v1.json"}],
  "decisions": [],
  "temporal_refs": []
}
```'''

    parsed = json.loads(_strip_output_hygiene(raw))

    assert parsed["facts"][0]["text"] == "Schema: https://synapt.dev/schemas/extract/v1.json"


@pytest.mark.parametrize(
    "case",
    FIXTURES["unknown_key_leaf_cases"],
    ids=lambda case: case["fixture_id"],
)
def test_real_unknown_leaf_keys_are_dropped(case):
    stage1 = _stage1()
    stage1[case["field"]] = [deepcopy(case["raw_leaf"])]

    normalized = _coerce_shape(stage1, RECALL_CAPABILITIES)

    assert normalized[case["field"]] == [case["expected_normalized_leaf"]]


@pytest.mark.parametrize(
    "case",
    FIXTURES["temporal_shape_cases"],
    ids=lambda case: case["fixture_id"],
)
def test_temporal_prompt_schema_conflict_is_explicitly_normalized(case):
    stage1 = _stage1(temporal_refs=[deepcopy(case["raw_temporal_ref"])])

    normalized = _coerce_shape(stage1, RECALL_CAPABILITIES)

    assert normalized["temporal_refs"] == [case["expected_normalized_temporal_ref"]]
    assert set(normalized["temporal_refs"][0]) <= {"raw", "resolved"}


def test_entity_refs_scalar_is_coerced_in_scope_and_dropped_out_of_scope():
    case = FIXTURES["contract_derived_cases"][0]
    assert case["must_not_be_reported_as_empirically_observed"] is True

    in_scope = _coerce_shape(
        _stage1(decisions=[deepcopy(case["raw_leaf"])]),
        ["entities", "decisions"],
    )
    out_of_scope = _coerce_shape(
        _stage1(decisions=[deepcopy(case["raw_leaf"])]),
        RECALL_CAPABILITIES,
    )

    assert in_scope["decisions"] == [case["expected_normalized_leaf"]]
    assert out_of_scope["decisions"] == [
        {"text": case["expected_normalized_leaf"]["text"]}
    ]


def test_extract_batch_runs_a_real_recorded_failure_to_strict_validity():
    case = _fixture_case(
        "raw_response_cases",
        "raw-response::sensitivity-raw_decision-r01",
    )
    source = case["input_units"][0]
    requests = []

    def infer(request):
        requests.append(request)
        return case["raw_model_output"]

    outputs = _run_batch(
        [BatchUnit(id=source["unit_id"], text=source["text"])],
        infer,
    )

    assert len(outputs) == 1
    _assert_success(outputs[0], source["unit_id"])
    for field in ("facts", "decisions", "temporal_refs"):
        assert outputs[0].extraction[field] == case["expected_normalized_stage1"][field]
    assert all(source["unit_id"] not in _model_visible_text(request) for request in requests)
    assert all("[UNIT" not in _model_visible_text(request) for request in requests)


def test_extract_batch_uses_per_call_capabilities_with_per_unit_overrides():
    units = [
        BatchUnit(
            id="fact-only",
            text="The extract library emits SynaptExtraction documents.",
            capabilities=["facts"],
        ),
        BatchUnit(
            id="decision-only",
            text="The team decided to keep unit boundaries out of band.",
            capabilities=["decisions"],
        ),
    ]
    responses = {
        units[0].text: json.dumps(_stage1(facts=[{"text": units[0].text}])),
        units[1].text: json.dumps(_stage1(decisions=[{"text": units[1].text}])),
    }
    seen_capabilities = {}

    def infer(request):
        prompt = _request_prompt(request)
        for unit in units:
            if unit.text in prompt:
                seen_capabilities[unit.id] = request["capabilities"]
        return _response_for_prompt(request, responses)

    outputs = _run_batch(units, infer, capabilities=["facts", "decisions"])

    assert seen_capabilities == {
        "fact-only": ["facts"],
        "decision-only": ["decisions"],
    }
    assert [output.source_unit_id for output in outputs] == ["fact-only", "decision-only"]
    assert len(outputs) == len(units)
    _assert_success(outputs[0], "fact-only")
    _assert_success(outputs[1], "decision-only")


def test_extract_batch_uses_the_standard_profile_when_call_capabilities_are_omitted():
    unit = BatchUnit(id="standard-default", text="The standard profile remains the default.")
    seen = []
    standard_stage1 = {
        "extracted_at": EXTRACTED_AT,
        "entities": [],
        "goals": [],
        "themes": [],
        "summary": unit.text,
        "sentiment": "neutral",
        "facts": [{"text": unit.text}],
        "temporal_refs": [],
    }

    def infer(request):
        seen.append(request["capabilities"])
        return json.dumps(standard_stage1)

    outputs = _run_batch(
        [unit],
        infer,
        capabilities=USE_STANDARD_DEFAULT,
    )

    assert seen == [profile_capabilities("standard")]
    _assert_success(outputs[0], unit.id)


def test_dropped_real_output_retries_through_the_inference_seam():
    case = _fixture_case(
        "raw_response_cases",
        "raw-response::sensitivity-raw_multi_clause-r01",
    )
    source = case["input_units"][0]
    completions = [
        case["raw_model_output"],
        json.dumps(_stage1(facts=[{"text": source["text"]}])),
    ]
    calls = 0

    def infer(_request):
        nonlocal calls
        response = completions[min(calls, len(completions) - 1)]
        calls += 1
        return response

    outputs = _run_batch(
        [BatchUnit(id=source["unit_id"], text=source["text"])],
        infer,
    )

    assert calls >= 2
    assert len(outputs) == 1
    _assert_success(outputs[0], source["unit_id"])
    assert outputs[0].extraction["facts"] == [{"text": source["text"]}]


@pytest.mark.parametrize(
    ("completion", "reason"),
    [
        ("not valid JSON", "unparseable"),
        (json.dumps(_stage1(facts=[{"text": 42}])), "schema_invalid"),
    ],
)
def test_terminal_failures_preserve_the_unit_slot(completion, reason):
    unit = BatchUnit(id=f"terminal-{reason}", text="This source unit remains attributable.")

    outputs = _run_batch([unit], lambda _request: completion)

    assert len(outputs) == 1
    _assert_failure(outputs[0], unit.id, reason)


@pytest.mark.parametrize(
    "case",
    FIXTURES["dropped_source_occurrence_cases"],
    ids=lambda case: case["fixture_id"],
)
def test_real_dropped_source_occurrences_never_disappear(case):
    source = case["dropped_source_unit"]
    unit = BatchUnit(id=source["source_unit_id"], text=source["text"])

    outputs = _run_batch([unit], lambda _request: json.dumps(_stage1()))

    assert len(outputs) == 1
    _assert_failure(outputs[0], unit.id, "dropped")


def test_failure_reason_contract_includes_merged_without_widening():
    assert set(get_args(BatchFailureReason)) == {
        "unparseable",
        "schema_invalid",
        "dropped",
        "merged",
    }


def test_one_bad_unit_never_voids_its_neighbors():
    units = [
        BatchUnit(id="good-before", text="The first durable fact is grounded."),
        BatchUnit(id="bad-middle", text="The malformed source remains attributable."),
        BatchUnit(id="good-after", text="The final durable fact is grounded."),
    ]
    responses = {
        units[0].text: json.dumps(_stage1(facts=[{"text": units[0].text}])),
        units[1].text: json.dumps(_stage1(facts=[{"text": {"not": "a string"}}])),
        units[2].text: json.dumps(_stage1(facts=[{"text": units[2].text}])),
    }

    outputs = _run_batch(
        units,
        lambda request: _response_for_prompt(request, responses),
    )

    assert len(outputs) == len(units)
    assert [output.source_unit_id for output in outputs] == [unit.id for unit in units]
    _assert_success(outputs[0], "good-before")
    _assert_failure(outputs[1], "bad-middle", "schema_invalid")
    _assert_success(outputs[2], "good-after")


def test_duplicate_unit_ids_are_rejected_before_inference():
    calls = 0

    def infer(_request):
        nonlocal calls
        calls += 1
        return json.dumps(_stage1())

    with pytest.raises(ValueError, match="duplicate.*id|id.*unique"):
        _run_batch(
            [
                BatchUnit(id="same", text="First source."),
                BatchUnit(id="same", text="Second source."),
            ],
            infer,
        )

    assert calls == 0


def test_empty_input_is_a_noop():
    outputs = _run_batch(
        [],
        lambda _request: pytest.fail("empty batches must not invoke inference"),
    )

    assert outputs == []
