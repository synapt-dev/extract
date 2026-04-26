from __future__ import annotations

import pytest

from .conftest import clone_doc, load_symbol


def test_finalize_extraction_runs_stage2_and_stage3_pipeline(
    stage1_output: dict[str, object],
) -> None:
    finalize_extraction = load_symbol("finalize", "finalize_extraction")
    FinalizeContext = load_symbol("finalize", "FinalizeContext")

    result = finalize_extraction(
        clone_doc(stage1_output),
        FinalizeContext(
            produced_by="openai://gpt-4o-mini",
            user_id="user-123",
            source_id="doc-456",
            source_type="session",
            kind="synapt/session_summary",
            capabilities_hint=["entities", "goals", "themes", "summary"],
            extensions={
                "synapt/session_summary": {
                    "focus": "Ship extract package",
                    "done": ["Wrote red specs"],
                }
            },
            embeddings=[
                {
                    "vector": [0.25, -0.5, 0.75],
                    "model": "openai://text-embedding-3-small",
                    "input": "source",
                }
            ],
        ),
    )

    extraction = getattr(result, "extraction", result)

    assert extraction["version"] == "1"
    assert extraction["produced_by"] == "openai://gpt-4o-mini"
    assert extraction["user_id"] == "user-123"
    assert extraction["source_id"] == "doc-456"
    assert extraction["source_type"] == "session"
    assert extraction["kind"] == "synapt/session_summary"

    assert set(extraction["capabilities"]) == {
        "entities",
        "entity_state",
        "entity_context",
        "entity_ids",
        "goals",
        "goal_entity_refs",
        "themes",
        "summary",
        "facts",
        "temporal_refs",
        "relations",
        "relation_origin",
    }
    assert "fake" not in extraction["capabilities"]

    first_entity = extraction["entities"][0]
    first_relation = first_entity["relations"][0]
    first_goal = extraction["goals"][0]
    first_fact = extraction["facts"][0]
    first_temporal = extraction["temporal_refs"][0]
    first_embedding = extraction["embeddings"][0]
    extension = extraction["extensions"]["synapt/session_summary"]

    assert "source" not in first_entity
    assert "signals" not in first_relation
    assert "source" not in first_goal
    assert "source" not in first_fact

    assert first_temporal["version"] == "1"
    assert first_embedding["version"] == "1"
    assert first_embedding["dimensions"] == 3
    assert extension["version"] == "1"


def test_finalize_extraction_prefers_observed_payload_over_requested_profile(
    stage1_output: dict[str, object],
) -> None:
    finalize_extraction = load_symbol("finalize", "finalize_extraction")
    FinalizeContext = load_symbol("finalize", "FinalizeContext")

    result = finalize_extraction(
        clone_doc(stage1_output),
        FinalizeContext(
            produced_by="openai://gpt-4o-mini",
            capabilities_hint=["entities", "goals", "themes", "summary"],
        ),
    )
    extraction = getattr(result, "extraction", result)

    assert "facts" in extraction["capabilities"]
    assert "relations" in extraction["capabilities"]
    assert "relation_origin" in extraction["capabilities"]
    assert "summary" in extraction["capabilities"]


def test_finalize_extraction_rejects_invalid_final_cross_references(
    stage1_output: dict[str, object],
) -> None:
    finalize_extraction = load_symbol("finalize", "finalize_extraction")
    FinalizeContext = load_symbol("finalize", "FinalizeContext")
    broken_output = clone_doc(stage1_output)
    broken_output["goals"] = [
        {
            "text": "Ship @synapt-dev/extract",
            "status": "open",
            "entity_refs": ["e404"],
        }
    ]

    with pytest.raises(Exception, match="entity_refs|e404"):
        finalize_extraction(
            broken_output,
            FinalizeContext(produced_by="openai://gpt-4o-mini"),
        )


def test_finalize_extraction_rejects_malformed_embeddings_instead_of_stripping_them(
    stage1_output: dict[str, object],
) -> None:
    finalize_extraction = load_symbol("finalize", "finalize_extraction")
    FinalizeContext = load_symbol("finalize", "FinalizeContext")

    with pytest.raises(Exception, match="embedding|model|dimensions|vector"):
        finalize_extraction(
            clone_doc(stage1_output),
            FinalizeContext(
                produced_by="openai://gpt-4o-mini",
                embeddings=[{"input": "source"}],
            ),
        )
