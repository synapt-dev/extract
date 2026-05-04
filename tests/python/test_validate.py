"""Tests for SynaptExtraction IL v1 validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "python" / "src"))

from synapt_extract.validate import validate_extraction


def _minimal_extraction(**overrides):
    base = {
        "version": "1",
        "extracted_at": "2026-04-26T00:00:00Z",
        "produced_by": "openai://gpt-4o-mini",
        "entities": [],
        "goals": [],
        "themes": [],
        "capabilities": ["entities", "goals", "themes"],
    }
    base.update(overrides)
    return base


class TestValidExtraction:

    def test_minimal_valid(self):
        result = validate_extraction(_minimal_extraction())
        assert result.valid
        assert result.errors == []

    def test_full_extraction_valid(self):
        doc = _minimal_extraction(
            entities=[{
                "id": "e1",
                "name": "Mom",
                "type": "person",
                "state": "recovering",
                "context": "family member",
                "date_hint": "2026-04-20",
                "source": {"version": "1", "snippet": "My mom is recovering"},
                "signals": {"version": "1", "confidence": 0.9, "negated": False},
                "relations": [{"target": "e2", "type": "parent_of"}],
            }, {
                "id": "e2",
                "name": "Surgery",
                "type": "event",
            }],
            goals=[{
                "text": "Mom's full recovery",
                "status": "open",
                "entity_refs": ["e1"],
                "stated_at": "2026-04-20T10:00:00Z",
                "source": {"version": "1", "snippet": "I hope mom recovers"},
                "signals": {"version": "1", "hedged": True},
            }],
            themes=["Health", "Family"],
            summary="Prayer for mom's recovery after surgery.",
            sentiment="hopeful",
            facts=[{
                "text": "Mom had surgery on April 20",
                "category": "Health",
                "source": {"version": "1", "snippet": "Mom had surgery"},
            }],
            temporal_refs=[{
                "version": "1",
                "raw": "April 20",
                "type": "point",
                "resolved": "2026-04-20",
            }],
            capabilities=[
                "entities", "entity_state", "entity_context", "entity_ids",
                "goals", "goal_timing", "goal_entity_refs",
                "themes", "summary", "sentiment", "facts",
                "temporal_refs", "temporal_classes",
                "relations", "assertion_signals", "evidence_anchoring",
            ],
        )
        result = validate_extraction(doc)
        assert result.valid, [f"{e.path}: {e.message}" for e in result.errors]


class TestMissingRequiredFields:

    def test_missing_version(self):
        doc = _minimal_extraction()
        del doc["version"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "version" for e in result.errors)

    def test_missing_extracted_at(self):
        doc = _minimal_extraction()
        del doc["extracted_at"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "extracted_at" for e in result.errors)

    def test_missing_produced_by(self):
        doc = _minimal_extraction()
        del doc["produced_by"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "produced_by" for e in result.errors)

    def test_missing_entities(self):
        doc = _minimal_extraction()
        del doc["entities"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "entities" for e in result.errors)

    def test_missing_goals(self):
        doc = _minimal_extraction()
        del doc["goals"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "goals" for e in result.errors)

    def test_missing_themes(self):
        doc = _minimal_extraction()
        del doc["themes"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "themes" for e in result.errors)

    def test_missing_capabilities(self):
        doc = _minimal_extraction()
        del doc["capabilities"]
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "capabilities" for e in result.errors)

    def test_not_an_object(self):
        result = validate_extraction("not an object")
        assert not result.valid
        assert result.errors[0].message == "must be an object"

    def test_null(self):
        result = validate_extraction(None)
        assert not result.valid


class TestEntityValidation:

    def test_entity_missing_name(self):
        doc = _minimal_extraction(entities=[{"type": "person"}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("name" in e.path for e in result.errors)

    def test_entity_missing_type(self):
        doc = _minimal_extraction(entities=[{"name": "Mom"}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("type" in e.path for e in result.errors)

    def test_entity_bad_source_ref_version(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "source": {"version": "2", "snippet": "test"},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("source.version" in e.path for e in result.errors)

    def test_entity_bad_signals_confidence(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "signals": {"version": "1", "confidence": 1.5},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("confidence" in e.path for e in result.errors)

    def test_entity_relation_missing_target(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "relations": [{"type": "knows"}],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("target" in e.path for e in result.errors)

    def test_entity_aliases_valid(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "aliases": ["Mother", "Mama"],
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_entity_aliases_empty_array(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "aliases": [],
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_entity_aliases_not_array(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "aliases": "Mother",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("aliases" in e.path for e in result.errors)

    def test_entity_aliases_empty_string(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "aliases": [""],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("aliases[0]" in e.path for e in result.errors)

    def test_entity_aliases_non_string(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "aliases": [123],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("aliases[0]" in e.path for e in result.errors)


class TestGoalValidation:

    def test_goal_missing_text(self):
        doc = _minimal_extraction(goals=[{
            "status": "open",
            "entity_refs": [],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("text" in e.path for e in result.errors)

    def test_goal_invalid_status(self):
        doc = _minimal_extraction(goals=[{
            "text": "recover",
            "status": "pending",
            "entity_refs": [],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("status" in e.path for e in result.errors)

    def test_goal_missing_entity_refs(self):
        doc = _minimal_extraction(goals=[{
            "text": "recover",
            "status": "open",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("entity_refs" in e.path for e in result.errors)


class TestCapabilityValidation:

    def test_unknown_capability(self):
        doc = _minimal_extraction(capabilities=["entities", "psychic_powers"])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("psychic_powers" in e.message for e in result.errors)

    def test_all_valid_capabilities(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        doc = _minimal_extraction(capabilities=sorted(EXTRACTION_CAPABILITIES))
        result = validate_extraction(doc)
        assert result.valid


class TestEmbeddingValidation:

    def test_valid_embedding(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "vector": [0.1, 0.2, 0.3],
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 3,
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_embedding_missing_vector(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 3,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("vector" in e.path for e in result.errors)


class TestTemporalRefValidation:

    def test_valid_temporal_ref(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "next Tuesday",
            "type": "point",
            "resolved": "2026-04-28",
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_invalid_temporal_type(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "sometime",
            "type": "vague",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("type" in e.path for e in result.errors)


class TestProducedByFormat:

    def test_produced_by_requires_scheme(self):
        doc = _minimal_extraction(produced_by="gpt-4o-mini")
        result = validate_extraction(doc)
        assert not result.valid
        assert any("produced_by" in e.path for e in result.errors)

    def test_produced_by_valid_uri(self):
        doc = _minimal_extraction(produced_by="openai://gpt-4o-mini")
        result = validate_extraction(doc)
        assert result.valid

    def test_produced_by_anthropic_uri(self):
        doc = _minimal_extraction(produced_by="anthropic://claude-sonnet-4-20250514")
        result = validate_extraction(doc)
        assert result.valid

    def test_produced_by_empty_string(self):
        doc = _minimal_extraction(produced_by="")
        result = validate_extraction(doc)
        assert not result.valid


class TestProducedByProducerObject:

    def test_produced_by_string_backwards_compat_valid(self):
        doc = _minimal_extraction(produced_by="anthropic://claude-sonnet-4-6")
        result = validate_extraction(doc)
        assert result.valid

    def test_produced_by_structured_minimal_valid(self):
        doc = _minimal_extraction(produced_by={
            "version": "1",
            "model": "anthropic://claude-sonnet-4-6",
        })
        result = validate_extraction(doc)
        assert result.valid

    def test_produced_by_structured_full_valid(self):
        doc = _minimal_extraction(produced_by={
            "version": "1",
            "model": "anthropic://claude-sonnet-4-6",
            "model_version": "claude-sonnet-4-6-20250514",
            "deployment": "bedrock",
            "configuration": {
                "reasoning_effort": "high",
                "system_prompt_hash": "abc123",
                "temperature": 0.2,
                "top_p": 0.95,
                "max_tokens": 2048,
                "vendor_flag": True,
            },
            "operator": "synapt-dev",
            "signature": "eyJhbGciOiJIUzI1NiJ9.payload.signature",
        })
        result = validate_extraction(doc)
        assert result.valid

    def test_produced_by_structured_missing_version_fails(self):
        doc = _minimal_extraction(produced_by={
            "model": "anthropic://claude-sonnet-4-6",
        })
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "produced_by.version" for e in result.errors)

    def test_produced_by_structured_missing_model_fails(self):
        doc = _minimal_extraction(produced_by={
            "version": "1",
        })
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "produced_by.model" for e in result.errors)

    def test_produced_by_structured_unknown_root_field_fails(self):
        doc = _minimal_extraction(produced_by={
            "version": "1",
            "model": "anthropic://claude-sonnet-4-6",
            "extra_field": "boom",
        })
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "produced_by.extra_field" for e in result.errors)

    def test_produced_by_structured_open_configuration_passes(self):
        doc = _minimal_extraction(produced_by={
            "version": "1",
            "model": "anthropic://claude-sonnet-4-6",
            "configuration": {
                "provider_sampling_mode": "adaptive",
                "vendor_flag": True,
            },
        })
        result = validate_extraction(doc)
        assert result.valid

    def test_produced_by_structured_known_configuration_fields_pass(self):
        doc = _minimal_extraction(produced_by={
            "version": "1",
            "model": "anthropic://claude-sonnet-4-6",
            "configuration": {
                "reasoning_effort": "medium",
                "system_prompt_hash": "f00dbabe",
                "temperature": 0.1,
                "top_p": 0.95,
                "max_tokens": 2048,
            },
        })
        result = validate_extraction(doc)
        assert result.valid

    def test_produced_by_structured_malformed_model_fails(self):
        doc = _minimal_extraction(produced_by={
            "version": "1",
            "model": "claude-sonnet-4-6",
        })
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "produced_by.model" for e in result.errors)

    def test_produced_by_structured_non_string_signature_fails(self):
        doc = _minimal_extraction(produced_by={
            "version": "1",
            "model": "anthropic://claude-sonnet-4-6",
            "signature": {"alg": "HS256"},
        })
        result = validate_extraction(doc)
        assert not result.valid
        assert any(e.path == "produced_by.signature" for e in result.errors)


class TestNonEmptyStrings:

    def test_entity_name_empty(self):
        doc = _minimal_extraction(entities=[{"name": "", "type": "person"}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("name" in e.path for e in result.errors)

    def test_entity_type_empty(self):
        doc = _minimal_extraction(entities=[{"name": "Mom", "type": ""}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("type" in e.path for e in result.errors)

    def test_goal_text_empty(self):
        doc = _minimal_extraction(goals=[{
            "text": "",
            "status": "open",
            "entity_refs": [],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("text" in e.path for e in result.errors)

    def test_theme_empty_string(self):
        doc = _minimal_extraction(themes=["Health", ""])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("themes" in e.path for e in result.errors)

    def test_fact_text_empty(self):
        doc = _minimal_extraction(facts=[{"text": ""}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("text" in e.path for e in result.errors)

    def test_relation_target_empty(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person",
            "relations": [{"target": "", "type": "knows"}],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("target" in e.path for e in result.errors)

    def test_relation_type_empty(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person",
            "relations": [{"target": "e2", "type": ""}],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("type" in e.path for e in result.errors)

    def test_temporal_ref_raw_empty(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("raw" in e.path for e in result.errors)


class TestTimestampValidation:

    def test_extracted_at_bad_timestamp(self):
        doc = _minimal_extraction(extracted_at="not-a-date")
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extracted_at" in e.path for e in result.errors)

    def test_extracted_at_date_only_rejected(self):
        doc = _minimal_extraction(extracted_at="2026-04-26")
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extracted_at" in e.path for e in result.errors)

    def test_extracted_at_valid_datetime(self):
        doc = _minimal_extraction(extracted_at="2026-04-26T10:30:00Z")
        result = validate_extraction(doc)
        assert result.valid

    def test_extracted_at_valid_datetime_with_offset(self):
        doc = _minimal_extraction(extracted_at="2026-04-26T10:30:00+05:30")
        result = validate_extraction(doc)
        assert result.valid

    def test_extracted_at_valid_datetime_no_seconds(self):
        doc = _minimal_extraction(extracted_at="2026-04-26T10:30Z")
        result = validate_extraction(doc)
        assert result.valid

    def test_goal_stated_at_bad(self):
        doc = _minimal_extraction(goals=[{
            "text": "Recovery",
            "status": "open",
            "entity_refs": [],
            "stated_at": "not-a-date",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("stated_at" in e.path for e in result.errors)

    def test_goal_resolved_at_bad(self):
        doc = _minimal_extraction(goals=[{
            "text": "Recovery",
            "status": "resolved",
            "entity_refs": [],
            "resolved_at": "whenever",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("resolved_at" in e.path for e in result.errors)

    def test_temporal_resolved_bad(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "next week",
            "type": "point",
            "resolved": "not-a-date",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("resolved" in e.path for e in result.errors)


class TestEmptySubSchemaWrappers:

    def test_source_version_only_rejected(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "source": {"version": "1"},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("source" in e.path for e in result.errors)

    def test_signals_version_only_rejected(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "signals": {"version": "1"},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("signals" in e.path for e in result.errors)

    def test_source_with_snippet_accepted(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "source": {"version": "1", "snippet": "My mom"},
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_signals_with_confidence_accepted(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom",
            "type": "person",
            "signals": {"version": "1", "confidence": 0.9},
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_goal_source_version_only_rejected(self):
        doc = _minimal_extraction(goals=[{
            "text": "Recovery",
            "status": "open",
            "entity_refs": [],
            "source": {"version": "1"},
        }])
        result = validate_extraction(doc)
        assert not result.valid

    def test_fact_signals_version_only_rejected(self):
        doc = _minimal_extraction(facts=[{
            "text": "Surgery happened",
            "signals": {"version": "1"},
        }])
        result = validate_extraction(doc)
        assert not result.valid


class TestTemporalRangeConstraints:

    def test_range_without_resolved_end_rejected(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "April 20 to May 1",
            "type": "range",
            "resolved": "2026-04-20",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("resolved_end" in e.path or "resolved_end" in e.message for e in result.errors)

    def test_range_with_resolved_end_accepted(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "April 20 to May 1",
            "type": "range",
            "resolved": "2026-04-20",
            "resolved_end": "2026-05-01",
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_point_without_resolved_end_accepted(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "April 20",
            "type": "point",
            "resolved": "2026-04-20",
        }])
        result = validate_extraction(doc)
        assert result.valid


class TestEmbeddingDimensionEquality:

    def test_dimensions_mismatch_rejected(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "vector": [0.1, 0.2],
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 99,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("dimensions" in e.path for e in result.errors)

    def test_dimensions_match_accepted(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "vector": [0.1, 0.2, 0.3],
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 3,
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_vector_non_number_rejected(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "vector": [0.1, "bad", 0.3],
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 3,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("vector" in e.path for e in result.errors)

    def test_vector_bool_rejected(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "vector": [0.1, True, 0.3],
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 3,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("vector" in e.path for e in result.errors)

    def test_vector_all_numbers_accepted(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "vector": [0.1, 0.2, 0.3],
            "model": "openai://text-embedding-3-small",
            "input": "source",
            "dimensions": 3,
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_embedding_model_requires_scheme(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1",
            "vector": [0.1, 0.2],
            "model": "text-embedding-3-small",
            "input": "source",
            "dimensions": 2,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("model" in e.path for e in result.errors)


class TestCrossRefIntegrity:

    def test_goal_entity_refs_to_missing_ids(self):
        doc = _minimal_extraction(
            entities=[{"name": "Mom", "type": "person"}],
            goals=[{
                "text": "Recovery",
                "status": "open",
                "entity_refs": ["e1"],
            }],
        )
        result = validate_extraction(doc)
        assert not result.valid
        assert any("entity_refs" in e.path for e in result.errors)

    def test_goal_entity_refs_to_valid_ids(self):
        doc = _minimal_extraction(
            entities=[{"id": "e1", "name": "Mom", "type": "person"}],
            goals=[{
                "text": "Recovery",
                "status": "open",
                "entity_refs": ["e1"],
            }],
        )
        result = validate_extraction(doc)
        assert result.valid

    def test_relation_target_to_missing_entity(self):
        doc = _minimal_extraction(entities=[{
            "id": "e1",
            "name": "Mom",
            "type": "person",
            "relations": [{"target": "e99", "type": "knows"}],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("target" in e.path for e in result.errors)

    def test_relation_target_to_valid_entity(self):
        doc = _minimal_extraction(entities=[
            {"id": "e1", "name": "Mom", "type": "person",
             "relations": [{"target": "e2", "type": "parent_of"}]},
            {"id": "e2", "name": "Dad", "type": "person"},
        ])
        result = validate_extraction(doc)
        assert result.valid

    def test_empty_entity_refs_accepted(self):
        doc = _minimal_extraction(goals=[{
            "text": "Recovery",
            "status": "open",
            "entity_refs": [],
        }])
        result = validate_extraction(doc)
        assert result.valid


class TestExtensionKeyFormat:

    def test_extension_key_requires_namespace(self):
        doc = _minimal_extraction(extensions={"badkey": {"foo": "bar"}})
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extensions" in e.path for e in result.errors)

    def test_extension_key_valid_namespace(self):
        doc = _minimal_extraction(extensions={"conversa/prayer": {"category": "Health"}})
        result = validate_extraction(doc)
        assert result.valid


class TestSummaryValidation:

    def test_empty_summary_rejected(self):
        doc = _minimal_extraction(summary="")
        result = validate_extraction(doc)
        assert not result.valid
        assert any("summary" in e.path for e in result.errors)

    def test_nonempty_summary_accepted(self):
        doc = _minimal_extraction(summary="A prayer for healing.")
        result = validate_extraction(doc)
        assert result.valid

    def test_summary_absent_accepted(self):
        doc = _minimal_extraction()
        assert "summary" not in doc
        result = validate_extraction(doc)
        assert result.valid


class TestTemporalUnresolvedConstraints:

    def test_unresolved_with_resolved_rejected(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "someday",
            "type": "unresolved",
            "resolved": "2026-04-20",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("resolved" in e.path or "unresolved" in e.message for e in result.errors)

    def test_unresolved_with_resolved_end_rejected(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "someday",
            "type": "unresolved",
            "resolved_end": "2026-05-01",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("resolved" in e.path or "unresolved" in e.message for e in result.errors)

    def test_unresolved_without_resolved_accepted(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "someday",
            "type": "unresolved",
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_point_with_resolved_accepted(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1",
            "raw": "April 20",
            "type": "point",
            "resolved": "2026-04-20",
        }])
        result = validate_extraction(doc)
        assert result.valid


class TestKindFormat:

    def test_kind_requires_namespace(self):
        doc = _minimal_extraction(kind="badkind")
        result = validate_extraction(doc)
        assert not result.valid
        assert any("kind" in e.path for e in result.errors)

    def test_kind_valid_namespace(self):
        doc = _minimal_extraction(kind="conversa/prayer")
        result = validate_extraction(doc)
        assert result.valid


class TestAdditionalProperties:

    def test_root_extra_property_rejected(self):
        doc = _minimal_extraction()
        doc["extra"] = True
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extra" in e.path and "additional" in e.message for e in result.errors)

    def test_entity_extra_property_rejected(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person", "extra_field": True,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extra_field" in e.path for e in result.errors)

    def test_goal_extra_property_rejected(self):
        doc = _minimal_extraction(goals=[{
            "text": "Recovery", "status": "open", "entity_refs": [],
            "custom": "value",
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("custom" in e.path for e in result.errors)

    def test_fact_extra_property_rejected(self):
        doc = _minimal_extraction(facts=[{"text": "A fact", "extra": 1}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extra" in e.path for e in result.errors)

    def test_relation_extra_property_rejected(self):
        doc = _minimal_extraction(entities=[{
            "id": "e1", "name": "Mom", "type": "person",
            "relations": [{"target": "e2", "type": "knows", "weight": 0.5}],
        }, {"id": "e2", "name": "Dad", "type": "person"}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("weight" in e.path for e in result.errors)

    def test_source_ref_extra_property_rejected(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person",
            "source": {"version": "1", "snippet": "text", "extra": True},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extra" in e.path for e in result.errors)

    def test_signals_extra_property_rejected(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person",
            "signals": {"version": "1", "confidence": 0.9, "extra": True},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extra" in e.path for e in result.errors)

    def test_temporal_ref_extra_property_rejected(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1", "raw": "April 20", "extra": True,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extra" in e.path for e in result.errors)

    def test_embedding_extra_property_rejected(self):
        doc = _minimal_extraction(embeddings=[{
            "version": "1", "vector": [0.1, 0.2], "model": "openai://emb",
            "input": "source", "dimensions": 2, "extra": True,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("extra" in e.path for e in result.errors)


class TestOptionalFieldTypes:

    def test_sentiment_must_be_string(self):
        doc = _minimal_extraction(sentiment=3)
        result = validate_extraction(doc)
        assert not result.valid
        assert any("sentiment" in e.path for e in result.errors)

    def test_source_id_must_be_string(self):
        doc = _minimal_extraction(source_id=123)
        result = validate_extraction(doc)
        assert not result.valid
        assert any("source_id" in e.path for e in result.errors)

    def test_source_type_must_be_string(self):
        doc = _minimal_extraction(source_type=True)
        result = validate_extraction(doc)
        assert not result.valid
        assert any("source_type" in e.path for e in result.errors)

    def test_user_id_must_be_string(self):
        doc = _minimal_extraction(user_id=42)
        result = validate_extraction(doc)
        assert not result.valid
        assert any("user_id" in e.path for e in result.errors)

    def test_entity_state_must_be_string(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person", "state": 7,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("state" in e.path for e in result.errors)

    def test_entity_context_must_be_string(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person", "context": 7,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("context" in e.path for e in result.errors)

    def test_entity_date_hint_must_be_string(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person", "date_hint": 7,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("date_hint" in e.path for e in result.errors)

    def test_fact_category_must_be_string(self):
        doc = _minimal_extraction(facts=[{"text": "A fact", "category": 1}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("category" in e.path for e in result.errors)

    def test_relation_origin_must_be_string(self):
        doc = _minimal_extraction(entities=[{
            "id": "e1", "name": "Mom", "type": "person",
            "relations": [{"target": "e2", "type": "knows", "origin": 1}],
        }, {"id": "e2", "name": "Dad", "type": "person"}])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("origin" in e.path for e in result.errors)

    def test_goal_entity_refs_items_must_be_strings(self):
        doc = _minimal_extraction(goals=[{
            "text": "Recovery", "status": "open", "entity_refs": [1],
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("entity_refs" in e.path for e in result.errors)

    def test_temporal_context_must_be_string(self):
        doc = _minimal_extraction(temporal_refs=[{
            "version": "1", "raw": "April 20", "context": 1,
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("context" in e.path for e in result.errors)


class TestSourceRefOffsets:

    def test_offset_start_negative_rejected(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person",
            "source": {"version": "1", "snippet": "text", "offset_start": -1},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("offset_start" in e.path for e in result.errors)

    def test_offset_start_string_rejected(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person",
            "source": {"version": "1", "snippet": "text", "offset_start": "1"},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("offset_start" in e.path for e in result.errors)

    def test_offset_start_valid(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person",
            "source": {"version": "1", "snippet": "text", "offset_start": 0, "offset_end": 4},
        }])
        result = validate_extraction(doc)
        assert result.valid

    def test_sentence_index_negative_rejected(self):
        doc = _minimal_extraction(entities=[{
            "name": "Mom", "type": "person",
            "source": {"version": "1", "snippet": "text", "sentence_index": -1},
        }])
        result = validate_extraction(doc)
        assert not result.valid
        assert any("sentence_index" in e.path for e in result.errors)


class TestV12Keywords:

    def test_valid_keywords(self):
        result = validate_extraction(_minimal_extraction(keywords=["prayer", "healing"]))
        assert result.valid

    def test_empty_keywords(self):
        result = validate_extraction(_minimal_extraction(keywords=[]))
        assert result.valid

    def test_keywords_empty_string_rejected(self):
        result = validate_extraction(_minimal_extraction(keywords=["prayer", ""]))
        assert not result.valid
        assert any("keywords[1]" in e.path for e in result.errors)

    def test_keywords_not_array_rejected(self):
        result = validate_extraction(_minimal_extraction(keywords="prayer"))
        assert not result.valid
        assert any(e.path == "keywords" for e in result.errors)


class TestV12Questions:

    def test_valid_question(self):
        result = validate_extraction(_minimal_extraction(questions=[{"text": "How are you?"}]))
        assert result.valid

    def test_question_with_directed_to(self):
        result = validate_extraction(_minimal_extraction(
            entities=[{"id": "e1", "name": "Mom", "type": "person"}],
            questions=[{"text": "How is Mom?", "directed_to": "e1"}],
        ))
        assert result.valid

    def test_question_missing_text_rejected(self):
        result = validate_extraction(_minimal_extraction(questions=[{"directed_to": "someone"}]))
        assert not result.valid
        assert any("questions[0].text" in e.path for e in result.errors)

    def test_question_empty_text_rejected(self):
        result = validate_extraction(_minimal_extraction(questions=[{"text": ""}]))
        assert not result.valid
        assert any("questions[0].text" in e.path for e in result.errors)

    def test_question_unknown_property_rejected(self):
        result = validate_extraction(_minimal_extraction(questions=[{"text": "Why?", "badProp": True}]))
        assert not result.valid
        assert any("questions[0].badProp" in e.path for e in result.errors)

    def test_questions_not_array_rejected(self):
        result = validate_extraction(_minimal_extraction(questions="why?"))
        assert not result.valid
        assert any(e.path == "questions" for e in result.errors)


class TestV12Actions:

    def test_valid_action(self):
        result = validate_extraction(_minimal_extraction(
            actions=[{"text": "Schedule appointment", "origin": "extracted"}],
        ))
        assert result.valid

    def test_action_proposed_from_goals(self):
        result = validate_extraction(_minimal_extraction(
            actions=[{"text": "Follow up", "origin": "proposed_from_goals"}],
        ))
        assert result.valid

    def test_action_missing_origin_rejected(self):
        result = validate_extraction(_minimal_extraction(actions=[{"text": "Do something"}]))
        assert not result.valid
        assert any("actions[0].origin" in e.path for e in result.errors)

    def test_action_invalid_origin_rejected(self):
        result = validate_extraction(_minimal_extraction(
            actions=[{"text": "Do something", "origin": "unknown"}],
        ))
        assert not result.valid
        assert any("actions[0].origin" in e.path for e in result.errors)

    def test_action_missing_text_rejected(self):
        result = validate_extraction(_minimal_extraction(actions=[{"origin": "extracted"}]))
        assert not result.valid
        assert any("actions[0].text" in e.path for e in result.errors)

    def test_action_with_entity_refs(self):
        result = validate_extraction(_minimal_extraction(
            entities=[{"id": "e1", "name": "Mom", "type": "person"}],
            actions=[{"text": "Call Mom", "origin": "extracted", "entity_refs": ["e1"]}],
        ))
        assert result.valid

    def test_action_dangling_entity_ref_rejected(self):
        result = validate_extraction(_minimal_extraction(
            entities=[{"id": "e1", "name": "Mom", "type": "person"}],
            actions=[{"text": "Call Dad", "origin": "extracted", "entity_refs": ["e99"]}],
        ))
        assert not result.valid
        assert any("actions[0].entity_refs[0]" in e.path for e in result.errors)

    def test_action_unknown_property_rejected(self):
        result = validate_extraction(_minimal_extraction(
            actions=[{"text": "Go", "origin": "extracted", "priority": "high"}],
        ))
        assert not result.valid
        assert any("actions[0].priority" in e.path for e in result.errors)


class TestV12Decisions:

    def test_valid_decision(self):
        result = validate_extraction(_minimal_extraction(decisions=[{"text": "Go forward"}]))
        assert result.valid

    def test_decision_with_entity_refs_and_decided_at(self):
        result = validate_extraction(_minimal_extraction(
            entities=[{"id": "e1", "name": "Team", "type": "organization"}],
            decisions=[{"text": "Chose option A", "entity_refs": ["e1"], "decided_at": "2026-05-01"}],
        ))
        assert result.valid

    def test_decision_missing_text_rejected(self):
        result = validate_extraction(_minimal_extraction(decisions=[{"entity_refs": []}]))
        assert not result.valid
        assert any("decisions[0].text" in e.path for e in result.errors)

    def test_decision_dangling_entity_ref_rejected(self):
        result = validate_extraction(_minimal_extraction(
            entities=[],
            decisions=[{"text": "Go with plan B", "entity_refs": ["e99"]}],
        ))
        assert not result.valid
        assert any("decisions[0].entity_refs[0]" in e.path for e in result.errors)

    def test_decision_unknown_property_rejected(self):
        result = validate_extraction(_minimal_extraction(
            decisions=[{"text": "Yes", "importance": "high"}],
        ))
        assert not result.valid
        assert any("decisions[0].importance" in e.path for e in result.errors)


class TestV12SentimentDualShape:

    def test_string_sentiment(self):
        result = validate_extraction(_minimal_extraction(sentiment="hopeful"))
        assert result.valid

    def test_structured_sentiment(self):
        result = validate_extraction(_minimal_extraction(
            sentiment={"version": "1", "valence": "positive", "intensity": 0.8, "confidence": 0.9},
        ))
        assert result.valid

    def test_structured_sentiment_minimal(self):
        result = validate_extraction(_minimal_extraction(
            sentiment={"version": "1", "valence": "negative"},
        ))
        assert result.valid

    def test_structured_sentiment_invalid_valence_rejected(self):
        result = validate_extraction(_minimal_extraction(
            sentiment={"version": "1", "valence": "angry"},
        ))
        assert not result.valid
        assert any("sentiment.valence" in e.path for e in result.errors)

    def test_structured_sentiment_missing_valence_rejected(self):
        result = validate_extraction(_minimal_extraction(
            sentiment={"version": "1", "intensity": 0.5},
        ))
        assert not result.valid
        assert any("sentiment.valence" in e.path for e in result.errors)

    def test_structured_sentiment_intensity_out_of_range_rejected(self):
        result = validate_extraction(_minimal_extraction(
            sentiment={"version": "1", "valence": "positive", "intensity": 1.5},
        ))
        assert not result.valid
        assert any("sentiment.intensity" in e.path for e in result.errors)

    def test_structured_sentiment_unknown_property_rejected(self):
        result = validate_extraction(_minimal_extraction(
            sentiment={"version": "1", "valence": "neutral", "mood": "calm"},
        ))
        assert not result.valid
        assert any("sentiment.mood" in e.path for e in result.errors)

    def test_sentiment_as_array_rejected(self):
        result = validate_extraction(_minimal_extraction(sentiment=["positive"]))
        assert not result.valid
        assert any(e.path == "sentiment" for e in result.errors)

    def test_sentiment_as_number_rejected(self):
        result = validate_extraction(_minimal_extraction(sentiment=0.8))
        assert not result.valid
        assert any(e.path == "sentiment" for e in result.errors)


class TestV12Language:

    def test_valid_language_tag(self):
        result = validate_extraction(_minimal_extraction(language="en-US"))
        assert result.valid

    def test_simple_language_tag(self):
        result = validate_extraction(_minimal_extraction(language="es"))
        assert result.valid

    def test_three_letter_language_tag(self):
        result = validate_extraction(_minimal_extraction(language="por"))
        assert result.valid

    def test_complex_bcp47_tag(self):
        result = validate_extraction(_minimal_extraction(language="zh-Hans-CN"))
        assert result.valid

    def test_invalid_language_tag_rejected(self):
        result = validate_extraction(_minimal_extraction(language="english"))
        assert not result.valid
        assert any(e.path == "language" for e in result.errors)

    def test_language_as_number_rejected(self):
        result = validate_extraction(_minimal_extraction(language=42))
        assert not result.valid
        assert any(e.path == "language" for e in result.errors)


class TestV12SourceMetadata:

    def test_valid_source_metadata(self):
        result = validate_extraction(_minimal_extraction(
            source_metadata={"version": "1", "token_count": 500, "character_count": 2000, "modality": "text", "format": "plain"},
        ))
        assert result.valid

    def test_source_metadata_version_only(self):
        result = validate_extraction(_minimal_extraction(source_metadata={"version": "1"}))
        assert result.valid

    def test_source_metadata_negative_token_count_rejected(self):
        result = validate_extraction(_minimal_extraction(source_metadata={"version": "1", "token_count": -1}))
        assert not result.valid
        assert any("source_metadata.token_count" in e.path for e in result.errors)

    def test_source_metadata_float_token_count_rejected(self):
        result = validate_extraction(_minimal_extraction(source_metadata={"version": "1", "token_count": 3.5}))
        assert not result.valid
        assert any("source_metadata.token_count" in e.path for e in result.errors)

    def test_source_metadata_unknown_property_rejected(self):
        result = validate_extraction(_minimal_extraction(source_metadata={"version": "1", "word_count": 100}))
        assert not result.valid
        assert any("source_metadata.word_count" in e.path for e in result.errors)

    def test_source_metadata_not_object_rejected(self):
        result = validate_extraction(_minimal_extraction(source_metadata="text"))
        assert not result.valid
        assert any(e.path == "source_metadata" for e in result.errors)


class TestV12Confidence:

    def test_valid_confidence(self):
        result = validate_extraction(_minimal_extraction(confidence=0.85))
        assert result.valid

    def test_confidence_zero(self):
        result = validate_extraction(_minimal_extraction(confidence=0))
        assert result.valid

    def test_confidence_one(self):
        result = validate_extraction(_minimal_extraction(confidence=1))
        assert result.valid

    def test_confidence_above_one_rejected(self):
        result = validate_extraction(_minimal_extraction(confidence=1.1))
        assert not result.valid
        assert any(e.path == "confidence" for e in result.errors)

    def test_confidence_below_zero_rejected(self):
        result = validate_extraction(_minimal_extraction(confidence=-0.1))
        assert not result.valid
        assert any(e.path == "confidence" for e in result.errors)

    def test_confidence_as_string_rejected(self):
        result = validate_extraction(_minimal_extraction(confidence="high"))
        assert not result.valid
        assert any(e.path == "confidence" for e in result.errors)


class TestV12Capabilities:

    def test_new_capabilities_accepted(self):
        result = validate_extraction(_minimal_extraction(capabilities=[
            "entities", "goals", "themes", "keywords", "summary",
            "sentiment", "structured_sentiment", "facts", "questions",
            "actions", "decisions", "language", "source_metadata", "confidence",
        ]))
        assert result.valid


class TestJsonSchemaFiles:

    def test_all_schema_files_are_valid_json(self):
        schema_dir = Path(__file__).resolve().parents[2] / "schemas"
        for schema_file in schema_dir.rglob("*.json"):
            content = schema_file.read_text()
            parsed = json.loads(content)
            assert "$schema" in parsed, f"{schema_file.name} missing $schema"
            assert "$id" in parsed, f"{schema_file.name} missing $id"

    def test_extraction_schema_references_sub_schemas(self):
        schema_path = Path(__file__).resolve().parents[2] / "schemas" / "extract" / "v1.json"
        schema = json.loads(schema_path.read_text())
        schema_str = json.dumps(schema)
        assert "source-ref/v1.json" in schema_str
        assert "embedding/v1.json" in schema_str
        assert "assertion-signals/v1.json" in schema_str
        assert "temporal-ref/v1.json" in schema_str
        assert "producer/v1.json" in schema_str

    def test_producer_schema_file_exists(self):
        producer_path = Path(__file__).resolve().parents[2] / "schemas" / "producer" / "v1.json"
        assert producer_path.exists()
        schema = json.loads(producer_path.read_text())
        assert schema["$id"] == "https://synapt.dev/schemas/producer/v1.json"

    def test_extraction_schema_required_fields(self):
        schema_path = Path(__file__).resolve().parents[2] / "schemas" / "extract" / "v1.json"
        schema = json.loads(schema_path.read_text())
        required = schema["required"]
        assert "version" in required
        assert "extracted_at" in required
        assert "produced_by" in required
        assert "entities" in required
        assert "goals" in required
        assert "themes" in required
        assert "capabilities" in required
