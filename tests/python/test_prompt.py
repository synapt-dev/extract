"""Tests for SynaptExtraction IL v1 composable prompt system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "python" / "src"))

from synapt_extract import (
    ExtractionBuilder,
    build_finalized_extraction_schema,
    build_extraction_response_format,
    build_extraction_schema,
    build_extraction_prompt,
    create_extraction_builder,
    resolve_capabilities,
)


SAMPLE_TEXT = "Please pray for my mom. She had surgery on April 20 and is recovering well."


class TestResolveCapabilities:

    def test_explicit_capabilities(self):
        result = resolve_capabilities(capabilities=["entities", "goals", "themes"])
        assert set(result) == {"entities", "goals", "themes"}

    def test_profile_minimal(self):
        result = resolve_capabilities(profile="minimal")
        assert "entities" in result
        assert "entity_state" in result
        assert "goals" in result
        assert "themes" in result
        assert "summary" in result
        assert "relations" not in result

    def test_profile_standard(self):
        result = resolve_capabilities(profile="standard")
        assert "entities" in result
        assert "entity_context" in result
        assert "goal_timing" in result
        assert "facts" in result
        assert "temporal_refs" in result
        assert "sentiment" in result
        assert "evidence_anchoring" in result
        assert "relations" not in result

    def test_profile_full(self):
        result = resolve_capabilities(profile="full")
        assert "entity_ids" in result
        assert "goal_entity_refs" in result
        assert "relations" in result
        assert "relation_origin" in result
        assert "assertion_signals" in result
        assert "temporal_classes" in result

    def test_profile_with_add(self):
        result = resolve_capabilities(profile="minimal", add=["relations"])
        assert "entities" in result
        assert "relations" in result
        assert "entity_ids" in result

    def test_profile_with_remove(self):
        result = resolve_capabilities(profile="standard", remove=["sentiment"])
        assert "entities" in result
        assert "sentiment" not in result

    def test_profile_with_add_and_remove(self):
        result = resolve_capabilities(
            profile="standard",
            add=["relations"],
            remove=["sentiment"],
        )
        assert "relations" in result
        assert "entity_ids" in result
        assert "sentiment" not in result

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            resolve_capabilities(profile="psychic")

    def test_no_capabilities_or_profile_raises(self):
        with pytest.raises(ValueError):
            resolve_capabilities()


class TestUnknownCapabilityRejection:

    def test_unknown_capability_raises(self):
        with pytest.raises(ValueError, match="Unknown capabilities"):
            resolve_capabilities(capabilities=["bogus"])

    def test_unknown_in_add_raises(self):
        with pytest.raises(ValueError, match="Unknown capabilities"):
            resolve_capabilities(capabilities=["entities"], add=["psychic"])

    def test_unknown_does_not_touch_filesystem(self):
        with pytest.raises(ValueError):
            resolve_capabilities(capabilities=["bogus"])

    def test_multiple_unknown_lists_all(self):
        with pytest.raises(ValueError, match="bogus") as exc_info:
            resolve_capabilities(capabilities=["bogus", "fake", "entities"])
        assert "fake" in str(exc_info.value)


class TestEmptyAndModifierOnlySets:

    def test_empty_after_remove_raises(self):
        with pytest.raises(ValueError, match="empty"):
            resolve_capabilities(capabilities=["entities"], remove=["entities"])

    def test_modifier_only_assertion_signals_raises(self):
        with pytest.raises(ValueError, match="base capability"):
            resolve_capabilities(capabilities=["assertion_signals"])

    def test_modifier_only_evidence_anchoring_raises(self):
        with pytest.raises(ValueError, match="base capability"):
            resolve_capabilities(capabilities=["evidence_anchoring"])

    def test_modifier_with_base_accepted(self):
        result = resolve_capabilities(capabilities=["entities", "assertion_signals"])
        assert "assertion_signals" in result
        assert "entities" in result

    def test_modifier_with_facts_accepted(self):
        result = resolve_capabilities(capabilities=["facts", "evidence_anchoring"])
        assert "evidence_anchoring" in result
        assert "facts" in result


class TestTemplateInjectionPrevention:

    def test_categories_not_double_expanded(self):
        result = build_extraction_prompt(
            "hello",
            capabilities=["entities"],
            categories=["A{{text}}B"],
        )
        assert "A{{text}}B" in result
        assert "AhelloB" not in result

    def test_source_type_not_expanded(self):
        result = build_extraction_prompt(
            "hello",
            capabilities=["entities"],
            source_type="{{date}}",
        )
        assert "{{date}}" in result


class TestDependencyClosure:

    def test_entity_state_adds_entities(self):
        result = resolve_capabilities(capabilities=["entity_state"])
        assert "entities" in result

    def test_entity_context_adds_entities(self):
        result = resolve_capabilities(capabilities=["entity_context"])
        assert "entities" in result

    def test_entity_ids_adds_entities(self):
        result = resolve_capabilities(capabilities=["entity_ids"])
        assert "entities" in result

    def test_goal_timing_adds_goals(self):
        result = resolve_capabilities(capabilities=["goal_timing"])
        assert "goals" in result

    def test_goal_entity_refs_adds_goals_and_entity_ids(self):
        result = resolve_capabilities(capabilities=["goal_entity_refs"])
        assert "goals" in result
        assert "entity_ids" in result
        assert "entities" in result

    def test_temporal_classes_adds_temporal_refs(self):
        result = resolve_capabilities(capabilities=["temporal_classes"])
        assert "temporal_refs" in result

    def test_relations_adds_entities_and_entity_ids(self):
        result = resolve_capabilities(capabilities=["relations"])
        assert "entities" in result
        assert "entity_ids" in result

    def test_relation_origin_adds_relations(self):
        result = resolve_capabilities(capabilities=["relation_origin"])
        assert "relations" in result
        assert "entities" in result
        assert "entity_ids" in result

    def test_transitive_closure(self):
        result = resolve_capabilities(capabilities=["relation_origin"])
        assert "relation_origin" in result
        assert "relations" in result
        assert "entity_ids" in result
        assert "entities" in result


class TestBuildPromptBasics:

    def test_returns_string(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="minimal")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_text(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="minimal")
        assert SAMPLE_TEXT in result

    def test_contains_extracted_at_instruction(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="minimal")
        assert "extracted_at" in result

    def test_contains_json_instruction(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="minimal")
        assert "JSON" in result


class TestBuildPromptCapabilities:

    def test_entities_fragment_present(self):
        result = build_extraction_prompt(SAMPLE_TEXT, capabilities=["entities"])
        assert '"entities"' in result
        assert '"name"' in result
        assert '"type"' in result

    def test_entity_state_fragment_present(self):
        result = build_extraction_prompt(
            SAMPLE_TEXT,
            capabilities=["entities", "entity_state"],
        )
        assert '"state"' in result or "state" in result

    def test_goals_fragment_present(self):
        result = build_extraction_prompt(SAMPLE_TEXT, capabilities=["goals"])
        assert '"goals"' in result
        assert '"status"' in result

    def test_themes_fragment_present(self):
        result = build_extraction_prompt(SAMPLE_TEXT, capabilities=["themes"])
        assert '"themes"' in result

    def test_summary_fragment_present(self):
        result = build_extraction_prompt(SAMPLE_TEXT, capabilities=["summary"])
        assert '"summary"' in result

    def test_sentiment_fragment_present(self):
        result = build_extraction_prompt(SAMPLE_TEXT, capabilities=["sentiment"])
        assert '"sentiment"' in result

    def test_facts_fragment_present(self):
        result = build_extraction_prompt(SAMPLE_TEXT, capabilities=["facts"])
        assert '"facts"' in result

    def test_temporal_refs_fragment_present(self):
        result = build_extraction_prompt(SAMPLE_TEXT, capabilities=["temporal_refs"])
        assert '"temporal_refs"' in result

    def test_relations_fragment_present(self):
        result = build_extraction_prompt(
            SAMPLE_TEXT,
            capabilities=["entities", "entity_ids", "relations"],
        )
        assert '"relations"' in result

    def test_assertion_signals_fragment_present(self):
        result = build_extraction_prompt(
            SAMPLE_TEXT,
            capabilities=["entities", "assertion_signals"],
        )
        assert '"signals"' in result or "signals" in result

    def test_evidence_anchoring_fragment_present(self):
        result = build_extraction_prompt(
            SAMPLE_TEXT,
            capabilities=["entities", "evidence_anchoring"],
        )
        assert '"source"' in result or "source" in result

    def test_absent_capability_not_in_prompt(self):
        result = build_extraction_prompt(SAMPLE_TEXT, capabilities=["entities"])
        assert '"relations"' not in result
        assert '"sentiment"' not in result


class TestBuildPromptOptions:

    def test_categories_included(self):
        result = build_extraction_prompt(
            SAMPLE_TEXT,
            profile="minimal",
            categories=["Health", "Family"],
        )
        assert "Health" in result
        assert "Family" in result

    def test_source_type_included(self):
        result = build_extraction_prompt(
            SAMPLE_TEXT,
            profile="minimal",
            source_type="prayer",
        )
        assert "prayer" in result

    def test_date_included(self):
        result = build_extraction_prompt(
            SAMPLE_TEXT,
            profile="minimal",
            date="2026-04-25",
        )
        assert "2026-04-25" in result

    def test_date_used_for_temporal_resolution(self):
        result = build_extraction_prompt(
            SAMPLE_TEXT,
            capabilities=["temporal_refs"],
            date="2026-04-25",
        )
        assert "2026-04-25" in result


class TestBuildPromptProfiles:

    def test_minimal_does_not_include_relations(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="minimal")
        assert '"relations"' not in result

    def test_standard_includes_facts(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="standard")
        assert '"facts"' in result

    def test_full_includes_relations(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="full")
        assert '"relations"' in result

    def test_profile_with_capabilities_raises(self):
        with pytest.raises(ValueError):
            build_extraction_prompt(
                SAMPLE_TEXT,
                profile="minimal",
                capabilities=["entities"],
            )


class TestBuildPromptCompositionOrder:

    def test_preamble_before_fragments(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="full")
        extract_idx = result.index("Extract structured data")
        entities_idx = result.index('"entities"')
        assert extract_idx < entities_idx

    def test_text_at_end(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="minimal")
        text_idx = result.rindex(SAMPLE_TEXT)
        rules_idx = result.index("Rules:")
        assert rules_idx < text_idx

    def test_primary_before_modifiers(self):
        result = build_extraction_prompt(SAMPLE_TEXT, profile="full")
        entities_idx = result.index('"entities"')
        entity_ids_section = result.find('"id"')
        assert entities_idx < entity_ids_section


class TestPromptFragmentFiles:

    def test_all_fragment_files_exist(self):
        prompts_dir = Path(__file__).resolve().parents[2] / "prompts" / "v1"
        expected = [
            "preamble.txt", "postamble.txt",
            "entities.txt", "entity_state.txt", "entity_context.txt", "entity_ids.txt",
            "goals.txt", "goal_timing.txt", "goal_entity_refs.txt",
            "themes.txt", "keywords.txt", "summary.txt", "sentiment.txt", "structured_sentiment.txt",
            "facts.txt", "questions.txt", "actions.txt", "decisions.txt",
            "temporal_refs.txt", "temporal_classes.txt",
            "relations.txt", "relation_origin.txt",
            "assertion_signals.txt", "evidence_anchoring.txt",
            "language.txt", "source_metadata.txt", "confidence.txt",
        ]
        for name in expected:
            assert (prompts_dir / name).exists(), f"Missing fragment: {name}"

    def test_all_fragment_files_non_empty(self):
        prompts_dir = Path(__file__).resolve().parents[2] / "prompts" / "v1"
        for txt_file in prompts_dir.glob("*.txt"):
            content = txt_file.read_text().strip()
            assert len(content) > 0, f"Empty fragment: {txt_file.name}"


class TestProfileFiles:

    def test_all_profile_files_exist(self):
        profiles_dir = Path(__file__).resolve().parents[2] / "prompts" / "profiles"
        for name in ["minimal.json", "standard.json", "full.json"]:
            assert (profiles_dir / name).exists(), f"Missing profile: {name}"

    def test_profile_files_are_valid_json(self):
        profiles_dir = Path(__file__).resolve().parents[2] / "prompts" / "profiles"
        for name in ["minimal.json", "standard.json", "full.json"]:
            content = (profiles_dir / name).read_text()
            data = json.loads(content)
            assert "capabilities" in data, f"{name} missing capabilities key"
            assert isinstance(data["capabilities"], list)

    def test_minimal_profile_contents(self):
        profiles_dir = Path(__file__).resolve().parents[2] / "prompts" / "profiles"
        data = json.loads((profiles_dir / "minimal.json").read_text())
        caps = set(data["capabilities"])
        assert caps == {"entities", "entity_state", "goals", "themes", "summary"}

    def test_standard_profile_contents(self):
        profiles_dir = Path(__file__).resolve().parents[2] / "prompts" / "profiles"
        data = json.loads((profiles_dir / "standard.json").read_text())
        caps = set(data["capabilities"])
        assert "entities" in caps
        assert "entity_context" in caps
        assert "goal_timing" in caps
        assert "facts" in caps
        assert "temporal_refs" in caps
        assert "sentiment" in caps
        assert "evidence_anchoring" in caps

    def test_full_profile_is_superset_of_standard(self):
        profiles_dir = Path(__file__).resolve().parents[2] / "prompts" / "profiles"
        standard = set(json.loads((profiles_dir / "standard.json").read_text())["capabilities"])
        full = set(json.loads((profiles_dir / "full.json").read_text())["capabilities"])
        assert standard.issubset(full)

    def test_full_profile_includes_all_capabilities(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        profiles_dir = Path(__file__).resolve().parents[2] / "prompts" / "profiles"
        data = json.loads((profiles_dir / "full.json").read_text())
        caps = set(data["capabilities"])
        assert caps == EXTRACTION_CAPABILITIES


class TestRegistryConsistency:

    def test_every_capability_has_fragment_file(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        prompts_dir = Path(__file__).resolve().parents[2] / "prompts" / "v1"
        for cap in EXTRACTION_CAPABILITIES:
            assert (prompts_dir / f"{cap}.txt").exists(), f"missing fragment for {cap}"

    def test_every_fragment_is_valid_capability(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        prompts_dir = Path(__file__).resolve().parents[2] / "prompts" / "v1"
        for txt_file in prompts_dir.glob("*.txt"):
            name = txt_file.stem
            if name in ("preamble", "postamble"):
                continue
            assert name in EXTRACTION_CAPABILITIES, f"orphan fragment: {name}"

    def test_canonical_order_covers_all_capabilities(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        from synapt_extract.prompt import CANONICAL_ORDER
        assert set(CANONICAL_ORDER) == EXTRACTION_CAPABILITIES

    def test_canonical_order_has_no_duplicates(self):
        from synapt_extract.prompt import CANONICAL_ORDER
        assert len(CANONICAL_ORDER) == len(set(CANONICAL_ORDER))

    def test_capability_deps_reference_valid_capabilities(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        from synapt_extract.prompt import CAPABILITY_DEPS
        for cap, deps in CAPABILITY_DEPS.items():
            assert cap in EXTRACTION_CAPABILITIES, f"dep key {cap} not a valid capability"
            for dep in deps:
                assert dep in EXTRACTION_CAPABILITIES, f"dep {dep} (from {cap}) not valid"

    def test_capability_rules_reference_valid_capabilities(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        from synapt_extract.prompt import CAPABILITY_RULES
        for cap in CAPABILITY_RULES:
            assert cap in EXTRACTION_CAPABILITIES, f"rule key {cap} not a valid capability"

    def test_full_profile_has_no_duplicates(self):
        profiles_dir = Path(__file__).resolve().parents[2] / "prompts" / "profiles"
        data = json.loads((profiles_dir / "full.json").read_text())
        caps = data["capabilities"]
        assert len(caps) == len(set(caps)), "full profile has duplicate capabilities"

    def test_build_prompt_succeeds_for_every_capability(self):
        from synapt_extract.schema import EXTRACTION_CAPABILITIES
        modifier_only = {"assertion_signals", "evidence_anchoring"}
        for cap in EXTRACTION_CAPABILITIES:
            caps = ["entities", cap] if cap in modifier_only else [cap]
            result = build_extraction_prompt("test", capabilities=caps)
            assert len(result) > 0, f"empty prompt for capability {cap}"

    def test_profiles_are_strict_subsets(self):
        profiles_dir = Path(__file__).resolve().parents[2] / "prompts" / "profiles"
        minimal = set(json.loads((profiles_dir / "minimal.json").read_text())["capabilities"])
        standard = set(json.loads((profiles_dir / "standard.json").read_text())["capabilities"])
        full = set(json.loads((profiles_dir / "full.json").read_text())["capabilities"])
        assert minimal.issubset(full), "minimal is not a subset of full"
        assert standard.issubset(full), "standard is not a subset of full"


class TestBuildExtractionSchema:

    def test_includes_only_fields_for_resolved_capabilities(self):
        schema = build_extraction_schema(capabilities=["entities"])
        properties = schema["properties"]
        assert "extracted_at" in properties
        assert "entities" in properties
        assert "goals" not in properties
        assert "temporal_refs" not in properties
        assert schema["required"] == ["extracted_at", "entities"]

    def test_applies_dependency_closure_to_schema(self):
        schema = build_extraction_schema(capabilities=["goal_entity_refs"])
        properties = schema["properties"]
        entities = properties["entities"]["items"]
        goals = properties["goals"]["items"]

        assert "entities" in properties
        assert "goals" in properties
        assert "id" in entities["properties"]
        assert "id" in entities["required"]
        assert "entity_refs" in goals["properties"]
        assert "entity_refs" in goals["required"]

    def test_adds_modifier_fields_only_when_requested(self):
        base = build_extraction_schema(capabilities=["entities"])
        enriched = build_extraction_schema(
            capabilities=["entities", "entity_context", "evidence_anchoring"]
        )
        base_entity = base["properties"]["entities"]["items"]["properties"]
        enriched_entity = enriched["properties"]["entities"]["items"]["properties"]

        assert "context" not in base_entity
        assert "source" not in base_entity
        assert "context" in enriched_entity
        assert "date_hint" in enriched_entity
        assert "source" in enriched_entity

    def test_includes_full_source_ref_fields_when_evidence_anchoring_requested(self):
        schema = build_extraction_schema(capabilities=["entities", "evidence_anchoring"])
        source = schema["properties"]["entities"]["items"]["properties"]["source"]

        assert source["required"] == ["snippet"]
        assert "snippet" in source["properties"]
        assert "offset_start" in source["properties"]
        assert "offset_end" in source["properties"]
        assert "sentence_index" in source["properties"]
        assert "version" not in source["properties"]

    def test_builds_finalized_schema_with_full_packet_fields(self):
        schema = build_finalized_extraction_schema(
            capabilities=["entities", "evidence_anchoring", "assertion_signals", "temporal_refs"]
        )
        properties = schema["properties"]
        entity = properties["entities"]["items"]
        source = entity["properties"]["source"]
        signals = entity["properties"]["signals"]
        temporal_ref = properties["temporal_refs"]["items"]

        assert schema["required"] == [
            "version",
            "extracted_at",
            "produced_by",
            "entities",
            "goals",
            "themes",
            "capabilities",
        ]
        assert "source_id" in properties
        assert "source_type" in properties
        assert "user_id" in properties
        assert "kind" in properties
        assert "embeddings" in properties
        assert "extensions" in properties
        assert "keywords" in properties
        assert "questions" in properties
        assert "actions" in properties
        assert "decisions" in properties
        assert "language" in properties
        assert "source_metadata" in properties
        assert "confidence" in properties
        assert source["required"] == ["version"]
        assert source["properties"]["version"] == {"const": "1"}
        assert "offset_start" in source["properties"]
        assert signals["required"] == ["version"]
        assert "confidence" in signals["properties"]
        assert temporal_ref["required"] == ["version", "raw"]

    def test_schema_covers_all_v12_capability_fields(self):
        schema = build_extraction_schema(profile="full")
        properties = schema["properties"]
        entity = properties["entities"]["items"]["properties"]
        question = properties["questions"]["items"]["properties"]
        action = properties["actions"]["items"]
        decision = properties["decisions"]["items"]["properties"]
        sentiment = properties["sentiment"]
        source_metadata = properties["source_metadata"]

        for field in ["keywords", "questions", "actions", "decisions", "language", "source_metadata", "confidence"]:
            assert field in properties
        assert "aliases" in entity
        assert "directed_to" in question
        assert action["required"] == ["text", "origin"]
        assert "entity_refs" in action["properties"]
        assert "due" in action["properties"]
        assert "entity_refs" in decision
        assert "decided_at" in decision
        assert sentiment["required"] == ["valence"]
        assert "version" not in sentiment["properties"]
        assert source_metadata["required"] == []
        assert "version" not in source_metadata["properties"]

    def test_wraps_schema_in_response_format(self):
        response_format = build_extraction_response_format(
            capabilities=["entities"],
            name="custom_stage1",
        )
        assert response_format["type"] == "json_schema"
        assert response_format["name"] == "custom_stage1"
        assert response_format["strict"] is True
        assert response_format["schema"] != build_extraction_schema(capabilities=["entities"])

    def test_strict_response_format_requires_every_object_property_for_openai(self):
        response_format = build_extraction_response_format(
            capabilities=["entities", "entity_context"],
        )
        entity = response_format["schema"]["properties"]["entities"]["items"]
        assert entity["required"] == ["name", "type", "aliases", "context", "date_hint"]

    def test_strict_response_format_makes_semantic_optional_fields_nullable(self):
        response_format = build_extraction_response_format(
            capabilities=["entities", "evidence_anchoring"],
        )
        entity = response_format["schema"]["properties"]["entities"]["items"]
        source = entity["properties"]["source"]

        assert source["type"] == ["object", "null"]
        assert source["required"] == ["snippet", "offset_start", "offset_end", "sentence_index"]
        assert source["properties"]["offset_start"]["type"] == ["integer", "null"]

    def test_non_strict_response_format_preserves_semantic_required_fields(self):
        response_format = build_extraction_response_format(
            capabilities=["entities", "entity_context"],
            strict=False,
        )
        assert response_format["schema"] == build_extraction_schema(
            capabilities=["entities", "entity_context"]
        )


class TestExtractionBuilder:

    def test_builds_prompt_and_schema_from_same_capabilities(self):
        builder = (
            create_extraction_builder(SAMPLE_TEXT, profile="minimal")
            .add_capabilities(["goal_entity_refs"])
            .with_extracted_at("2026-05-11T12:00:00Z")
        )
        result = builder.build(name="synapt_test")
        properties = result["schema"]["properties"]

        assert "goal_entity_refs" in result["capabilities"]
        assert "Additional run constraints" in result["prompt"]
        assert "Use exactly this extracted_at value: 2026-05-11T12:00:00Z." in result["prompt"]
        assert "Produce Stage 1 content only." in result["prompt"]
        assert "Every goal.entity_refs entry must refer to one of the entity IDs you emit." in result["prompt"]
        assert "Omit temporal_refs for this run." in result["prompt"]
        assert "entities" in properties
        assert "goals" in properties
        assert result["response_format"]["name"] == "synapt_test"

    def test_supports_fluent_construction(self):
        builder = (
            ExtractionBuilder()
            .with_text(SAMPLE_TEXT)
            .with_capabilities(["entities"])
            .with_source_type("prayer")
            .with_date("2026-05-11")
        )

        assert builder.resolved_capabilities() == ["entities"]
        assert "prayer" in builder.prompt()
        assert builder.schema()["required"] == ["extracted_at", "entities"]

    def test_carries_finalization_context_and_can_finalize_without_live_model_call(self):
        builder = (
            ExtractionBuilder(
                SAMPLE_TEXT,
                capabilities=["entities", "goals", "themes", "evidence_anchoring"],
            )
            .with_extracted_at("2026-05-11T12:00:00Z")
            .with_produced_by(
                {
                    "model": "openai://gpt-5.5",
                    "model_version": "gpt-5.5-2026-04-23",
                    "configuration": {"reasoning_effort": "medium"},
                    "operator": "synapt-dev",
                }
            )
            .with_source(source_id="fixture-1", source_type="note")
            .with_user_id("user-1")
            .with_kind("synapt/test")
            .with_extensions({"synapt/source_binding": {"source_sha256": "abc"}})
            .with_embeddings(
                [
                    {
                        "vector": [0.1, 0.2],
                        "model": "openai://text-embedding-3-small",
                        "input": "source",
                        "dimensions": 2,
                        "computed_at": "2026-05-11T12:00:01Z",
                    }
                ]
            )
        )

        built = builder.build()
        assert built["finalize_context"]["source_id"] == "fixture-1"
        assert "evidence_anchoring" in built["finalize_context"]["capabilities_hint"]
        assert "produced_by" in built["finalized_schema"]["properties"]

        result = builder.finalize(
            {
                "extracted_at": "2026-05-11T12:00:00Z",
                "entities": [{"name": "Mom", "type": "person", "source": {"snippet": "mom", "offset_start": None}}],
                "goals": [],
                "themes": [],
            }
        )

        assert result.validation.valid
        assert result.extraction["source_id"] == "fixture-1"
        assert result.extraction["produced_by"]["version"] == "1"
        assert result.extraction["produced_by"]["model"] == "openai://gpt-5.5"
        assert result.extraction["embeddings"][0]["version"] == "1"
        assert result.extraction["embeddings"][0]["dimensions"] == 2
        assert result.extraction["entities"][0]["source"] == {"version": "1", "snippet": "mom"}
