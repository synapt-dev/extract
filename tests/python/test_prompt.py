"""Tests for SynaptExtraction IL v1 composable prompt system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "python" / "src"))

from synapt_extract.prompt import build_extraction_prompt, resolve_capabilities


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
            "themes.txt", "summary.txt", "sentiment.txt", "facts.txt",
            "temporal_refs.txt", "temporal_classes.txt",
            "relations.txt", "relation_origin.txt",
            "assertion_signals.txt", "evidence_anchoring.txt",
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
