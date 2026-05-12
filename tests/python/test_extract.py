"""Tests for provider-hooked extraction and embeddings pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "python" / "src"))

from synapt_extract import create_extraction_builder, extract


SAMPLE_TEXT = (
    "On May 10, 2026, Layne told Mark that Synapt should ship the extraction builder by Friday. "
    "Mark asked whether embeddings should cover the source and summary. "
    "Layne said the first version might use local entity IDs, and if validation passes, they will publish the gist."
)

STAGE1_FULL = {
    "extracted_at": "2026-05-12T14:00:00Z",
    "entities": [
        {
            "id": "e1",
            "name": "Layne",
            "type": "person",
            "aliases": ["Layne Penney"],
            "state": "coordinating Synapt extraction work",
            "context": "Asked Mark to review builder and embedding behavior.",
            "date_hint": "2026-05-10",
            "relations": [
                {
                    "target": "e2",
                    "type": "collaborates_with",
                    "origin": "explicit",
                    "signals": {"confidence": 0.91},
                }
            ],
            "source": {"snippet": "Layne told Mark", "sentence_index": 0},
            "signals": {"confidence": 0.93},
        },
        {
            "id": "e2",
            "name": "Mark",
            "type": "person",
            "aliases": ["Mark Hendrickson"],
            "state": "reviewing Synapt extraction ideas",
            "context": "Asked about embedding coverage.",
            "date_hint": "2026-05-10",
            "relations": [],
            "source": {"snippet": "Mark asked", "sentence_index": 1},
            "signals": {"confidence": 0.9},
        },
    ],
    "goals": [
        {
            "text": "Ship the extraction builder by Friday.",
            "status": "open",
            "entity_refs": ["e1", "e2"],
            "stated_at": "2026-05-10T00:00:00Z",
            "source": {"snippet": "ship the extraction builder by Friday", "sentence_index": 0},
            "signals": {"confidence": 0.88},
        }
    ],
    "themes": ["extraction pipeline", "embeddings", "schema validation"],
    "keywords": ["Synapt", "extraction builder", "embeddings", "gist"],
    "summary": "Layne and Mark discussed shipping a Synapt extraction builder with embedding coverage and validation.",
    "sentiment": {"valence": "positive", "intensity": 0.55, "confidence": 0.72},
    "facts": [
        {
            "text": "Mark asked whether embeddings should cover the source and summary.",
            "category": "technical_question",
            "source": {"snippet": "embeddings should cover the source and summary", "sentence_index": 1},
            "signals": {"confidence": 0.95},
        }
    ],
    "questions": [
        {
            "text": "Should embeddings cover the source and summary?",
            "directed_to": "Layne",
            "source": {"snippet": "whether embeddings should cover the source and summary", "sentence_index": 1},
            "signals": {"confidence": 0.95},
        }
    ],
    "actions": [
        {
            "text": "Publish the gist if validation passes.",
            "origin": "extracted",
            "entity_refs": ["e1"],
            "due": "2026-05-15T00:00:00Z",
            "source": {"snippet": "if validation passes, they will publish the gist", "sentence_index": 2},
            "signals": {"confidence": 0.82, "condition": "validation passes"},
        }
    ],
    "decisions": [
        {
            "text": "The first version may use local entity IDs.",
            "entity_refs": ["e1"],
            "decided_at": "2026-05-10T00:00:00Z",
            "source": {"snippet": "first version might use local entity IDs", "sentence_index": 2},
            "signals": {"confidence": 0.7, "hedged": True},
        }
    ],
    "temporal_refs": [
        {
            "raw": "Friday",
            "type": "point",
            "resolved": "2026-05-15T00:00:00Z",
            "context": "ship the extraction builder by Friday",
        }
    ],
    "language": "en-US",
    "source_metadata": {
        "token_count": 47,
        "character_count": len(SAMPLE_TEXT),
        "modality": "text",
        "format": "plain",
    },
    "confidence": 0.86,
}


def test_plans_fluent_full_minus_embed_capability_ux():
    plan = (
        create_extraction_builder(SAMPLE_TEXT)
        .full(embed=True)
        .minus("questions")
        .embed("summary", False)
        .plan()
    )

    assert "entities" in plan["capabilities"]
    assert "questions" not in plan["capabilities"]
    assert "questions" in plan["excluded"]
    assert "entities" in plan["embedded_inputs"]
    assert "questions" not in plan["embedded_inputs"]
    assert "summary" not in plan["embedded_inputs"]
    assert plan["required_callbacks"] == {"call_llm": True, "get_embedding": True}
    assert plan["prompt_characters"] > 0


def test_extract_runs_full_profile_llm_and_embedding_callbacks():
    llm_requests = []
    embedding_requests = []

    async def call_llm(request):
        llm_requests.append(request)
        return {
            "output": STAGE1_FULL,
            "produced_by": {
                "model": "openai://gpt-5.5",
                "model_version": "gpt-5.5-2026-04-23",
                "configuration": {"reasoning_effort": "medium"},
                "operator": "test",
            },
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }

    async def get_embedding(request):
        embedding_requests.append(request)
        return {
            "vector": [len(request["text"]) / 1000, len(embedding_requests)],
            "model": "openai://text-embedding-3-small",
            "space": "cosine",
            "computed_at": "2026-05-12T14:00:01Z",
        }

    result = asyncio.run(extract(
        SAMPLE_TEXT,
        {"call_llm": call_llm, "get_embedding": get_embedding},
        profile="full",
        source_id="fixture-full-1",
        source_type="message",
        user_id="user-1",
        kind="synapt/test",
        embedding_inputs="all",
    ))

    assert len(llm_requests) == 1
    assert llm_requests[0]["response_format"]["type"] == "json_schema"
    assert "relation_origin" in llm_requests[0]["capabilities"]
    assert [request["input"] for request in embedding_requests] == [
        "source",
        "summary",
        "entities",
        "goals",
        "themes",
        "keywords",
        "facts",
        "questions",
        "actions",
        "decisions",
        "temporal_refs",
        "sentiment",
    ]
    assert result.validation.valid
    assert len(result.extraction["embeddings"]) == 12
    assert result.extraction["embeddings"][0]["version"] == "1"
    assert result.extraction["embeddings"][0]["input"] == "source"
    assert result.extraction["embeddings"][0]["model"] == "openai://text-embedding-3-small"
    assert result.extraction["embeddings"][0]["dimensions"] == 2
    assert set(llm_requests[0]["capabilities"]).issubset(set(result.extraction["capabilities"]))
    assert result.extraction["produced_by"]["version"] == "1"
    assert result.extraction["produced_by"]["model"] == "openai://gpt-5.5"
    assert result.usage.llm_calls == 1
    assert result.usage.embedding_calls == 12
    assert result.usage.total_tokens == 150
    assert result.warnings == []


def test_extract_requires_embedding_callback_when_embedding_inputs_are_requested():
    with pytest.raises(ValueError, match="get_embedding"):
        asyncio.run(extract(
            SAMPLE_TEXT,
            {"call_llm": lambda _request: {"output": dict(STAGE1_FULL)}},
            profile="full",
            produced_by="openai://gpt-5.5",
            embedding_inputs=["source"],
        ))


def test_extract_derives_embedding_inputs_from_capability_specs():
    embedding_inputs = []

    def get_embedding(request):
        embedding_inputs.append(request["input"])
        return {
            "vector": [0.1, 0.2],
            "model": "openai://text-embedding-3-small",
        }

    result = asyncio.run(extract(
        SAMPLE_TEXT,
        {
            "call_llm": lambda _request: {
                "output": dict(STAGE1_FULL),
                "produced_by": "openai://gpt-5.5",
            },
            "get_embedding": get_embedding,
        },
        capabilities=[
            {"name": "entity_context", "embed": True},
            {"name": "summary", "embed": True},
            "goals",
        ],
        embedding_inputs=["source"],
    ))

    assert embedding_inputs == ["source", "entities", "summary"]
    assert [embedding["input"] for embedding in result.extraction["embeddings"]] == ["source", "entities", "summary"]
    assert result.validation.valid


def test_extract_dynamic_extensions_use_normalized_response_context():
    def extend(context):
        response = context["response"]
        return {
            "synapt/response_binding": {
                "response_id": response["id"],
                "response_status": response["status"],
                "response_model": response["model"],
                "stage1_fields": len(context["stage1"]),
                "embedding_count": len(context["embeddings"]),
            }
        }

    result = asyncio.run(extract(
        SAMPLE_TEXT,
        {
            "call_llm": lambda _request: {
                "output": dict(STAGE1_FULL),
                "produced_by": "openai://gpt-5.5",
                "id": "resp_test_123",
                "status": "completed",
                "model": "gpt-5.5-2026-04-23",
            },
        },
        profile="full",
        extend=extend,
    ))

    assert result.validation.valid
    assert result.extraction["extensions"]["synapt/response_binding"]["version"] == "1"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_id"] == "resp_test_123"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_status"] == "completed"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_model"] == "gpt-5.5-2026-04-23"
    assert result.extraction["extensions"]["synapt/response_binding"]["embedding_count"] == 0
