"""Tests for provider-hooked extraction and embeddings pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "python" / "src"))

from synapt_extract import create_extraction_builder, extract, extract_openai


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
    assert plan["embedded_inputs"][0] == "source"
    assert "entities" in plan["embedded_inputs"]
    assert "questions" not in plan["embedded_inputs"]
    assert "summary" not in plan["embedded_inputs"]
    assert plan["required_callbacks"] == {"call_llm": True, "get_embedding": True}
    assert plan["prompt_characters"] > 0

    assert create_extraction_builder(SAMPLE_TEXT).full(embed=True).extract_options()["embedding_inputs"] == [
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

    builder = (
        create_extraction_builder(SAMPLE_TEXT)
        .full(embed=True)
        .with_source(source_id="fixture-full-1", source_type="message")
        .with_user_id("user-1")
        .with_kind("synapt/test")
    )

    result = asyncio.run(extract(
        SAMPLE_TEXT,
        {"call_llm": call_llm, "get_embedding": get_embedding},
        **builder.extract_options(),
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


def test_extract_translates_openai_raw_response_context():
    def extend(context):
        response = context["response"]
        return {
            "synapt/response_binding": {
                "provider": response["provider"],
                "response_id": response["id"],
                "response_status": response["status"],
                "response_model": response["model"],
            }
        }

    result = asyncio.run(extract(
        SAMPLE_TEXT,
        {
            "call_llm": lambda _request: {
                "output": dict(STAGE1_FULL),
                "raw": {
                    "object": "response",
                    "id": "resp_raw_123",
                    "status": "completed",
                    "model": "gpt-5.5-2026-04-23",
                    "usage": {"input_tokens": 42, "output_tokens": 18},
                },
            },
        },
        profile="full",
        extend=extend,
    ))

    assert result.validation.valid
    assert result.extraction["produced_by"]["version"] == "1"
    assert result.extraction["produced_by"]["model"] == "openai://gpt-5.5-2026-04-23"
    assert result.extraction["produced_by"]["model_version"] == "gpt-5.5-2026-04-23"
    assert result.usage.input_tokens == 42
    assert result.usage.output_tokens == 18
    assert result.usage.total_tokens == 60
    assert result.extraction["extensions"]["synapt/response_binding"]["provider"] == "openai"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_id"] == "resp_raw_123"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_status"] == "completed"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_model"] == "gpt-5.5-2026-04-23"


def test_extract_translates_anthropic_raw_response_context():
    def extend(context):
        response = context["response"]
        return {
            "synapt/response_binding": {
                "provider": response["provider"],
                "response_id": response["id"],
                "response_status": response["status"],
                "response_model": response["model"],
                "stop_reason": response["stop_reason"],
            }
        }

    result = asyncio.run(extract(
        SAMPLE_TEXT,
        {
            "call_llm": lambda _request: {
                "output": dict(STAGE1_FULL),
                "raw": {
                    "type": "message",
                    "id": "msg_raw_123",
                    "model": "claude-sonnet-4-20250514",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 25, "output_tokens": 15},
                    "content": [{"type": "text", "text": "{}"}],
                },
            },
        },
        profile="full",
        extend=extend,
    ))

    assert result.validation.valid
    assert result.extraction["produced_by"]["version"] == "1"
    assert result.extraction["produced_by"]["model"] == "anthropic://claude-sonnet-4-20250514"
    assert result.extraction["produced_by"]["model_version"] == "claude-sonnet-4-20250514"
    assert result.usage.input_tokens == 25
    assert result.usage.output_tokens == 15
    assert result.usage.total_tokens == 40
    assert result.extraction["extensions"]["synapt/response_binding"]["provider"] == "anthropic"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_id"] == "msg_raw_123"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_status"] == "completed"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_model"] == "claude-sonnet-4-20250514"
    assert result.extraction["extensions"]["synapt/response_binding"]["stop_reason"] == "end_turn"


def test_extract_uses_custom_response_translator():
    def response_translator(context):
        raw = context["raw"]
        return {
            "provider": "local",
            "id": raw["request_id"],
            "status": "ok",
            "model": raw["engine"],
            "usage": raw["tokens"],
        }

    def extend(context):
        response = context["response"]
        return {
            "synapt/response_binding": {
                "provider": response["provider"],
                "response_id": response["id"],
                "response_status": response["status"],
                "response_model": response["model"],
            }
        }

    result = asyncio.run(extract(
        SAMPLE_TEXT,
        {
            "call_llm": lambda _request: {
                "output": dict(STAGE1_FULL),
                "raw": {
                    "request_id": "local_raw_123",
                    "engine": "fixture-engine",
                    "tokens": {"input_tokens": 7, "output_tokens": 5},
                },
            },
        },
        profile="full",
        response_translator=response_translator,
        extend=extend,
    ))

    assert result.validation.valid
    assert result.extraction["produced_by"]["version"] == "1"
    assert result.extraction["produced_by"]["model"] == "local://fixture-engine"
    assert result.extraction["produced_by"]["model_version"] == "fixture-engine"
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 12
    assert result.extraction["extensions"]["synapt/response_binding"]["provider"] == "local"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_id"] == "local_raw_123"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_status"] == "ok"
    assert result.extraction["extensions"]["synapt/response_binding"]["response_model"] == "fixture-engine"


def test_extract_openai_adapter_writes_artifacts(tmp_path):
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload
            self.output_text = payload.get("output_text")

        def model_dump(self, mode="json"):
            return self.payload

    class FakeResponses:
        def __init__(self):
            self.requests = []

        def create(self, **body):
            self.requests.append(body)
            return FakeResponse({
                "object": "response",
                "id": "resp_adapter_123",
                "status": "completed",
                "model": "gpt-5.5-2026-04-23",
                "output_text": json_dumps(STAGE1_FULL),
                "usage": {"input_tokens": 101, "output_tokens": 55},
            })

    class FakeEmbeddings:
        def __init__(self):
            self.requests = []

        def create(self, **body):
            self.requests.append(body)
            return FakeResponse({
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
            })

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()
            self.embeddings = FakeEmbeddings()

    client = FakeClient()
    result = asyncio.run(extract_openai(
        SAMPLE_TEXT,
        client,
        profile="full",
        source_id="fixture-openai-adapter",
        source_type="message",
        kind="synapt/test",
        model="gpt-5.5",
        reasoning_effort="medium",
        max_output_tokens=2048,
        text_verbosity="low",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=3,
        embedding_inputs=["source"],
        deployment="test-suite",
        operator="synapt-dev",
        response_format_name="synapt_adapter_stage1",
        artifact_dir=tmp_path,
    ))

    assert len(client.responses.requests) == 1
    assert client.responses.requests[0]["model"] == "gpt-5.5"
    assert client.responses.requests[0]["reasoning"] == {"effort": "medium"}
    assert client.responses.requests[0]["max_output_tokens"] == 2048
    assert client.responses.requests[0]["text"]["verbosity"] == "low"
    assert client.responses.requests[0]["text"]["format"]["name"] == "synapt_adapter_stage1"
    assert client.embeddings.requests == [
        {"model": "text-embedding-3-small", "input": SAMPLE_TEXT, "dimensions": 3}
    ]
    assert result.validation.valid
    assert result.extraction["produced_by"]["model"] == "openai://gpt-5.5"
    assert result.extraction["produced_by"]["model_version"] == "gpt-5.5-2026-04-23"
    assert result.extraction["produced_by"]["deployment"] == "test-suite"
    assert result.extraction["produced_by"]["operator"] == "synapt-dev"
    assert result.extraction["produced_by"]["configuration"]["reasoning_effort"] == "medium"
    assert result.extraction["produced_by"]["configuration"]["max_tokens"] == 2048
    assert result.extraction["produced_by"]["configuration"]["response_format"] == "synapt_adapter_stage1"
    assert result.extraction["embeddings"][0]["input"] == "source"
    assert result.extraction["embeddings"][0]["model"] == "openai://text-embedding-3-small"
    assert result.extraction["embeddings"][0]["dimensions"] == 3
    assert result.usage.llm_calls == 1
    assert result.usage.embedding_calls == 1
    assert result.usage.total_tokens == 156
    assert result.artifact_bundle["source"]["text"] == SAMPLE_TEXT
    assert result.artifact_bundle["llm"]["request"]["model"] == "gpt-5.5"
    assert result.artifact_bundle["llm"]["request"]["max_output_tokens"] == 2048
    assert result.artifact_bundle["llm"]["response"]["id"] == "resp_adapter_123"
    assert result.artifact_bundle["llm"]["response"]["model"] == "gpt-5.5-2026-04-23"
    assert (tmp_path / "source.txt").exists()
    assert (tmp_path / "prompt.md").exists()
    assert (tmp_path / "extraction.json").exists()


def json_dumps(value):
    import json

    return json.dumps(value, sort_keys=True)
