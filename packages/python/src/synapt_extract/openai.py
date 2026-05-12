"""OpenAI-compatible convenience adapter for caller-owned clients."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from synapt_extract.artifacts import create_artifact_bundle, write_artifact_bundle
from synapt_extract.extract import (
    EmbeddingRequest,
    EmbeddingResponse,
    ExtractResult,
    LlmRequest,
    LlmResponse,
    extract,
)


JsonObject = dict[str, Any]


@dataclass
class OpenAIExtractResult(ExtractResult):
    artifact_bundle: JsonObject = field(default_factory=dict)


async def extract_openai(
    text: str,
    client: Any,
    *,
    model: str = "gpt-5.5",
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    text_verbosity: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    embedding_space: str = "cosine",
    deployment: str | None = None,
    operator: str | None = None,
    artifact_dir: str | Path | None = None,
    include_artifact_source_text: bool = True,
    **options: Any,
) -> OpenAIExtractResult:
    prompt_text: str | None = None
    response_format: JsonObject | None = None
    provider_llm_request: JsonObject | None = None
    provider_llm_response: JsonObject | None = None
    embedding_runs: list[JsonObject] = []

    async def call_llm(request: LlmRequest) -> LlmResponse:
        nonlocal prompt_text, provider_llm_request, provider_llm_response, response_format
        prompt_text = request["prompt"]
        response_format = request["response_format"]
        prompt_hash = sha256_text(request["prompt"])
        output_limit = max_output_tokens if max_output_tokens is not None else request.get("max_tokens")
        request_body = prune_none({
            "model": model,
            "reasoning": {"effort": reasoning_effort} if reasoning_effort else None,
            "max_output_tokens": output_limit,
            "temperature": temperature if temperature is not None else request.get("temperature"),
            "top_p": top_p,
            "input": [{"role": "developer", "content": request["prompt"]}],
            "text": prune_none({
                "format": request["response_format"],
                "verbosity": text_verbosity,
            }),
        })
        provider_llm_request = request_body

        response = await maybe_await(client.responses.create(**request_body))
        raw = to_json_object(response)
        provider_llm_response = raw
        content = output_text(response, raw)
        if not content.strip():
            raise ValueError("OpenAI response did not include output text")

        configuration = prune_none({
            "reasoning_effort": reasoning_effort,
            "system_prompt_hash": prompt_hash,
            "temperature": temperature if temperature is not None else request.get("temperature"),
            "top_p": top_p,
            "max_tokens": output_limit,
            "response_format": optional_str(response_format.get("name")),
        })
        llm_response: LlmResponse = {
            "content": content,
            "provider": "openai",
            "id": optional_str(raw.get("id")),
            "status": optional_str(raw.get("status")),
            "model": optional_str(raw.get("model")) or model,
            "produced_by": {
                "model": f"openai://{model}",
                "model_version": optional_str(raw.get("model")) or model,
                **({"deployment": deployment} if deployment else {}),
                "configuration": configuration,
                **({"operator": operator} if operator else {}),
            },
            "usage": usage_summary(raw.get("usage")),
            "raw": raw,
        }
        return llm_response

    callbacks: JsonObject = {"call_llm": call_llm}

    if embedding_model is not None:
        async def get_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
            embedding_request = prune_none({
                "model": embedding_model,
                "input": request["text"],
                "dimensions": embedding_dimensions,
            })
            embeddings_client = getattr(client, "embeddings", None)
            create = getattr(embeddings_client, "create", None)
            if create is None:
                raise ValueError("embedding_model was set but client.embeddings.create is unavailable")
            response = await maybe_await(create(**embedding_request))
            raw = to_json_object(response)
            vector = embedding_vector(raw)
            embedding_response: EmbeddingResponse = {
                "vector": vector,
                "model": f"openai://{optional_str(raw.get('model')) or embedding_model}",
                "dimensions": len(vector),
                "space": embedding_space,
                "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "raw": raw,
            }
            embedding_runs.append({
                "input": request["input"],
                "request": embedding_request,
                "response": raw,
            })
            return embedding_response

        callbacks["get_embedding"] = get_embedding

    runner_options = dict(options)
    if max_output_tokens is not None and "max_tokens" not in runner_options:
        runner_options["max_tokens"] = max_output_tokens

    result = await extract(text, callbacks, **runner_options)
    artifact_bundle = create_artifact_bundle(
        source_text=text,
        result=result,
        prompt=prompt_text,
        response_format=response_format,
        llm_request=provider_llm_request,
        llm_response=provider_llm_response,
        embedding_runs=embedding_runs,
        include_source_text=include_artifact_source_text,
    )
    if artifact_dir is not None:
        write_artifact_bundle(artifact_dir, artifact_bundle)

    return OpenAIExtractResult(
        extraction=result.extraction,
        validation=result.validation,
        warnings=result.warnings,
        stage1=result.stage1,
        embeddings=result.embeddings,
        usage=result.usage,
        artifact_bundle=artifact_bundle,
    )


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def prune_none(value: JsonObject) -> JsonObject:
    return {key: child for key, child in value.items() if child is not None}


def optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def to_json_object(value: Any) -> JsonObject:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return value
    try:
        dumped = json.loads(json.dumps(value, default=str))
    except TypeError:
        return {}
    return dumped if isinstance(dumped, dict) else {}


def output_text(response: Any, raw: JsonObject) -> str:
    raw_output = raw.get("output_text")
    if isinstance(raw_output, str):
        return raw_output
    attr_output = getattr(response, "output_text", None)
    if isinstance(attr_output, str):
        return attr_output
    output = raw.get("output")
    if not isinstance(output, list):
        return ""
    parts = []
    for item in output:
        if not isinstance(item, dict) or not isinstance(item.get("content"), list):
            continue
        for content in item["content"]:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts)


def usage_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    usage = dict(value)
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    if isinstance(input_tokens, int):
        usage["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        usage["output_tokens"] = output_tokens
    if isinstance(value.get("total_tokens"), int):
        usage["total_tokens"] = value["total_tokens"]
    elif isinstance(input_tokens, int) and isinstance(output_tokens, int):
        usage["total_tokens"] = input_tokens + output_tokens
    return usage


def embedding_vector(raw: JsonObject) -> list[float]:
    data = raw.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError("OpenAI embedding response did not include data[0].embedding")
    vector = data[0].get("embedding")
    if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
        raise ValueError("OpenAI embedding vector must contain only numbers")
    return [float(value) for value in vector]
