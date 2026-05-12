"""Provider-hooked extraction pipeline with optional embeddings."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol, TypedDict

from synapt_extract.builder import DEFAULT_RESPONSE_FORMAT_NAME, ExtractionBuilder
from synapt_extract.finalize import FinalizeContext
from synapt_extract.prompt import capability_embedding_preference, capability_name
from synapt_extract.validate import ValidationResult


JsonObject = dict[str, Any]
MaybeAwaitable = Awaitable[Any] | Any


class LlmMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


class LlmUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class LlmRequest(TypedDict, total=False):
    prompt: str
    messages: list[LlmMessage]
    capabilities: list[str]
    schema: JsonObject
    response_format: JsonObject
    temperature: float
    max_tokens: int


class LlmResponse(TypedDict, total=False):
    content: str
    json: JsonObject
    output: JsonObject
    produced_by: str | JsonObject
    id: str
    response_id: str
    status: str
    model: str
    model_version: str
    usage: LlmUsage
    raw: Any


class NormalizedLlmResponse(TypedDict, total=False):
    id: str
    status: str
    model: str
    model_version: str
    produced_by: str | JsonObject
    content: str
    usage: LlmUsage
    raw: Any


NamedEmbeddingInput = Literal[
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


class CustomEmbeddingInput(TypedDict):
    input: str
    text: str


EmbeddingInputSelector = NamedEmbeddingInput | CustomEmbeddingInput
EmbeddingInputSelection = Literal["all"] | list[EmbeddingInputSelector]


class EmbeddingRequest(TypedDict):
    text: str
    input: str


class EmbeddingResponse(TypedDict, total=False):
    vector: list[float]
    model: str
    dimensions: int
    space: str
    computed_at: str
    raw: Any


class LogEntry(TypedDict, total=False):
    level: Literal["debug", "info", "warn", "error"]
    stage: Literal["prompt_build", "llm_call", "parse", "embed", "finalize"]
    message: str
    data: dict[str, Any]


class ExtractCallbacks(Protocol):
    def call_llm(self, request: LlmRequest) -> MaybeAwaitable: ...
    def get_embedding(self, request: EmbeddingRequest) -> MaybeAwaitable: ...
    def log(self, entry: LogEntry) -> None: ...


@dataclass
class UsageSummary:
    llm_calls: int = 0
    embedding_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ExtractResult:
    extraction: dict[str, Any]
    validation: ValidationResult
    warnings: list[str] = field(default_factory=list)
    stage1: dict[str, Any] = field(default_factory=dict)
    embeddings: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageSummary = field(default_factory=UsageSummary)


ExtensionResolver = Callable[[dict[str, Any]], MaybeAwaitable]


SYSTEM_MESSAGE = "You are a deterministic information extraction engine. Return only JSON matching the supplied schema."
STANDARD_EMBEDDING_INPUTS: list[NamedEmbeddingInput] = [
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
CAPABILITY_EMBEDDING_INPUTS: dict[str, str] = {
    "entities": "entities",
    "entity_state": "entities",
    "entity_context": "entities",
    "entity_ids": "entities",
    "relations": "entities",
    "relation_origin": "entities",
    "goals": "goals",
    "goal_timing": "goals",
    "goal_entity_refs": "goals",
    "themes": "themes",
    "keywords": "keywords",
    "summary": "summary",
    "sentiment": "sentiment",
    "structured_sentiment": "sentiment",
    "facts": "facts",
    "questions": "questions",
    "actions": "actions",
    "decisions": "decisions",
    "temporal_refs": "temporal_refs",
    "temporal_classes": "temporal_refs",
}

_BUILDER_KEYS = {
    "capabilities",
    "profile",
    "add",
    "remove",
    "categories",
    "source_type",
    "date",
    "stage",
    "extracted_at",
    "compact",
    "produced_by",
    "user_id",
    "source_id",
    "kind",
    "extensions",
    "embeddings",
    "capabilities_hint",
}


async def _maybe_await(value: MaybeAwaitable) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _get_callback(callbacks: Any, snake_name: str, camel_name: str | None = None) -> Callable[..., Any] | None:
    if isinstance(callbacks, dict):
        fn = callbacks.get(snake_name)
        if fn is None and camel_name is not None:
            fn = callbacks.get(camel_name)
        return fn if callable(fn) else None
    fn = getattr(callbacks, snake_name, None)
    if fn is None and camel_name is not None:
        fn = getattr(callbacks, camel_name, None)
    return fn if callable(fn) else None


def _safe_log(callbacks: Any, entry: LogEntry) -> None:
    log = _get_callback(callbacks, "log")
    if log is None:
        return
    try:
        log(entry)
    except Exception:
        pass


def _parse_llm_output(response: LlmResponse | dict[str, Any]) -> dict[str, Any]:
    if "output" in response:
        output = response["output"]
        if isinstance(output, dict):
            return output
    if "json" in response:
        payload = response["json"]
        if isinstance(payload, dict):
            return payload
    content = response.get("content")
    if not isinstance(content, str):
        raise ValueError("LLM response must include output, json, or JSON string content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Failed to parse LLM JSON response: parsed JSON must be an object")
    return parsed


def _model_uri_from_response(response: LlmResponse | dict[str, Any]) -> str | dict[str, Any] | None:
    if "produced_by" in response:
        produced_by = response["produced_by"]
        if isinstance(produced_by, (str, dict)):
            return produced_by
    model = response.get("model")
    if isinstance(model, str) and "://" in model:
        producer: dict[str, Any] = {"model": model}
        model_version = response.get("model_version")
        if isinstance(model_version, str):
            producer["model_version"] = model_version
        return producer
    return None


def _normalize_llm_response(response: LlmResponse | dict[str, Any]) -> NormalizedLlmResponse:
    raw = response.get("raw")
    raw_obj = raw if isinstance(raw, dict) else {}
    normalized: NormalizedLlmResponse = {}
    response_id = response.get("id") or response.get("response_id") or raw_obj.get("id")
    if isinstance(response_id, str):
        normalized["id"] = response_id
    status = response.get("status") or raw_obj.get("status")
    if isinstance(status, str):
        normalized["status"] = status
    model = response.get("model") or raw_obj.get("model")
    if isinstance(model, str):
        normalized["model"] = model
    model_version = response.get("model_version")
    if isinstance(model_version, str):
        normalized["model_version"] = model_version
    produced_by = response.get("produced_by")
    if isinstance(produced_by, (str, dict)):
        normalized["produced_by"] = produced_by
    content = response.get("content")
    if isinstance(content, str):
        normalized["content"] = content
    usage = response.get("usage")
    if isinstance(usage, dict):
        normalized["usage"] = usage
    if raw is not None:
        normalized["raw"] = raw
    return normalized


def _object_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _compact_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _text_for_named_embedding(input_name: str, source_text: str, stage1: dict[str, Any]) -> str | None:
    if input_name == "source":
        return source_text
    if input_name == "summary":
        summary = stage1.get("summary")
        return summary if isinstance(summary, str) and summary.strip() else None
    if input_name == "entities":
        entities = _object_items(stage1.get("entities"))
        if not entities:
            return None
        lines = []
        for entity in entities:
            prefix = f"{entity.get('id')}: " if isinstance(entity.get("id"), str) else ""
            typ = f" ({entity.get('type')})" if isinstance(entity.get("type"), str) else ""
            state = f" - {entity.get('state')}" if isinstance(entity.get("state"), str) else ""
            context = f" {entity.get('context')}" if isinstance(entity.get("context"), str) else ""
            lines.append(f"{prefix}{entity.get('name', 'unknown')}{typ}{state}{context}".strip())
        return "\n".join(lines)
    if input_name == "goals":
        goals = _object_items(stage1.get("goals"))
        return "\n".join(str(goal.get("text", "")) for goal in goals if goal.get("text")) or None
    if input_name in {"themes", "keywords"}:
        values = stage1.get(input_name)
        return ", ".join(str(value) for value in values) if isinstance(values, list) and values else None
    if input_name in {"facts", "questions", "actions", "decisions"}:
        items = _object_items(stage1.get(input_name))
        return "\n".join(str(item.get("text", "")) for item in items if item.get("text")) or None
    if input_name == "temporal_refs":
        refs = _object_items(stage1.get("temporal_refs"))
        return "\n".join(_compact_json(ref) for ref in refs) if refs else None
    if input_name == "sentiment":
        return _compact_json(stage1["sentiment"]) if "sentiment" in stage1 else None
    return None


def _resolve_embedding_inputs(
    selection: EmbeddingInputSelection | None,
    source_text: str,
    stage1: dict[str, Any],
    warnings: list[str],
) -> list[EmbeddingRequest]:
    selectors: list[EmbeddingInputSelector]
    if selection == "all":
        selectors = list(STANDARD_EMBEDDING_INPUTS)
    else:
        selectors = selection or []

    requests: list[EmbeddingRequest] = []
    for selector in selectors:
        if isinstance(selector, str):
            text = _text_for_named_embedding(selector, source_text, stage1)
            if text is None or text.strip() == "":
                warnings.append(f'embedding input "{selector}" was requested but no text was available; skipped')
                continue
            requests.append({"input": selector, "text": text})
        else:
            text = selector.get("text", "")
            input_name = selector.get("input", "")
            if text.strip() == "":
                warnings.append(f'embedding input "{input_name}" was empty; skipped')
                continue
            requests.append({"input": input_name, "text": text})
    return requests


def _derive_embedding_selection_from_capability_inputs(capabilities: list[Any] | None) -> list[str]:
    if not capabilities:
        return []
    selected: list[str] = []
    for capability in capabilities:
        if capability_embedding_preference(capability) is not True:
            continue
        input_name = CAPABILITY_EMBEDDING_INPUTS.get(capability_name(capability))
        if input_name is not None and input_name not in selected:
            selected.append(input_name)
    return selected


def _embedding_selection_for_options(
    options: dict[str, Any],
    explicit_embedding_inputs: EmbeddingInputSelection | None,
) -> EmbeddingInputSelection | None:
    selected: list[Any] = [
        *_derive_embedding_selection_from_capability_inputs(options.get("capabilities")),
        *_derive_embedding_selection_from_capability_inputs(options.get("add")),
    ]

    if explicit_embedding_inputs == "all":
        return explicit_embedding_inputs
    if isinstance(explicit_embedding_inputs, list):
        selected = [*explicit_embedding_inputs, *selected]

    overrides = options.get("embed")
    if isinstance(overrides, dict):
        for capability, enabled in overrides.items():
            input_name = CAPABILITY_EMBEDDING_INPUTS.get(str(capability))
            if input_name is None:
                continue
            if enabled is True and input_name not in selected:
                selected.append(input_name)
            if enabled is False and input_name in selected:
                selected.remove(input_name)

    unique = [input_name for index, input_name in enumerate(selected) if input_name not in selected[:index]]
    return unique or None


def _build_finalize_context(
    options: dict[str, Any],
    capabilities: list[str],
    llm_response: LlmResponse | dict[str, Any],
    embeddings: list[dict[str, Any]],
    dynamic_extensions: dict[str, Any] | None = None,
) -> FinalizeContext:
    produced_by = options.get("produced_by") or _model_uri_from_response(llm_response)
    if produced_by is None:
        raise ValueError("Cannot finalize without produced_by; pass produced_by or return produced_by/model URI from call_llm()")

    context: dict[str, Any] = {
        "produced_by": produced_by,
        "capabilities_hint": options.get("capabilities_hint") or capabilities,
    }
    for key in ("user_id", "source_id", "source_type", "kind"):
        if options.get(key) is not None:
            context[key] = options[key]

    extensions = {**(options.get("extensions") or {}), **(dynamic_extensions or {})}
    if extensions:
        context["extensions"] = extensions

    all_embeddings = [*(options.get("embeddings") or []), *embeddings]
    if all_embeddings:
        context["embeddings"] = all_embeddings

    return FinalizeContext(**context)


async def _resolve_dynamic_extensions(
    options: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    resolver = options.get("extend") or options.get("extension_resolver")
    if resolver is None:
        return None
    if not callable(resolver):
        raise ValueError("extend/extension_resolver must be callable")
    try:
        result = await _maybe_await(resolver(context))
    except Exception as exc:
        if options.get("extension_errors") == "warn":
            context["warnings"].append(f"extension resolver failed: {exc}")
            return None
        raise
    if result is None:
        return None
    if not isinstance(result, dict):
        raise ValueError("extension resolver must return a dictionary")
    return result


async def extract(
    text: str,
    callbacks: ExtractCallbacks | dict[str, Any],
    **options: Any,
) -> ExtractResult:
    """Run prompt construction, LLM extraction, optional embeddings, finalization, and validation."""
    response_format_name = options.pop("response_format_name", DEFAULT_RESPONSE_FORMAT_NAME)
    strict = options.pop("strict", True)
    embedding_inputs = options.pop("embedding_inputs", None)
    temperature = options.pop("temperature", None)
    max_tokens = options.pop("max_tokens", None)

    builder_options = {key: value for key, value in options.items() if key in _BUILDER_KEYS}
    builder = ExtractionBuilder(text=text, **builder_options)
    built = builder.build(name=response_format_name, strict=strict)
    usage = UsageSummary()
    warnings: list[str] = []

    _safe_log(callbacks, {
        "level": "info",
        "stage": "prompt_build",
        "message": "Built extraction prompt and response schema",
        "data": {"capabilities": built["capabilities"]},
    })

    call_llm = _get_callback(callbacks, "call_llm", "callLlm")
    if call_llm is None:
        raise ValueError("callbacks must provide call_llm")

    llm_request: LlmRequest = {
        "prompt": built["prompt"],
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": built["prompt"]},
        ],
        "capabilities": built["capabilities"],
        "schema": built["schema"],
        "response_format": built["response_format"],
    }
    if temperature is not None:
        llm_request["temperature"] = temperature
    if max_tokens is not None:
        llm_request["max_tokens"] = max_tokens

    _safe_log(callbacks, {
        "level": "info",
        "stage": "llm_call",
        "message": "Calling LLM extraction callback",
        "data": {"capabilities": built["capabilities"]},
    })
    usage.llm_calls += 1
    llm_response = await _maybe_await(call_llm(llm_request))
    if not isinstance(llm_response, dict):
        raise ValueError("call_llm must return a dictionary response")
    normalized_response = _normalize_llm_response(llm_response)

    llm_usage = llm_response.get("usage")
    if isinstance(llm_usage, dict):
        usage.input_tokens = llm_usage.get("input_tokens")
        usage.output_tokens = llm_usage.get("output_tokens")
        usage.total_tokens = llm_usage.get("total_tokens")

    stage1 = _parse_llm_output(llm_response)
    _safe_log(callbacks, {
        "level": "info",
        "stage": "parse",
        "message": "Parsed LLM extraction JSON",
        "data": {"fields": list(stage1.keys())},
    })

    embedding_requests = _resolve_embedding_inputs(
        _embedding_selection_for_options(options, embedding_inputs),
        text,
        stage1,
        warnings,
    )
    get_embedding = _get_callback(callbacks, "get_embedding", "getEmbedding")
    if embedding_requests and get_embedding is None:
        raise ValueError("embedding_inputs requested but get_embedding callback was not provided")

    embeddings: list[dict[str, Any]] = []
    for request in embedding_requests:
        _safe_log(callbacks, {
            "level": "info",
            "stage": "embed",
            "message": "Calling embedding callback",
            "data": {"input": request["input"]},
        })
        usage.embedding_calls += 1
        response = await _maybe_await(get_embedding(request))
        if not isinstance(response, dict):
            raise ValueError("get_embedding must return a dictionary response")
        vector = response.get("vector")
        if not isinstance(vector, list):
            raise ValueError("get_embedding response must include vector")
        model = response.get("model")
        if not isinstance(model, str):
            raise ValueError("get_embedding response must include model")
        embedding = {
            "vector": vector,
            "model": model,
            "input": request["input"],
            "dimensions": response.get("dimensions", len(vector)),
        }
        if response.get("space") is not None:
            embedding["space"] = response["space"]
        if response.get("computed_at") is not None:
            embedding["computed_at"] = response["computed_at"]
        embeddings.append(embedding)

    dynamic_extensions = await _resolve_dynamic_extensions(options, {
        "source_text": text,
        "capabilities": built["capabilities"],
        "prompt": built["prompt"],
        "schema": built["schema"],
        "response_format": built["response_format"],
        "llm_request": llm_request,
        "response": normalized_response,
        "llm_response": normalized_response,
        "stage1": stage1,
        "embeddings": embeddings,
        "usage": usage,
        "warnings": warnings,
    })

    context = _build_finalize_context(options, built["capabilities"], llm_response, embeddings, dynamic_extensions)
    finalized = builder.with_finalize_context(context).finalize(stage1)
    _safe_log(callbacks, {
        "level": "info" if finalized.validation.valid else "warn",
        "stage": "finalize",
        "message": "Finalized extraction",
        "data": {"valid": finalized.validation.valid, "errors": len(finalized.validation.errors)},
    })

    return ExtractResult(
        extraction=finalized.extraction,
        validation=finalized.validation,
        warnings=[*warnings, *finalized.warnings],
        stage1=stage1,
        embeddings=embeddings,
        usage=usage,
    )


run_extraction = extract
