"""Artifact bundle helpers for extraction fixtures and audits."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from synapt_extract.extract import ExtractResult


JsonObject = dict[str, Any]


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def create_artifact_bundle(
    *,
    source_text: str,
    result: ExtractResult,
    prompt: str | None = None,
    response_format: JsonObject | None = None,
    llm_request: JsonObject | None = None,
    llm_response: JsonObject | None = None,
    embedding_runs: list[JsonObject] | None = None,
    include_source_text: bool = True,
    created_at: str | None = None,
) -> JsonObject:
    prompt_payload = None
    prompt_text = prompt
    if isinstance(llm_request, dict) and isinstance(llm_request.get("prompt"), str):
        prompt_text = llm_request["prompt"]
    if isinstance(prompt_text, str):
        prompt_payload = {
            "text": prompt_text,
            "sha256": sha256_text(prompt_text),
        }
        resolved_response_format = response_format
        if resolved_response_format is None and isinstance(llm_request, dict):
            candidate = llm_request.get("response_format") or llm_request.get("responseFormat")
            resolved_response_format = candidate if isinstance(candidate, dict) else None
        if isinstance(resolved_response_format, dict):
            prompt_payload["response_format"] = resolved_response_format

    source: JsonObject = {
        "sha256": sha256_text(source_text),
        "characters": len(source_text),
    }
    if include_source_text:
        source["text"] = source_text

    bundle: JsonObject = {
        "version": "1",
        "created_at": created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source,
        "embeddings": embedding_runs or [],
        "stage1": result.stage1,
        "extraction": result.extraction,
        "validation": _jsonable(result.validation),
        "warnings": result.warnings,
        "usage": _jsonable(result.usage),
    }
    if prompt_payload is not None:
        bundle["prompt"] = prompt_payload
    if llm_request is not None or llm_response is not None:
        bundle["llm"] = {}
        if llm_request is not None:
            bundle["llm"]["request"] = _jsonable(llm_request)
        if llm_response is not None:
            bundle["llm"]["response"] = _jsonable(llm_response)
    return bundle


def write_artifact_bundle(
    directory: str | Path,
    bundle: JsonObject,
    *,
    prefix: str | None = None,
) -> dict[str, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    file_prefix = f"{prefix}." if prefix else ""

    def write_json(name: str, value: Any) -> None:
        path = root / f"{file_prefix}{name}"
        path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[name] = path

    def write_text(name: str, value: str) -> None:
        path = root / f"{file_prefix}{name}"
        path.write_text(value, encoding="utf-8")
        written[name] = path

    write_json("bundle.json", bundle)
    source = bundle.get("source")
    if isinstance(source, dict) and isinstance(source.get("text"), str):
        write_text("source.txt", source["text"])
    prompt = bundle.get("prompt")
    if isinstance(prompt, dict) and isinstance(prompt.get("text"), str):
        write_text("prompt.md", prompt["text"])
    llm = bundle.get("llm")
    if isinstance(llm, dict):
        if llm.get("request") is not None:
            write_json("llm-request.json", llm["request"])
        if llm.get("response") is not None:
            write_json("llm-response.json", llm["response"])
    embeddings = bundle.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        write_json("embedding-runs.json", embeddings)
    write_json("stage1.json", bundle.get("stage1", {}))
    write_json("extraction.json", bundle.get("extraction", {}))
    write_json("validation.json", {
        "validation": bundle.get("validation"),
        "warnings": bundle.get("warnings", []),
        "usage": bundle.get("usage", {}),
    })
    return written


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_jsonable(child) for child in value]
    if isinstance(value, tuple):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value
