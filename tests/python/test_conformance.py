"""Cross-language conformance fixtures shared by Python and TypeScript suites."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "python" / "src"))

from synapt_extract.finalize import finalize_extraction, FinalizeContext
from synapt_extract.prompt import build_extraction_prompt, resolve_capabilities
from synapt_extract.validate import validate_extraction


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "conformance"


def _load(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())


class TestValidationConformance:

    def test_validate_cases(self):
        for case in _load("validate_cases.json"):
            result = validate_extraction(case["input"])
            assert result.valid is case["expected_valid"], case["name"]
            for path in case["expected_error_paths"]:
                assert any(err.path == path for err in result.errors), case["name"]


class TestFinalizeConformance:

    def test_finalize_cases(self):
        for case in _load("finalize_cases.json"):
            result = finalize_extraction(
                case["llm_output"],
                FinalizeContext(**case["context"]),
            )
            assert result.validation.valid is case["expected_valid"], case["name"]
            if case["expected_valid"]:
                assert result.extraction["version"] == "1"
                assert set(case["expected_capabilities"]).issubset(set(result.extraction["capabilities"]))
            else:
                for path in case["expected_error_paths"]:
                    assert any(err.path == path for err in result.validation.errors), case["name"]


class TestPromptConformance:

    def test_prompt_cases(self):
        for case in _load("prompt_cases.json"):
            resolve_opts = {
                key: case["options"][key]
                for key in ("capabilities", "profile", "add", "remove")
                if key in case["options"]
            }
            resolved = resolve_capabilities(**resolve_opts)
            assert resolved == case["expected_capabilities"], case["name"]
            prompt = build_extraction_prompt(case["text"], **case["options"])
            for snippet in case["prompt_includes"]:
                assert snippet in prompt, case["name"]
            for snippet in case["prompt_excludes"]:
                assert snippet not in prompt, case["name"]
