"""Composable prompt system for SynaptExtraction IL v1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from synapt_extract.schema import EXTRACTION_CAPABILITIES

_INSTALLED_PROMPTS = Path(__file__).resolve().parent / "prompts"
_REPO_PROMPTS = Path(__file__).resolve().parents[4] / "prompts"
PROMPTS_DIR = _INSTALLED_PROMPTS if _INSTALLED_PROMPTS.is_dir() else _REPO_PROMPTS


def _duplicates(values: list[str]) -> list[str]:
    return [value for index, value in enumerate(values) if value in values[:index]]


def _validate_capability_registry(registry: dict[str, Any], path: Path) -> None:
    if registry.get("version") != "1":
        raise ValueError(f"Unsupported capability registry version in {path}: {registry.get('version')}")

    definitions = registry.get("capabilities")
    if not isinstance(definitions, list):
        raise ValueError(f"Capability registry {path} must define capabilities")

    names = [definition.get("name") for definition in definitions if isinstance(definition, dict)]
    if len(names) != len(definitions) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Capability registry {path} contains malformed capability definitions")

    duplicates = _duplicates(names)
    if duplicates:
        raise ValueError(f"Capability registry has duplicate capabilities: {', '.join(sorted(set(duplicates)))}")

    registry_capabilities = set(names)
    unknown = registry_capabilities - EXTRACTION_CAPABILITIES
    if unknown:
        raise ValueError(f"Capability registry contains unknown capabilities: {', '.join(sorted(unknown))}")

    missing = EXTRACTION_CAPABILITIES - registry_capabilities
    if missing:
        raise ValueError(f"Capability registry is missing schema capabilities: {', '.join(sorted(missing))}")

    for definition in definitions:
        deps = definition.get("depends_on", [])
        if not isinstance(deps, list):
            raise ValueError(f"Capability registry dependency list for {definition.get('name')} must be a list")
        for dep in deps:
            if dep not in registry_capabilities:
                raise ValueError(f"Capability registry dependency {definition.get('name')}.{dep} is not valid")

    profiles = registry.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"Capability registry {path} must define profiles")
    for profile, capabilities in profiles.items():
        if not isinstance(capabilities, list):
            raise ValueError(f"Profile {profile} capabilities must be a list")
        for capability in capabilities:
            if capability not in registry_capabilities:
                raise ValueError(f"Profile {profile} references unknown capability {capability}")

    omit_when_absent = registry.get("omit_when_absent", [])
    if not isinstance(omit_when_absent, list):
        raise ValueError("Capability registry omit_when_absent must be a list")
    for capability in omit_when_absent:
        if capability not in registry_capabilities:
            raise ValueError(f"omit_when_absent references unknown capability {capability}")

    standard_embedding_inputs = registry.get("standard_embedding_inputs")
    if not isinstance(standard_embedding_inputs, list):
        raise ValueError("Capability registry standard_embedding_inputs must be a list")
    embedding_inputs = {
        "source",
        *(
            definition.get("embedding_input")
            for definition in definitions
            if isinstance(definition.get("embedding_input"), str)
        ),
    }
    for input_name in standard_embedding_inputs:
        if input_name not in embedding_inputs:
            raise ValueError(f"standard_embedding_inputs references unknown input {input_name}")


def _load_capability_registry() -> dict[str, Any]:
    path = PROMPTS_DIR / "capabilities.json"
    if not path.exists():
        raise ValueError(f"Missing capability registry: {path}")
    registry = json.loads(path.read_text())
    _validate_capability_registry(registry, path)
    return registry


CAPABILITY_REGISTRY = _load_capability_registry()
_CAPABILITY_DEFINITIONS: list[dict[str, Any]] = CAPABILITY_REGISTRY["capabilities"]
CANONICAL_ORDER = [definition["name"] for definition in _CAPABILITY_DEFINITIONS]
VALID_CAPABILITIES = frozenset(CANONICAL_ORDER)
CAPABILITY_DEPS: dict[str, list[str]] = {
    definition["name"]: list(definition["depends_on"])
    for definition in _CAPABILITY_DEFINITIONS
    if definition.get("depends_on") is not None
}
CAPABILITY_RULES: dict[str, str] = {
    definition["name"]: definition["rule"]
    for definition in _CAPABILITY_DEFINITIONS
    if isinstance(definition.get("rule"), str)
}
BASE_CAPABILITIES = frozenset(
    definition["name"]
    for definition in _CAPABILITY_DEFINITIONS
    if definition.get("base") is True
)
MODIFIER_ONLY_CAPABILITIES = frozenset(
    definition["name"]
    for definition in _CAPABILITY_DEFINITIONS
    if definition.get("modifier_only") is True
)
CAPABILITY_EMBEDDING_INPUTS: dict[str, str] = {
    definition["name"]: definition["embedding_input"]
    for definition in _CAPABILITY_DEFINITIONS
    if isinstance(definition.get("embedding_input"), str)
}
OMIT_WHEN_ABSENT = tuple(CAPABILITY_REGISTRY["omit_when_absent"])
STANDARD_EMBEDDING_INPUTS = tuple(CAPABILITY_REGISTRY["standard_embedding_inputs"])


def _build_run_constraint_rules(
    resolved: list[str],
    *,
    stage: str | None = None,
    extracted_at: str | None = None,
    compact: bool | None = None,
) -> list[str]:
    rules: list[str] = []
    capabilities = set(resolved)

    if compact:
        rules.append("Keep this extraction compact and high signal.")
    if extracted_at:
        rules.append(f"Use exactly this extracted_at value: {extracted_at}.")
    if stage == "stage1":
        rules.append(
            "Produce Stage 1 content only. Do not include version, produced_by, source_id, "
            "source_type, kind, capabilities, extensions, or embeddings."
        )
        rules.append("Only include fields represented in the requested capability set and response schema.")
    if "entity_ids" in capabilities:
        rules.append('Entity IDs are extraction-local only. Use short IDs like "e1", "e2", "e3".')
    if "goal_entity_refs" in capabilities:
        rules.append("Every goal.entity_refs entry must refer to one of the entity IDs you emit.")
    if "temporal_refs" not in capabilities:
        rules.append("Omit temporal_refs for this run.")
    if "relations" not in capabilities:
        rules.append("Omit relations for this run.")
    if "assertion_signals" not in capabilities:
        rules.append("Omit signals fields for this run.")
    for capability in OMIT_WHEN_ABSENT:
        if capability not in capabilities:
            rules.append(f"Omit {capability} for this run.")

    return rules


def _load_profile(name: str) -> list[str]:
    profiles = CAPABILITY_REGISTRY["profiles"]
    if name not in profiles:
        raise ValueError(f"Unknown profile: {name}")
    return list(profiles[name])


def profile_capabilities(profile: str) -> list[str]:
    return list(_load_profile(profile))


def _load_fragment(name: str) -> str:
    path = PROMPTS_DIR / "v1" / f"{name}.txt"
    return path.read_text()


def _render_template(template: str, context: dict[str, Any]) -> str:
    def replace_if(match: re.Match) -> str:
        var = match.group(1)
        body = match.group(2)
        if context.get(var):
            return body
        return ""

    result = re.sub(r"\{\{#if (\w+)\}\}(.*?)\{\{/if\}\}", replace_if, template, flags=re.DOTALL)
    return _render_vars(result, context)


def _render_vars(template: str, context: dict[str, Any]) -> str:
    def replace_var(match: re.Match) -> str:
        var = match.group(1)
        val = context.get(var, "")
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val)

    return re.sub(r"\{\{(\w+)\}\}", replace_var, template)


def capability_name(capability: Any) -> str:
    if isinstance(capability, str):
        return capability
    if isinstance(capability, dict):
        name = capability.get("name") or capability.get("capability")
        if isinstance(name, str):
            return name
    raise ValueError("Capability object must include name or capability")


def normalize_capability_inputs(capabilities: list[Any]) -> list[str]:
    return [capability_name(capability) for capability in capabilities]


def capability_requests_embedding(capability: Any) -> bool:
    return isinstance(capability, dict) and (capability.get("embed") is True or capability.get("embedding") is True)


def capability_embedding_preference(capability: Any) -> bool | None:
    if not isinstance(capability, dict):
        return None
    if isinstance(capability.get("embed"), bool):
        return capability["embed"]
    if isinstance(capability.get("embedding"), bool):
        return capability["embedding"]
    return None


def capability_embedding_input(capability: str) -> str | None:
    return CAPABILITY_EMBEDDING_INPUTS.get(capability)


def _validate_capability_names(caps: list[Any], source: str) -> None:
    names = set(normalize_capability_inputs(caps))
    unknown = names - VALID_CAPABILITIES
    if unknown:
        raise ValueError(f"Unknown {source}: {', '.join(sorted(unknown))}")


def resolve_capabilities(
    *,
    capabilities: list[Any] | None = None,
    profile: str | None = None,
    add: list[Any] | None = None,
    remove: list[Any] | None = None,
) -> list[str]:
    if capabilities is None and profile is None:
        raise ValueError("Either capabilities or profile must be provided")

    if capabilities is not None:
        _validate_capability_names(capabilities, "capabilities")
        caps = set(normalize_capability_inputs(capabilities))
    else:
        caps = set(_load_profile(profile))

    if add:
        _validate_capability_names(add, "capabilities in add")
        caps.update(normalize_capability_inputs(add))
    removed = set(expand_capability_exclusions(normalize_capability_inputs(remove))) if remove else set()

    changed = True
    while changed:
        changed = False
        for cap in list(caps):
            for dep in CAPABILITY_DEPS.get(cap, []):
                if dep not in caps:
                    caps.add(dep)
                    changed = True

    caps -= removed

    if not caps:
        raise ValueError("Resolved capability set is empty")

    modifiers_present = caps & MODIFIER_ONLY_CAPABILITIES
    if modifiers_present and not (caps & BASE_CAPABILITIES):
        raise ValueError(
            f"Modifier capabilities {sorted(modifiers_present)} require at least one "
            f"base capability ({', '.join(sorted(BASE_CAPABILITIES))})"
        )

    return sorted(caps, key=lambda c: CANONICAL_ORDER.index(c) if c in CANONICAL_ORDER else len(CANONICAL_ORDER))


def expand_capability_exclusions(capabilities: list[str]) -> list[str]:
    excluded = set(capabilities)
    changed = True
    while changed:
        changed = False
        for capability, deps in CAPABILITY_DEPS.items():
            if capability not in excluded and any(dep in excluded for dep in deps):
                excluded.add(capability)
                changed = True
    return sorted(excluded, key=lambda c: CANONICAL_ORDER.index(c) if c in CANONICAL_ORDER else len(CANONICAL_ORDER))


def build_extraction_prompt(
    text: str,
    *,
    capabilities: list[Any] | None = None,
    profile: str | None = None,
    add: list[Any] | None = None,
    remove: list[Any] | None = None,
    categories: list[str] | None = None,
    source_type: str | None = None,
    date: str | None = None,
    stage: str | None = None,
    extracted_at: str | None = None,
    compact: bool | None = None,
) -> str:
    if capabilities is not None and profile is not None:
        raise ValueError("Cannot specify both capabilities and profile")

    resolved = resolve_capabilities(
        capabilities=capabilities,
        profile=profile,
        add=add,
        remove=remove,
    )

    template_ctx: dict[str, Any] = {
        "text": text,
        "categories": categories,
        "source_type": source_type,
        "date": date,
    }

    parts: list[str] = []

    preamble = _render_template(_load_fragment("preamble"), template_ctx)
    parts.append(preamble.strip())

    for cap in resolved:
        fragment = _render_template(_load_fragment(cap), template_ctx)
        parts.append(fragment.rstrip())

    rules_section: list[str] = []
    for cap in resolved:
        rule = CAPABILITY_RULES.get(cap)
        if rule:
            rules_section.append(rule)
    run_constraint_rules = _build_run_constraint_rules(
        resolved,
        stage=stage,
        extracted_at=extracted_at,
        compact=compact,
    )

    postamble_template = _load_fragment("postamble")
    if rules_section or run_constraint_rules:
        extra_blocks: list[str] = []
        if rules_section:
            extra_blocks.append("\n".join(f"- {r}" for r in rules_section))
        if run_constraint_rules:
            extra_blocks.append(
                "Additional run constraints:\n"
                + "\n".join(f"- {r}" for r in run_constraint_rules)
            )
        extra_rules = "\n".join(extra_blocks)
        postamble_rendered = _render_template(postamble_template, template_ctx).rstrip()
        idx = postamble_rendered.find("\nText:")
        if idx >= 0:
            postamble_rendered = postamble_rendered[:idx] + "\n" + extra_rules + postamble_rendered[idx:]
        parts.append(postamble_rendered)
    else:
        parts.append(_render_template(postamble_template, template_ctx).rstrip())

    return "\n".join(parts) + "\n"
