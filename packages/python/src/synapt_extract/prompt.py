"""Composable prompt system for SynaptExtraction IL v1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parents[4] / "prompts"

CAPABILITY_DEPS: dict[str, list[str]] = {
    "entity_state": ["entities"],
    "entity_context": ["entities"],
    "entity_ids": ["entities"],
    "goal_timing": ["goals"],
    "goal_entity_refs": ["goals", "entity_ids"],
    "temporal_classes": ["temporal_refs"],
    "relations": ["entities", "entity_ids"],
    "relation_origin": ["relations"],
}

CANONICAL_ORDER = [
    "entities", "goals", "themes", "summary", "sentiment", "facts", "temporal_refs",
    "entity_state", "entity_context", "entity_ids",
    "goal_timing", "goal_entity_refs",
    "temporal_classes",
    "relations", "relation_origin",
    "assertion_signals", "evidence_anchoring",
]

CAPABILITY_RULES: dict[str, str] = {
    "entity_ids": 'Assign each entity a short local ID ("e1", "e2", etc.). Goals and relations reference entities by ID.',
    "temporal_refs": "Resolve all relative dates to absolute dates.",
    "relation_origin": 'Mark relation origin: "explicit" if stated in text, "inferred" if deduced from context, "dependent" if derived from another relation.',
    "assertion_signals": 'Preserve negation, hedging, and conditions in signals. "I might move" → hedged=true. "No longer using Redis" → negated=true. "If we get funding" → condition="we get funding".',
}


def _load_profile(name: str) -> list[str]:
    path = PROMPTS_DIR / "profiles" / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown profile: {name}")
    data = json.loads(path.read_text())
    return data["capabilities"]


def _load_fragment(name: str) -> str:
    path = PROMPTS_DIR / "v1" / f"{name}.txt"
    return path.read_text()


def _render_template(template: str, context: dict[str, Any]) -> str:
    def replace_if(match: re.Match) -> str:
        var = match.group(1)
        body = match.group(2)
        if context.get(var):
            return _render_vars(body, context)
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


def resolve_capabilities(
    *,
    capabilities: list[str] | None = None,
    profile: str | None = None,
    add: list[str] | None = None,
    remove: list[str] | None = None,
) -> list[str]:
    if capabilities is None and profile is None:
        raise ValueError("Either capabilities or profile must be provided")

    if capabilities is not None:
        caps = set(capabilities)
    else:
        caps = set(_load_profile(profile))

    if add:
        caps.update(add)
    if remove:
        caps -= set(remove)

    changed = True
    while changed:
        changed = False
        for cap in list(caps):
            for dep in CAPABILITY_DEPS.get(cap, []):
                if dep not in caps:
                    caps.add(dep)
                    changed = True

    return sorted(caps, key=lambda c: CANONICAL_ORDER.index(c) if c in CANONICAL_ORDER else len(CANONICAL_ORDER))


def build_extraction_prompt(
    text: str,
    *,
    capabilities: list[str] | None = None,
    profile: str | None = None,
    add: list[str] | None = None,
    remove: list[str] | None = None,
    categories: list[str] | None = None,
    source_type: str | None = None,
    date: str | None = None,
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

    postamble_template = _load_fragment("postamble")
    if rules_section:
        extra_rules = "\n".join(f"- {r}" for r in rules_section)
        postamble_rendered = _render_template(postamble_template, template_ctx).rstrip()
        idx = postamble_rendered.find("\nText:")
        if idx >= 0:
            postamble_rendered = postamble_rendered[:idx] + "\n" + extra_rules + postamble_rendered[idx:]
        parts.append(postamble_rendered)
    else:
        parts.append(_render_template(postamble_template, template_ctx).rstrip())

    return "\n".join(parts) + "\n"
