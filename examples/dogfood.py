#!/usr/bin/env python3
"""Dogfooding: produce a real LLM-generated SynaptExtraction document.

Uses the local extract package's prompt system, calls Claude Sonnet,
validates and finalizes the output, saves to dogfood-2026-04-27.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "python" / "src"))

import anthropic
from synapt_extract.prompt import build_extraction_prompt
from synapt_extract.finalize import FinalizeContext, finalize_extraction
from synapt_extract.validate import validate_extraction

CONVERSATION = """\
Session: Weekly check-in with Marcus, April 22, 2026

Marcus: Hey, thanks for meeting with me. It's been a rough couple weeks.

Counselor: Of course. What's been going on?

Marcus: So my mom was diagnosed with early-stage Alzheimer's last month. We got the results back March 28th. My sister Elena and I have been trying to figure out care options, but we keep disagreeing. She wants to move Mom into assisted living at Sunrise Senior Care in Portland, but I think we should hire a home aide first.

Counselor: That sounds like a lot of weight to carry. How has this affected your day-to-day?

Marcus: Honestly, I've been losing sleep. I started seeing Dr. Patel — she's a therapist my friend recommended — on April 10th. That's been helping a little. But work at Meridian Tech has been intense too. We have a product launch May 15th and my manager Jake keeps piling on more tasks.

Counselor: So you're managing family stress, starting therapy, and dealing with work pressure all at once. What feels most urgent right now?

Marcus: The thing with Mom. Elena and I need to make a decision by end of May because the lease on Mom's apartment in Lake Oswego expires June 30th. We had a big argument last Saturday about it. I said some things I regret.

Counselor: It sounds like you want to repair that relationship with Elena too.

Marcus: Yeah, definitely. I texted her an apology yesterday but haven't heard back. I'm also thinking about whether I should take FMLA leave from work to help with the transition, whatever we decide. I haven't talked to Jake or HR about it yet.

Counselor: Those are concrete steps. Let's talk about prioritizing. What would you say your top three goals are right now?

Marcus: First, get aligned with Elena on Mom's care plan. Second, figure out the FMLA situation at work. Third, keep up with therapy — Dr. Patel wants me to come weekly and I've only been going biweekly.

Counselor: Good. And how are you feeling overall?

Marcus: Stressed but hopeful, I think. Starting therapy was a good move. And Mom is still pretty independent right now — it's early stage, so we have some time. I just don't want to waste it fighting with my sister.
"""

MODEL = "claude-sonnet-4-6"

TEMPORAL_REF_ALLOWED = {"version", "raw", "type", "resolved", "resolved_end", "context"}


def parse_llm_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return json.loads(text)


def conformance_cleanup(doc: dict) -> list[str]:
    """Fix systematic LLM-vs-schema conformance gaps. Returns list of fixes applied."""
    fixes = []

    for goal in doc.get("goals", []):
        if "entity_refs" not in goal:
            goal["entity_refs"] = []
            fixes.append("Added empty entity_refs to goal (required by schema, not prompted)")

    for i, ref in enumerate(doc.get("temporal_refs", [])):
        extra = set(ref.keys()) - TEMPORAL_REF_ALLOWED
        if extra:
            for k in extra:
                del ref[k]
            fixes.append(f"Stripped extra properties from temporal_refs[{i}]: {extra}")

    return fixes


def main():
    print("Building extraction prompt (standard profile)...")
    prompt = build_extraction_prompt(
        CONVERSATION,
        profile="standard",
        source_type="counseling_session",
        date="2026-04-22",
    )
    print(f"Prompt length: {len(prompt)} chars")

    print(f"\nCalling {MODEL}...")
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text
    print(f"Response length: {len(raw_text)} chars")
    llm_output = parse_llm_json(raw_text)

    # Apply conformance cleanup before finalization
    print("\nApplying conformance cleanup...")
    fixes = conformance_cleanup(llm_output)
    for fix in fixes:
        print(f"  FIX: {fix}")
    if not fixes:
        print("  (none needed)")

    # Finalize with v1.1 structured producer
    print("\nFinalizing with SynaptProducer (v1.1 object form)...")
    ctx = FinalizeContext(
        produced_by={
            "model": f"anthropic://{MODEL}",
            "model_version": MODEL,
            "deployment": "dogfood-extract-v020",
            "configuration": {
                "temperature": 0,
                "max_tokens": 4096,
            },
            "operator": "synapt-dev/extract-dogfood",
        },
        source_type="counseling_session",
        source_id="dogfood-session-2026-04-22",
        user_id="dogfood-user",
        kind="synapt/counseling",
    )
    result = finalize_extraction(llm_output, ctx)

    print(f"\nFinalized valid: {result.validation.valid}")
    print(f"Validation errors: {len(result.validation.errors)}")
    for err in result.validation.errors:
        print(f"  {err.path}: {err.message}")
    print(f"Warnings: {len(result.warnings)}")
    for w in result.warnings:
        print(f"  {w}")

    if not result.validation.valid:
        # Retry: feed errors back to LLM
        print("\n--- RETRY: feeding errors back to LLM ---")
        error_lines = [f"- {e.path}: {e.message}" for e in result.validation.errors]
        retry_prompt = (
            "The JSON you produced has validation errors. Fix them and output only the corrected JSON.\n\n"
            "Errors:\n" + "\n".join(error_lines) + "\n\n"
            "Original JSON:\n" + json.dumps(llm_output, indent=2)
        )
        retry_response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": retry_prompt}],
        )
        retry_output = parse_llm_json(retry_response.content[0].text)
        fixes2 = conformance_cleanup(retry_output)
        for fix in fixes2:
            print(f"  FIX (retry): {fix}")
        result = finalize_extraction(retry_output, ctx)
        print(f"\nRetry valid: {result.validation.valid}")
        print(f"Retry errors: {len(result.validation.errors)}")
        for err in result.validation.errors:
            print(f"  {err.path}: {err.message}")

    doc = result.extraction
    print(f"\n{'='*60}")
    print("DOGFOOD SUMMARY")
    print(f"{'='*60}")
    print(f"Capabilities: {doc.get('capabilities', [])}")
    print(f"Entities: {len(doc.get('entities', []))}")
    for e in doc.get("entities", []):
        print(f"  - {e.get('name')} ({e.get('type')}){' [state: ' + e['state'] + ']' if e.get('state') else ''}")
    print(f"Goals: {len(doc.get('goals', []))}")
    for g in doc.get("goals", []):
        print(f"  - [{g.get('status')}] {g.get('text', '')[:80]}")
    print(f"Themes: {doc.get('themes', [])}")
    print(f"Facts: {len(doc.get('facts', []))}")
    print(f"Temporal refs: {len(doc.get('temporal_refs', []))}")
    print(f"Summary: {doc.get('summary', 'MISSING')}")
    print(f"Sentiment: {doc.get('sentiment', 'MISSING')}")
    pb = doc.get("produced_by")
    if isinstance(pb, dict):
        print(f"Produced by: {pb.get('model')} (v1.1 object form)")
    else:
        print(f"Produced by: {pb} (v1.0 string form)")

    # Save
    out_path = Path(__file__).resolve().parent / "dogfood-2026-04-27.json"
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    if not result.validation.valid:
        print("\nFAILED: Document still has validation errors after retry!")
        sys.exit(1)

    print("\nDogfooding complete. Document is valid v1.1 SynaptExtraction.")


if __name__ == "__main__":
    main()
