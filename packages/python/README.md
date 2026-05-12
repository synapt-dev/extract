# synapt-extract

SynaptExtraction is the intermediate language (IL) for [synapt](https://synapt.dev)'s product stack. It is the universal exchange format between text extraction and intelligence operations.

```
Any text + Any LLM  ->  SynaptExtraction (IL)  ->  @synapt/memory (intelligence)
```

## Install

```bash
pip install synapt-extract
```

## Quick start

```python
from synapt_extract import (
    build_extraction_prompt,
    finalize_extraction,
    FinalizeContext,
)

# 1. Build a prompt
prompt = build_extraction_prompt(text, profile="standard")

# 2. Send to any LLM, parse JSON response
llm_output = json.loads(llm.complete(prompt))

# 3. Finalize
result = finalize_extraction(llm_output, FinalizeContext(
    produced_by="openai://gpt-4o-mini",
    user_id=user_id,
    kind="conversa/prayer",
))

assert result.validation.valid
```

## Prompt profiles

| Profile | Model class | Capabilities |
|---------|------------|--------------|
| `minimal` | 3B-7B local | entities, entity_state, goals, themes, summary |
| `standard` | GPT-4o-mini, Haiku | + entity_context, goal_timing, facts, temporal_refs, sentiment, evidence_anchoring |
| `full` | GPT-4o, Sonnet, Opus | + entity_ids, goal_entity_refs, relations, relation_origin, assertion_signals, temporal_classes |

## Prompt and schema builder

Use the builder when the model API supports structured output. It resolves capabilities once, then builds the matching prompt, Stage 1 JSON schema, OpenAI response format, finalized packet schema, and optional finalization context.

```python
from synapt_extract import create_extraction_builder

builder = (
    create_extraction_builder(text, profile="standard")
    .add_capabilities(["entity_ids", "goal_entity_refs"])
    .with_extracted_at("2026-05-11T18:00:00Z")
    .with_produced_by({
        "model": "openai://gpt-5.5",
        "model_version": "gpt-5.5-2026-04-23",
        "configuration": {"reasoning_effort": "medium"},
        "operator": "synapt-dev",
    })
    .with_source(source_id="note-1", source_type="note")
)

built = builder.build(name="synapt_extract_stage1")

# Send built["prompt"] and built["response_format"] to the model.
# Then call builder.finalize(stage1_json) or finalize_extraction(stage1_json, builder.finalize_context()).
```

`build_extraction_schema()` returns the semantic Stage 1 schema. `build_extraction_response_format()` returns an OpenAI-compatible `json_schema` response format; strict mode requires every object property as OpenAI expects and represents semantic optional fields as nullable. `build_finalized_extraction_schema()` returns the finalized packet shape, including `produced_by`, source context, capabilities, embeddings, and extensions.

## Full extraction runner

Use `extract()` when you want synapt-extract to run prompt construction, the LLM callback, optional embedding callbacks, finalization, and validation while your application owns provider credentials and routing.

```python
from synapt_extract import extract

result = await extract(
    text,
    {
        "call_llm": call_llm,
        "get_embedding": get_embedding,
    },
    capabilities=[
        {"name": "entities", "embed": True},
        {"name": "goals", "embed": True},
        {"name": "summary", "embed": True},
        "themes",
    ],
    source_id="note-1",
    source_type="note",
    embedding_inputs=["source"],
    extend=lambda ctx: {
        "synapt/response_binding": {
            "response_id": ctx["response"].get("id"),
            "response_model": ctx["response"].get("model"),
            "stage1_fields": len(ctx["stage1"]),
            "embedding_count": len(ctx["embeddings"]),
        }
    },
)
```

Capability entries can be plain strings or `{"name": ..., "embed": True}` specs. The runner derives embeddings from embedded capability specs and merges them with explicit embedding inputs such as `"source"`. `embedding_inputs="all"` remains available for exhaustive tests and computes embeddings for source, summary, entities, goals, themes, keywords, facts, questions, actions, decisions, temporal refs, and sentiment when those fields exist. Embeddings are opt-in; no embedding API call is made unless requested.

The `extend` resolver runs after the LLM response is parsed and embeddings are computed, but before finalization. It receives a normalized response envelope (`provider`, `id`, `status`, `model`, `stop_reason`, `usage`, and `raw`), so extensions can depend on provider output without knowing the provider's raw response shape. Raw OpenAI Responses and Anthropic Messages objects are translated automatically when returned as `raw`; callers can pass `response_translator` for custom providers. If `produced_by` is omitted, the runner can derive `openai://...` or `anthropic://...` producer metadata from normalized provider/model fields.

Builders also expose profile helpers and a preflight plan:

```python
plan = (
    create_extraction_builder(text)
    .full(embed=True)
    .minus("questions")
    .embed("summary", False)
    .plan()
)
```

## Links

- [Repository](https://github.com/synapt-dev/extract)
- [JSON Schema](https://synapt.dev/schemas/extract/v1.json)
- [TypeScript package](https://www.npmjs.com/package/@synapt-dev/extract)

## License

MIT
