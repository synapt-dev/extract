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

## Links

- [Repository](https://github.com/synapt-dev/extract)
- [JSON Schema](https://synapt.dev/schemas/extract/v1.json)
- [TypeScript package](https://www.npmjs.com/package/@synapt-dev/extract)

## License

MIT
