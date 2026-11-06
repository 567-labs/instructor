---
title: Mode Comparison Guide
description: Compare different modes available in Instructor and understand when to use each
---

## Instructor Mode Comparison Guide

Instructor uses **core modes** that work across providers. Provider-specific
modes still work, but they are deprecated and will show warnings.

Mode handling now lives in provider handlers. The DSL no longer stores
mode-specific streaming logic.

## Core Modes

- `TOOLS`: Tool or function calling for structured extraction.
- `JSON_SCHEMA`: Native schema support when a provider has it.
- `MD_JSON`: JSON from text or code blocks for simple or fallback cases.
- `PARALLEL_TOOLS`: Multiple tool calls in one response.
- `RESPONSES_TOOLS`: OpenAI Responses API tools.

## Legacy Modes (Deprecated)

These legacy modes map to core modes:

- `FUNCTIONS` -> `TOOLS`
- `TOOLS_STRICT` -> `TOOLS`
- `ANTHROPIC_TOOLS` -> `TOOLS`
- `ANTHROPIC_JSON` -> `MD_JSON`
- `GENAI_TOOLS` -> `TOOLS`
- `GENAI_JSON` -> `JSON`
- `MISTRAL_TOOLS` -> `TOOLS`
- `MISTRAL_STRUCTURED_OUTPUTS` -> `JSON_SCHEMA`
- `BEDROCK_TOOLS` -> `TOOLS`
- `BEDROCK_JSON` -> `MD_JSON`
- `FIREWORKS_TOOLS` -> `TOOLS`
- `FIREWORKS_JSON` -> `MD_JSON`
- `CEREBRAS_TOOLS` -> `TOOLS`
- `CEREBRAS_JSON` -> `MD_JSON`
- `WRITER_TOOLS` -> `TOOLS`
- `WRITER_JSON` -> `MD_JSON`
- `PERPLEXITY_JSON` -> `MD_JSON`
- `VERTEXAI_TOOLS` -> `TOOLS`
- `VERTEXAI_JSON` -> `MD_JSON`
- `VERTEXAI_PARALLEL_TOOLS` -> `PARALLEL_TOOLS`

## Mode Selection Tips

- Use `TOOLS` for most structured output cases.
- Use `JSON_SCHEMA` when the provider supports native schema enforcement.
- Use `MD_JSON` if tools are not supported or outputs are simple.
- Use `PARALLEL_TOOLS` for multiple tasks in one response.

## Examples

### TOOLS Mode (Recommended)

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "openai/gpt-4o-mini",
    mode=Mode.TOOLS,
)
```

### MD_JSON Mode (Fallback)

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    mode=Mode.MD_JSON,
)
```

### JSON_SCHEMA Mode (Native Schema)

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "openai/gpt-4o-mini",
    mode=Mode.JSON_SCHEMA,
)
```

See the [Mode Migration Guide](concepts/mode-migration.md) for more details.

### Google/Gemini

For complex structures:

```python
import instructor

client = instructor.from_provider(
    "google/gemini-2.5-flash",
    mode=instructor.Mode.TOOLS,
)
```

For structured outputs with JSON:

```python
import instructor

client = instructor.from_provider(
    "google/gemini-2.5-flash",
    mode=instructor.Mode.JSON,
)
```

## Mode Compatibility List

Legacy modes are shown for compatibility only. Prefer core modes in new code.

- OpenAI: TOOLS, TOOLS_STRICT, PARALLEL_TOOLS, FUNCTIONS; JSON, MD_JSON, JSON_O1.
- Anthropic: ANTHROPIC_TOOLS, ANTHROPIC_PARALLEL_TOOLS; ANTHROPIC_JSON.
- Gemini: TOOLS; JSON.
- Vertex AI: VERTEXAI_TOOLS; VERTEXAI_JSON.
- Cohere: COHERE_TOOLS; JSON, MD_JSON.
- Mistral: MISTRAL_TOOLS; MISTRAL_STRUCTURED_OUTPUTS.
- Anyscale: (none); JSON, MD_JSON, JSON_SCHEMA.
- Databricks: TOOLS; JSON, MD_JSON.
- Together: (none); JSON, MD_JSON.
- Fireworks: FIREWORKS_TOOLS; FIREWORKS_JSON.
- Cerebras: (none); CEREBRAS_JSON.
- Writer: WRITER_TOOLS; JSON.
- Perplexity: (none); PERPLEXITY_JSON.
- GenAI: TOOLS; JSON.
- LiteLLM: depends on provider for both tool-based and JSON-based modes.

## Best Practices

1. **Start with the recommended mode for your provider**
    - For OpenAI: `TOOLS`
    - For Anthropic: `ANTHROPIC_TOOLS` (Claude 3+) or `ANTHROPIC_JSON`
    - For Gemini: `TOOLS` or `JSON`

2. **Try JSON modes for simple structures or if you encounter issues**
   - JSON modes often work with simpler schemas
   - They may be more token-efficient
   - They work with more models

3. **Use provider-specific modes when available**
   - Provider-specific modes are optimized for that provider
   - They handle special cases and requirements

4. **Test and validate**
   - Different modes may perform differently for your specific use case
   - Always test with your actual data and models
