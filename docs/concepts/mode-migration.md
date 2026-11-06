---
title: Mode Migration Guide
description: Migrate from provider-specific modes to the core modes in Instructor.
---

# Mode Migration Guide

This guide helps you move from provider-specific modes to the core modes.
Core modes work across providers and are the recommended choice for new code.

!!! note "V2 Preview"

    Provider-specific modes are deprecated in v2. They still work, emit warnings, and map to core modes.

## Core Modes

These are the core modes you should use:

- `TOOLS`: Tool or function calling
- `JSON_SCHEMA`: Native schema support when a provider has it
- `MD_JSON`: JSON extracted from text or code blocks
- `PARALLEL_TOOLS`: Multiple tool calls in one response
- `RESPONSES_TOOLS`: OpenAI Responses API tools

## Quick Mapping

Use this table to replace legacy modes:

| Legacy Mode | Core Mode |
|------------|-----------|
| `FUNCTIONS` | `TOOLS` |
| `TOOLS_STRICT` | `TOOLS` |
| `ANTHROPIC_TOOLS` | `TOOLS` |
| `ANTHROPIC_JSON` | `MD_JSON` |
| `COHERE_TOOLS` | `TOOLS` |
| `COHERE_JSON_SCHEMA` | `JSON_SCHEMA` |
| `XAI_TOOLS` | `TOOLS` |
| `XAI_JSON` | `MD_JSON` |
| `MISTRAL_TOOLS` | `TOOLS` |
| `MISTRAL_STRUCTURED_OUTPUTS` | `JSON_SCHEMA` |
| `FIREWORKS_TOOLS` | `TOOLS` |
| `FIREWORKS_JSON` | `MD_JSON` |
| `CEREBRAS_TOOLS` | `TOOLS` |
| `CEREBRAS_JSON` | `MD_JSON` |
| `WRITER_TOOLS` | `TOOLS` |
| `WRITER_JSON` | `MD_JSON` |
| `BEDROCK_TOOLS` | `TOOLS` |
| `BEDROCK_JSON` | `MD_JSON` |
| `PERPLEXITY_JSON` | `MD_JSON` |
| `VERTEXAI_TOOLS` | `TOOLS` |
| `VERTEXAI_JSON` | `MD_JSON` |
| `VERTEXAI_PARALLEL_TOOLS` | `PARALLEL_TOOLS` |

## Example: Anthropic

**Before:**

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "anthropic/claude-3-5-haiku-latest",
    mode=Mode.ANTHROPIC_TOOLS,
)
```

**After:**

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "anthropic/claude-3-5-haiku-latest",
    mode=Mode.TOOLS,
)
```

## Example: Bedrock

**Before:**

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    mode=Mode.BEDROCK_TOOLS,
)
```

**After:**

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    mode=Mode.BEDROCK_TOOLS,
)
```

## Notes

- Legacy modes still work but show a deprecation warning.
- Use core modes for new code and docs.
- Core tests are parameterized by provider and mode for consistent coverage.
- Streaming extraction is now handled by provider handlers instead of the DSL.
- Legacy `ResponseSchema.parse_*` helpers are deprecated. Use `process_response` or
  `ResponseSchema.from_response` with core modes so the v2 registry handles parsing.
- See [Mode Comparison](../modes-comparison.md) for details.
