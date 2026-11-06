---
title: How Instructor Patches LLM Clients
description: Learn how Instructor adds structured output capabilities to LLM clients through patching.
---

# Patching

Patching adds structured output features to LLM client libraries. This page explains how it works. For most users, [`from_provider`](./from_provider.md) is simpler than manual patching.

!!! tip "Recommended Approach"
    Use [`from_provider`](./from_provider.md) instead of manual patching. It works the same way across all providers. See the [Migration Guide](./migration.md) if you're using older patching patterns.

## What is Patching?

Patching adds new features to LLM client objects without changing their original code. When Instructor patches a client, it adds:

- New parameters: `response_model`, `max_retries`, and `context` to completion methods
- Validation: Checks responses against Pydantic models
- Retry logic: Retries when validation fails
- Compatibility: The patched client still works with all original methods

## How Patching Works

When Instructor patches a client, it:

1. Wraps the completion method: Intercepts calls to `create()` or `chat.completions.create()`
2. Converts schemas: Changes Pydantic models into provider-specific formats (JSON schema, tool definitions, etc.)
3. Validates responses: Checks LLM outputs against your Pydantic model
4. Handles retries: Retries with validation feedback if needed
5. Returns typed objects: Converts validated JSON into Pydantic model instances

## Patching Modes

Different providers support different modes for structured extraction. Instructor automatically selects the best mode for each provider, but you can override it:

### Tool Calling (TOOLS)

Uses the provider's function/tool calling API. This is the default for OpenAI.

Supported by: OpenAI, Anthropic, Google, Ollama (for supported models)

### JSON Mode

Instructs the model to return JSON directly. Works with most providers.

Supported by: OpenAI, Anthropic, Google, Ollama, and most providers

### Markdown JSON (MD_JSON)

Asks for JSON wrapped in markdown. Only use for specific providers like Databricks.

Supported by: Databricks, some vision models

## Default Modes by Provider

Each provider uses a recommended default mode:

- **OpenAI**: `Mode.TOOLS` (function calling)
- **Anthropic**: `Mode.TOOLS` (tool use)
- **Google**: `Mode.TOOLS` (function calling)
- **Ollama**: `Mode.TOOLS` (if model supports it) or `Mode.JSON`
- **Others**: Provider-specific defaults

When using `from_provider`, these defaults are applied automatically. You can override them with the `mode` parameter.

## Manual Patching (Advanced)

If you need to patch a client manually (not recommended for most users):

```python
import openai
import instructor
from pydantic import BaseModel


class YourModel(BaseModel):
    message: str


# Create the base client
openai_client = openai.OpenAI()

# Patch it manually
client = instructor.patch(openai_client, mode=instructor.Mode.TOOLS)

# Now use it
response = client.chat.completions.create(
    response_model=YourModel,
    messages=[{"role": "user", "content": "Say hello"}],
)
```

However, using `from_provider` is simpler and recommended:

```python
import instructor
from pydantic import BaseModel


# Simpler approach
class YourModel(BaseModel):
    message: str


client = instructor.from_provider("openai/gpt-4o-mini")
_response = client.create(
    response_model=YourModel,
    messages=[{"role": "user", "content": "Say hello"}],
)
```

## What Gets Patched?

Instructor adds these features to patched clients:

### New Parameters

- `response_model`: A Pydantic model or type that defines the expected output structure
- `max_retries`: Number of retry attempts if validation fails (default: 0)
- `context`: Additional context for validation hooks

### Enhanced Methods

The patched client's `create()` method:
- Accepts `response_model` parameter
- Validates responses automatically
- Retries on validation failures
- Returns typed Pydantic objects instead of raw responses

## Provider-Specific Considerations

### OpenAI

- Default mode: `TOOLS` (function calling)
- Supports streaming with structured outputs

### Anthropic

- Default mode: `Mode.TOOLS` (tool use)
- Uses Claude's native tool calling API

### Google Gemini

- Default mode: `Mode.TOOLS` (function calling)
- Requires `jsonref` package for tool calling
- Union types (except `Optional`) are not supported

### Ollama (Local Models)

- Default mode: `TOOLS` (if model supports it) or `JSON`
- Models like llama3.1, llama3.2, mistral-nemo support tools
- Older models fall back to JSON mode

## When to Use Manual Patching

Manual patching is rarely needed. Use it only if:

1. You need fine-grained control over the patching process
2. You're working with a custom client implementation
3. You're debugging patching behavior

For 99% of use cases, `from_provider` is the better choice.

## Related Documentation

- [from_provider Guide](./from_provider.md) - Recommended way to create patched clients
- [Migration Guide](./migration.md) - Migrating from manual patching to from_provider
- [Modes Comparison](../modes-comparison.md) - Detailed comparison of different modes
- [Integrations](../integrations/index.md) - Provider-specific documentation
