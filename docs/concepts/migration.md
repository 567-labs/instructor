---
title: Migration Guide
description: Migrate from older Instructor patterns to the modern from_provider approach.
---

# Migration Guide

This guide helps you migrate from older Instructor patterns to `from_provider`, the recommended approach for all providers.

## Why Migrate?

- **Simpler code**: Less boilerplate, easier to read
- **Consistent interface**: Same pattern works for all providers
- **Better type safety**: Improved IDE support
- **Future-proof**: Recommended pattern going forward

## Quick Reference

| Old Pattern | New Pattern |
|-------------|-------------|
| `instructor.patch(openai.OpenAI())` | `instructor.from_provider("openai/model")` |
| `instructor.apatch(openai.AsyncOpenAI())` | `instructor.from_provider("openai/model", async_client=True)` |
| `from_openai(client)` | `instructor.from_provider("openai/model")` |
| `from_anthropic(client)` | `instructor.from_provider("anthropic/model")` |
| `from_genai(client)` | `instructor.from_provider("google/model")` |
| `client.chat.completions.create(...)` | `client.create(...)` |
| `client.messages.create(...)` | `client.create(...)` |

## Basic Migration

**Before:**

```python
import openai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


openai_client = openai.OpenAI()
client = instructor.patch(openai_client)

user = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**After:**

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4o-mini")

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

## Async Migration

**Before:**

```python
import openai
import instructor

openai_client = openai.AsyncOpenAI()
client = instructor.apatch(openai_client)

user = await client.chat.completions.create(...)
```

**After:**

```python
import instructor

client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

user = await client.create(...)
```

## Provider-Specific Migrations

### Anthropic

```python
# Before (removed)
import anthropic
from instructor import from_anthropic

client = from_anthropic(anthropic.Anthropic())
user = client.messages.create(..., model="claude-3-5-sonnet")

# After
client = instructor.from_provider("anthropic/claude-3-5-sonnet")
user = client.create(...)
```

### Google/Gemini

```python
# Before (removed)
import google.genai as genai
from instructor import from_genai

client = from_genai(genai.Client(), model="gemini-pro")
user = client.generate_content(...)

# After
client = instructor.from_provider("google/gemini-pro")
user = client.create(messages=[...])
```

## Configuration Options

Pass configuration directly to `from_provider`:

```python
import instructor

# Mode configuration
client = instructor.from_provider("openai/gpt-4o-mini", mode=instructor.Mode.JSON)

# Custom API settings
client = instructor.from_provider(
    "openai/gpt-4o-mini",
    api_key="custom-key",
    organization="org-id",
    timeout=30.0,
)
```

## Multiple Providers

**Before:**

```python
import openai
import anthropic
import instructor
from instructor import from_anthropic

openai_client = instructor.patch(openai.OpenAI())
anthropic_client = from_anthropic(anthropic.Anthropic())
```

**After:**

```python
import instructor

openai_client = instructor.from_provider("openai/gpt-4o-mini")
anthropic_client = instructor.from_provider("anthropic/claude-3-5-sonnet")
```

## Migration Checklist

1. **Identify your current pattern**: `patch()`, `apatch()`, or `from_*()` functions
2. **Find your model name**: e.g., `gpt-4o-mini`, `claude-3-5-sonnet`
3. **Replace client creation**: Use `from_provider("provider/model")`
4. **Update method calls**: Change to `client.create(...)`
5. **Use standard message format**: `[{"role": "user", "content": "..."}]`
6. **Test your code**

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `'Instructor' object has no attribute 'chat'` | Using old method call | Use `client.create()` instead of `client.chat.completions.create()` |
| Invalid model string | Wrong format | Use `"provider/model-name"` format |
| Message format error | Provider-specific format | Use standard `messages` list format |

## Backward Compatibility

Legacy helpers have been removed:

- `instructor.patch()` → Use `from_provider` instead
- `instructor.apatch()` → Use `from_provider` with `async_client=True`
- `from_openai()`, `from_anthropic()`, etc. → Use `from_provider`

Update all call sites before upgrading.

## See Also

- [from_provider Guide](./from_provider.md) - Complete guide to using from_provider
- [Patching](./patching.md) - How Instructor enhances clients
