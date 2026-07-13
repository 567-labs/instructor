---
title: "Structured outputs with Requesty, a complete guide with instructor"
description: "Learn how to use Instructor with Requesty to access multiple LLM providers through a unified, OpenAI-compatible API. Get type-safe, structured outputs from various models."
---

# Structured outputs with Requesty, a complete guide with instructor

[Requesty](https://requesty.ai) provides a unified, OpenAI-compatible API to access 400+ LLM models through a single endpoint, with smart routing, fallbacks, caching, and cost tracking. This guide shows you how to use Instructor with Requesty for type-safe, validated responses across various LLM providers.

## Quick Start

Instructor works with Requesty through the OpenAI client, so you don't need to install anything extra beyond the base package.

```bash
pip install instructor
```

You'll need a Requesty API key, which you can create at [app.requesty.ai/api-keys](https://app.requesty.ai/api-keys). Set it as the `REQUESTY_API_KEY` environment variable.

```bash
export REQUESTY_API_KEY="your-api-key"
```

Requesty uses `provider/model` naming (e.g. `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-5`, `google/gemini-2.5-flash`).

## Simple User Example (Sync)

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("requesty/openai/gpt-4o-mini")

resp = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Ivan is 28 years old",
        },
    ],
    response_model=User,
)

print(resp)
#> name='Ivan' age=28
```

## Simple User Example (Async)

```python
import asyncio
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("requesty/openai/gpt-4o-mini", async_client=True)


async def extract_user():
    return await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28 years old",
            },
        ],
        response_model=User,
    )


user = asyncio.run(extract_user())
print(user)
#> name='Ivan' age=28
```

## Using the OpenAI client directly

You can also point the OpenAI client at Requesty explicitly:

```python
from openai import OpenAI
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_openai(
    OpenAI(
        base_url="https://router.requesty.ai/v1",
        api_key="<your-requesty-api-key>",
    )
)

resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "Ivan is 28 years old"}],
    response_model=User,
)

print(resp)
#> name='Ivan' age=28
```

## Supported Modes

Requesty supports the standard OpenAI-compatible modes:

- `Mode.TOOLS`: tool/function calling (default)
- `Mode.JSON`: JSON output
- `Mode.MD_JSON`: Markdown-fenced JSON
- `Mode.PARALLEL_TOOLS`: parallel tool calls

## Related Documentation

- [OpenAI](./openai.md): The base OpenAI integration
- [OpenRouter](./openrouter.md): Another unified LLM router

The [Requesty models listing](https://app.requesty.ai/router/list) shows the full catalogue. Keep in mind that some models do not support `json_schema` response formats.
