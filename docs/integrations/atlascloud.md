---
title: "Structured outputs with Atlas Cloud, a complete guide with instructor"
description: "Learn how to use Instructor with Atlas Cloud to get type-safe, validated structured outputs from DeepSeek, GLM, Kimi, Qwen and MiniMax models through one OpenAI-compatible API."
---

# Structured outputs with Atlas Cloud, a complete guide with instructor

[Atlas Cloud](https://www.atlascloud.ai/) serves DeepSeek, GLM, Kimi, Qwen and MiniMax models behind
a single OpenAI-compatible API. This guide shows you how to use Instructor with Atlas Cloud for
type-safe, validated responses.

## Quick Start

Instructor works with Atlas Cloud through the OpenAI client, so you don't need to install anything
extra beyond the base package.

```bash
export ATLASCLOUD_API_KEY=<your-api-key>
```

## Simple User Example (Sync)

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("atlascloud/deepseek-ai/deepseek-v4-pro")

resp = client.create(
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


client = instructor.from_provider(
    "atlascloud/zai-org/glm-5",
    async_client=True,
)


async def extract_user():
    return await client.create(
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28 years old",
            },
        ],
        response_model=User,
    )


print(asyncio.run(extract_user()))
#> name='Ivan' age=28
```

## Modes

Atlas Cloud supports the OpenAI-compatible modes:

| Mode | Notes |
|---|---|
| `Mode.TOOLS` | Default. Tool calling. |
| `Mode.JSON_SCHEMA` | `response_format` with a strict JSON schema. |
| `Mode.MD_JSON` | Markdown-fenced JSON, for models without tool calling. |
| `Mode.PARALLEL_TOOLS` | Multiple tool calls in one response. |

```python
import instructor

client = instructor.from_provider(
    "atlascloud/deepseek-ai/deepseek-v4-pro",
    mode=instructor.Mode.JSON_SCHEMA,
)
```

## A note on reasoning models

Some Atlas models — `deepseek-ai/deepseek-v4-pro` among them — are reasoning models that spend
completion tokens on a hidden chain of thought before the answer. If you pass a small
`max_tokens`, the response can come back empty with `finish_reason="length"`. Leave `max_tokens`
unset or give it room. Non-reasoning ids such as `deepseek-ai/DeepSeek-V3.1` are unaffected.

## Custom base URL

The default base URL is `https://api.atlascloud.ai/v1`. Override it with `base_url` if you are
pointed at a different endpoint:

```python
import instructor

client = instructor.from_provider(
    "atlascloud/deepseek-ai/deepseek-v4-pro",
    base_url="https://api.atlascloud.ai/v1",
)
```
