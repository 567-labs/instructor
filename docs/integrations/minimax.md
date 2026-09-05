---
title: Structured Outputs with MiniMax and Pydantic
description: Learn how to use MiniMax with Instructor for structured JSON outputs using Pydantic models. Create type-safe, validated responses from MiniMax models with Python.
---

# Structured Outputs with MiniMax

This guide demonstrates how to use MiniMax with Instructor to generate structured outputs. You'll learn how to use MiniMax's models with Pydantic to create type-safe, validated responses.

## Prerequisites

Sign up for a MiniMax account and get an API key at [minimax.io](https://www.minimax.io/).

```bash
export MINIMAX_API_KEY=<your-api-key-here>
pip install "instructor[openai]"
```

MiniMax provides an OpenAI-compatible API, so no additional SDK is needed beyond `openai`.

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers

## Basic Usage

### One-line setup with `from_provider`

```python
import os
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("minimax/MiniMax-M2.7")

user = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

### Explicit client setup

```python
import os
import openai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


raw_client = openai.OpenAI(
    api_key=os.environ["MINIMAX_API_KEY"],
    base_url="https://api.minimax.io/v1",
)
client = instructor.from_minimax(raw_client)

user = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

### Async example

```python
import asyncio
import os
import openai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


async def main():
    raw_client = openai.AsyncOpenAI(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url="https://api.minimax.io/v1",
    )
    client = instructor.from_minimax(raw_client)

    user = await client.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": "Extract: Bob is 42 years old"}],
        response_model=User,
    )
    print(user)
    # > User(name='Bob', age=42)


asyncio.run(main())
```

## Supported Modes

MiniMax supports two modes with Instructor:

| Mode | Description |
|------|-------------|
| `MINIMAX_TOOLS` (default) | Uses the model's tool-calling feature for structured output |
| `MINIMAX_JSON` | Requests JSON via system prompt — use when tool calling is not desired |

```python
import os
import openai
import instructor
from instructor import Mode
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


raw_client = openai.OpenAI(
    api_key=os.environ["MINIMAX_API_KEY"],
    base_url="https://api.minimax.io/v1",
)

# Tool calling mode (default)
tools_client = instructor.from_minimax(raw_client, mode=Mode.MINIMAX_TOOLS)

# JSON mode via system prompt
json_client = instructor.from_minimax(raw_client, mode=Mode.MINIMAX_JSON)
```

### Note on reasoning models

MiniMax reasoning models (such as `MiniMax-M2.7`) may prepend `<think>...</think>` blocks
to their response when using `MINIMAX_JSON` mode.  Instructor strips these automatically
before parsing the JSON, so your code does not need to handle them.

## Nested Objects

```python
import os
import instructor
from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


client = instructor.from_provider("minimax/MiniMax-M2.7")

user = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[
        {
            "role": "user",
            "content": (
                "Extract: Jason is 25 years old. "
                "He lives at 123 Main St, New York, USA "
                "and has a summer house at 456 Beach Rd, Miami, USA."
            ),
        }
    ],
    response_model=User,
)

print(user)
#> User(
#>     name='Jason',
#>     age=25,
#>     addresses=[
#>         Address(street='123 Main St', city='New York', country='USA'),
#>         Address(street='456 Beach Rd', city='Miami', country='USA'),
#>     ]
#> )
```

## Available Models

| Model | Description |
|-------|-------------|
| `MiniMax-M2.7` | Full-size MiniMax model, GPT-4-class performance |
| `MiniMax-M2.7-highspeed` | Faster, lower-latency variant |

## Additional Resources

- [MiniMax API Documentation](https://www.minimax.io/docs)
- [MiniMax API Reference](https://www.minimax.io/docs/api-reference)
