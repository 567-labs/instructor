---
title: "Structured outputs with Hubris, a complete guide with instructor"
description: "Learn how to use Instructor with Hubris, an OpenAI-compatible gateway that serves Anthropic, OpenAI, Google, DeepSeek, Moonshot and xAI models behind one API key."
---

# Structured outputs with Hubris, a complete guide with instructor

[Hubris](https://hubris.pw) is an OpenAI-compatible gateway that serves models from Anthropic, OpenAI, Google, DeepSeek, Moonshot, xAI, Z.ai and Qwen behind a single API key. It is aimed at users in Russia, where several of those vendors are not directly reachable, and bills in Russian roubles.

Because the wire format is plain OpenAI, Instructor works through the OpenAI client with nothing extra to install.

## Quick Start

Create an API key at [hubris.pw/keys](https://hubris.pw/keys) and export it:

```bash
export HUBRIS_API_KEY=sk-gw-...
```

Model IDs take the form `vendor/model` and are resolved exactly, with no aliasing, so pass the full ID. The catalogue is public — you can browse it without a key at [api.hubris.pw/v1/models](https://api.hubris.pw/v1/models).

## Simple User Example (Sync)

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("hubris/openai/gpt-5.6-luna")

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
#> User(name='Ivan', age=28)
```

The provider prefix is split on the first slash only, so `hubris/openai/gpt-5.6-luna` selects the Hubris provider and the `openai/gpt-5.6-luna` model.

## Simple User Example (Async)

```python
import asyncio
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "hubris/anthropic/claude-sonnet-5",
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
#> User(name='Ivan', age=28)
```

## Bring your own client

If you would rather construct the OpenAI client yourself:

```python
from openai import OpenAI
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_hubris(
    OpenAI(
        api_key="sk-gw-...",
        base_url="https://api.hubris.pw/v1",
    ),
    model="google/gemini-3.7-flash",
)

print(
    client.create(
        messages=[{"role": "user", "content": "Ivan is 28 years old"}],
        response_model=User,
    )
)
#> User(name='Ivan', age=28)
```

## Nested Example

```python
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


client = instructor.from_provider("hubris/openai/gpt-5.6-luna")

user = client.create(
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """,
        },
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
#>     ],
#> )
```

## Streaming Support

Partial and iterable streaming both work over the standard OpenAI streaming format:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("hubris/openai/gpt-5.6-luna")

for partial in client.create_partial(
    messages=[{"role": "user", "content": "Ivan is 28 years old"}],
    response_model=User,
):
    print(partial)
#> User(name=None, age=None)
#> User(name='Ivan', age=None)
#> User(name='Ivan', age=28)
```

## Supported Modes

| Mode | Supported | Notes |
|------|-----------|-------|
| `Mode.TOOLS` | ✅ | Default. Standard OpenAI tool calling. |
| `Mode.JSON_SCHEMA` | ✅ | `response_format` with a JSON schema. |
| `Mode.MD_JSON` | ✅ | Prompt-based JSON, works on every model. |
| `Mode.PARALLEL_TOOLS` | ✅ | Several tool calls in one response. |
| `Mode.RESPONSES_TOOLS` | ❌ | Not wired up for this provider. |

Not every model in the catalogue supports every mode. Each entry in [`GET /v1/models`](https://api.hubris.pw/v1/models) carries a `supported_parameters` list — look for `tools` and `structured_outputs` before choosing a mode.

## Choosing a model

| Model | Good for |
|-------|----------|
| `google/gemini-3.7-flash` | cheap, fast extraction at volume |
| `openai/gpt-5.6-luna` | general-purpose structured output |
| `anthropic/claude-sonnet-5` | long documents, images, harder schemas |
| `deepseek/deepseek-v4-pro` | inexpensive reasoning and code |

## Related Resources

- [Hubris documentation](https://hubris.pw/docs)
- [Model catalogue](https://hubris.pw/models)
- [Instructor's core concepts](../concepts/index.md)

## Updates and Compatibility

Instructor reaches Hubris through the OpenAI client, so it inherits OpenAI SDK compatibility. New models appear in the catalogue without an Instructor release.
