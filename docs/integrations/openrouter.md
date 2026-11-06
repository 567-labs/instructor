---
title: "Structured outputs with OpenRouter, a complete guide with instructor"
description: "Learn how to use Instructor with OpenRouter to access multiple LLM providers through a unified API. Get type-safe, structured outputs from various models including Qwen, Gemini, Mistral, and Cohere."
---

# Structured outputs with OpenRouter, a complete guide with instructor

OpenRouter provides a unified API to access multiple LLM providers, allowing you to easily switch between different models. This guide shows you how to use Instructor with OpenRouter for type-safe, validated responses across various LLM providers.

To set Provider specific configuration on the `openai` client, make sure to use the `extra_body` kwarg.

## Quick Start

⚠️ **Important**: Make sure that the model you're using has support for `Tool Calling` and/or `Structured Outputs` in the [OpenRouter models listing](https://openrouter.ai/models)

Instructor works with OpenRouter through the OpenAI client, so you don't need to install anything extra beyond the base package.

## Simple User Example (Sync)

We support simple tool calling with this

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "openrouter/google/gemini-3.8-flash",
    base_url="https://openrouter.ai/api/v1",
    async_client=False,
)

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Ivan is 28 years old",
        },
    ],
    response_model=User,
    extra_body={"provider": {"require_parameters": True}},
)

print(resp)
#> name='Ivan' age=20
```

## Simple User Example ( Async )

```python
import instructor
from pydantic import BaseModel
import asyncio


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "openrouter/google/gemini-3.8-flash",
    async_client=True,
)


async def extract_user():
    user = await client.create(
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
        extra_body={"provider": {"require_parameters": True}},
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)
```

## Nested Object Example ( Sync )

```python
from pydantic import BaseModel
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


# Initialize with API key
# Initialize client with base URL
client = instructor.from_provider(
    "openrouter/google/gemini-3.8-flash",
    base_url="https://openrouter.ai/api/v1",
    async_client=False,
)

# Create structured output with nested objects
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
    extra_body={"provider": {"require_parameters": True}},
    response_model=User,
)

print(user)
#> name='Jason' age=25 addresses=[Address(street='123 Main St', city='New York', country='USA'), Address(street='456 Beach Rd', city='Miami', country='USA')]
```

## Structured Outputs (Sync)

⚠️ **Important**: Check that your chosen model supports `Structured Outputs` in the [OpenRouter models listing](https://openrouter.ai/models). Structured Outputs is a subset of Tool Calling that constrains the model's output to match your schema in order to produce valid JSON Schema.

Instructor also supports Structured Outputs with OpenRouter as documented in their API [here](https://openrouter.ai/docs/features/structured-outputs). Note that the following User model will throw an error if we use the OpenAI GPT-4o model like `openai/gpt-4o-2024-11-20` because OpenAI does not support using a regex pattern as part of their structured output schema.

```python
from pydantic import BaseModel, Field
import instructor


class User(BaseModel):
    name: str
    age: int
    phone_number: str = Field(
        pattern=r"^\+?1?\s*\(?(\d{3})\)?[-.\s]*(\d{3})[-.\s]*(\d{4})$"
    )


# Initialize with API key
# Initialize client with base URL
client = instructor.from_provider(
    "openrouter/google/gemini-3.8-flash",
    base_url="https://openrouter.ai/api/v1",
    async_client=False,
)

# Create structured output with nested objects
user = client.create(
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old and his number is 1-212-456-7890
        """,
        },
    ],
    response_model=User,
    extra_body={"provider": {"require_parameters": True}},
)

print(user)
#> name='Jason' age=25 phone_number='+1 (212) 456-7890'
```

## JSON Mode

In the event that your model doesn't support tool calling, you will see the following error when you try to use `mode.TOOLS`

> instructor.exceptions.InstructorRetryException: Error code: 404 - {'error': {'message': 'No endpoints found that support tool use. To learn more about provider routing, visit: https://openrouter.ai/docs/provider-routing', 'code': 404}}

In this case, we recommend using the `JSON` mode instead as seen below.

```python
from pydantic import BaseModel, Field
import instructor


class User(BaseModel):
    name: str
    age: int
    phone_number: str = Field(
        pattern=r"^\+?1?\s*\(?(\d{3})\)?[-.\s]*(\d{3})[-.\s]*(\d{4})$"
    )


# Initialize with API key
# Initialize client with base URL
client = instructor.from_provider(
    "openrouter/google/gemini-3.8-flash",
    base_url="https://openrouter.ai/api/v1",
    async_client=False,
)

# Create structured output with nested objects
user = client.create(
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old and his number is 1-212-456-7890
        """,
        },
    ],
    response_model=User,
)

print(user)
```

## Streaming

You can also use streaming with as seen below using the `create_partial` method. While we're using JSON mode here, this should work with tool calling and structured outputs too.

```python
from pydantic import BaseModel
import instructor


class User(BaseModel):
    name: str
    age: int


# Initialize with API key
# Initialize client with base URL
client = instructor.from_provider(
    "openrouter/google/gemini-3.8-flash",
    base_url="https://openrouter.ai/api/v1",
)

# Create structured output with nested objects
user = client.create_partial(
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old and his number is 1-212-456-7890
        """,
        },
    ],
    response_model=User,
)

for chunk in user:
    print(chunk)
    #> name=None age=None
    #> name='Jason' age=None
    #> name='Jason' age=25
```
