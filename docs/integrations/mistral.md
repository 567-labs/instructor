---
draft: False
date: 2025-03-11
title: "Structured outputs with Mistral, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Mistral. Learn how to generate structured, type-safe outputs with Mistral."
slug: mistral
tags:
  - patching
authors:
  - shanktt
  - ivanleomk
---

# Structured outputs with Mistral, a complete guide w/ instructor

This guide demonstrates how to use Mistral with Instructor to generate structured outputs. You'll learn how to use function calling with Mistral Large to create type-safe responses.

Mistral Large is the flagship model from Mistral AI, supporting 32k context windows and functional calling abilities. Mistral Large's addition of [function calling](https://docs.mistral.ai/guides/function-calling/) makes it possible to obtain structured outputs using JSON schema.

## Quick Start

To get started with Instructor and Mistral, you'll need to install the required packages:

```bash
pip install "instructor[mistral]"
```

⚠️ **Important**: You must set your Mistral API key by setting it explicitly on the client

```python
try:
    from mistralai.client import Mistral  # mistralai 2.x
except ImportError:
    from mistralai import Mistral  # mistralai 1.x

client = Mistral(api_key="your-api-key-here")
```

## Available Modes

Instructor provides two modes for working with Mistral:

1. `instructor.Mode.TOOLS`: Uses Mistral's function calling API to return structured outputs (default)
2. `instructor.Mode.JSON_SCHEMA`: Uses Mistral's structured output capabilities

To set the mode for your mistral client, simply use the code snippet below

```python
import instructor

# Initialize with API key
instructor_client = instructor.from_provider(
    "mistral/mistral-large-latest",
    mode=instructor.Mode.TOOLS,
)
```

## Simple User Example (Sync)

```python
from pydantic import BaseModel
import instructor
from instructor import Mode


class UserDetails(BaseModel):
    name: str
    age: int


# Initialize the client
instructor_client = instructor.from_provider(
    "mistral/mistral-large-latest",
    mode=Mode.TOOLS,
)

# Extract a single user
user = instructor_client.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Jason is 25 years old"}],
    temperature=0,
)

print(user)
# Output: UserDetails(name='Jason', age=25)
```

## Async Example

For asynchronous operations, use `async_client=True` with `from_provider()`:

```python
import asyncio
from pydantic import BaseModel
import instructor
from instructor import Mode


class User(BaseModel):
    name: str
    age: int


# Initialize the async client
instructor_client = instructor.from_provider(
    "mistral/mistral-large-latest",
    async_client=True,
    mode=Mode.TOOLS,
)


async def extract_user():
    user = await instructor_client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jack is 28 years old."}],
        temperature=0,
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)
# Output: User(name='Jack', age=28)
```

## Nested Example

You can also work with nested models:

```python
from pydantic import BaseModel
from typing import List
import instructor
from instructor import Mode


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]


# Initialize the client
instructor_client = instructor.from_provider(
    "mistral/mistral-large-latest",
    mode=Mode.TOOLS,
)

# Create structured output with nested objects
user = instructor_client.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """,
        }
    ],
    temperature=0,
)

print(user)
# Output:
# User(
#     name='Jason',
#     age=25,
#     addresses=[
#         Address(street='123 Main St', city='New York', country='USA'),
#         Address(street='456 Beach Rd', city='Miami', country='USA')
#     ]
# )
```

## Streaming Support

Instructor now supports streaming capabilities with Mistral! You can use both `create_partial` for incremental model building and `create_iterable` for streaming collections.

### Streaming Partial Responses

```python
from pydantic import BaseModel
import instructor
from instructor.dsl.partial import Partial


class UserExtract(BaseModel):
    name: str
    age: int


# Create an Instructor client for Mistral
instructor_client = instructor.from_provider("mistral/mistral-small-latest")

# Stream partial responses
model = instructor_client.create(
    response_model=Partial[UserExtract],
    stream=True,
    messages=[
        {"role": "user", "content": "Jason Liu is 25 years old"},
    ],
)

for partial_user in model:
    print(f"Received update: {partial_user}")
# Output might show:
# Received update: UserExtract(name='Jason', age=None)
# Received update: UserExtract(name='Jason Liu', age=None)
# Received update: UserExtract(name='Jason Liu', age=25)
```

### Streaming Iterable Collections

```python
from pydantic import BaseModel
import instructor


class UserExtract(BaseModel):
    name: str
    age: int


# Create an Instructor client for Mistral
instructor_client = instructor.from_provider("mistral/mistral-small-latest")

# Stream iterable responses
users = instructor_client.create_iterable(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Make up two people"},
    ],
)

for user in users:
    print(f"Generated user: {user}")
# Output:
# Generated user: UserExtract(name='Emily Johnson', age=32)
# Generated user: UserExtract(name='Michael Chen', age=28)
```

### Async Streaming

You can also use async versions of both streaming approaches:

```python
import asyncio
from pydantic import BaseModel
import instructor
from instructor.dsl.partial import Partial


class UserExtract(BaseModel):
    name: str
    age: int


instructor_client = instructor.from_provider(
    "mistral/mistral-small-latest",
    async_client=True,
)


async def stream_partial():
    model = await instructor_client.create(
        response_model=Partial[UserExtract],
        stream=True,
        messages=[
            {"role": "user", "content": "Jason Liu is 25 years old"},
        ],
    )

    async for partial_user in model:
        print(f"Received update: {partial_user}")


async def stream_iterable():
    users = instructor_client.create_iterable(
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Make up two people"},
        ],
    )

    async for user in users:
        print(f"Generated user: {user}")


# Run async functions
asyncio.run(stream_partial())
asyncio.run(stream_iterable())
```

## Related Resources

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Mistral Function Calling Guide](https://docs.mistral.ai/guides/function-calling/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest Mistral API versions and models. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates on Mistral integration features.

## Multimodal

Instructor makes it easy to analyse and extract semantic information from PDFs using Mistral's models. Let's see an example below with the sample PDF above where we'll load it in using our `from_url` method. Note that for now Mistral only supports document URLs.

```python
from instructor.processing.multimodal import PDF
from pydantic import BaseModel
import instructor


class Receipt(BaseModel):
    total: int
    items: list[str]


client = instructor.from_provider("mistral/mistral-small-latest")

url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"

response = client.create(
    response_model=Receipt,
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                "Extract out the total and line items from the invoice",
                PDF.from_url(
                    url
                ),  # Also supports PDF.from_path() and PDF.from_base64()
            ],
        },
    ],
)

print(response)
#> Receipt(total=220, items=['English Tea', 'Tofu'])
```
