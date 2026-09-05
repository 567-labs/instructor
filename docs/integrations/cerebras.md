---
title: "Structured outputs with Cerebras, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Cerebras's hardware-accelerated AI models. Learn how to generate structured, type-safe outputs with high-performance computing."
---

# Structured outputs with Cerebras, a complete guide w/ instructor

Cerebras provides hardware-accelerated AI models optimized for high-performance computing environments. This guide shows you how to use Instructor with Cerebras's models for type-safe, validated responses.

## Quick Start

Install Instructor with Cerebras support:

```bash
pip install "instructor[cerebras_cloud_sdk]"
```

## Cerebras request parameters

Instructor forwards additional keyword arguments from `client.create(...)` to
the Cerebras SDK. This lets you use Cerebras request parameters without falling
back to a generic OpenAI client. For example:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("cerebras/gpt-oss-120b")

resp = client.create(
    messages=[{"role": "user", "content": "Return a short user record."}],
    response_model=User,
    max_completion_tokens=256,
    reasoning_effort="none",
    seed=42,
)
```

Use `max_completion_tokens` for the canonical output limit. The compatibility
alias `max_tokens` is forwarded separately, so do not send both fields in the
same request. Other Cerebras parameters, including `temperature`, `top_p`,
`stop`, `response_format`, `parallel_tool_calls`, log probabilities, and
penalties, can be supplied in the same way when supported by the selected
model. See the [Cerebras API reference](https://inference-docs.cerebras.ai/api-reference/chat-completions)
for the current request schema.

## Simple User Example (Sync)

```python
import instructor
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel

client = instructor.from_provider("cerebras/gpt-oss-120b")

class User(BaseModel):
    name: str
    age: int

# Create structured output
resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract the name and age of the person in this sentence: John Smith is 29 years old.",
        }
    ],
    response_model=User,
)

print(resp)
#> User(name='John Smith', age=29)
```

## Simple User Example (Async)

```python
import instructor
from pydantic import BaseModel
import asyncio

client = instructor.from_provider(
    "cerebras/gpt-oss-120b",
    async_client=True,
)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    resp = await client.create(
        messages=[
            {
                "role": "user",
                "content": "Extract the name and age of the person in this sentence: John Smith is 29 years old.",
            }
        ],
        response_model=User,
    )
    return resp

# Run async function
resp = asyncio.run(extract_user())
print(resp)
#> User(name='John Smith', age=29)
```

## Nested Example

```python
from pydantic import BaseModel
import instructor
from cerebras.cloud.sdk import Cerebras

client = instructor.from_provider("cerebras/gpt-oss-120b")


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


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
        }
    ],
    response_model=User,
)

print(user)
#> {
#>     'name': 'Jason',
#>     'age': 25,
#>     'addresses': [
#>         {
#>             'street': '123 Main St',
#>             'city': 'New York',
#>             'country': 'USA'
#>         },
#>         {
#>             'street': '456 Beach Rd',
#>             'city': 'Miami',
#>             'country': 'USA'
#>         }
#>     ]
#> }
```

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

We currently support partial streaming for Cerebras by parsing the raw text completion. We have not implemented streaming for function calling at this point in time yet. Please make sure you have `mode=instructor.Mode.MD_JSON` set when using partial streaming.

```python
import instructor
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel
from typing import Iterable

client = instructor.from_provider(
    "cerebras/gpt-oss-120b",
    mode=instructor.Mode.MD_JSON,
)


class Person(BaseModel):
    name: str
    age: int


resp = client.create_partial(
    messages=[
        {
            "role": "user",
            "content": "Ivan is 27 and lives in Singapore",
        }
    ],
    response_model=Person,
    stream=True,
)

for person in resp:
    print(person)
    # > name=None age=None
    # > name='Ivan' age=None
    # > name='Ivan' age=27

```

## Iterable Example

```python
import instructor
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel
from typing import Iterable

client = instructor.from_provider(
    "cerebras/gpt-oss-120b",
    mode=instructor.Mode.MD_JSON,
)


class Person(BaseModel):
    name: str
    age: int


resp = client.create_iterable(
    messages=[
        {
            "role": "user",
            "content": "Extract all users from this sentence : Chris is 27 and lives in San Francisco, John is 30 and lives in New York while their college roommate Jessica is 26 and lives in London",
        }
    ],
    response_model=Person,
    stream=True,
)

for person in resp:
    print(person)
    # > Person(name='Chris', age=27)
    # > Person(name='John', age=30)
    # > Person(name='Jessica', age=26)

```

## Instructor Hooks

Instructor provides several hooks to customize behavior:

### Validation Hook

```python
from instructor import Instructor

def validation_hook(value, retry_count, exception):
    print(f"Validation failed {retry_count} times: {exception}")
    return retry_count < 3  # Retry up to 3 times

instructor.patch(client, validation_hook=validation_hook)
```

## Instructor Modes

We provide several modes to make it easy to work with the different response models that Cerebras Supports

1. `instructor.Mode.MD_JSON` : This parses the raw completions as a valid JSON object.
2. `instructor.Mode.TOOLS` : This uses Cerebras's tool calling mode to return structured outputs to the client.

In general, we recommend using `Mode.TOOLS` because it's the most flexible and future-proof mode. It has the largest set of features that you can specify your schema in and makes things significantly easier to work with.
