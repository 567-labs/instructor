---
title: "Structured outputs with Cortex, a complete guide w/ instructor"
description: "Learn how to use Cortex with Instructor for structured outputs. Complete guide with examples and best practices."
---

# Structured outputs with Cortex

Cortex.cpp is a runtime that helps you run open source LLMs out of the box. It supports a wide variety of models and powers their [Jan](https://jan.ai) platform. This guide provides a quickstart on how to use Cortex with instructor for structured outputs.

## Quick Start

Instructor comes with support for the OpenAI client out of the box, so you don't need to install anything extra.

```bash
pip install "instructor"
```

Once you've done so, make sure to pull the model that you'd like to use. In this example, we'll be using a quantized llama3.2 model.

```bash
cortex run llama3.2:3b-gguf-q4-km
```

Let's start by initializing the client below - note that we need to provide a base URL and an API key here. The API key isn't important, it's just so the OpenAI client doesn't throw an error.

```python
import instructor

client = instructor.from_provider(
    "cortex/llama3.2:3b-gguf-q4-km",
    base_url="http://localhost:39281/v1",
    api_key="this is a fake api key that doesn't matter",
)
```

## Simple User Example (Sync)

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider(
    "cortex/llama3.2:3b-gguf-q4-km",
    base_url="http://localhost:39281/v1",
    api_key="this is a fake api key that doesn't matter",
)


class User(BaseModel):
    name: str
    age: int


resp = client.create(
    messages=[{"role": "user", "content": "Ivan is 27 and lives in Singapore"}],
    response_model=User,
)

print(resp)
# > name='Ivan', age=27
```

## Simple User Example (Async)

```python
import instructor
from pydantic import BaseModel
import asyncio

# Initialize with API key
client = instructor.from_provider(
    "cortex/llama3.2:3b-gguf-q4-km",
    async_client=True,
    base_url="http://localhost:39281/v1",
    api_key="this is a fake api key that doesn't matter",
)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.create(
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )
    return user

# Run async function
user = asyncio.run(extract_user())
print(user)
#> User(name='Jason', age=25)
```

## Nested Example

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider(
    "cortex/llama3.2:3b-gguf-q4-km",
    base_url="http://localhost:39281/v1",
    api_key="this is a fake api key that doesn't matter",
)


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


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

In this tutorial we've seen how we can run local models with Cortex while simplifying a lot of the logic around managing retries and function calling with our simple interface.

We'll be publishing a lot more content on Cortex and how to work with local models moving forward so do keep an eye out for that.

## Related Resources

- [Cortex Documentation](https://cortex.so/docs/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest OpenAI API versions and models. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
