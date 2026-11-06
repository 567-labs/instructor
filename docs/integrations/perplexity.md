---
title: Structured Outputs with Perplexity AI and Pydantic
description: Learn how to use Perplexity AI with Instructor for structured JSON outputs using Pydantic models. Create type-safe, validated responses from Perplexity's Sonar models with Python.
---

# Structured Outputs with Perplexity AI

This guide demonstrates how to use Perplexity AI with Instructor to generate structured outputs. You'll learn how to use Perplexity's Sonar models with Pydantic to create type-safe, validated responses.

## Prerequisites

You'll need to sign up for a Perplexity account and get an API key. You can do that [here](https://www.perplexity.ai/).

```bash
export PERPLEXITY_API_KEY=<your-api-key-here>
pip install "instructor[perplexity]"
```

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers
- [Search Examples](../examples/search.md) - Search query processing examples

# Perplexity AI

Perplexity AI provides access to powerful language models through their API. Instructor supports structured outputs with Perplexity's models using the OpenAI-compatible API.

### Sync Example

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider(
    "perplexity/sonar",
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.create(
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
#> User(name='Jason', age=25)
```

### Async Example

```python
import instructor
from pydantic import BaseModel
import asyncio

async_client = instructor.from_provider(
    "perplexity/sonar",
    async_client=True,
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

### Nested Objects

```python
import os
import instructor
from pydantic import BaseModel

# Initialize with API key
client = instructor.from_provider(
    "perplexity/sonar",
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)


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
#>         Address(street='456 Beach Rd', city='Miami', country='USA')
#>     ]
#> )
```

## Supported Modes

Perplexity AI currently supports the following mode with Instructor:

- `PERPLEXITY_JSON`: Direct JSON response generation

```python
import os
import instructor
from pydantic import BaseModel

# Initialize client with base URL
client = instructor.from_provider(
    "perplexity/sonar",
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.create(
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
#> User(name='Jason', age=25)
```

## Additional Resources

- [Perplexity API Documentation](https://docs.perplexity.ai/)
- [Perplexity API Reference](https://docs.perplexity.ai/reference/post_chat_completions)