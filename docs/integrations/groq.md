---
title: Structured Outputs with Groq AI and Pydantic
description: Learn how to use Groq AI for structured outputs with Pydantic in Python and enhance API interactions.
---

# Structured Outputs with Groq AI

This guide demonstrates how to use Groq AI with Instructor to generate structured outputs. You'll learn how to use Groq's LLM models to create type-safe responses.

you'll need to sign up for an account and get an API key. You can do that [here](https://console.groq.com/docs/quickstart).

```bash
export GROQ_API_KEY=<your-api-key-here>
pip install "instructor[groq]"
```

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [Groq Examples](../examples/groq.md) - Practical Groq examples
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers

# Groq AI

Groq supports structured outputs with their new `llama-3-groq-70b-8192-tool-use-preview` model.

### Sync Example

```python
import os
from groq import Groq
import instructor
from pydantic import BaseModel

# Initialize with API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Enable instructor patches for Groq client
client = instructor.from_provider("groq/llama-3.3-70b-versatile")


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

# Initialize async client using provider string
client = instructor.from_provider(
    "groq/llama-3.3-70b-versatile",
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

### Nested Object

```python
import os
from groq import Groq
import instructor
from pydantic import BaseModel

# Initialize with API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Enable instructor patches for Groq client
client = instructor.from_provider("groq/llama-3.3-70b-versatile")


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
