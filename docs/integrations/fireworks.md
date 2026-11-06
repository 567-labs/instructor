---
title: "Structured outputs with Fireworks, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Fireworks AI models. Learn how to generate structured, type-safe outputs with high-performance, cost-effective AI capabilities."
---

# Structured outputs with Fireworks, a complete guide w/ instructor

Fireworks provides efficient and cost-effective AI models with enterprise-grade reliability. This guide shows you how to use Instructor with Fireworks's models for type-safe, validated responses.

## Quick Start

Install Instructor with Fireworks support:

```bash
pip install "instructor[fireworks-ai]"
```

## Simple User Example (Sync)

```python
from fireworks.client import Fireworks
import instructor
from pydantic import BaseModel

# Initialize the client
client = Fireworks()

# Enable instructor patches
client = instructor.from_provider("fireworks/accounts/fireworks/models/kimi-k2p5")


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract: Jason is 25 years old",
        }
    ],
    response_model=User,
)

print(user)
#> User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import instructor
from pydantic import BaseModel
import asyncio

client = instructor.from_provider(
    "fireworks/accounts/fireworks/models/kimi-k2p5",
    async_client=True,
)


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    user = await client.create(
        messages=[
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old",
            }
        ],
        response_model=User,
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)  # User(name='Jason', age=25)
```

## Nested Example

```python
import instructor
from pydantic import BaseModel

# Enable instructor patches
client = instructor.from_provider("fireworks/accounts/fireworks/models/kimi-k2p5")


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

### Partial Streaming Example

```python
import instructor
from pydantic import BaseModel

# Enable instructor patches
client = instructor.from_provider("fireworks/accounts/fireworks/models/kimi-k2p5")


class User(BaseModel):
    name: str
    age: int
    bio: str


user = client.create_partial(
    messages=[
        {
            "role": "user",
            "content": "Create a user profile for Jason + 1 sentence bio, age 25",
        },
    ],
    response_model=User,
)

for user_partial in user:
    print(user_partial)
    # name=None age=None bio=None
    # name='Jason' age=None bio=None
    # name='Jason' age=25 bio="When he's"
    # name='Jason' age=25 bio="When he's not working as a graphic designer, Jason can usually be found trying out new craft beers or attempting to cook something other than ramen noodles."
```

## Iterable Example

```python
import instructor
from pydantic import BaseModel

# Enable instructor patches
client = instructor.from_provider("fireworks/accounts/fireworks/models/kimi-k2p5")


class User(BaseModel):
    name: str
    age: int


# Extract multiple users from text
users = client.create_iterable(
    messages=[
        {
            "role": "user",
            "content": """
            Extract users:
            1. Jason is 25 years old
            2. Sarah is 30 years old
            3. Mike is 28 years old
        """,
        },
    ],
    response_model=User,
)

for user in users:
    print(user)

    # name='Jason' age=25
    # name='Sarah' age=30
    # name='Mike' age=28
```

## Instructor Modes

We provide several modes to make it easy to work with the different response models that Fireworks supports

1. `instructor.Mode.MD_JSON` : This parses the raw text completion into a pydantic object
2. `instructor.Mode.TOOLS` : This uses Fireworks's tool calling API to return structured outputs to the client

## Related Resources

- [Fireworks Documentation](https://docs.fireworks.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Fireworks's latest API versions. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.

Note: Always verify model-specific features and limitations before implementing streaming functionality in production environments.
