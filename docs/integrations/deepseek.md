---
title: "Structured outputs with DeepSeek, a complete guide with instructor"
description: "Learn how to use Instructor with DeepSeek's models for type-safe, structured outputs."
---

# Structured outputs with DeepSeek, a complete guide with instructor

DeepSeek is a Chinese company that provides AI models and services. They're most notable for the deepseek coder and chat model and most recently, the R1 reasoning model.

This guide covers everything you need to know about using DeepSeek with Instructor for type-safe, validated responses.

## Quick Start

Instructor comes with support for the OpenAI Client out of the box, so you don't need to install anything extra.

```bash
pip install "instructor"
```

⚠️ **Important**: You must set your DeepSeek API key before using the client. You can do this in two ways:

1. Set the environment variable:

```bash
export DEEPSEEK_API_KEY='your-api-key-here'
```

2. Or provide it directly to the client:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com"
)
```

## Simple User Example (Sync)

```python
from pydantic import BaseModel
import instructor

client = instructor.from_provider(
    "deepseek/deepseek-v4-flash",
    base_url="https://api.deepseek.com",
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
#> name='Jason' age=25
```

## Simple User Example (Async)

```python
import asyncio
from pydantic import BaseModel
import instructor

client = instructor.from_provider(
    "deepseek/deepseek-v4-flash",
    async_client=True,
    base_url="https://api.deepseek.com",
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
#> name='Jason' age=25
```

## Nested Example

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
client = instructor.from_provider(
    "deepseek/deepseek-v4-flash",
    base_url="https://api.deepseek.com",
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

### Partials

```python
from pydantic import BaseModel
import instructor
from pydantic import BaseModel

# Initialize with API key
client = instructor.from_provider(
    "deepseek/deepseek-v4-flash",
    base_url="https://api.deepseek.com",
)


class User(BaseModel):
    name: str
    age: int
    bio: str


user = client.create_partial(
    messages=[
        {
            "role": "user",
            "content": "Create a user profile for Jason and a one sentence bio, age 25",
        },
    ],
    response_model=User,
)

for user_partial in user:
    print(user_partial)


#> name='Jason' age=None bio='None'
#> name='Jason' age=25 bio='A tech'
#> name='Jason' age=25 bio='A tech enthusiast'
#> name='Jason' age=25 bio='A tech enthusiast who loves coding, gaming, and exploring new'
#> name='Jason' age=25 bio='A tech enthusiast who loves coding, gaming, and exploring new technologies'
```

### Iterable Example

```python
from pydantic import BaseModel
import instructor
from pydantic import BaseModel

# Initialize with API key
client = instructor.from_provider(
    "deepseek/deepseek-v4-flash",
    base_url="https://api.deepseek.com",
)


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

    #> name='Jason' age=25
    #> name='Sarah' age=30
    #> name='Mike' age=28
```

## Reasoning Models

Because Instructor is built on top of the OpenAI API, we can get our reasoning traces from the `deepseek-reasoner` model. Make sure to configure the `MD_JSON` mode here to get the best experience.

```python
from pydantic import BaseModel
import instructor
from rich import print

client = instructor.from_provider(
    "deepseek/deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    mode=instructor.Mode.MD_JSON,
)


class User(BaseModel):
    name: str
    age: int


# Create structured output
completion, raw_completion = client.create_with_completion(
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(completion)
#> User(name='Jason', age=25)
print(raw_completion.choices[0].message.reasoning_content)
#> Okay, let's see. The user wants me to extract information from the sentence "Jason is 25 years old" and format it into a JSON object that matches the given schema. The schema requires a "name" and an "age", both of which are required.
# >
#> First, I need to identify the name. The sentence starts with "Jason", so that's the name. Then the age is given as "25 years old". The age should be an integer, so I need to convert "25" from a string to a number.
# >
#> So putting that together, the JSON should have "name": "Jason" and "age": 25. Let me double-check the schema to make sure there are no other requirements. The properties are "name" (string) and "age" (integer), both required. Yep, that's all.
# >
#> I need to make sure the JSON is correctly formatted, with commas and braces. Also, the user specified to return it in a json codeblock, not the schema itself. So the final answer should be a JSON object with those key-value pairs.
```

## Instructor Modes

We suggest using the `Mode.Tools` mode for Deepseek which is the default when initializing via `from_provider`.

## Related Resources

- [DeepSeek Documentation](https://api-docs.deepseek.com/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest OpenAI API versions and models. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
