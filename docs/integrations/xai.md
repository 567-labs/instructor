---
title: "Structured outputs with xAI, a complete guide with instructor"
description: "Learn how to use Instructor with xAI's Grok models for type-safe, structured outputs. Complete guide with examples and best practices."
---

# Structured outputs with xAI, a complete guide with instructor

xAI provides access to Grok models through the `xai-sdk` package, enabling structured outputs with Instructor. This guide covers everything you need to know about using xAI's Grok models with Instructor for type-safe, validated responses.

## Quick Start

Instructor is distributed without xAI dependencies by default. The `xai` extra
requires Python 3.10 or later because it installs `xai-sdk`. Install xAI support
with the optional `xai` extra:

```bash
pip install "instructor[xai]"
```

Or using uv:

```bash
uv pip install "instructor[xai]"
```

⚠️ **Important**: You must set your xAI API key before using the client. You can do this in two ways:

1. Set the environment variable:

```bash
export XAI_API_KEY='your-api-key-here'
```

2. The xAI SDK will use this environment variable automatically.

## Simple User Example (Sync)

```python
import instructor
from pydantic import BaseModel

# Auto-configure xAI client
client = instructor.from_provider("xai/grok-4.20-reasoning")


class User(BaseModel):
    name: str
    age: int


# Create structured output
user = client.create(
    response_model=User,
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
)

print(user)
#> User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import instructor
from pydantic import BaseModel
import asyncio

# Auto-configure async xAI client
client = instructor.from_provider("xai/grok-4.20-reasoning", async_client=True)


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    user = await client.create(
        response_model=User,
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)
#> User(name='Jason', age=25)
```

## Nested Example

```python
from pydantic import BaseModel
from typing import List
import instructor


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]


# Auto-configure xAI client
client = instructor.from_provider("xai/grok-4.20-reasoning")

# Create structured output with nested objects
user = client.create(
    response_model=User,
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

## Instructor Modes

xAI supports the following modes:

1. `instructor.Mode.JSON` : Forces the model to return JSON output (default)
2. `instructor.Mode.TOOLS` : Uses function calling for structured outputs

```python
import instructor
from instructor import Mode

# Using JSON mode (default)
client = instructor.from_provider("xai/grok-4.20-reasoning", mode=Mode.JSON)

# Using TOOLS mode
client = instructor.from_provider("xai/grok-4.20-reasoning", mode=Mode.TOOLS)
```

## Available Models

These examples use `grok-4.20-reasoning`. Check the [current xAI model catalog](https://docs.x.ai/developers/models/grok-4.20-reasoning) for available aliases and capabilities.

## Limitations

### Streaming Support

⚠️ **Note**: Streaming responses (`create_iterable` and `create_partial`) are not yet supported due to differences in xAI's streaming API. See [issue #1663](https://github.com/567-labs/instructor/issues/1663) for updates.

### Python Version

⚠️ **Requires Python 3.10+**: The xAI SDK requires Python 3.10 or higher.

## Best Practices

### 1. API Key Management

Store your xAI API key securely using environment variables:

```bash
export XAI_API_KEY="your-api-key-here"
```

### 2. Model Selection

Evaluate extraction accuracy, latency, and cost on representative inputs before
changing models. A model's availability does not guarantee support for every
Instructor mode.

### 3. Error Handling

Always handle potential API errors gracefully:

```python
try:
    user = client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract user data"}],
    )
except Exception as e:
    print(f"Error: {e}")
```

## Common Use Cases

- Data Extraction from unstructured text
- Form parsing and validation
- Content classification
- Entity recognition
- Structured data generation

## Related Resources

- [xAI Documentation](https://docs.x.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest xAI SDK versions. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
