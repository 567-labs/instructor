---
title: "Structured outputs with LiteLLM, a complete guide w/ instructor"
description: "Complete guide to using Instructor with LiteLLM's unified interface. Learn how to generate structured, type-safe outputs across multiple LLM providers."
---

# Structured outputs with LiteLLM, a complete guide w/ instructor

LiteLLM provides a unified interface for multiple LLM providers, making it easy to switch between different models and providers. This guide shows you how to use Instructor with LiteLLM for type-safe, validated responses across various LLM providers.

## Quick Start

Install Instructor with LiteLLM support:

```bash
pip install "instructor[litellm]"
```

## Simple User Example (Sync)

```python
from litellm import completion
import instructor
from pydantic import BaseModel

# Enable instructor patches
client = instructor.from_provider("litellm/gpt-5.4-mini")

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

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import instructor
from pydantic import BaseModel
import asyncio

client = instructor.from_provider(
    "litellm/gpt-5.4-mini",
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
print(user)  # User(name='Jason', age=25)

```

## Cost Calculation

In order to calculate the cost of the response, LiteLLM provides a simple `response_cost` attribute on the response object's `_hidden_params` attribute. This is recorded in their documentation [here](https://docs.litellm.ai/docs/completion/token_usage#6-completion_cost).

Here is a code snippet using instructor to calculate the cost of the response:

```python
import instructor
from litellm import completion
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("litellm/gpt-5.4-mini")
instructor_resp, raw_completion = client.create_with_completion(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

print(raw_completion._hidden_params["response_cost"])
#> 0.00189
```

## Related Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with LiteLLM's latest releases. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.

Note: Always verify provider-specific features and limitations in their respective documentation before implementation.
