---
title: Anyscale
description: Guide to using instructor with Anyscale
---

# Structured outputs with Anyscale, a complete guide w/ instructor

[Anyscale](https://www.anyscale.com/) is a platform that provides access to various open-source LLMs like Mistral and Llama models. This guide shows how to use instructor with Anyscale to get structured outputs from these models.

## Quick Start

First, install the required packages:

```bash
pip install instructor
```

You'll need an Anyscale API key which you can set as an environment variable:

```bash
export ANYSCALE_API_KEY=your_api_key_here
```

## Basic Example

Here's how to extract structured data from Anyscale models:

```python
import instructor
from pydantic import BaseModel

# Initialize the client with Anyscale base URL
client = instructor.from_provider(
    "anyscale/Mixtral-8x7B-Instruct-v0.1",
    mode=instructor.Mode.JSON_SCHEMA,
)
class UserExtract(BaseModel):
    name: str
    age: int

# Extract structured data
user = client.create(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user)
# Output: UserExtract(name='Jason', age=25)
```

### Async Example

```python
import asyncio
import instructor
from pydantic import BaseModel

async_client = instructor.from_provider(
    "anyscale/Mixtral-8x7B-Instruct-v0.1",
    async_client=True,
    mode=instructor.Mode.JSON_SCHEMA,
)

class UserExtract(BaseModel):
    name: str
    age: int

async def fetch_user():
    return await async_client.create(
        messages=[{"role": "user", "content": "Extract jason is 25 years old"}],
        response_model=UserExtract,
    )

user = asyncio.run(fetch_user())
print(user)
```

## Supported Modes

Anyscale supports the following instructor modes:

- `Mode.TOOLS`
- `Mode.JSON`
- `Mode.JSON_SCHEMA`
- `Mode.MD_JSON`

## Models

Anyscale provides access to various models, including:

- Mistral models (e.g., `mistralai/Mixtral-8x7B-Instruct-v0.1`)
- Llama models
- Other open-source LLMs available through their platform

