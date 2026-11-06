---
draft: False
date: 2024-02-08
title: "Structured outputs with Ollama, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Ollama. Learn how to generate structured, type-safe outputs with Ollama."
slug: ollama
tags:
  - patching
  - open source
authors:
  - jxnl
---

# Structured outputs with Ollama, a complete guide w/ instructor

This guide demonstrates how to use Ollama with Instructor to generate structured outputs. You'll learn how to use JSON schema mode with local LLMs to create type-safe responses.

Open-source LLMS are gaining popularity, and the release of Ollama's OpenAI compatibility later it has made it possible to obtain structured outputs using JSON schema.

By the end of this blog post, you will learn how to effectively utilize instructor with ollama. But before we proceed, let's first explore the concept of patching.

<!-- more -->

## Patching

Instructor's patch enhances a openai api it with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy
- `timeout` parameter for controlling total retry duration (especially important for Ollama)

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## Timeout Handling with Ollama

Ollama integration now properly supports timeout parameters to ensure reliable request handling:

```python
from pydantic import BaseModel
import instructor

class Character(BaseModel):
    name: str
    age: int

client = instructor.from_provider(
    "ollama/llama2",
    mode=instructor.Mode.JSON,
)

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me about Harry Potter",
        }
    ],
    response_model=Character,
    max_retries=2,
    timeout=10.0,  # Total timeout across all retry attempts
)
```

The timeout parameter ensures that:

- **Total timeout control**: Limits the total time spent across all retry attempts, not per individual attempt
- **Ollama compatibility**: Prevents timeout issues where retries would multiply the total wait time
- **Predictable behavior**: A 3-second timeout stays 3 seconds total, not 9+ seconds when retrying

!!! tip "Timeout Best Practices"

    When using Ollama, especially with larger models, set appropriate timeout values based on your model's response time. The timeout applies to the total retry duration, making response times more predictable.

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [Ollama Examples](../examples/ollama.md) - Practical Ollama examples
- [Open Source Models](../examples/open_source.md) - More open-source model examples
- [Local Deployment](../examples/index.md#local-deployment) - Local model deployment guide

# Ollama

Start by downloading [Ollama](https://ollama.ai/download), and then pull a model such as Llama 2 or Mistral.

!!! tip "Make sure you update your `ollama` to the latest version!"

```
ollama pull llama2
```

## Quick Start with Auto Client

You can use Ollama with Instructor's auto client for a simple setup:

```python
import instructor
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int

# Simple setup - automatically configured for Ollama
client = instructor.from_provider("ollama/llama2")

resp = client.create(
    messages=[{"role": "user", "content": "Tell me about Harry Potter"}],
    response_model=Character,
)
```

### Async Example

```python
import instructor
from pydantic import BaseModel
import asyncio

async_client = instructor.from_provider(
    "ollama/llama2",
    async_client=True,
)

class Character(BaseModel):
    name: str
    age: int

async def get_character():
    return await async_client.create(
        messages=[{"role": "user", "content": "Tell me about Harry Potter"}],
        response_model=Character,
    )

print(asyncio.run(get_character()))
```

### Intelligent Mode Selection

The auto client automatically selects the best mode based on your model:

- **Function Calling Models** (llama3.1, llama3.2, llama4, mistral-nemo, qwen2.5, etc.): Uses `TOOLS` mode for enhanced function calling support
- **Other Models**: Uses `JSON` mode for structured output

```python
# These models automatically use TOOLS mode
client = instructor.from_provider("ollama/llama3.1")
client = instructor.from_provider("ollama/qwen2.5")

# Other models use JSON mode
client = instructor.from_provider("ollama/llama2")
```

You can also override the mode manually:

```python
import instructor

# Force JSON mode
client = instructor.from_provider("ollama/llama3.1", mode=instructor.Mode.JSON)

# Force TOOLS mode
client = instructor.from_provider("ollama/llama2", mode=instructor.Mode.TOOLS)
```

## Manual Setup

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

import instructor


class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


# enables `response_model` in create call
client = instructor.from_provider(
    "ollama/llama2",
    mode=instructor.Mode.JSON,
)

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me about the Harry Potter",
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))
"""
{
  "name": "Harry James Potter",
  "age": 37,
  "fact": [
    "He is the chosen one.",
    "He has a lightning-shaped scar on his forehead.",
    "He is the son of James and Lily Potter.",
    "He attended Hogwarts School of Witchcraft and Wizardry.",
    "He is a skilled wizard and sorcerer.",
    "He fought against Lord Voldemort and his followers.",
    "He has a pet owl named Snowy."
  ]
}
"""
```
