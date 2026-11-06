---
title: Structured outputs with Cohere, a complete guide w/ instructor
description: Learn how to leverage Cohere's command models with Python's instructor library for structured data outputs.
---

# Structured outputs with Cohere, a complete guide w/ instructor

This guide demonstrates how to use Cohere with Instructor to generate structured outputs. You'll learn how to use Cohere's command models to create type-safe responses.

You can now use any of the Cohere's [command models](https://docs.cohere.com/docs/models) with the `instructor` library to get structured outputs.

You'll need a cohere API key which can be obtained by signing up [here](https://dashboard.cohere.com/) and gives you [free](https://cohere.com/pricing), rate-limited usage for learning and prototyping.

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Document Segmentation](../examples/document_segmentation.md) - Cohere example for document processing
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers

# Cohere V2 API Support

As of version 1.12.0, Instructor supports both Cohere V1 and V2 SDK clients. The V2 API provides an OpenAI-compatible interface with support for the latest Cohere models.

**Key differences:**
- **V2 API** (recommended): Uses `cohere.ClientV2` / `cohere.AsyncClientV2` with OpenAI-compatible message format
- **V1 API** (legacy): Uses `cohere.Client` / `cohere.AsyncClient` with Cohere-specific message format

The V2 API is recommended for new projects as it provides better compatibility with the OpenAI SDK interface and supports the latest models like `command-a-03-2025`.

## Setup

```
pip install "instructor[cohere]"

```

This installs `cohere>=5.1.8`, which includes both V1 and V2 client support.

Export your key:

```
export CO_API_KEY=<YOUR_COHERE_API_KEY>
```

## Example (V2 API - Recommended)

The easiest way to use Cohere with Instructor is through the `from_provider` factory, which automatically uses the V2 API:

```python
from pydantic import BaseModel, Field
from typing import List
import instructor

# Using from_provider automatically uses Cohere V2 API
client = instructor.from_provider(
    "cohere/command-a-03-2025",
    max_tokens=1000,
)


class Person(BaseModel):
    name: str = Field(description="name of the person")
    country_of_origin: str = Field(description="country of origin of the person")


class Group(BaseModel):
    group_name: str = Field(description="name of the group")
    members: List[Person] = Field(description="list of members in the group")


task = """\
Given the following text, create a Group object for 'The Beatles' band

Text:
The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time. The group were integral to the development of 1960s counterculture and popular music's recognition as an art form.
"""
group = client.create(
    response_model=Group,
    messages=[{"role": "user", "content": task}],
    temperature=0,
)

print(group.model_dump_json(indent=2))
"""
{
  "group_name": "The Beatles",
  "members": [
    {
      "name": "John Lennon",
      "country_of_origin": "England"
    },
    {
      "name": "Paul McCartney",
      "country_of_origin": "England"
    },
    {
      "name": "George Harrison",
      "country_of_origin": "England"
    },
    {
      "name": "Ringo Starr",
      "country_of_origin": "England"
    }
  ]
}
"""
```

### Async Example

```python
import instructor

async_client = instructor.from_provider(
    "cohere/command-a-03-2025",
    async_client=True,
    max_tokens=1000,
)
```

## Using Cohere SDK Directly

You can also explicitly create a Cohere client and patch it with Instructor:

### V2 API (Recommended)

```python
import instructor

# Use from_provider for simplified setup
client = instructor.from_provider(
    "cohere/command-a-03-2025", mode=instructor.Mode.TOOLS
)

# Now use it with structured outputs
response = client.create(
    response_model=YourModel,
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "Extract..."}],
)
```

### V1 API (Legacy Support)

The V1 API is still supported for backward compatibility:

```python
import instructor

# Use from_provider for simplified setup (works with both V1 and V2)
client = instructor.from_provider(
    "cohere/command-a-03-2025", mode=instructor.Mode.TOOLS
)

# V1 uses different message format internally but instructor handles the conversion
response = client.create(
    response_model=YourModel,
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "Extract..."}],
)
```

**Note**: Instructor automatically detects whether you're using V1 or V2 client and handles message format conversion accordingly. The V2 API uses OpenAI-compatible message format (`messages`), while V1 uses Cohere's legacy format (`message` + `chat_history`).
