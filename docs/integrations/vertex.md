---
title: "Gemini on Vertex AI"
description: "Configure a Google Cloud project and location, migrate to google-genai, and stream Pydantic models from Gemini on Vertex AI with Instructor."
---

# Gemini structured outputs on Vertex AI with Python

Use the [Google Gen AI SDK](genai.md) with `vertexai=True` to run Gemini in a Google Cloud project. Configure [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc) and select a location where your model is available. For API key setup, see the [Gemini API guide](google.md).

## Quick Start

Install the Google Gen AI SDK and set [your project and location](#environment-variables) before running the examples.

```bash
pip install "instructor[google-genai]"
```

## Simple User Example (Sync)

```python
import instructor
from pydantic import BaseModel
import os

# Set your project ID and location
os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"


class User(BaseModel):
    name: str
    age: int


# Using from_provider (recommended)
client = instructor.from_provider(
    "google/gemini-3.8-flash",
    vertexai=True,
)

resp = client.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
)

print(resp)
#> User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import asyncio
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "google/gemini-3.8-flash",
    vertexai=True,
    async_client=True,
    mode=instructor.Mode.TOOLS,
)


async def extract_user():
    user = await client.create(
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=User,
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)  # User(name='Jason', age=25)
```

## Streaming Support

Instructor now supports streaming capabilities with Vertex AI! You can use both `create_partial` for incremental model building and `create_iterable` for streaming collections.

### Streaming Partial Responses

```python
import instructor
from pydantic import BaseModel
from instructor.dsl.partial import Partial


class UserExtract(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "google/gemini-3.8-flash",
    vertexai=True,
    mode=instructor.Mode.TOOLS,
)

# Stream partial responses
response_stream = client.create(
    response_model=Partial[UserExtract],
    stream=True,
    messages=[
        {"role": "user", "content": "Anibal is 23 years old"},
    ],
)

for partial_user in response_stream:
    print(f"Received update: {partial_user}")
# Output might show:
# Received update: UserExtract(name='Anibal', age=None)
# Received update: UserExtract(name='Anibal', age=23)
```

### Streaming Iterable Collections

```python
import instructor
from pydantic import BaseModel


class UserExtract(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "google/gemini-3.8-flash",
    vertexai=True,
    mode=instructor.Mode.TOOLS,
)

# Stream iterable responses
response_stream = client.create_iterable(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Make up two people"},
    ],
)

for user in response_stream:
    print(f"Generated user: {user}")
# Output:
# Generated user: UserExtract(name='Sarah Johnson', age=32)
# Generated user: UserExtract(name='David Chen', age=27)
```

### Async Streaming

You can also use async versions of both streaming approaches:

```python
import asyncio
import instructor
from pydantic import BaseModel
from instructor.dsl.partial import Partial


class UserExtract(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "google/gemini-3.8-flash",
    vertexai=True,
    async_client=True,
    mode=instructor.Mode.TOOLS,
)


async def stream_partial():
    response_stream = await client.create(
        response_model=Partial[UserExtract],
        stream=True,
        messages=[
            {"role": "user", "content": "Anibal is 23 years old"},
        ],
    )

    async for partial_user in response_stream:
        print(f"Received update: {partial_user}")


async def stream_iterable():
    response_stream = client.create_iterable(
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Make up two people"},
        ],
    )

    async for user in response_stream:
        print(f"Generated user: {user}")


# Run async functions
asyncio.run(stream_partial())
asyncio.run(stream_iterable())
```

## Related Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Migration to Google GenAI

Replace the deprecated `vertexai/<model>` prefix with `google/<model>` and pass `vertexai=True`. If you previously wrapped a Vertex AI client with `from_vertexai`, wrap a `google.genai.Client` with `from_genai` instead.

### New Way (Recommended)
```python
import instructor

# Option 1: Using from_provider (simplest)
client = instructor.from_provider(
    "google/gemini-3.8-flash",
    vertexai=True,
    project="your-project",  # Optional if set in environment
    location="us-central1",  # Optional if set in environment
)

# Option 2: Using from_genai with Google GenAI SDK
from google import genai
from instructor import from_genai

client = from_genai(
    genai.Client(
        vertexai=True,
        project="your-project",
        location="us-central1",
    ),
    model="gemini-3.8-flash",
)
```

### Environment Variables

You can also set these environment variables to avoid passing project/location each time:
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```
