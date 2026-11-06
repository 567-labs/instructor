---
authors:
- ivanleomk
categories:
- LLM Techniques
comments: true
date: 2024-09-03
description: Introducing structured outputs for Gemini tool calling support in the
  instructor library, enhancing interactions with Gemini and VertexAI SDKs.
draft: false
tags:
- Gemini
- VertexAI
- Tool Calling
- Instructor Library
- AI SDKs
---

# Structured Outputs for Gemini now supported

We're excited to announce that `instructor` now supports structured outputs using tool calling for both the Gemini SDK and the VertexAI SDK.

A special shoutout to [Sonal](https://x.com/sonalsaldanha) for his contributions to the Gemini Tool Calling support.

Let's walk through a simple example of how to use these new features

## Installation

To get started, install the latest version of `instructor`. Depending on whether you're using Gemini or VertexAI, you should install the following:

=== "Gemini"

    ```bash
    pip install "instructor[google-generativeai]"
    ```

=== "VertexAI"

    ```bash
    pip install "instructor[vertexai]"
    ```

This ensures that you have the necessary dependencies to use the Gemini or VertexAI SDKs with instructor.

We recommend using the Gemini SDK over the VertexAI SDK for two main reasons.

1. Compared to the VertexAI SDK, the Gemini SDK comes with a free daily quota of 1.5 billion tokens to use for developers.
2. The Gemini SDK is significantly easier to setup, all you need is a `GOOGLE_API_KEY` that you can generate in your GCP console. THe VertexAI SDK on the other hand requires a credentials.json file or an OAuth integration to use.

## Getting Started

With our provider agnostic API, you can use the same interface to interact with both SDKs, the only thing that changes here is how we initialise the client itself.

Before running the following code, you'll need to make sure that you have your Gemini API Key set in your shell under the alias `GOOGLE_API_KEY`.

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("google/gemini-2.5-flash")

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

print(resp)
#> name='Jason' age=25
```

1. Current Gemini models that support tool calling are `gemini-3-flash` and `gemini-1.5-pro-latest`.

We can achieve a similar thing with the VertexAI SDK. For this to work, you'll need to authenticate to VertexAI.

There are some instructions [here](https://cloud.google.com/vertex-ai/docs/authentication) but the easiest way I found was to simply download the GCloud cli and run `gcloud auth application-default login`.

```python
import instructor
import vertexai  # type: ignore
from pydantic import BaseModel

vertexai.init()


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("google/gemini-2.5-flash", vertexai=True)  # (1)!


resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

print(resp)
#> name='Jason' age=25
```

1. Current Gemini models that support tool calling are `gemini-3-flash` and `gemini-1.5-pro-latest`.
