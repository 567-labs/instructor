---
authors:
  - ivanleomk
categories:
  - Google
  - OpenAI
comments: true
date: 2024-11-10
description: Learn why Instructor remains essential even with Google's new OpenAI-compatible client for Gemini
draft: false
tags:
  - Gemini
---

# Do I Still Need Instructor with Google's New OpenAI Integration?

Google recently launched OpenAI client compatibility for Gemini.

While this is a significant step forward for developers by simplifying Gemini model interactions, **you absolutely still need instructor**.

If you're unfamiliar with instructor, we provide a simple interface to get structured outputs from LLMs across different providers.

This makes it easy to switch between providers, get reliable outputs from language models and ultimately build production grade LLM applications.

<!-- more -->

## The current state

The new integration provides an easy integration with the Open AI Client, this means that using function calling with Gemini models has become much easier. We don't need to use a gemini specific library like `vertexai` or `google.generativeai` anymore to define response models.

This looks something like this:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/", api_key="YOUR_API_KEY"
)

response = client.create(
    model="gemini-3-flash",
    messages=[{"role": "user", "content": "Extract name and age from: John is 30"}],
)
```

While this seems convenient, there are three major limitations that make `instructor` still essential:

### 1. Limited Schema Support

The current implementation only supports simple, single-level schemas. This means you can't use complex nested schemas that are common in real-world applications. For example, this won't work:

```python
class User(BaseModel):
    name: str
    age: int


class Users(BaseModel):
    users: list[User]  # Nested schema - will throw an error
```

### 2. No Streaming Support for Function Calling

The integration doesn't support streaming for function calling. This is a significant limitation if your application relies on streaming responses, which is increasingly common for:

- Real-time user interfaces
- Progressive rendering
- Long-running extractions

### 3. No Multimodal Support

Perhaps the biggest limitation is the lack of multimodal support. Gemini's strength lies in its ability to process multiple types of inputs (images, video, audio), but the OpenAI compatibility layer doesn't support this. This means you can't:

- Perform visual question answering
- Extract structured data from images
- Analyze video content
- Process audio inputs

## Why Instructor Remains Essential

Let's see how instructor solves these issues.

### 1. Easy Schema Management

It's easy to define and experiment with different response models when you're building your application up. In our [own experiments](./bad-schemas-could-break-llms.md), we found that changing a single field name from `final_choice` to `answer` improved model accuracy from 4.5% to 95%.

The way we structure and name fields in our response models can fundamentally alter how the model interprets and responds to queries. Manually editing schemas constrains your ability to iterate on your response models, introduces room for catastrophic errors and limits what you can squeeze out of your models.

You can get the full power of Pydantic with `instructor` with gemini using our `from_gemini` and `from_vertexai` integration instead of the limited support in the OpenAI integration.

### 2. Streaming Support

`instructor` provides built in support for streaming, allowing you to stream partial results as they're generated.

A common use case for streaming is to extract multiple items that have the same structure - Eg. extracting multiple users, extracting multiple products, extracting multiple events, etc.

This is relatively easy to do with `instructor`

```python
from instructor import from_openai
from openai import OpenAI
from instructor import Mode
from pydantic import BaseModel
import os

client = from_openai(
    OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/",
    ),
    mode=Mode.MD_JSON,
)


class User(BaseModel):
    name: str
    age: int


resp = client.create_iterable(
    model="gemini-3-flash",
    messages=[
        {
            "role": "user",
            "content": "Generate 10 random users",
        }
    ],
    response_model=User,
)

for r in resp:
    print(r)
# name='Alice' age=25
# name='Bob' age=32
# name='Charlie' age=19
# name='David' age=48
# name='Emily' age=28
# name='Frank' age=36
# name='Grace' age=22
# name='Henry' age=41
# name='Isabella' age=30
# name='Jack' age=27
```

If you want to instead stream out an item as it's being generated, you can do so by using the `create_partial` method instead

```python
from instructor import from_openai
from openai import OpenAI
from instructor import Mode
from pydantic import BaseModel
import os

client = from_openai(
    OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/",
    ),
    mode=Mode.MD_JSON,
)


class Story(BaseModel):
    title: str
    summary: str


resp = client.create_partial(
    model="gemini-3-flash",
    messages=[
        {
            "role": "user",
            "content": "Generate a random bedtime story + 1 sentence summary",
        }
    ],
    response_model=Story,
)

for r in resp:
    print(r)


# title = None summary = None
# title='The Little Firefly Who Lost His Light' summary=None
# title='The Little Firefly Who Lost His Light' summary='A tiny firefly learns the true meaning of friendship when he loses his glow and a wise old owl helps him find it again.'
```

### 3. Multimodal Support

`instructor` supports multimodal inputs for Gemini models, allowing you to perform tasks like visual question answering, image analysis, and more.

You can see an example of how to use instructor with Gemini to [extract travel recommendations from videos](./multimodal-gemini.md) post.

## What else does Instructor offer?

Beyond solving the core limitations of Gemini's new OpenAI integration, instructor provides a list of features that make it indispensable for production grade applications.

### 1. Provider Agnostic API

Switching between providers shouldn't require rewriting your entire codebase. With instructor, it's as simple as changing just a few lines of code.

```
from openai import OpenAI
from instructor import from_openai

client = from_openai(
    OpenAI()
)

# rest of code
```

If we wanted to switch to Anthropic, all it takes is changing the following lines of code

```python
from anthropic import Anthropic
from instructor import from_anthropic

client = from_anthropic(Anthropic())

# rest of code
```

### 2. Automatic Validation and Retries

Production applications need reliable outputs. Instructor handles this by validating all outputs against your desired response model and automatically retrying outputs that fail validation.

With [our tenacity integration](../../concepts/retrying.md), you get full control over the retries if needed, allowing you to mechanisms like exponential backoff and other retry strategies easily.

```python
import instructor
from pydantic import BaseModel
from tenacity import Retrying, stop_after_attempt, wait_fixed

client = instructor.from_provider("openai/gpt-5-nano", mode=instructor.Mode.TOOLS)


class UserDetail(BaseModel):
    name: str
    age: int


response = client.create(
    model="gpt-4o-mini",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract `jason is 12`"},
    ],
    # Stop after the second attempt and wait a fixed 1 second between attempts
    max_retries=Retrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
    ),
)
print(response.model_dump_json(indent=2))
"""
{
  "name": "jason",
  "age": 12
}
"""
```

## Conclusion

While Google's OpenAI compatibility layer is a welcome addition, there are still a few reasons why you might want to stick with instructor for now.

Within a single package, you get features such as a provider agnostic API, streaming capabilities, multimodal support, automatic re-asking and more.

Give us a try today by installing with `pip install instructor` and see why Pydantic is all you need for a production grade LLM application..
