---
authors:
  - ivanleomk
categories:
  - instructor
comments: true
date: 2025-05-11
description: Take advantage of OpenAI's latest offerings with the new responses API
draft: false
tags:
  - LLMs
  - OpenAI
  - Instructor
---

# Announcing Responses API support

We're excited to announce Instructor's integration with OpenAI's new Responses API. This integration brings a more streamlined approach to working with structured outputs from OpenAI models. Let's see what makes this integration special and how it can improve your LLM applications.

<!-- more -->

## What's New?

The Responses API represents a significant shift in how we interact with OpenAI models. With Instructor's integration, you can leverage this new API with our familiar, type-safe interface.

For our full documentation of the features we support, check out our full [OpenAI integration guide](../../integrations/openai.md).

Getting started is now easier than ever. With our unified provider interface, you can initialize your client with a single line of code. This means less time dealing with configuration and more time building features that matter.

```python
import instructor

# Initialize the client with Responses mode
client = instructor.from_provider(
    "openai/gpt-4.1-mini", mode=instructor.Mode.RESPONSES_TOOLS
)
```

The Responses API brings several improvements to structured data handling. You get access to built-in tools like web search and file search directly through the API. There's more efficient validation of structured outputs and improved error messages with better recovery mechanisms.

Here's a quick example showing how it works:

```python
class User(BaseModel):
    name: str
    age: int


# Create structured output
profile = client.responses.create(
    input="Extract out Ivan is 28 years old",
    response_model=User,
)

print(profile)
#> name='Ivan' age=28
```

## Key Benefits

The integration maintains Instructor's core strength of type safety while adding the power of the Responses API. You get full Pydantic model validation, automatic type checking, and clear error messages when validation fails. This gives you confidence that your outputs meet the constraints you've defined.

One of the most exciting features is the built-in tools support. You can now easily perform web searches with automatic citations, search through your knowledge base, and get real-time information with proper attribution. This significantly expands what you can build without having to integrate multiple APIs.

Here's an example using web search:

```python
class Citation(BaseModel):
    id: int
    url: str


class Summary(BaseModel):
    citations: list[Citation]
    summary: str


response = client.responses.create(
    input="What are some of the best places to visit in New York for Latin American food?",
    tools=[{"type": "web_search_preview"}],
    response_model=Summary,
)
```

The integration supports multiple ways to get structured outputs. You can use basic creation for simple, straightforward structured outputs. If you need real-time updates, partial creation lets you stream them as they come in. For handling multiple instances of the same object, iterable creation works great. And when you need both structured output and raw completion, completion with raw response gives you exactly that.

For production applications, we've maintained full async support. This lets you build responsive applications that can handle multiple requests efficiently:

```python
async def get_user_profile():
    async_client = instructor.from_provider(
        "openai/gpt-4.1-mini", mode=instructor.Mode.RESPONSES_TOOLS, async_client=True
    )

    profile = await async_client.responses.create(
        input="Extract: Maria lives in Spain.", response_model=UserProfile
    )
```

## Why This Matters

The integration of Instructor with OpenAI's Responses API brings two major benefits that will transform how you work with LLMs.

First, it makes working with inline citations significantly easier. When your LLM needs to reference external information, you get structured citation data that's ready to integrate into downstream applications. No more parsing messy text or manually extracting references - they come as properly typed objects that you can immediately use in your code.

Second, it works seamlessly with your existing chat completions code. You can add powerful capabilities like file search and web search without modifying your codebase. Just add the tool definition, and you're ready to go. Here's how simple it is:

```python
from pydantic import BaseModel
import instructor


class Citation(BaseModel):
    id: int
    url: str


class Summary(BaseModel):
    citations: list[Citation]
    summary: str


client = instructor.from_provider(
    "openai/gpt-4.1-mini",
    mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
)

response = client.create(
    messages=[
        {
            "role": "user",
            "content": "What are some of the best places to visit in New York for Latin American food?",
        }
    ],
    tools=[{"type": "web_search_preview"}],
    response_model=Summary,
)
print(response)
"""
citations=[Citation(id=1, url='https://www.nycgo.com/restaurants/best-latin-american-restaurants-in-nyc/'), Citation(id=2, url='https://www.timeout.com/newyork/restaurants/best-latin-american-restaurants-in-nyc'), Citation(id=3, url='https://www.thrillist.com/eat/nation/best-latin-american-restaurants-nyc')] summary="Some of the best places to visit in New York for Latin American food include neighborhoods and restaurants known for authentic and diverse offerings. In Manhattan, areas like the East Village and Lower East Side have excellent Latin American restaurants. Popular spots include Casa Enrique, known for Mexican cuisine; Tia Pol, offering Spanish and Latin flavors; and La Contenta, serving dishes from various Latin American countries. Brooklyn's Williamsburg and Bushwick have emerged as vibrant spots for Latin American eats, with restaurants such as La Esquina and Fonda not to miss. These places are celebrated for delicious food, lively atmospheres, and cultural authenticity, making them top choices for anyone looking to enjoy Latin American cuisine in New York City."
"""
```

This makes the path forward clear - you can enhance your existing applications with the latest OpenAI features while maintaining the type safety and validation Instructor is known for. No need to learn a new API or refactor your code. It just works.

## Getting Started

To start using the new Responses API integration, update to the latest version of Instructor, set up your OpenAI API key, initialize your client with the Responses mode, and start creating structured outputs.

This integration represents a significant step forward in making LLM development more accessible and powerful. We're excited to see what you'll build with these new capabilities.

For more detailed information about using the Responses API with Instructor, check out our [OpenAI integration guide](../../integrations/openai.md).

Happy coding!
