---
title: "OpenAI Responses API Guide"
description: "Learn how to use Instructor's new Responses API with OpenAI models for structured outputs. Complete guide with examples and best practices."
---

# OpenAI Responses API Guide

The Responses API provides a more streamlined way to work with OpenAI models through Instructor. This guide covers everything you need to know about using the new Responses API for type-safe, validated outputs.

## Quick Start

```python
import instructor
from pydantic import BaseModel

# Initialize the client
client = instructor.from_provider(
    "openai/gpt-4.1-mini", mode=instructor.Mode.RESPONSES_TOOLS
)


# Define your response model
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

## Response Modes

The Responses API supports two main modes:

1. `instructor.Mode.RESPONSES_TOOLS`: Standard mode for structured outputs
2. `instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS`: Enhanced mode that includes built-in tools like web search and file search

```python
# Initialize the client
client = instructor.from_provider(
    "openai/gpt-4.1-mini", mode=instructor.Mode.RESPONSES_TOOLS
)
```

## Core Methods

The Responses API provides several methods for creating structured outputs. Here's how to use each one:

### Basic Creation

The `create` method is the simplest way to get a structured output:

=== "Sync"

    ```python
    from pydantic import BaseModel
    import instructor

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    profile = client.responses.create(
        input="Extract: Jason is 25 years old",
        response_model=User,
    )
    print(profile)  # User(name='Jason', age=25)
    ```

=== "Async"

    ```python
    from pydantic import BaseModel
    import instructor
    import asyncio

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS,
        async_client=True
    )

    async def main():
        profile = await client.responses.create(
            input="Extract: Jason is 25 years old",
            response_model=User,
        )
        print(profile)  # User(name='Jason', age=25)

    asyncio.run(main())
    ```

### Create with Completion

If you need the original completion object from OpenAI, you can do so with the `create_with_completion` method. This is useful when you have specific methods and data that you need to work from.

=== "Sync"

    ```python
    from pydantic import BaseModel
    import instructor

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, completion = client.responses.create_with_completion(
        input="Extract: Jason is 25 years old",
        response_model=User,
    )
    print(response)  # User(name='Jason', age=25)
    print(completion)  # Raw completion object
    ```

=== "Async"

    ```python
    from pydantic import BaseModel
    import instructor
    import asyncio

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS,
        async_client=True
    )

    async def main():
        response, completion = await client.responses.create_with_completion(
            input="Extract: Jason is 25 years old",
            response_model=User,
        )
        print(response)  # User(name='Jason', age=25)
        print(completion)  # Raw completion object

    asyncio.run(main())
    ```

### Iterable Creation

If you're interested in extracting multiple instances of the same object, we provide a convenient wrapper to be able to do so.

=== "Sync"

    ```python
    from pydantic import BaseModel
    from typing import Iterable
    import instructor

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    profiles = client.responses.create(
        input="Generate three fake profiles",
        response_model=Iterable[User],
    )

    for profile in profiles:
        print(profile)

    ```

=== "Async"

    ```python
    from pydantic import BaseModel
    from typing import Iterable
    import instructor
    import asyncio

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS,
        async_client=True
    )

    async def main():
        profiles = await client.responses.create_iterable(
            input="Generate three fake profiles",
            response_model=User,
        )

        async for profile in profiles:
            print(profile)

    asyncio.run(main())
    ```

### Partial Creation

We also provide validated outputs that you can stream in real time. This is incredibly useful for working with dynamic generative UI.

=== "Sync"

    ```python
    from pydantic import BaseModel
    import instructor

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    resp = client.responses.create_partial(
        input="Generate a fake profile",
        response_model=User,
    )

    for user in resp:
        print(user)  # Will show partial updates as they come in
    ```

=== "Async"

    ```python
    from pydantic import BaseModel
    import instructor
    import asyncio

    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS,
        async_client=True
    )

    async def main():
        resp = client.responses.create_partial(
            input="Generate a fake profile",
            response_model=User,
        )

        async for user in resp:
            print(user)  # Will show partial updates as they come in

    asyncio.run(main())
    ```

## Built-In Tools

The Responses API comes with powerful built-in tools that enhance the model's capabilities. These tools are managed by OpenAI, so you don't need to implement any additional code to use them.

For the most up-to-date documentation on how to use these tools, please refer to the [OpenAI Documentation](https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses)

### Web Search

The web search tool allows models to search the internet for real-time information. This is particularly useful for getting up-to-date information or verifying facts.

Model responses that use the web search tool will include two parts:

- A web_search_call output item with the ID of the search call.
- A message output item containing:
    1. The text result in message.content[0].text
    2. Annotations message.content[0].annotations for the cited URLs

By default, the model's response will include inline citations for URLs found in the web search results.

In addition to this, the url_citation annotation object will contain the URL, title and location of the cited source. You can extract this information using the `create_with_completion` method.

=== "Sync"

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
        async_client=False,
    )

    response, completion = client.responses.create_with_completion(
        input="What are some of the best places to visit in New York for Latin American food?",
        tools=[{"type": "web_search_preview"}],
        response_model=Summary,
    )

    print(response)
    # > citations=[Citation(id=1,url=....)]
    # > summary = New York City offers a rich variety of ...
    ```

=== "Async"

    ```python
    from pydantic import BaseModel
    import instructor
    import asyncio


    class Citation(BaseModel):
        id: int
        url: str


    class Summary(BaseModel):
        citations: list[Citation]
        summary: str


    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        async_client=True,
    )


    async def main():
        response = await client.responses.create(
            input="What are some of the best places to visit in New York for Latin American food?",
            tools=[{"type": "web_search_preview"}],
            response_model=Summary,
        )
        print(response)


    asyncio.run(main())
    # > citations=[Citation(id=1,url=....)]
    # > summary = New York City offers a rich variety of ...
    ```

You can customize the web search behavior with additional parameters:

```python
response = client.responses.create(
    input="What are the best restaurants around Granary Square?",
    tools=[{
        "type": "web_search_preview",
        "user_location": {
            "type": "approximate",
            "country": "GB",
            "city": "London",
            "region": "London",
        }
    }],
    response_model=Summary,
)
```

### File Search

The file search tool enables models to retrieve information from your knowledge base through semantic and keyword search. This is useful for augmenting the model's knowledge with your own documents.

This makes it easy to build RAG applications out of the box

=== "Sync"
    ```python
    from pydantic import BaseModel
    import instructor

    class Citation(BaseModel):
        file_id: int
        file_name: str
        excerpt: str

    class Response(BaseModel):
        citations: list[Citation]
        response: str

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
    )

    response = client.responses.create(
        input="How much does the Kyoto itinerary cost?",
        tools=[{
            "type": "file_search",
            "vector_store_ids": ["your_vector_store_id"],
            "max_num_results": 2,
        }],
        response_model=Response,
    )
    ```

=== "Async"
    ```python
    from pydantic import BaseModel
    import instructor
    import asyncio

    class Citation(BaseModel):
        file_id: int
        file_name: str
        excerpt: str

    class Response(BaseModel):
        citations: list[Citation]
        response: str

    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        async_client=True
    )

    async def main():
        response = await client.responses.create(
            input="How much does the Kyoto itinerary cost?",
            tools=[{
                "type": "file_search",
                "vector_store_ids": ["your_vector_store_id"],
                "max_num_results": 2,
            }],
            response_model=Response,
        )

    asyncio.run(main())
    ```

## Related Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)
