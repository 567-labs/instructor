---
authors:
  - jxnl
categories:
  - LLM Techniques
comments: true
date: 2024-06-15
description:
  Learn how to easily get structured JSON data from LLMs using the Instructor
  library with Pydantic models in Python.
draft: false
slug: zero-cost-abstractions
tags:
  - Instructor
  - JSON
  - LLM
  - Pydantic
  - Python
---

# Why Instructor is the best way to get JSON from LLMs

Large Language Models (LLMs) like GPT are incredibly powerful, but getting them to return well-formatted JSON can be challenging. This is where the Instructor library shines. Instructor allows you to easily map LLM outputs to JSON data using Python type annotations and Pydantic models.

Instructor makes it easy to get structured data like JSON from LLMs like GPT-3.5, GPT-4, GPT-4-Vision, and open-source models including [Mistral/Mixtral](../../integrations/together.md), [Ollama](../../integrations/ollama.md), and [llama-cpp-python](../../integrations/llama-cpp-python.md).

It stands out for its simplicity, transparency, and user-centric design, built on top of Pydantic. Instructor helps you manage [validation context](../../concepts/reask_validation.md), retries with [Tenacity](../../concepts/retrying.md), and streaming [Lists](../../concepts/lists.md) and [Partial](../../concepts/partial.md) responses.

- Instructor provides support for a wide range of programming languages, including:
  - [Python](https://python.useinstructor.com)
  - [TypeScript](https://js.useinstructor.com)
  - [Ruby](https://ruby.useinstructor.com)
  - [Go](https://go.useinstructor.com)
  - [Elixir](https://hex.pm/packages/instructor)

<!-- more -->

## The Simple Patch for JSON LLM Outputs

Instructor works as a lightweight patch over the OpenAI Python SDK. To use it, you simply apply the patch to your OpenAI client:

```python
import instructor

client = instructor.from_provider("openai/gpt-5-nano")
```

Then, you can pass a `response_model` parameter to the `completions.create` or `chat.completions.create` methods. This parameter takes in a Pydantic model class that defines the JSON structure you want the LLM output mapped to. Just like `response_model` when using FastAPI.

Here's an example of a `response_model` for a simple user profile:

```python
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int
    email: str


client = instructor.from_provider("openai/gpt-5-nano")

user = client.create(
    model="gpt-5.4-mini",
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the user's name, age, and email from this: John Doe is 25 years old. His email is john@example.com",
        }
    ],
)

print(user.model_dump())
#> {
#     "name": "John Doe",
#     "age": 25,
#     "email": "john@example.com"
#   }
```

Instructor extracts the JSON data from the LLM output and returns an instance of your specified Pydantic model. You can then use the `model_dump()` method to serialize the model instance to a JSON string.

Some key benefits of Instructor:

- Zero new syntax to learn - it builds on standard Python type hints
- Seamless integration with existing OpenAI SDK code
- Incremental, zero-overhead adoption path
- Direct access to the `messages` parameter for flexible prompt engineering
- Broad compatibility with any OpenAI SDK-compatible platform or provider

## Pydantic: More Powerful than Plain Dictionaries

You might be wondering, why use Pydantic models instead of just returning a dictionary of key-value pairs? While a dictionary could hold JSON data, Pydantic models provide several powerful advantages:

1. Type validation: Pydantic models enforce the types of the fields. If the LLM returns an incorrect type (e.g. a string for an int field), it will raise a validation error.

2. Field requirements: You can mark fields as required or optional. Pydantic will raise an error if a required field is missing.

3. Default values: You can specify default values for fields that aren't always present.

4. Advanced types: Pydantic supports more advanced field types like dates, UUIDs, URLs, lists, nested models, and more.

5. Serialization: Pydantic models can be easily serialized to JSON, which is helpful for saving results or passing them to other systems.

6. IDE support: Because Pydantic models are defined as classes, IDEs can provide autocompletion, type checking, and other helpful features when working with the JSON data.

So while dictionaries can work for very simple JSON structures, Pydantic models are far more powerful for working with complex, validated JSON in a maintainable way.

## JSON from LLMs Made Easy

Instructor and Pydantic together provide a fantastic way to extract and work with JSON data from LLMs. The lightweight patching of Instructor combined with the powerful validation and typing of Pydantic models makes it easy to integrate JSON outputs into your LLM-powered applications. Give Instructor a try and see how much easier it makes getting JSON from LLMs!
