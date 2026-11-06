---
authors:
- jxnl
categories:
- LLM Techniques
comments: true
date: 2024-03-05
description: Discover how the Instructor library simplifies structured LLM outputs
  using Python type annotations for seamless data mapping.
draft: false
slug: zero-cost-abstractions
tags:
- Instructor
- LLM Outputs
- Python
- Pydantic
- Data Mapping
---

# Why Instructor is the Best Library for Structured LLM Outputs

Large language models (LLMs) like GPTs are incredibly powerful, but working with their open-ended text outputs can be challenging. This is where the Instructor library shines - it allows you to easily map LLM outputs to structured data using Python type annotations.

<!-- more -->

The core idea behind Instructor is incredibly simple: it's just a patch over the OpenAI Python SDK that adds a response_model parameter. This parameter lets you pass in a Pydantic model that describes the structure you want the LLM output mapped to. Pydantic models are defined using standard Python type hints, so there's zero new syntax to learn.

Here's an example of extracting structured user data from an LLM:

```python
from pydantic import BaseModel
import instructor


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-5-nano")

user = client.create(
    model="gpt-5.4-mini",
    response_model=User,  # (1)!
    messages=[
        {
            "role": "user",
            "content": "Extract the user's name and age from this: John is 25 years old",
        }
    ],
)

print(user)  # (2)!
#> name='John' age=25
```

1. Notice that now we have a new response_model parameter that we pass in to the completions.create method. This parameter lets us specify the structure we want the LLM output to be mapped to. In this case, we're using a Pydantic model called User that describes a user's name and age.
2. The output of the completions.create method is a User object that matches the structure we specified in the response_model parameter, rather than a ChatCompletion.

## Other Features

Other features on instructor, in and out of the llibrary are:

1. Ability to use [Tenacity in retrying logic](../../concepts/retrying.md)
2. Ability to use [Pydantic's validation context](../../concepts/reask_validation.md)
3. [Parallel Tool Calling](../../concepts/parallel.md) with correct types
4. Streaming [Partial](../../concepts/partial.md) and [Iterable](../../concepts/iterable.md) data.
5. Returning [Primitive](../../concepts/types.md) Types and [Unions](../../concepts/unions.md) as well!
6. Lots of [Cookbooks](../../examples/index.md), [Tutorials](../../tutorials/1-introduction.ipynb), and comprehensive Documentation in our [Integration Guides](../../integrations/index.md)

## Instructor's Broad Applicability

One of the key strengths of Instructor is that it's designed as a lightweight patch over the official OpenAI Python SDK. This means it can be easily integrated not just with OpenAI's hosted API service, but with any provider or platform that exposes an interface compatible with the OpenAI SDK.

For example, providers like [Together](../../integrations/together.md), [Ollama](../../integrations/ollama.md), [Groq](../../integrations/groq.md), and [llama-cpp-python](../../integrations/llama-cpp-python.md) all either use or mimic the OpenAI Python SDK under the hood. With Instructor's zero-overhead patching approach, teams can immediately start deriving structured data outputs from any of these providers. There's no need for custom integration work.

## Direct access to the messages array

Unlike other libraries that abstract away the `messages=[...]` parameter, Instructor provides direct access. This direct approach facilitates intricate prompt engineering, ensuring compatibility with OpenAI's evolving message types, including future support for images, audio, or video, without the constraints of string formatting.

## Low Abstraction

What makes Instructor so powerful is how seamlessly it integrates with existing OpenAI SDK code. To use it, you literally just call instructor.from_openai() on your OpenAI client instance, then use response_model going forward. There's no complicated refactoring or new abstractions to wrap your head around.

This incremental, zero-overhead adoption path makes Instructor perfect for sprinkling structured LLM outputs into an existing OpenAI-based application. You can start extracting data models from simple prompts, then incrementally expand to more complex hierarchical models, streaming outputs, and custom validations.

And if you decide Instructor isn't a good fit after all, removing it is as simple as not applying the patch! The familiarity and flexibility of working directly with the OpenAI SDK is a core strength.

Instructor solves the "string hellll" of unstructured LLM outputs. It allows teams to easily realize the full potential of tools like GPTs by mapping their text to type-safe, validated data structures. If you're looking to get more structured value out of LLMs, give Instructor a try!

## Related Concepts

- [Philosophy](../../concepts/philosophy.md) - Understand Instructor's design principles
- [Patching](../../concepts/patching.md) - Learn how Instructor patches LLM clients
- [Retrying](../../concepts/retrying.md) - Handle validation failures gracefully
- [Streaming](../../concepts/partial.md) - Work with streaming responses

## See Also

- [Introduction to Instructor](introduction.md) - Get started with structured outputs
- [Integration Guides](../../integrations/index.md) - See all supported providers
- [Type Examples](../../concepts/types.md) - Explore different response types
