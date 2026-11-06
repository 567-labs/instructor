---
draft: False
date: 2024-01-27
slug: together
title: "Structured outputs with Together AI, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Together AI. Learn how to generate structured, type-safe outputs with Together AI."
tags:
  - patching
  - open source
authors:
  - jxnl
---

# Structured outputs with Together AI, a complete guide with instructor

This guide demonstrates how to use Together AI with Instructor to generate structured outputs. You'll learn how to use function calling with Together's models to create type-safe responses.

Open-source LLMS are gaining popularity, and with the release of Together's Function calling models, its been easier than ever to get structured outputs.

By the end of this blog post, you will learn how to effectively utilize instructor with Together AI. But before we proceed, let's first explore the concept of patching.

!!! note "Other Languages"

    This blog post is written in Python, but the concepts are applicable to other languages as well, as we currently have support for [Javascript](https://js.useinstructor.com/), [Elixir](https://hexdocs.pm/instructor/Instructor.html) and [PHP](https://github.com/cognesy/instructor-php/).

<!-- more -->

## Patching

Instructor's patch enhances the openai api it with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers
- [Open Source Models](../examples/open_source.md) - More open-source model examples

# Together AI

The good news is that Together employs the same OpenAI client, and its models support some of these output modes too!

!!! note "Getting access"

    If you want to try this out for yourself check out the [Together AI](https://www.together.ai/) website. You can get started [here](http://api.together.ai/).

```python
import os
from pydantic import BaseModel
import instructor

client = instructor.from_provider(
    "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter


# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.create(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25

print(user.model_dump_json(indent=2))
"""
{
  "name": "jason",
  "age": 25
}
"""
{
    "name": "Jason",
    "age": 25,
}
```

### Async Example

```python
import instructor
from pydantic import BaseModel
import os
import asyncio

async_client = instructor.from_provider(
    "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    async_client=True,
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)


class UserExtract(BaseModel):
    name: str
    age: int


async def extract_user():
    return await async_client.create(
        response_model=UserExtract,
        messages=[{"role": "user", "content": "Extract jason is 25 years old"}],
    )


print(asyncio.run(extract_user()))
```

You can find more information about Together's function calling support [here](https://docs.together.ai/docs/function-calling).
