---
authors:
  - jxnl
categories:
  - Anthropic
comments: true
date: 2024-10-23
description: Learn how to leverage Anthropic's Claude with Instructor for structured outputs and prompt caching, enhancing AI application development.
draft: false
tags:
  - Anthropic
  - API Development
  - Pydantic
  - Python
  - LLM Techniques
  - Prompt Caching
---

# Structured Outputs and Prompt Caching with Anthropic

Anthropic's ecosystem now offers two powerful features for AI developers: structured outputs and prompt caching. These advancements enable more efficient use of large language models (LLMs). This guide demonstrates how to leverage these features with the Instructor library to enhance your AI applications.

## Structured Outputs with Anthropic and Instructor

Instructor now offers seamless integration with Anthropic's powerful language models, allowing developers to easily create structured outputs using Pydantic models. This integration simplifies the process of extracting specific information from AI-generated responses.

<!-- more -->

To get started, you'll need to install Instructor with Anthropic support:

```bash
pip install instructor[anthropic]
```

Here's a basic example of how to use Instructor with Anthropic:

```python
from pydantic import BaseModel
from typing import List
import anthropic
import instructor

# Patch the Anthropic client with Instructor
anthropic_client = instructor.from_anthropic(create=anthropic.Anthropic())


# Define your Pydantic models
class Properties(BaseModel):
    name: str
    value: str


class User(BaseModel):
    name: str
    age: int
    properties: List[Properties]


# Use the patched client to generate structured output
user_response = anthropic_client(
    model="claude-3-7-sonnet-latest",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Create a user for a model with a name, age, and properties.",
        }
    ],
    response_model=User,
)

print(user_response.model_dump_json(indent=2))
"""
{
  "name": "John Doe",
  "age": 30,
  "properties": [
    { "name": "favorite_color", "value": "blue" }
  ]
}
"""
```

This approach allows you to easily extract structured data from Claude's responses, making it simpler to integrate AI-generated content into your applications.

## Prompt Caching: Boosting Performance and Reducing Costs

Anthropic has introduced a new prompt caching feature that can significantly improve response times and reduce costs for applications dealing with large context windows. This feature is particularly useful when making multiple calls with similar large contexts over time.

Here's how you can implement prompt caching with Instructor and Anthropic:

```python
from pydantic import BaseModel

# Set up the client with prompt caching
client = instructor.from_provider("anthropic/claude-3-5-haiku-latest")


# Define your Pydantic model
class Character(BaseModel):
    name: str
    description: str


# Load your large context
with open("./book.txt") as f:
    book = f.read()

# Make multiple calls using the cached context
for _ in range(2):
    resp, completion = client.create_with_completion(
        model="claude-3-7-sonnet-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<book>" + book + "</book>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "Extract a character from the text given above",
                    },
                ],
            },
        ],
        response_model=Character,
        max_tokens=1000,
    )
```

In this example, the large context (the book content) is cached after the first request and reused in subsequent requests. This can lead to significant time and cost savings, especially when working with extensive context windows.

## Conclusion

By combining Anthropic's Claude with Instructor's structured output capabilities and leveraging prompt caching, developers can create more efficient, cost-effective, and powerful AI applications. These features open up new possibilities for building sophisticated AI systems that can handle complex tasks with ease.

As the AI landscape continues to evolve, staying up-to-date with the latest tools and techniques is crucial. We encourage you to explore these features and share your experiences with the community. Happy coding!

## Related Documentation
- [How Patching Works](../../concepts/patching.md) - Understand provider integration
- [Anthropic Integration](../../integrations/anthropic.md) - Complete setup guide

## See Also
- [Anthropic Prompt Caching](anthropic-prompt-caching.md) - Optimize Anthropic costs
- [Unified Provider Interface](announcing-unified-provider-interface.md) - Switch providers easily
- [Framework Comparison](best_framework.md) - Why Instructor excels
