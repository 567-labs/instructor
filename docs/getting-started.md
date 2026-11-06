---
title: Getting Started
description: A step-by-step guide to getting started with Instructor for structured outputs from LLMs
---

# Getting Started with Instructor

This guide will walk you through the basics of using Instructor to extract structured data from language models. By the end, you'll understand how to:

1. Install and set up Instructor
2. Extract basic structured data
3. Handle validation and errors
4. Work with streaming responses
5. Use different LLM providers

## Installation

First, install Instructor:

```bash
pip install instructor
```

To use a specific provider, install the appropriate extras:

> Instructor's core install contains only required dependencies. Provider SDKs are optional and must be added explicitly.

```bash
# For OpenAI (included by default)
pip install instructor

# For Anthropic
pip install "instructor[anthropic]"

# For other providers
pip install "instructor[google-genai]"         # For Google/Gemini
pip install "instructor[vertexai]"             # For Vertex AI
pip install "instructor[cohere]"               # For Cohere
pip install "instructor[litellm]"              # For LiteLLM (multiple providers)
pip install "instructor[mistral]"               # For Mistral
pip install "instructor[xai]"                  # For xAI
```

## Setting Up Environment

Set your API keys as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# For Anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key

# For other providers, set relevant API keys
```

## Your First Structured Output

Let's start with a simple example using OpenAI:

```python
import instructor
from pydantic import BaseModel

# Define your output structure
class UserInfo(BaseModel):
    name: str
    age: int

# Create an instructor client with from_provider
client = instructor.from_provider("openai/gpt-5-nano")

# Extract structured data
user_info = client.create(
    response_model=UserInfo,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old."}
    ],
)

print(f"Name: {user_info.name}, Age: {user_info.age}")
# Output: Name: John Doe, Age: 30
```

This example demonstrates the core workflow:
1. Define a Pydantic model for your output structure
2. Create an Instructor client with `from_provider`
3. Request structured output using the `response_model` parameter

## Validation and Error Handling

Instructor leverages Pydantic's validation to ensure your data meets requirements:

```python
from pydantic import BaseModel, Field, field_validator

class User(BaseModel):
    name: str
    age: int = Field(gt=0, lt=120)  # Age must be between 0 and 120

    @field_validator('name')
    def name_must_have_space(cls, v):
        if ' ' not in v:
            raise ValueError('Name must include first and last name')
        return v

# This will make the LLM retry if validation fails
user = client.create(
    response_model=User,
    messages=[
        {"role": "user", "content": "Extract: Tom is 25 years old."}
    ],
)
```

## Working with Complex Models

Instructor works seamlessly with nested Pydantic models:

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

person = client.create(
    response_model=Person,
    messages=[
        {"role": "user", "content": """
        Extract: John Smith is 35 years old.
        He has homes at 123 Main St, Springfield, IL 62704 and
        456 Oak Ave, Chicago, IL 60601.
        """}
    ],
)
```

## Streaming Responses

For larger responses or better user experience, use streaming:

```python
from instructor import Partial

# Stream the response as it's being generated
stream = client.create_partial(
    response_model=Person,
    messages=[
        {"role": "user", "content": "Extract a detailed person profile for John Smith, 35, who lives in Chicago and Springfield."}
    ],
)

for partial in stream:
    # This will incrementally show the response being built
    print(partial)
```

## Using Different Providers

Instructor supports multiple LLM providers. Here's how to use Anthropic:

```python
import instructor
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Create an instructor client with from_provider
client = instructor.from_provider("anthropic/claude-3-opus-20240229")

user_info = client.create(
    response_model=UserInfo,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old."}
    ],
)

print(f"Name: {user_info.name}, Age: {user_info.age}")
```

## Frequently Asked Questions

### What's the difference between `start-here.md` and `getting-started.md`?

- **[Start Here](./start-here.md)**: Explains what Instructor is and why you'd use it (conceptual overview)
- **Getting Started**: This guide - shows you how to install and use Instructor (practical steps)

### Which provider should I start with?

OpenAI is the most popular choice for beginners due to reliability and wide support. Once comfortable, you can explore Anthropic Claude, Google Gemini, or open-source models.

### Do I need to understand Pydantic?

Basic knowledge helps, but you can start with simple models. Instructor works with any Pydantic BaseModel. Learn more advanced features as you need them.

### Can I use Instructor with async code?

Yes! Use `async_client=True` when creating your client: `client = instructor.from_provider("openai/gpt-4o", async_client=True)`, then use `await client.create()`.

### What if validation fails?

Instructor automatically retries with validation feedback. You can configure retry behavior with `max_retries` parameter. See [retry mechanisms](./learning/validation/retry_mechanisms.md) for details.

[View all FAQs →](./faq.md)

## Next Steps

Now that you've mastered the basics, here are some next steps:

- Learn about [client setup with from_provider](./concepts/from_provider.md) for different LLM providers
- Explore [advanced validation](./concepts/reask_validation.md) to ensure data quality
- Check out the [Cookbook examples](./examples/index.md) for real-world applications
- See how to [use hooks](./concepts/hooks.md) for monitoring and debugging

**Using older patterns?** If you're using `instructor.patch()` or provider-specific functions like `from_openai()`, check out the [Migration Guide](./concepts/migration.md) to modernize your code.

**New to Instructor?** Start with [Start Here](./start-here.md) for a conceptual overview.

For more detailed information on any topic, visit the [Concepts](./concepts/index.md) section.

If you have questions or need help, join our [Discord community](https://discord.gg/bD9YE9JArw) or check the [GitHub repository](https://github.com/jxnl/instructor).
