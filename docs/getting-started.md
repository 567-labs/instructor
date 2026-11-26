---
title: Extract and validate LLM responses in Python with Instructor
description: Build a Pydantic response model, extract a record with OpenAI, and add field validation, retries, and streaming. Then try the same model with Claude.
---

# Extract and validate your first LLM response {#getting-started-with-instructor}

This guide starts with two fields, a name and an age. You'll add a validation rule, extract a list of addresses, and stream the result as it arrives.

## Installation

Use Python 3.10 or later. Installing Instructor also installs Pydantic and the OpenAI SDK.

```bash
pip install instructor
```

For the Claude example, run `pip install "instructor[anthropic]"`. See [provider setup](./learning/getting_started/installation.md) for other SDKs and API keys.

## Setting Up Environment

Set the API key for the provider you plan to use in the same shell that runs Python.

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
client = instructor.from_provider("openai/gpt-6-luna")

# Extract structured data
user_info = client.create(
    response_model=UserInfo,
    reasoning_effort="none",
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

Reuse the client above. A field constraint checks the age, and a custom validator checks that the name contains a space.

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

user = client.create(
    response_model=User,
    max_retries=2,
    messages=[
        {"role": "user", "content": "Tom Smith is 25 years old."}
    ],
)
```

If the LLM returns only `Tom`, the validator raises an error. Instructor can send that error back to the LLM and try again. A name that passes the check can still be wrong. When a source omits a value, consider [making the field optional](./learning/patterns/optional_fields.md).

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

Call `create_partial` with the same `Person` model to see fields appear as the provider sends more data. Early snapshots may have missing fields.

```python
# Stream the response as it's being generated
stream = client.create_partial(
    response_model=Person,
    messages=[
        {
            "role": "user",
            "content": "John Smith, 35, lives at 456 Oak Ave, Chicago, IL 60601.",
        }
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
client = instructor.from_provider("anthropic/claude-haiku-5-5")

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

Yes! Use `async_client=True` when creating your client: `client = instructor.from_provider("openai/gpt-6-luna", async_client=True)`, then use `await client.create()`.

### What if validation fails?

Instructor automatically retries with validation feedback. You can configure retry behavior with `max_retries` parameter. See [retry mechanisms](./learning/validation/retry_mechanisms.md) for details.

[View all FAQs →](./faq.md)

## Next Steps

- Use [typed decisions](./concepts/decisions.md) to classify or score the same input against several questions in one request.
- Learn about [client setup with from_provider](./concepts/from_provider.md) for different LLM providers
- Explore [advanced validation](./concepts/reask_validation.md) to ensure data quality
- Check out the [Cookbook examples](./examples/index.md) for real-world applications
- See how to [use hooks](./concepts/hooks.md) for monitoring and debugging

**Using older patterns?** If you're using `instructor.patch()` or provider-specific functions like `from_openai()`, check out the [Migration Guide](./concepts/migration.md) to modernize your code.

**New to Instructor?** Start with [Start Here](./start-here.md) for a conceptual overview.

For more detailed information on any topic, visit the [Concepts](./concepts/index.md) section.

If you have questions or need help, join our [Discord community](https://discord.gg/bD9YE9JArw) or check the [GitHub repository](https://github.com/567-labs/instructor).
