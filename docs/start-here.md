---
title: How Instructor turns LLM responses into Python objects
description: Learn how Pydantic response models, provider clients, and validation work together in Instructor. Follow a simple example that extracts a person’s details.
---

# How Instructor turns LLM responses into Python objects {#start-here-instructor-for-beginners}

## What is Instructor?

Instructor is a Python library for getting structured data from language models. Suppose your application needs a name, age, and city from a sentence. You define those fields in a Python class. Instructor uses the class as the output schema and returns an instance you can read with `person.name`, `person.age`, and `person.city`.

### Where Instructor Fits

Here's how Instructor fits into your application:

```mermaid
flowchart LR
    A[Your Application] --> B[Instructor]
    B --> C[LLM Provider]
    C --> B
    B --> A

    style B fill:#e2f0fb,stroke:#b8daff,color:#004085
```

The provider generates a response. Instructor validates its fields and can retry with feedback if a field is missing or invalid. Your schema determines which errors validation can catch.

## A Simple Example

[Install Instructor](./installation.md) and set `OPENAI_API_KEY` before running this example.

```python
import instructor
from pydantic import BaseModel

# Define the structure you want
class Person(BaseModel):
    name: str
    age: int
    city: str

# Connect to the LLM with Instructor
client = instructor.from_provider("openai/gpt-6-luna")

# Extract structured data
person = client.create(
    response_model=Person,
    reasoning_effort="none",
    messages=[
        {"role": "user", "content": "Extract: John is 30 years old and lives in New York."}
    ]
)

# Now you have a structured object
print(f"Name: {person.name}")  # Name: John
print(f"Age: {person.age}")    # Age: 30
print(f"City: {person.city}")  # City: New York
```

That's it! Instructor handled all the complexity of getting the LLM to format the data correctly.

**Ready to get started?** [Follow our step-by-step guide →](./getting-started.md)

## Key Concepts

Here are the main concepts you need to know:

### 1. Response Models

Response models define the structure you want the LLM to return. They are built using Pydantic, which is a data validation library.

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    # The descriptions help the LLM understand what to extract
```

### 2. Client Setup

The `from_provider` function connects Instructor to your LLM provider. It automatically handles provider-specific configurations:

```python
# For OpenAI
client = instructor.from_provider("openai/gpt-6-luna")

# For Anthropic
client = instructor.from_provider("anthropic/claude-haiku-5-5")

# For Google Gemini
client = instructor.from_provider("google/gemini-3.5-flash")
```

### 3. Modes

Modes control how Instructor gets structured data from the LLM. Each provider has a default mode. You can [choose a supported mode](./modes-comparison.md) or use [typed decisions](./concepts/decisions.md) for classification and scoring.

[Learn more about client setup →](./concepts/from_provider.md)

## Common Use Cases

Here are some popular ways people use Instructor:

1. **Data extraction**: Pull structured information from text documents
2. **Form filling**: Convert free-text into form fields
3. **Classification**: Sort content into predefined categories
4. **Content generation**: Create structured content like articles or product descriptions
5. **API integration**: Format LLM outputs to match API requirements

## Next Steps

Now that you understand the basics, here are some suggested next steps:

1. **Try the [Getting Started Guide](getting-started.md)** for a more in-depth tutorial
2. **Explore the [Cookbook Examples](examples/index.md)** for practical use cases
3. **Learn about [Validation](concepts/validation.md)** to ensure data quality
4. **Check out [Streaming](concepts/partial.md)** for handling large responses
5. **Understand [Providers](integrations/index.md)** to use different LLM services

## Common Questions

### Do I need to understand Pydantic?

While knowing Pydantic helps, you don't need to be an expert. The basic patterns shown above will get you started. You can learn more advanced features as you need them.

### Which LLM provider should I use?

OpenAI is the most popular choice for beginners because of its reliability and wide support. As you grow more comfortable, you can explore other providers like Anthropic Claude, Gemini, or open-source models.

### Is Instructor hard to learn?

No! If you're familiar with Python classes and working with APIs, you'll find Instructor straightforward. The core concepts are simple, and you can gradually explore advanced features.

### How does Instructor compare to other libraries?

Instructor focuses specifically on structured outputs with a simple, clean API. Unlike larger frameworks that try to do everything, Instructor does one thing very well: getting structured data from LLMs.

## Getting Help

If you get stuck:

- Check the [FAQ](faq.md) for common questions
- Browse the [Examples](examples/index.md) for similar use cases
- Join our [Discord community](https://discord.gg/bD9YE9JArw) for real-time help
- Look for related topics in the [Concepts](concepts/index.md) section

Welcome aboard, and happy extracting!
