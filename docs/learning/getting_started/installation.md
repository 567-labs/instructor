---
title: Installing Instructor for LLM Structured Outputs
description: Complete installation guide for Instructor with support for OpenAI, Anthropic, Google, and 15+ LLM providers. Get started in minutes.
---

# Instructor Installation Guide: Setup for LLM Structured Outputs

Learn how to install Instructor, the leading Python library for extracting structured data from LLMs like GPT-4, Claude, and Gemini. This comprehensive installation tutorial covers all major LLM providers and gets you ready for production use.

## Quick Start: Install Instructor for LLM Development

Get started with structured LLM outputs in seconds. Install Instructor using pip:

```shell
pip install instructor
```

Instructor leverages Pydantic for type-safe LLM data extraction:

```shell
pip install pydantic
```

> **Pro Tip**: Use `uv` for faster installation: `uv pip install instructor`

## LLM Provider Installation Guide

Instructor supports 15+ LLM providers. Here's how to install and configure each:

### OpenAI (GPT-4, GPT-3.5)

OpenAI is the default LLM provider for Instructor. Perfect for GPT-4 and GPT-3.5-turbo structured outputs:

```shell
pip install instructor
```

Configure your OpenAI API key for LLM access:

```shell
export OPENAI_API_KEY=your_openai_key
```

### Anthropic Claude LLM Setup

Extract structured data from Claude 3 models (Opus, Sonnet, Haiku) with native tool support:

```shell
pip install "instructor[anthropic]"
```

Configure Claude API access:

```shell
export ANTHROPIC_API_KEY=your_anthropic_key
```

### Google Gemini LLM Integration

Use Gemini Pro and Flash models for structured outputs with function calling:

```shell
pip install "instructor[google-genai]"
```

Set up Gemini API access:

```shell
export GOOGLE_API_KEY=your_google_key
```

### Cohere

To use with Cohere's models:

```shell
pip install "instructor[cohere]"
```

Set up your Cohere API key:

```shell
export COHERE_API_KEY=your_cohere_key
```

### Mistral

To use with Mistral AI's models:

```shell
pip install "instructor[mistral]"
```

Set up your Mistral API key:

```shell
export MISTRAL_API_KEY=your_mistral_key
```

### LiteLLM (Multiple Providers)

To use LiteLLM for accessing multiple providers:

```shell
pip install "instructor[litellm]"
```

Set up API keys for the providers you want to use.

## Verify Your Instructor LLM Setup

Test your Instructor installation with this simple LLM structured output example:

```python
import instructor
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-5-nano")
person = client.create(
    model="gpt-5.4-mini",
    response_model=Person,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old"}
    ]
)

print(f"Name: {person.name}, Age: {person.age}")
```

## Next Steps in Your LLM Tutorial Journey

With Instructor installed, you're ready to build powerful LLM applications:

1. **[Create Your First LLM Extraction](first_extraction.md)** - Build structured outputs with any LLM
2. **[Master Response Models](response_models.md)** - Learn Pydantic models for LLM data validation
3. **[Configure LLM Clients](../../concepts/from_provider.md)** - Set up OpenAI, Anthropic, Google, and more

## Common Installation Issues

- **Import Errors**: Ensure you've installed the provider-specific extras (e.g., `instructor[anthropic]`)
- **API Key Issues**: Verify your environment variables are set correctly
- **Version Conflicts**: Use `pip install --upgrade instructor` to get the latest version

Ready to extract structured data from LLMs? Continue to [Your First Extraction](first_extraction.md) →