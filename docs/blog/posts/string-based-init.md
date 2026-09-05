---
draft: false
date: 2024-04-20
authors:
  - jxnl
categories:
  - Tutorial
---

# Unified Provider Interface with String-Based Initialization

Instructor now offers a simplified way to initialize any supported LLM provider with a single consistent interface. This approach makes it easier than ever to switch between different LLM providers while maintaining the same structured output functionality you rely on.

## The Problem

As the number of LLM providers grows, so does the complexity of initializing and working with different client libraries. Each provider has its own initialization patterns, API structures, and quirks. This leads to code that isn't portable between providers and requires significant refactoring when you want to try a new model.

## The Solution: String-Based Initialization

We've introduced a new unified interface that allows you to initialize any supported provider with a simple string format:

```python
import instructor
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


# Initialize any provider with a single consistent interface
client = instructor.from_provider("openai/gpt-5.4-mini")
client = instructor.from_provider("anthropic/claude-3-sonnet")
client = instructor.from_provider("google/gemini-pro")
client = instructor.from_provider("mistral/mistral-large")
```

The `from_provider` function takes a string in the format `"provider/model-name"` and handles all the details of setting up the appropriate client with the right model. This provides several key benefits:

- **Simplified Initialization**: No need to manually create provider-specific clients
- **Consistent Interface**: Same syntax works across all providers
- **Reduced Dependency Exposure**: You don't need to import specific provider libraries in your application code
- **Easy Experimentation**: Switch between providers with a single line change

## Supported Providers

The string-based initialization currently supports all major providers in the ecosystem:

- OpenAI: `"openai/gpt-5.4-mini"`, `"openai/gpt-4o"`, `"openai/gpt-5-nano"`
- Anthropic: `"anthropic/claude-3-opus-20240229"`, `"anthropic/claude-3-sonnet-20240229"`, `"anthropic/claude-3-5-haiku-latest"`
- Google Gemini: `"google/gemini-pro"`, `"google/gemini-pro-vision"`
- Mistral: `"mistral/mistral-small-latest"`, `"mistral/mistral-medium-latest"`, `"mistral/mistral-large-latest"`
- Cohere: `"cohere/command"`, `"cohere/command-r"`, `"cohere/command-light"`
- Perplexity: `"perplexity/sonar-small-online"`, `"perplexity/sonar-medium-online"`
- Groq: `"groq/llama2-70b-4096"`, `"groq/mixtral-8x7b-32768"`, `"groq/gemma-7b-it"`
- Writer: `"writer/palmyra-instruct"`, `"writer/palmyra-instruct-v2"`
- AWS Bedrock: `"bedrock/anthropic.claude-v2"`, `"bedrock/amazon.titan-text-express-v1"`
- Cerebras: `"cerebras/cerebras-gpt"`, `"cerebras/cerebras-gpt-2.7b"`
- Fireworks: `"fireworks/llama-v2-70b"`, `"fireworks/firellama-13b"`
- Vertex AI: `"vertexai/gemini-pro"`, `"vertexai/text-bison"`
- Google GenAI: `"genai/gemini-pro"`, `"genai/gemini-pro-vision"`

Each provider will be initialized with sensible defaults, but you can also pass additional keyword arguments to customize the configuration. For model-specific details, consult each provider's documentation.

## Async Support

The unified interface fully supports both synchronous and asynchronous clients:

```python
# Synchronous client (default)
client = instructor.from_provider("openai/gpt-5.4-mini")

# Asynchronous client
async_client = instructor.from_provider("anthropic/claude-3-sonnet", async_client=True)

# Use like any other async client
response = await async_client.create(
    response_model=UserInfo,
    messages=[
        {
            "role": "user",
            "content": "Extract information about John who is 30 years old",
        }
    ],
)
```

## Mode Selection

You can also specify which structured output mode to use with the provider:

```python
import instructor
from instructor import Mode

# Override the default mode for a provider
client = instructor.from_provider("anthropic/claude-3-sonnet", mode=Mode.TOOLS)

# Use JSON mode instead of the default tools mode
client = instructor.from_provider("mistral/mistral-large", mode=Mode.JSON_SCHEMA)

# Use reasoning tools instead of regular tools for Anthropic
client = instructor.from_provider("anthropic/claude-3-opus", mode=Mode.TOOLS)
```

If not specified, each provider will use its recommended default mode:

- OpenAI: `Mode.OPENAI_FUNCTIONS`
- Anthropic: `Mode.TOOLS`
- Google Gemini: `Mode.MD_JSON`
- Mistral: `Mode.TOOLS`
- Cohere: `Mode.TOOLS`
- Perplexity: `Mode.JSON`
- Groq: `Mode.GROQ_TOOLS`
- Writer: `Mode.MD_JSON`
- Bedrock: `Mode.TOOLS` (for Claude on Bedrock)
- Vertex AI: `Mode.TOOLS`

You can always customize this based on your specific needs and model capabilities.

## Error Handling

The `from_provider` function includes robust error handling to help you quickly identify and fix issues:

```python
# Missing dependency
try:
    client = instructor.from_provider("anthropic/claude-3-sonnet")
except ImportError as e:
    print("Error: Install the anthropic package first")
    # pip install anthropic

# Invalid provider format
try:
    client = instructor.from_provider("invalid-format")
except ValueError as e:
    print(e)  # Model string must be in format "provider/model-name"

# Unsupported provider
try:
    client = instructor.from_provider("unknown/model")
except ValueError as e:
    print(e)  # Unsupported provider: unknown. Supported providers are: ...
```

The function validates the provider string format, checks if the provider is supported, and ensures the necessary packages are installed.

## Environment Variables

Like the native client libraries, `from_provider` respects environment variables set for each provider:

```python
# Set environment variables
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
os.environ["MISTRAL_API_KEY"] = "your-mistral-key"

# No need to pass API keys directly
client = instructor.from_provider("openai/gpt-5.4-mini")
```

## Troubleshooting

Here are some common issues and solutions when using the unified provider interface:

### Model Not Found Errors

If you receive a 404 error, check that you're using the correct model name format:

```
Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-3-haiku'}}
```

For Anthropic models, always include the version date:
- ✅ Correct: `anthropic/claude-3-haiku-20240307`
- ❌ Incorrect: `anthropic/claude-3-haiku`

### Provider-Specific Parameters

Some providers require specific parameters for API calls:

```python
# Anthropic requires max_tokens
anthropic_client = instructor.from_provider(
    "anthropic/claude-3-5-haiku-latest", max_tokens=400  # Required for Anthropic
)

# Use models with vision capabilities for multimodal content
gemini_client = instructor.from_provider(
    "google/gemini-pro-vision"  # Required for image processing
)
```

### Working Example

Here's a complete example that demonstrates the automodel functionality with multiple providers:

```python
import os
import asyncio
import instructor
from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    """User information extraction model."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    occupation: str = Field(description="The user's job or profession")


async def main():
    # Test OpenAI
    openai_client = instructor.from_provider("openai/gpt-5-nano")
    openai_result = openai_client.create(
        response_model=UserInfo,
        messages=[
            {"role": "user", "content": "Jane Doe is a 28-year-old data scientist."}
        ],
    )
    print(f"OpenAI result: {openai_result.model_dump()}")

    # Test Anthropic with async client
    if os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_client = instructor.from_provider(
            model="anthropic/claude-3-5-haiku-latest",
            async_client=True,
            max_tokens=400,  # Required for Anthropic
        )
        anthropic_result = await anthropic_client.create(
            response_model=UserInfo,
            messages=[
                {
                    "role": "user",
                    "content": "John Smith is a 35-year-old software engineer.",
                }
            ],
        )
        print(f"Anthropic result: {anthropic_result.model_dump()}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

String-based initialization is a significant step toward making Instructor even more user-friendly and flexible. It reduces the learning curve for working with multiple providers and makes it easier than ever to experiment with different models.

Benefits include:
- Simplified initialization with a consistent interface
- Automatic selection of appropriate default modes
- Support for both synchronous and asynchronous clients
- Clear error messages to quickly identify issues
- Respect for provider-specific environment variables
- Comprehensive model selection across the entire LLM ecosystem

Whether you're building a new application or migrating an existing one, the unified provider interface offers a cleaner, more maintainable way to work with structured outputs across the LLM ecosystem.

Try it today with `instructor.from_provider()` and check out the [complete example code](https://github.com/instructor-ai/instructor/tree/main/examples/automodel) in our repository!