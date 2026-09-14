---
title: Using from_provider for Unified Client Creation
description: Learn how to use from_provider to create Instructor clients for any LLM provider.
---

# Using from_provider

The `from_provider` function creates Instructor clients for any LLM provider. It uses the same interface across all providers, making it easy to switch between models.

!!! note "V2 Preview"

    `from_provider` routes to the v2 implementation by default for supported providers. Legacy provider-specific modes are deprecated, emit warnings, and map to generic modes (`Mode.TOOLS`, `Mode.JSON`, `Mode.JSON_SCHEMA`, `Mode.MD_JSON`).

## Why Use from_provider?

`from_provider` provides:

- Simple syntax: One function works for all providers
- Automatic setup: Handles provider-specific configuration automatically
- Consistent interface: Same code works across different providers
- Type safety: Full IDE support with proper type inference
- Easy switching: Change providers with a single string change

## Basic Usage

The basic syntax is simple: `instructor.from_provider("provider/model-name")`

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Create a client for any provider
client = instructor.from_provider("openai/gpt-4o-mini")
# Or: instructor.from_provider("anthropic/claude-3-5-sonnet")
# Or: instructor.from_provider("google/gemini-2.5-flash")

# Use the client as usual
user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
)
```

## Supported Providers

`from_provider` supports all major LLM providers:

### Cloud Providers

- OpenAI: `"openai/gpt-4o"`, `"openai/gpt-4o-mini"`, `"openai/gpt-5.4-mini"`
- Anthropic: `"anthropic/claude-3-5-sonnet"`, `"anthropic/claude-3-opus"`
- Google: `"google/gemini-2.5-flash"`, `"google/gemini-pro"`
- Azure OpenAI: `"azure_openai/gpt-4o"`
- AWS Bedrock: `"bedrock/claude-3-5-sonnet"`
- Vertex AI: `"vertexai/gemini-pro"` (or use `"google/gemini-pro"` with `vertexai=True`)

### Fast Inference Providers

- Groq: `"groq/llama-3.1-70b"`
- Fireworks: `"fireworks/mixtral-8x7b"`
- Together: `"together/meta-llama/Llama-3-70b"`
- Anyscale: `"anyscale/meta-llama/Llama-3-70b"`

### Other Providers

- Mistral: `"mistral/mistral-large"`
- Cohere: `"cohere/command-r-plus"`
- Perplexity: `"perplexity/llama-3.1-sonar"`
- DeepSeek: `"deepseek/deepseek-chat"`
- xAI: `"xai/grok-beta"`
- OpenRouter: `"openrouter/meta-llama/llama-3.1-70b"`
- Hubris: `"hubris/anthropic/claude-sonnet-5"`
- Ollama: `"ollama/llama3.2"` (local models)
- LiteLLM: `"litellm/gpt-4o"` (meta-provider)

See the [Integrations](../integrations/index.md) section for complete provider documentation.

## Provider String Format

The provider string follows the format: `"provider/model-name"`

```python
# Correct formats
"openai/gpt-4o"
"anthropic/claude-3-5-sonnet-20241022"
"google/gemini-2.5-flash"

# Incorrect formats (will raise errors)
"gpt-4o"  # Missing provider prefix
"openai"  # Missing model name
"openai/gpt-4o/mini"  # Too many slashes
```

## Async Clients

Create async clients by setting `async_client=True`:

```python
import asyncio
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


async def main() -> None:
    # Create async client
    async_client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

    # Use with await
    await async_client.create(
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Alice is 25"}],
    )


asyncio.run(main())
```

## Advanced Configuration

### Custom API Keys

Pass API keys directly or use environment variables:

```python
import instructor

# Pass API key directly
client = instructor.from_provider("openai/gpt-4o-mini", api_key="sk-your-key-here")

# Or use environment variables (recommended)
# export OPENAI_API_KEY=sk-your-key-here
client = instructor.from_provider("openai/gpt-4o-mini")
```

### Mode Overrides

Override the default mode for a provider:

```python
import instructor

# OpenAI defaults to TOOLS mode, but you can override
client = instructor.from_provider(
    "openai/gpt-4o-mini", mode=instructor.Mode.JSON  # Use JSON mode instead
)
```

### Caching

Enable response caching:

```python
from instructor.cache import AutoCache
import instructor

cache = AutoCache(maxsize=1000)

client = instructor.from_provider("openai/gpt-4o-mini", cache=cache)
```

### Provider-Specific Options

Pass provider-specific options through `**kwargs`:

```python
import os
import instructor

# For OpenAI
client = instructor.from_provider(
    "openai/gpt-4o-mini", organization="org-your-org-id", timeout=30.0
)

# OpenAI-compatible multi-model gateways (same client pattern via base_url)
# Example: DaoXE at https://api.daoxe.com/v1
client = instructor.from_provider(
    "openai/gpt-4o-mini",
    api_key="YOUR_GATEWAY_KEY",
    base_url="https://api.daoxe.com/v1",
)

# For Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet", max_tokens=4096)

# For Google with Vertex AI
google_api_key = os.environ.pop("GOOGLE_API_KEY", None)

client = instructor.from_provider(
    "google/gemini-pro",
    vertexai=True,
    project="your-project-id",
    location="us-central1",
)

if google_api_key is not None:
    os.environ["GOOGLE_API_KEY"] = google_api_key
```

## Default Modes

Each provider uses a recommended default mode:

- OpenAI: `Mode.TOOLS`
- Anthropic: `Mode.TOOLS`
- Google: `Mode.TOOLS` or `Mode.JSON` based on the model
- Ollama: `Mode.TOOLS` (if supported) or `Mode.JSON`
- Others: `Mode.TOOLS` or `Mode.MD_JSON` depending on capability

Legacy provider-specific modes still work but are deprecated. See the [Mode Migration Guide](./mode-migration.md) for details.

Override these defaults with the `mode` parameter.

## Error Handling

`from_provider` raises clear errors for common issues:

```python
import instructor
from instructor.core.exceptions import ConfigurationError

try:
    # Invalid provider format
    client = instructor.from_provider("invalid-format")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    """
    Configuration error: Model string must be in format "provider/model-name" (e.g. "openai/gpt-5.4-mini" or "anthropic/claude-3-sonnet")
    """

try:
    # Unsupported provider
    client = instructor.from_provider("unsupported/provider")
except ConfigurationError as e:
    print(f"Unsupported provider: {e}")
    """
    Unsupported provider: Unsupported provider: unsupported. Supported providers are: ['openai', 'azure_openai', 'databricks', 'anthropic', 'google', 'generative-ai', 'vertexai', 'mistral', 'cohere', 'perplexity', 'groq', 'writer', 'bedrock', 'cerebras', 'deepseek', 'fireworks', 'ollama', 'openrouter', 'xai', 'litellm']
    """

try:
    # Missing required package
    client = instructor.from_provider("anthropic/claude-3")
except ImportError as e:
    print(f"Missing package: {e}")
    # Install with: pip install anthropic
```

## Environment Variables

Most providers support environment variables for configuration:

```bash
# OpenAI
export OPENAI_API_KEY=sk-your-key

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-your-key

# Google
export GOOGLE_API_KEY=your-key

# Azure OpenAI
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# AWS Bedrock
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Others
export MISTRAL_API_KEY=your-key
export COHERE_API_KEY=your-key
export GROQ_API_KEY=your-key
export DEEPSEEK_API_KEY=your-key
export OPENROUTER_API_KEY=your-key
```

## Switching Between Providers

One of the biggest advantages of `from_provider` is easy provider switching:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Easy to switch providers
PROVIDER = "openai/gpt-4o-mini"  # Change this to switch
# PROVIDER = "anthropic/claude-3-5-sonnet"
# PROVIDER = "google/gemini-2.5-flash"

client = instructor.from_provider(PROVIDER)

# Same code works for all providers
user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: Bob is 40"}],
)
```

## Best Practices

1. Use environment variables: Store API keys in environment variables, not in code
2. Use type hints: Let your IDE help with autocomplete and type checking
3. Handle errors: Wrap provider creation in try-except blocks
4. Cache when appropriate: Use caching for repeated requests
5. Choose the right mode: Let defaults work, but override when needed

## Comparison with Other Methods

### from_provider vs. Manual Patching

```python
# Old way (still works, but more verbose)
import openai
import instructor

openai_client = openai.OpenAI()
client = instructor.patch(openai_client)

# New way (recommended)
client = instructor.from_provider("openai/gpt-4o-mini")
```

### from_provider vs. Provider-Specific Functions

Provider-specific helpers were removed. Use `from_provider` for all clients:

```python
import instructor

openai_client = instructor.from_provider("openai/gpt-4o-mini")
anthropic_client = instructor.from_provider("anthropic/claude-3-5-sonnet")
```

## Troubleshooting

### Provider Not Found

If you get an error about an unsupported provider:

1. Check the provider name spelling
2. Verify the provider is in the supported list
3. Check if you need to install an extra package: `uv pip install "instructor[provider-name]"`

### Import Errors

If you get import errors:

```bash
# Install the required package
# For Anthropic
uv pip install anthropic

# For Google
uv pip install google-genai

# For others, see integration docs
```

### Invalid Model String

The model string must be in format `"provider/model-name"`:

```python
# Correct
"openai/gpt-4o"

# Incorrect
"gpt-4o"  # Missing provider
"openai"  # Missing model
```

## Related Documentation

- [Getting Started](../getting-started.md) - Quick start guide
- [Patching](./patching.md) - How Instructor enhances clients
- [Integrations](../integrations/index.md) - Provider-specific documentation
- [Migration Guide](./migration.md) - Migrating from old patterns
