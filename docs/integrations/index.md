---
title: "LLM Provider Integration Tutorials - Instructor"
description: "Complete tutorials for integrating Instructor with 15+ LLM providers. Learn structured data extraction with OpenAI, Anthropic Claude, Google Gemini, local models with Ollama, and more."
---

# LLM Provider Integration Tutorials

Learn how to integrate Instructor with various AI model providers. These comprehensive tutorials cover everything from cloud-based services like OpenAI and Anthropic to local open-source models, helping you extract structured outputs from any LLM.

<div class="grid cards" markdown>

- :material-cloud: **Major Cloud Providers**

    Leading AI providers with comprehensive features

    [:octicons-arrow-right-16: OpenAI](./openai.md)          ·
    [:octicons-arrow-right-16: OpenAI Responses](./openai-responses.md)          ·
    [:octicons-arrow-right-16: Azure](./azure.md)            ·
    [:octicons-arrow-right-16: Anthropic](./anthropic.md)    ·
    [:octicons-arrow-right-16: Google.GenerativeAI](./google.md)          ·
    [:octicons-arrow-right-16: Vertex AI](./vertex.md)       ·
    [:octicons-arrow-right-16: AWS Bedrock](./bedrock.md)    ·
    [:octicons-arrow-right-16: Google.GenAI](./genai.md)     ·
    [:octicons-arrow-right-16: xAI](./xai.md)

- :material-cloud-outline: **Additional Cloud Providers**

    Other commercial AI providers with specialized offerings

    [:octicons-arrow-right-16: Cohere](./cohere.md)          ·
    [:octicons-arrow-right-16: Mistral](./mistral.md)        ·
    [:octicons-arrow-right-16: DeepSeek](./deepseek.md)      ·
    [:octicons-arrow-right-16: Together AI](./together.md)    ·
    [:octicons-arrow-right-16: Groq](./groq.md)              ·
    [:octicons-arrow-right-16: Fireworks](./fireworks.md)    ·
    [:octicons-arrow-right-16: Cerebras](./cerebras.md)      ·
    [:octicons-arrow-right-16: Writer](./writer.md)          ·
    [:octicons-arrow-right-16: Perplexity](./perplexity.md)
    [:octicons-arrow-right-16: SambaNova](./sambanova.md)

- :material-open-source-initiative: **Open Source**

    Run open-source models locally or in the cloud

    [:octicons-arrow-right-16: Ollama](./ollama.md)                  ·
    [:octicons-arrow-right-16: llama-cpp-python](./llama-cpp-python.md)

- :material-router-wireless: **Routing**

    Unified interfaces for multiple providers

    [:octicons-arrow-right-16: LiteLLM](./litellm.md)
    [:octicons-arrow-right-16: OpenRouter](./openrouter.md)

</div>

## Common Features

All integrations support these core features:

| Feature | Description | Documentation |
|---------|-------------|---------------|
| **Model Patching** | Enhance provider clients with structured output capabilities | [Patching](../concepts/patching.md) |
| **Response Models** | Define expected response schema with Pydantic | [Models](../concepts/models.md) |
| **Validation** | Ensure responses match your schema definition | [Validation](../concepts/validation.md) |
| **Streaming** | Stream partial or iterative responses | [Partial](../concepts/partial.md), [Iterable](../concepts/iterable.md) |
| **Hooks** | Add callbacks for monitoring and debugging | [Hooks](../concepts/hooks.md) |

However, each provider has different capabilities and limitations. Refer to the specific provider documentation for details.

## Provider Modes

Providers support different methods for generating structured outputs:

| Mode | Description | Providers |
|------|-------------|-----------|
| `TOOLS` | Uses OpenAI-style tools/function calling | OpenAI, Anthropic, Mistral |
| `PARALLEL_TOOLS` | Multiple simultaneous tool calls | OpenAI |
| `JSON` | Direct JSON response generation | OpenAI, Gemini, Cohere, GenAI |
| `MD_JSON` | JSON embedded in markdown | Most providers |

See the [Modes Comparison](../modes-comparison.md) guide for details.

## Getting Started

There are two ways to use providers with Instructor:

### 1. Using Provider Initialization (Recommended)

The simplest way to get started is using the provider initialization:

```python
import instructor
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Initialize any provider with a simple string
client = instructor.from_provider("openai/gpt-5.4-mini")
# Or use async client
async_client = instructor.from_provider("anthropic/claude-3-sonnet", async_client=True)

# Use the same interface for all providers
response = client.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "Your prompt"}]
)
```

Supported provider strings:
- `openai/model-name`: OpenAI models
- `anthropic/model-name`: Anthropic models
- `google/model-name`: Google models
- `mistral/model-name`: Mistral models
- `cohere/model-name`: Cohere models
- `perplexity/model-name`: Perplexity models
- `groq/model-name`: Groq models
- `writer/model-name`: Writer models
- `bedrock/model-name`: AWS Bedrock models
- `cerebras/model-name`: Cerebras models
- `fireworks/model-name`: Fireworks models
- `vertexai/model-name`: Vertex AI models
- `genai/model-name`: Google GenAI models
- `ollama/model-name`: Ollama models

### Provider Checklist

Use these example strings with `from_provider` to quickly get started:

- [x] `instructor.from_provider("openai/gpt-5-nano")`
- [x] `instructor.from_provider("anthropic/claude-3-sonnet")`
- [x] `instructor.from_provider("google/gemini-2.5-flash")`
- [x] `instructor.from_provider("mistral/mistral-large-latest")`
- [x] `instructor.from_provider("cohere/command-r")`
- [x] `instructor.from_provider("perplexity/sonar-small")`
- [x] `instructor.from_provider("groq/llama3-8b-8192")`
- [x] `instructor.from_provider("writer/palmyra-x-004")`
- [x] `instructor.from_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")`
- [x] `instructor.from_provider("cerebras/llama3.1-70b")`
- [x] `instructor.from_provider("fireworks/llama-v3-70b-instruct")`
- [x] `instructor.from_provider("vertexai/gemini-3-flash")`
- [x] `instructor.from_provider("genai/gemini-3-flash")`
- [x] `instructor.from_provider("ollama/llama3.2")`

### 2. Manual Client Setup

Alternatively, you can manually set up the client:

1. Install the required dependencies:
   ```bash
   pip install "instructor[provider]"  # e.g., instructor[anthropic]
   ```

2. Import the provider client and patch it with Instructor:
   ```python
   import instructor
   from provider_package import Client

   client = instructor.from_provider(Client())
   ```

3. Use the patched client with your Pydantic model:
   ```python
   response = client.create(
       response_model=YourModel,
       messages=[{"role": "user", "content": "Your prompt"}]
   )
   ```

For provider-specific setup and examples, visit each provider's documentation page.

## Need Help?

If you need assistance with a specific integration:

1. Check the provider-specific documentation
2. Browse the [examples](../examples/index.md) and [cookbooks](../examples/index.md)
3. Search existing [GitHub issues](https://github.com/jxnl/instructor/issues)
4. Join our [Discord community](https://discord.gg/bD9YE9JArw)
