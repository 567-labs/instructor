---
title: "LLM providers"
description: "Find the Python SDK, credentials, and model ID for your LLM provider."
---

# LLM provider integrations for structured outputs in Python

Choose the service that hosts your model to find its SDK and credentials. For the string to pass to `from_provider`, see [model IDs by provider](model-selection.md).

<div class="grid cards" markdown>

- :material-cloud: **Major Cloud Providers**

    Leading AI providers with comprehensive features

    [:octicons-arrow-right-16: OpenAI Chat Completions](./openai.md) ·
    [:octicons-arrow-right-16: OpenAI-compatible gateways](./openai-compatible-gateways.md) ·
    [:octicons-arrow-right-16: OpenAI Responses](./openai-responses.md) ·
    [:octicons-arrow-right-16: Azure OpenAI](./azure.md) ·
    [:octicons-arrow-right-16: Anthropic Claude](./anthropic.md) ·
    [:octicons-arrow-right-16: Google Gemini API](./google.md) ·
    [:octicons-arrow-right-16: Vertex AI](./vertex.md) ·
    [:octicons-arrow-right-16: AWS Bedrock](./bedrock.md) ·
    [:octicons-arrow-right-16: Google Gen AI SDK](./genai.md) ·
    [:octicons-arrow-right-16: xAI](./xai.md)

- :material-cloud-outline: **Additional Cloud Providers**

    Other commercial AI providers with specialized offerings

    [:octicons-arrow-right-16: Cohere](./cohere.md) ·
    [:octicons-arrow-right-16: Mistral](./mistral.md) ·
    [:octicons-arrow-right-16: DeepSeek](./deepseek.md) ·
    [:octicons-arrow-right-16: Together AI](./together.md) ·
    [:octicons-arrow-right-16: Groq](./groq.md) ·
    [:octicons-arrow-right-16: Fireworks](./fireworks.md) ·
    [:octicons-arrow-right-16: Cerebras](./cerebras.md) ·
    [:octicons-arrow-right-16: Writer](./writer.md) ·
    [:octicons-arrow-right-16: Perplexity](./perplexity.md) ·
    [:octicons-arrow-right-16: SambaNova](./sambanova.md)

- :material-open-source-initiative: **Open Source**

    Run open-source models locally or in the cloud

    [:octicons-arrow-right-16: Ollama](./ollama.md) ·
    [:octicons-arrow-right-16: llama-cpp-python](./llama-cpp-python.md)

- :material-router-wireless: **Routing**

    Unified interfaces for multiple providers

    [:octicons-arrow-right-16: LiteLLM](./litellm.md) ·
    [:octicons-arrow-right-16: OpenRouter](./openrouter.md)

</div>

## Common Features

| Feature | Description | Documentation |
|---------|-------------|---------------|
| **Model Patching** | Enhance provider clients with structured output capabilities | [Patching](../concepts/patching.md) |
| **Response Models** | Define expected response schema with Pydantic | [Models](../concepts/models.md) |
| **Validation** | Ensure responses match your schema definition | [Validation](../concepts/validation.md) |
| **Streaming** | Stream partial or iterative responses | [Partial](../concepts/partial.md), [Iterable](../concepts/iterable.md) |
| **Hooks** | Add callbacks for monitoring and debugging | [Hooks](../concepts/hooks.md) |

## Provider Modes

Use the [mode comparison](../modes-comparison.md) to check whether your provider accepts tool calls, a JSON schema, or JSON instructions in the prompt.

## Getting Started

### 1. Using Provider Initialization (Recommended)

Install `instructor` and set `OPENAI_API_KEY` to run this example.

```python
import instructor
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-6-luna")
response = client.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "Extract the name and age. Mina is 32."}],
    reasoning_effort="none",
)
```

### Provider Checklist

1. Install the SDK and configure credentials as shown in your provider's guide.
2. Choose a [model ID](model-selection.md) and pass `"provider/model"` to [`from_provider`](../concepts/from_provider.md).
3. For asynchronous requests, set `async_client=True` and await `client.create(...)`.

### 2. Manual Client Setup

If you already have an SDK client, use the matching Instructor factory. For example, wrap an OpenAI client with `from_openai`.

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI(), model="gpt-6-luna")
```

## Need Help?

Browse the [examples](../examples/index.md), search [GitHub issues](https://github.com/567-labs/instructor/issues), or join the [Discord community](https://discord.gg/bD9YE9JArw).
