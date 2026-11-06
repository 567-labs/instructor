---
title: Harnessing Structured Outputs with Ollama and Instructor
description: Discover how to utilize Ollama's Instructor library for structured outputs in LLM applications using Pydantic models.
---

## See Also

- [Ollama Integration](../integrations/ollama.md) - Complete Ollama setup guide
- [Open Source Models](./open_source.md) - More open-source model examples
- [Local Deployment](./index.md#local-deployment) - Local model deployment options
- [Response Models](../concepts/models.md) - Working with Pydantic models

# Structured Outputs with Ollama

Open-source Large Language Models (LLMs) are rapidly gaining popularity in the AI community. With the recent release of Ollama's OpenAI compatibility layer, it has become possible to obtain structured outputs using JSON schema from these open-source models. This development opens up exciting possibilities for developers and researchers alike.

In this blog post, we'll explore how to effectively utilize the Instructor library with Ollama to harness the power of structured outputs with [Pydantic models](../concepts/models.md). We'll cover everything from setup to implementation, providing you with practical insights and code examples.

## Why use Instructor?

Instructor offers several key benefits:

- :material-code-tags: **Simple API with Full Prompt Control**: Instructor provides a straightforward API that gives you complete ownership and control over your prompts. This allows for fine-tuned customization and optimization of your LLM interactions. [:octicons-arrow-right-16: Explore Concepts](../concepts/models.md)

- :material-refresh: **Reasking and Validation**: Automatically reask the model when validation fails, ensuring high-quality outputs. Leverage Pydantic's validation for robust error handling. [:octicons-arrow-right-16: Learn about Reasking](../concepts/reask_validation.md)

- :material-repeat-variant: **Streaming Support**: Stream partial results and iterables with ease, allowing for real-time processing and improved responsiveness in your applications. [:octicons-arrow-right-16: Learn about Streaming](../concepts/partial.md)

- :material-code-braces: **Powered by Type Hints**: Leverage Pydantic for schema validation, prompting control, less code, and IDE integration. [:octicons-arrow-right-16: Learn more](https://docs.pydantic.dev/)

- :material-lightning-bolt: **Simplified LLM Interactions**: Support for various LLM providers including OpenAI, Anthropic, Google, Vertex AI, Mistral/Mixtral, Anyscale, Ollama, llama-cpp-python, Cohere, and LiteLLM. [:octicons-arrow-right-16: See Examples](../examples/index.md)

For more details on these features, check out the [Concepts](../concepts/models.md) section of the documentation.

## Patching

Instructor's [patch](../concepts/patching.md) enhances an openai api with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## Ollama

Start by downloading [Ollama](https://ollama.ai/download), and then pull a model such as Llama 3 or Mistral.

!!! tip "Make sure you update your `ollama` to the latest version!"

```
ollama pull llama3.2
```

```python
import instructor
from pydantic import BaseModel, Field
from typing import List


class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


# Use from_provider with base_url for Ollama
client = instructor.from_provider(
    "ollama/llama3.2",
    base_url="http://localhost:11434/v1",
    mode=instructor.Mode.JSON,
)

resp = client.create(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Tell me about the Harry Potter",
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))
"""
{
  "name": "Harry James Potter",
  "age": 37,
  "fact": [
    "He is the chosen one.",
    "He has a lightning-shaped scar on his forehead.",
    "He is the son of James and Lily Potter.",
    "He attended Hogwarts School of Witchcraft and Wizardry.",
    "He is a skilled wizard and sorcerer.",
    "He fought against Lord Voldemort and his followers.",
    "He has a pet owl named Snowy."
  ]
}
"""
```

This example demonstrates how to use Instructor with Ollama, a local LLM server, to generate structured outputs. By leveraging Instructor's capabilities, we can easily extract structured information from the LLM's responses, making it simpler to work with the generated data in our applications.

## Further Reading

To explore more about Instructor and its various applications, consider checking out the following resources:

1. [Why use Instructor?](../why.md) - Learn about the benefits and use cases of Instructor.

2. [Concepts](../concepts/models.md) - Dive deeper into the core concepts of Instructor, including models, retrying, and validation.

3. [Examples](../examples/index.md) - Explore our comprehensive collection of examples and integrations with various LLM providers.

4. [Tutorials](../tutorials/1-introduction.ipynb) - Step-by-step tutorials to help you get started with Instructor.

5. [Learn Prompting](../prompting/index.md) - Techniques and strategies for effective prompt engineering with Instructor.

By exploring these resources, you'll gain a comprehensive understanding of Instructor's capabilities and how to leverage them in your projects.
