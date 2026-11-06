---
title: "Instructor - Multi-Language Library for Structured LLM Outputs | Python, TypeScript, Go, Ruby"
description: "Get structured, validated data from any LLM with Instructor - the #1 library for LLM data extraction. Supports 15+ providers (OpenAI, Anthropic, Google, Ollama, DeepSeek) in 6 languages. Built on type-safe schemas with automatic retries, streaming, and nested object support."
keywords: "LLM structured outputs, structured data extraction, OpenAI structured data, Pydantic LLM validation, Python LLM library, TypeScript LLM, Go LLM, Ruby LLM, Anthropic structured outputs, GPT structured data extraction, LLM response validation, AI data extraction, Ollama structured outputs, open source LLM, DeepSeek validation, Instructor vs Guardrails, LLM validation library, JSON schema validation, nested LLM schemas"
---

# Instructor: Top Multi-Language Library for Structured LLM Outputs

_Extract structured data from any LLM with type safety, validation, and automatic retries. Available in Python, TypeScript, Go, Ruby, Elixir, and Rust._

[![PyPI - Version](https://img.shields.io/pypi/v/instructor?style=flat-square&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/instructor/)
[![License](https://img.shields.io/github/license/instructor-ai/instructor?style=flat-square&color=blue)](https://github.com/instructor-ai/instructor/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square&logo=github&logoColor=white)](https://github.com/instructor-ai/instructor)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square&logo=pypi&logoColor=white&label=Downloads)](https://pypi.org/project/instructor/)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square&logo=discord&logoColor=white&label=Discord)](https://discord.gg/bD9YE9JArw)
[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/jxnlco)

> **Instructor for extraction, PydanticAI for agents.** Instructor shines when you need fast, schema-first extraction without extra agents. When your project needs quality gates, shareable runs, or built-in observability, try [PydanticAI](https://ai.pydantic.dev/). PydanticAI is the official agent runtime from the Pydantic team: it adds typed tools, dataset replays, and production dashboards while keeping your existing Instructor models. Read the [PydanticAI docs](https://ai.pydantic.dev/) to see how to bring those capabilities into your stack.

## What is Instructor?

Instructor is the **most popular Python library** for extracting structured data from Large Language Models (LLMs). With over **3 million monthly downloads, 11k stars, and 100+ contributors**, it's the go-to solution for developers who need reliable, validated outputs from AI models.

Built on top of **Pydantic**, Instructor provides type-safe data extraction with automatic validation, retries, and streaming support. Whether you're using OpenAI's GPT models, Anthropic's Claude, Google's Gemini, **open source models with Ollama**, **DeepSeek**, or any of 15+ supported providers, Instructor ensures your LLM outputs are always structured and validated.

## Key Features for LLM Data Extraction

- **Structured Outputs**: Define Pydantic models to specify exactly what data you want from your LLM
- **Automatic Retries**: Built-in retry logic when validation fails - no more manual error handling
- **Data Validation**: Leverage Pydantic's powerful validation to ensure response quality
- **Streaming Support**: Real-time processing of partial responses and lists
- **Multi-Provider**: Works with OpenAI, Anthropic, Google, Mistral, Cohere, Ollama, DeepSeek, and 15+ LLM providers
- **Type Safety**: Full IDE support with proper type inference and autocompletion
- **Open Source Support**: Run any open source model locally with Ollama, llama-cpp-python, or vLLM

## Quick Start

Install Instructor and start extracting structured data in minutes:

=== "pip"
    ```bash
    pip install instructor
    ```

=== "uv"
    ```bash
    uv add instructor
    ```

=== "poetry"
    ```bash
    poetry add instructor
    ```

### Extract Structured Data

Instructor's **`from_provider`** function provides a unified interface to work with any LLM provider. Switch between OpenAI, Anthropic, Google, Ollama, DeepSeek, and 15+ providers with the same code:

```python
import instructor
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    occupation: str


# Works with any provider - same interface everywhere
client = instructor.from_provider("openai/gpt-5-nano")
# Or: instructor.from_provider("anthropic/claude-3")
# Or: instructor.from_provider("google/gemini-pro")
# Or: instructor.from_provider("ollama/llama3.2")  # local

# Extract structured data from natural language
person = client.create(
    response_model=Person,
    messages=[
        {"role": "user", "content": "Extract: John is a 30-year-old software engineer"}
    ],
)
print(person)  # Person(name='John', age=30, occupation='software engineer')
```

The **`from_provider`** API supports both sync and async usage (`async_client=True`) and automatically handles provider-specific configurations. [See all supported providers →](./integrations/index.md)

## Complex Schemas & Validation

Instructor excels at extracting complex, nested data structures with custom validation rules. Here's a concise example:

```python
import instructor
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Ticket(BaseModel):
    title: str = Field(..., min_length=5, max_length=100)
    priority: Priority
    estimated_hours: Optional[float] = Field(None, gt=0, le=100)

    @field_validator('estimated_hours')
    @classmethod
    def validate_hours(cls, v):
        if v is not None and v % 0.5 != 0:
            raise ValueError('Hours must be in 0.5 increments')
        return v


class CustomerSupport(BaseModel):
    customer_name: str
    tickets: List[Ticket] = Field(..., min_items=1)


client = instructor.from_provider("openai/gpt-4o")

support_case = client.create(
    response_model=CustomerSupport,
    messages=[{"role": "user", "content": "Extract support case details..."}],
    max_retries=3,
)
```

**Key Features:**
- Deep nesting with nested models and lists
- Custom validation with Pydantic validators
- Automatic retries on validation failures
- Type-safe extraction with full IDE support

[Learn more about validation and complex schemas →](./concepts/reask_validation.md)

## Supported LLM Providers

Instructor works seamlessly with **15+ popular LLM providers**, giving you the flexibility to use any model while maintaining consistent structured output handling. From OpenAI's GPT models to **open source alternatives with Ollama**, **DeepSeek models**, and local inference, get validated data extraction everywhere.

It stands out for its simplicity, transparency, and user-centric design, built on top of Pydantic. Instructor helps you manage [validation context](./concepts/reask_validation.md), retries with [Tenacity](./concepts/retrying.md), and streaming [Lists](./concepts/lists.md) and [Partial](./concepts/partial.md) responses.

[:material-star: Star the Repo](https://github.com/jxnl/instructor){: .md-button .md-button--primary } [:material-book-open-variant: Cookbooks](./examples/index.md){: .md-button } [:material-lightbulb: Prompting Guide](./prompting/index.md){: .md-button }

If you ever get stuck, you can always run `instructor docs` to open the documentation in your browser. It even supports searching for specific topics.

```bash
instructor docs [QUERY]
```

### Provider Examples

All providers use the same simple interface. Here are quick examples for the most popular providers:

=== "OpenAI"
    ```python
    import instructor
    from pydantic import BaseModel


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_provider("openai/gpt-5-nano")
    res = client.create(
        response_model=ExtractUser,
        messages=[{"role": "user", "content": "John Doe is 30 years old."}],
    )
    ```

    [Full OpenAI docs →](./integrations/openai.md)

=== "Anthropic"
    ```python
    import instructor
    from pydantic import BaseModel


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_provider("anthropic/claude-3-5-sonnet-20240620")
    resp = client.create(
        response_model=ExtractUser,
        messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
    )
    ```

    [Full Anthropic docs →](./integrations/anthropic.md)

=== "Google Gemini"
    ```python
    import instructor
    from pydantic import BaseModel


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_provider("google/gemini-2.5-flash")
    resp = client.create(
        response_model=ExtractUser,
        messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
    )
    ```

    [Full Google docs →](./integrations/google.md)

=== "Ollama (Local)"
    ```python
    import instructor
    from pydantic import BaseModel


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_provider("ollama/llama3.2")
    resp = client.create(
        response_model=ExtractUser,
        messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
    )
    ```

    [Full Ollama docs →](./integrations/ollama.md)

[View all 15+ providers →](./integrations/index.md)

## Citation

If you use Instructor in your research or project, please cite it using:

```bibtex
@software{liu2024instructor,
  author = {Jason Liu and Contributors},
  title = {Instructor: A library for structured outputs from large language models},
  url = {https://github.com/instructor-ai/instructor},
  year = {2024},
  month = {3}
}
```

## Why use Instructor?

<div class="grid cards" markdown>

- :material-code-tags: **Simple API with Full Prompt Control**

    Instructor provides a straightforward API that gives you complete ownership and control over your prompts. This allows for fine-tuned customization and optimization of your LLM interactions.

    [:octicons-arrow-right-16: Explore Concepts](./concepts/models.md)

- :material-translate: **Multi-Language Support**

    Simplify structured data extraction from LLMs with type hints and validation.

    [:simple-python: Python](https://python.useinstructor.com) · [:simple-typescript: TypeScript](https://js.useinstructor.com) · [:simple-ruby: Ruby](https://ruby.useinstructor.com) · [:simple-go: Go](https://go.useinstructor.com) · [:simple-elixir: Elixir](https://hex.pm/packages/instructor) · [:simple-rust: Rust](https://rust.useinstructor.com)

- :material-refresh: **Reasking and Validation**

    Automatically reask the model when validation fails, ensuring high-quality outputs. Leverage Pydantic's validation for robust error handling.

    [:octicons-arrow-right-16: Learn about Reasking](./concepts/reask_validation.md)

- :material-repeat-variant: **Streaming Support**

    Stream partial results and iterables with ease, allowing for real-time processing and improved responsiveness in your applications.

    [:octicons-arrow-right-16: Learn about Streaming](./concepts/partial.md)

- :material-code-braces: **Powered by Type Hints**

    Leverage Pydantic for schema validation, prompting control, less code, and IDE integration.

    [:octicons-arrow-right-16: Learn more](https://docs.pydantic.dev/)

- :material-lightning-bolt: **Simplified LLM Interactions**

    Support for [OpenAI](./integrations/openai.md), [Anthropic](./integrations/anthropic.md), [Google](./integrations/google.md), [Vertex AI](./integrations/vertex.md), [Mistral/Mixtral](./integrations/together.md), [Ollama](./integrations/ollama.md), [llama-cpp-python](./integrations/llama-cpp-python.md), [Cohere](./integrations/cohere.md), [LiteLLM](./integrations/litellm.md).

    [:octicons-arrow-right-16: See Hub](./integrations/index.md)

</div>


### Using Hooks

Instructor's hooks system lets you intercept and handle events during LLM interactions. Use hooks for logging, monitoring, or custom error handling:

```python
import instructor
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4o-mini")

# Attach hooks for logging and error handling
client.on("completion:kwargs", lambda **kw: print("Called with:", kw))
client.on("completion:error", lambda e: print(f"Error: {e}"))

user_info = client.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "Extract: John is 20 years old"}],
)
```

[Learn more about hooks →](./concepts/hooks.md)

## Type Inference & Advanced Methods

Instructor provides full type inference for better IDE support and type safety. The client includes specialized methods for different use cases:

**Basic extraction:**
```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4o-mini")
user = client.create(response_model=User, messages=[...])  # Type: User
```

**Async support:**
```python
client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)
user = await client.create(...)  # Type: User
```

**Access original completion:**
```python
user, completion = client.create_with_completion(...)  # Returns tuple
```

**Stream partial objects:**
```python
for partial in client.create_partial(...):  # Type: Generator[User, None]
    print(partial)
```

**Stream multiple objects:**
```python
for user in client.create_iterable(...):  # Type: Generator[User, None]
    print(user)
```

All methods provide full type inference for better IDE autocomplete and type checking.

## Frequently Asked Questions

### What is Instructor?

Instructor is a Python library that extracts structured, validated data from Large Language Models (LLMs). It uses Pydantic models to define output schemas and automatically handles validation, retries, and error handling.

### Which LLM providers does Instructor support?

Instructor supports 15+ providers including OpenAI, Anthropic, Google Gemini, Mistral, Cohere, Ollama, DeepSeek, and many more. See our [integrations page](./integrations/index.md) for the complete list.

### Do I need to know Pydantic to use Instructor?

Basic Pydantic knowledge helps, but you can get started with simple models. Instructor works with any Pydantic BaseModel, and you can learn advanced features as you need them.

### How does Instructor compare to other libraries?

Instructor focuses specifically on structured outputs with automatic validation and retries. Unlike larger frameworks, Instructor does one thing very well: getting reliable, validated data from LLMs.

### Can I use Instructor with open source models?

Yes! Instructor works with Ollama, llama-cpp-python, and other local models. See our [Ollama integration guide](./integrations/ollama.md) to get started.

### Does Instructor work with async code?

Yes, Instructor fully supports async/await. Use `async_client=True` when creating your client, then use `await client.create()`.

[View all FAQs →](./faq.md)

## Templating

Instructor supports templating with Jinja, which lets you create dynamic prompts. This is useful when you want to fill in parts of a prompt with data. Here's a simple example:

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4o-mini")


class User(BaseModel):
    name: str
    age: int


# Create a completion using a Jinja template in the message content
response = client.create(
    messages=[
        {
            "role": "user",
            "content": """Extract the information from the
            following text: {{ data }}`""",
        },
    ],
    response_model=User,
    context={"data": "John Doe is thirty years old"},
)

print(response)
#> User(name='John Doe', age=30)
```

[Learn more about templating :octicons-arrow-right:](./concepts/templating.md){: .md-button .md-button-primary }
## Validation

You can also use Pydantic to validate your outputs and get the llm to retry on failure. Check out our docs on [retrying](./concepts/retrying.md) and [validation context](./concepts/reask_validation.md).

```python
import instructor
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator

# Create instructor client
client = instructor.from_provider("openai/gpt-4o-mini")


class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(llm_validator("don't say objectionable things", client=client)),
    ]


try:
    qa = QuestionAnswer(
        question="What is the meaning of life?",
        answer="The meaning of life is to be evil and steal",
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for QuestionAnswer
    answer
      Assertion failed, The statement promotes objectionable behavior by encouraging evil and stealing. [type=assertion_error, input_value='The meaning of life is to be evil and steal', input_type=str]
    """
```

## Contributing

If you want to help out, checkout some of the issues marked as `good-first-issue` or `help-wanted`. Found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cook book.

## License

This project is licensed under the terms of the MIT License.
