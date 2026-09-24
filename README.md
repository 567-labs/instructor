# Instructor: Structured Outputs for LLMs

Get reliable JSON from any LLM. Built on Pydantic for validation, type safety, and IDE support.

```python
import instructor
from pydantic import BaseModel


# Define what you want
class User(BaseModel):
    name: str
    age: int


# Extract it from natural language
client = instructor.from_provider("openai/gpt-4o-mini")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user)  # User(name='John', age=25)
```

**That's it.** No JSON parsing, no error handling, no retries. Just define a model and get structured data.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/567-labs/instructor?style=flat-square)](https://github.com/567-labs/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

> **Use Instructor for fast extraction, reach for PydanticAI when you need agents.** Instructor keeps schema-first flows simple and cheap. If your app needs richer agent runs, built-in observability, or shareable traces, try [PydanticAI](https://ai.pydantic.dev/). PydanticAI is the official agent runtime from the Pydantic team, adding typed tools, replayable datasets, evals, and production dashboards while using the same Pydantic models. Dive into the [PydanticAI docs](https://ai.pydantic.dev/) to see how it extends Instructor-style workflows.

## Why Our Instructor?

Getting structured data from LLMs is hard. You need to:

1. Write complex JSON schemas
2. Handle validation errors  
3. Retry failed extractions
4. Parse unstructured responses
5. Deal with different provider APIs

**Instructor handles all of this with one simple interface:**

<table>
<tr>
<td><b>Without Instructor</b></td>
<td><b>With Instructor</b></td>
</tr>
<tr>
<td>

```python
response = openai.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "..."}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "extract_user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            },
        }
    ],
)

# Parse response
tool_call = response.choices[0].message.tool_calls[0]
user_data = json.loads(tool_call.function.arguments)

# Validate manually
if "name" not in user_data:
    # Handle error...
    pass
```

</td>
<td>

```python
client = instructor.from_provider("openai/gpt-5.4-mini")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)

# That's it! user is validated and typed
```

</td>
</tr>
</table>

## Install in seconds

```bash
pip install instructor
```

Or with your package manager:
```bash
uv add instructor
poetry add instructor
```

## Works with every major provider

Use the same code with any LLM provider:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# With API keys directly (no environment variables needed)
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")

# All use the same API!
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Production-ready features

### Automatic retries

Failed validations are automatically retried with the error message:

```python
from pydantic import BaseModel, field_validator


class User(BaseModel):
    name: str
    age: int

    @field_validator('age')
    def validate_age(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v


# Instructor automatically retries when validation fails
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
    max_retries=3,
)
```

### Streaming support

Stream partial objects as they're generated:

```python
from instructor import Partial

for partial_user in client.chat.completions.create(
    response_model=Partial[User],
    messages=[{"role": "user", "content": "..."}],
    stream=True,
):
    print(partial_user)
    # User(name=None, age=None)
    # User(name="John", age=None)
    # User(name="John", age=25)
```

### Nested objects

Extract complex, nested data structures:

```python
from typing import List


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]


# Instructor handles nested objects automatically
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Used in production by

Trusted by over 100,000 developers and companies building AI applications:

- **3M+ monthly downloads**
- **10K+ GitHub stars**  
- **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get started

### Basic extraction

Extract structured data from any text:

```python
from pydantic import BaseModel
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")


class Product(BaseModel):
    name: str
    price: float
    in_stock: bool


product = client.chat.completions.create(
    response_model=Product,
    messages=[{"role": "user", "content": "iPhone 15 Pro, $999, available now"}],
)

print(product)
# Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

### Multiple languages

Instructor's simple API is available in many languages:

- [Python](https://python.useinstructor.com) - The original
- [TypeScript](https://js.useinstructor.com) - Full TypeScript support
- [Ruby](https://ruby.useinstructor.com) - Ruby implementation  
- [Go](https://go.useinstructor.com) - Go implementation
- [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
- [Rust](https://rust.useinstructor.com) - Rust implementation

### Learn more

- [Documentation](https://python.useinstructor.com) - Comprehensive guides
- [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes  
- [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
- [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why use Instructor over alternatives?

**vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.

**vs LangChain/LlamaIndex**: Instructor is focused on one thing - structured extraction. It's lighter, faster, and easier to debug.

**vs Custom solutions**: Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/567-labs/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/567-labs/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/567-labs/instructor/graphs/contributors">contributors</a>.
</p>
