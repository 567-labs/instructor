---
title: Frequently Asked Questions
description: Common questions and answers about using Instructor
---

# Frequently Asked Questions

This page answers common questions about using Instructor with various LLM providers.

## General Questions

### What is Instructor?

Instructor is a library that makes it easy to get structured data from Large Language Models (LLMs). It uses Pydantic to define output schemas and provides a consistent interface across different LLM providers.

### How does Instructor work?

Instructor "patches" LLM clients to add a `response_model` parameter that accepts a Pydantic model. When you make a request, Instructor:

1. Converts your Pydantic model to a schema the LLM can understand
2. Formats the prompt appropriately for the provider
3. Validates the LLM's response against your model
4. Retries automatically if validation fails
5. Returns a properly typed Pydantic object

### Which LLM providers does Instructor support?

Instructor supports many providers, including:

- OpenAI (GPT models)
- Anthropic (Claude models)
- Google (Gemini models)
- Cohere
- Mistral AI
- Groq
- LiteLLM (meta-provider)
- TrueFoundry AI Gateway
- Various open-source models via Ollama, llama.cpp, etc.

See the [Integrations](./integrations/index.md) section for the complete list.

### What's the difference between various modes?

Instructor supports generic modes across providers:

- `Mode.TOOLS` - Tool/function calling when supported
- `Mode.JSON` - JSON generation for providers that support it (GenAI)
- `Mode.JSON_SCHEMA` - JSON schema enforcement (OpenAI, Mistral, Cohere)
- `Mode.MD_JSON` - JSON embedded in markdown
- `Mode.PARALLEL_TOOLS` - Parallel tool calls where supported

The optimal mode depends on your provider and use case. See [Patching](./concepts/patching.md) for details.

## Installation and Setup

### How do I install Instructor?

Basic installation:
```bash
pip install instructor
```

For specific providers:
```bash
pip install "instructor[anthropic]"  # For Anthropic
pip install "instructor[google-generativeai]"  # For Google/Gemini
```

### What environment variables do I need?

This depends on your provider:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

Each provider has specific requirements documented in their integration guide.

## Common Issues

### Why is my model not returning structured data?

Common reasons include:

1. Using the wrong mode for your provider
2. Complex schema that confuses the model
3. Insufficient context in your prompt
4. Using a model that doesn't support function/tool calling

Try simplifying your schema or providing clearer instructions in your prompt.

### How do I handle validation errors?

Instructor automatically retries when validation fails. You can customize this behavior:

```python
from tenacity import stop_after_attempt

result = client.create(
    response_model=MyModel,
    max_retries=stop_after_attempt(5),  # Retry up to 5 times
    messages=[...]
)
```

### Can I see the raw response from the LLM?

Yes, use `create_with_completion`:

```python
result, completion = client.create_with_completion(
    response_model=MyModel,
    messages=[...]
)
```

`result` is your Pydantic model, and `completion` is the raw response.

### How do I stream large responses?

Use `create_partial` for partial updates as the response is generated:

```python
stream = client.create_partial(
    response_model=MyModel,
    messages=[...]
)

for partial in stream:
    print(partial)  # Partial model with fields filled in as they're generated
```

## Performance and Costs

### How can I optimize token usage?

1. Use concise prompts
2. Use smaller models for simpler tasks
3. Use the `MD_JSON` or `JSON` mode for simple schemas
4. Cache responses for repeated queries

### How do I handle rate limits?

Instructor uses the `tenacity` library for retries, which you can configure:

```python
from tenacity import retry_if_exception_type, wait_exponential
from openai.error import RateLimitError

result = client.create(
    response_model=MyModel,
    max_retries=retry_if_exception_type(RateLimitError),
    messages=[...],
)
```

## Advanced Usage

### How do I use Instructor with FastAPI?

Instructor works seamlessly with FastAPI:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import instructor
app = FastAPI()
client = instructor.from_provider("openai/gpt-5-nano")

class UserInfo(BaseModel):
    name: str
    age: int

@app.post("/extract")
async def extract_user_info(text: str) -> UserInfo:
    return client.create(
        model="gpt-5.4-mini",
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

### How do I use Instructor with async code?

Use the async client:

```python
import instructor
import asyncio
client = instructor.from_provider("openai/gpt-5-nano", async_client=True)

async def extract_data():
    result = await client.create(
        response_model=MyModel,
        messages=[...]
    )
    return result

asyncio.run(extract_data())
```

### Where can I get more help?

- [Discord community](https://discord.gg/bD9YE9JArw)
- [GitHub issues](https://github.com/jxnl/instructor/issues)
- [Twitter @jxnl](https://twitter.com/jxnlco)
