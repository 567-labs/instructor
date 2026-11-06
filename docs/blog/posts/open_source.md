---
authors:
- jxnl
categories:
- API Development
comments: true
date: 2024-03-07
description: Discover how Instructor integrates with OpenAI and local LLMs for structured
  outputs using Pydantic and JSON schema.
draft: false
slug: open-source-local-structured-output-pydantic-json-openai
tags:
- OpenAI
- Pydantic
- LLMs
- Structured Outputs
- API Integration
---

# Structured Output for Open Source and Local LLMs

Instructor has expanded its capabilities for language models. It started with API interactions via the OpenAI SDK, using [Pydantic](https://pydantic-docs.helpmanual.io/) for structured data validation. Now, Instructor supports multiple models and platforms.

The integration of [JSON mode](../../concepts/patching.md#json-mode) improved adaptability to vision models and open source alternatives. This allows support for models from [GPT](https://openai.com/api/) and [Mistral](https://mistral.ai) to models on [Ollama](https://ollama.ai) and [Hugging Face](https://huggingface.co/models), using [llama-cpp-python](../../integrations/llama-cpp-python.md).

Instructor now works with cloud-based APIs and local models for structured data extraction. Developers can refer to our guide on [Patching](../../concepts/patching.md) for information on using JSON mode with different models.

For learning about Instructor and Pydantic, we offer a course on [Steering language models towards structured outputs](https://www.wandb.courses/courses/steering-language-models).

The following sections show examples of Instructor's integration with platforms and local setups for structured outputs in AI projects.

<!-- more -->


## Exploring Different OpenAI Clients with Instructor

OpenAI clients offer functionalities for different needs. We explore clients integrated with Instructor, providing structured outputs and capabilities. Examples show how to initialize and patch each client.

## Local Models

### Ollama: A New Frontier for Local Models

Ollama enables structured outputs with local models using JSON schema. See our [Ollama documentation](../../integrations/ollama.md) for details.

For setup and features, refer to the documentation. The [Ollama website](https://ollama.ai/download) provides resources, models, and support.

```
ollama run llama2
```

```python
from openai import OpenAI
from pydantic import BaseModel
import instructor


class UserDetail(BaseModel):
    name: str
    age: int


# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)


user = client.create(
    model="llama2",
    messages=[
        {
            "role": "user",
            "content": "Jason is 30 years old",
        }
    ],
    response_model=UserDetail,
)

print(user)
#> name='Jason' age=30
```

### llama-cpp-python

llama-cpp-python provides the `llama-cpp` model for structured outputs using JSON schema. It uses [constrained sampling](https://llama-cpp-python.readthedocs.io/en/latest/#json-schema-mode) and [speculative decoding](https://llama-cpp-python.readthedocs.io/en/latest/#speculative-decoding). An [OpenAI compatible client](https://llama-cpp-python.readthedocs.io/en/latest/#openai-compatible-web-server) allows in-process structured output without network dependency.

Example of using llama-cpp-python for structured outputs:


```python
import llama_cpp
import instructor
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from pydantic import BaseModel

llama = llama_cpp.Llama(
    model_path="../../models/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    n_gpu_layers=-1,
    chat_format="chatml",
    n_ctx=2048,
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2),
    logits_all=True,
    verbose=False,
)


create = instructor.patch(
    create=llama.create_chat_completion_openai_v1,
    mode=instructor.Mode.JSON_SCHEMA,
)


class UserDetail(BaseModel):
    name: str
    age: int


user = create(
    messages=[
        {
            "role": "user",
            "content": "Extract `Jason is 30 years old`",
        }
    ],
    response_model=UserDetail,
)

print(user)
#> name='Jason' age=30
```

## Alternative Providers

### Groq

Groq's platform, detailed further in our [Groq documentation](../../integrations/groq.md) and on [Groq's official documentation](https://groq.com/), offers a unique approach to processing with its tensor architecture. This innovation significantly enhances the performance of structured output processing.

```bash
export GROQ_API_KEY="your-api-key"
```

```python
import os
from pydantic import BaseModel

import groq
import instructor

client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods
# to support the response_model parameter
client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)


# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.create(
    model="mixtral-8x7b-32768",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"

print(user)
#> name='jason' age=25
```

### Together AI

Together AI, when combined with Instructor, offers a seamless experience for developers looking to leverage structured outputs in their applications. For more details, refer to our [Together AI documentation](../../integrations/together.md) and explore the [patching guide](../../concepts/patching.md) to enhance your applications.

```bash
export TOGETHER_API_KEY="your-api-key"
```

```python
import os
from pydantic import BaseModel

import instructor
import openai

client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
)

client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)


class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"

print(user)
#> name='jason' age=25
```

### Mistral

For those interested in exploring the capabilities of Mistral Large with Instructor, we highly recommend checking out our comprehensive guide on [Mistral Large](../../integrations/mistral.md).

```python
import instructor
from pydantic import BaseModel

try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral

client = Mistral(api_key="your-api-key-here")
patched_chat = instructor.from_mistral(
    client=client,
    model="mistral-large-latest",
    mode=instructor.Mode.TOOLS,
)


class UserDetails(BaseModel):
    name: str
    age: int


resp = patched_chat.create(
    response_model=UserDetails,
    messages=[
        {
            "role": "user",
            "content": f'Extract the following entities: "Jason is 20"',
        },
    ],
)

print(resp)
#> name='Jason' age=20
```
