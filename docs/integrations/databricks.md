---
title: Databricks
description: Guide to using instructor with Databricks models
---

# Structured outputs with Databricks, a complete guide w/ instructor

[Databricks](https://www.databricks.com/) provides an AI platform with access to various models. This guide shows how to use instructor with Databricks to get structured outputs.

## Quick Start

First, install the required packages:

```bash
uv pip install instructor openai
```

Set your Databricks workspace URL and token as environment variables:

```bash
export DATABRICKS_TOKEN="your_personal_access_token"
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
```

`DATABRICKS_API_KEY` and `DATABRICKS_WORKSPACE_URL` are also supported if you prefer those names. The provider appends `/serving-endpoints` automatically, so the host only needs the base workspace URL.

## Basic Example

Here's how to extract structured data from Databricks models:

```python
import instructor
from pydantic import BaseModel

# Initialize the client; host and token are read from the environment
client = instructor.from_provider(
    "databricks/dbrx-instruct",
    mode=instructor.Mode.TOOLS,
)

# Define your data structure
class UserExtract(BaseModel):
    name: str
    age: int

# Extract structured data
user = client.create(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user)
# Output: UserExtract(name='Jason', age=25)
```

If you need to point at a different workspace or testing endpoint, pass `base_url="https://alt-workspace.cloud.databricks.com/serving-endpoints"`. The helper will use that value as-is without adding another suffix.

### Async Example

```python
async_client = instructor.from_provider(
    "databricks/dbrx-instruct",
    async_client=True,
    mode=instructor.Mode.TOOLS,
)
```

## Supported Modes

Databricks supports the same modes as OpenAI:

- `Mode.TOOLS`
- `Mode.JSON`
- `Mode.FUNCTIONS`
- `Mode.PARALLEL_TOOLS`
- `Mode.MD_JSON`
- `Mode.TOOLS_STRICT`
- `Mode.JSON_O1`

## Models

Databricks provides access to various models depending on your setup, including:

- Foundation models hosted on Databricks
- Custom fine-tuned models
- Open source models deployed on Databricks

