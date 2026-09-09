---
title: Handling Non-Streaming Requests in OpenAI with Usage Tracking
description: Learn how to manage non-streaming requests in OpenAI, track token usage, and handle exceptions with Python.
---

## See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](./from_provider.md) - Detailed client configuration
- [Response Models](./models.md) - Working with Pydantic models
- [Raw Response](./raw_response.md) - Access original LLM responses

The easiest way to inspect usage for a non-streaming request is to access the completion returned by `create_with_completion`.

```python
import instructor

from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserExtract(BaseModel):
    name: str
    age: int


user, completion = client.create_with_completion(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(completion.usage)
"""
CompletionUsage(
    completion_tokens=10,
    prompt_tokens=79,
    total_tokens=89,
    completion_tokens_details=CompletionTokensDetails(
        accepted_prediction_tokens=None,
        audio_tokens=0,
        reasoning_tokens=0,
        rejected_prediction_tokens=None,
    ),
    prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
)
"""
```

You can catch an IncompleteOutputException whenever the context length is exceeded and react accordingly, such as by trimming your prompt by the number of exceeding tokens.

```python
from instructor.core.exceptions import IncompleteOutputException
import instructor
from pydantic import BaseModel

client = instructor.from_provider("openai/gpt-4.1-mini")


class UserExtract(BaseModel):
    name: str
    age: int


try:
    client.create_with_completion(
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
except IncompleteOutputException as e:
    token_count = e.last_completion.usage.total_tokens  # type: ignore
    # your logic here
```

## Retry totals and provider differences

For structured responses with OpenAI Chat Completions or stable Anthropic
Messages usage, Instructor accumulates usage across validation retries and
updates the returned completion's usage in place. After retries,
`completion.usage` therefore contains cumulative values, not just the final
attempt's original usage. Do not sum usage from multiple retained retry
completions. Successful Pydantic models and `ListResponse` results also expose
`_total_usage` when all observed responses have compatible usage metadata.

This accounting is not available for every SDK or API. OpenAI Responses uses a
different usage type, as do Anthropic beta Messages and most native provider
SDKs. Their raw usage may be available without a cumulative `_total_usage`.
Missing optional usage fields are not evidence of zero consumption; current
accumulation can fill some absent fields with zero or carry forward a known
subtotal. Streaming parsed results do not provide a final cumulative usage
record, and an interrupted stream may never receive final usage.

Token categories have different relationships across providers. OpenAI cached
prompt tokens and reasoning completion tokens are breakdowns of the enclosing
totals. Anthropic cache-read and cache-creation input tokens must be added to
its uncached `input_tokens` to get inclusive input. Do not sum every numeric
usage field or derive a bill from total tokens alone.

Use [`completion:usage`](hooks.md#cumulative-usage) for copied cumulative
snapshots. Replace the previous snapshot instead of summing these events.
To preserve per-attempt provider metadata, copy it immediately inside a
`completion:response` hook, before Instructor updates the response:

```python
from copy import deepcopy

attempt_usage = []


def record_attempt(response):
    attempt_usage.append(deepcopy(getattr(response, "usage", None)))


client.on("completion:response", record_attempt)
```

This example applies to non-streaming responses with a `usage` attribute;
Google responses use `usage_metadata`, and native xAI follows a separate path.
A local response-cache hit can restore historical completion usage without a
new provider request. It is different from provider prompt caching.
