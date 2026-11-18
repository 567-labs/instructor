---
title: Batch Processing
description: Process multiple LLM requests efficiently using batch processing for 50% cost savings.
---

# Batch Processing

Batch processing lets you send multiple requests in a single operation, saving up to 50% on costs. Instructor supports batch processing across multiple providers.

## Supported Providers

| Provider | Models | Cost Savings |
|----------|--------|--------------|
| OpenAI | `gpt-6-luna` | 50% |
| Anthropic | `claude-haiku-5-5` | 50% |
| Mistral | mistral-small-latest, mistral-medium-latest, mistral-large-latest | 50% |

`BatchProcessor` supports OpenAI, Anthropic, and Mistral. Other providers supported
by `instructor.from_provider`, including Google GenAI, are not available through
`BatchProcessor`.

## Basic Usage

```python
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


processor = BatchProcessor("openai/gpt-6-luna", User)

messages_list = [
    [
        {"role": "system", "content": "Extract user information from text."},
        {"role": "user", "content": "Hi, I'm Alice and I'm 28 years old."},
    ],
    [
        {"role": "system", "content": "Extract user information from text."},
        {"role": "user", "content": "Hello, I'm Bob, 35 years old."},
    ],
]

# Create batch file
processor.create_batch_from_messages(
    file_path="batch_requests.jsonl",
    messages_list=messages_list,
    max_tokens=200,
    temperature=0.1,
)

# Submit batch job
batch_id = processor.submit_batch("batch_requests.jsonl")
print(f"Batch job submitted: {batch_id}")

# Check status and retrieve results
status = processor.get_batch_status(batch_id)
if status["status"] == "completed":
    from instructor.batch import extract_results

    all_results = processor.retrieve_results(batch_id)
    for user in extract_results(all_results):
        print(f"Name: {user.name}, Age: {user.age}")
```

## In-Memory Processing

For serverless deployments, use in-memory mode by setting `file_path=None`:

```python
import time
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


processor = BatchProcessor("openai/gpt-6-luna", User)

messages_list = [
    [{"role": "user", "content": "Extract: John is 25 years old"}],
    [{"role": "user", "content": "Extract: Jane is 30 years old"}],
]

# Create in-memory buffer (no file_path)
buffer = processor.create_batch_from_messages(
    messages_list,
    file_path=None,
    max_tokens=150,
)

# Submit and poll for results
batch_id = processor.submit_batch(buffer)

while True:
    status = processor.get_batch_status(batch_id)
    if status.get("status") in ["completed", "failed", "cancelled"]:
        break
    time.sleep(10)

if status.get("status") == "completed":
    results = processor.get_results(batch_id)
    for r in results:
        if hasattr(r, "result"):
            print(f"{r.result.name}, {r.result.age}")
```

### When to Use Each Approach

| Use Case | Approach |
|----------|----------|
| Serverless (Lambda, Cloud Functions) | In-memory |
| Large batch jobs | File-based |
| Security-sensitive environments | In-memory |
| Debugging/audit requirements | File-based |

## Provider Setup

### OpenAI

```bash
export OPENAI_API_KEY="your-openai-key"
```

```python
processor = BatchProcessor("openai/gpt-6-luna", User)
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

```python
processor = BatchProcessor("anthropic/claude-haiku-5-5", User)
```

### Mistral

```bash
export MISTRAL_API_KEY="your-mistral-key"
```

```python
processor = BatchProcessor("mistral/mistral-small-latest", User)
```

Mistral sets the model on the batch job rather than on each request, so the model
you pass to `BatchProcessor` is applied to the whole job. The supported Mistral SDK range starts at `mistralai>=1.5.1`.
Deleting a job needs `mistralai>=2.0.0`, because version 1.x has no delete
endpoint.

Mistral batch jobs report their own status values, so poll for `SUCCESS`:

```python
import time

batch_id = processor.submit_batch("batch_requests.jsonl")

status = processor.get_batch_status(batch_id)
while status["status"] not in ["SUCCESS", "FAILED", "CANCELLED", "TIMEOUT_EXCEEDED"]:
    time.sleep(10)
    status = processor.get_batch_status(batch_id)

results = processor.retrieve_results(batch_id)
```

## Processing Results

OpenAI batch requests use strict JSON schemas. All named fields are required in
the output, even when the Pydantic model has defaults. Nullable fields still
allow `null`. Arbitrary-key dictionaries are not supported by this strict batch
path; use named model fields or a list of key/value objects instead. Anthropic
batch requests keep their provider-specific tool schema, including mapping
value types.

Results use a Maybe/Result pattern for type-safe handling:

For completed or expired OpenAI batches, retrieval reads both the output and error
files. Expiration does not discard responses that finished within the time window;
unfinished requests are recorded in the error file.
For ended Anthropic batches, it reads every result, including errors. A batch
where every request failed returns `BatchError` records rather than raising
because no requests succeeded. Errors retain the request's `custom_id`, provider
error details, and original record in `raw_data`. Jobs that are still running
cannot be retrieved.

```python
from instructor.batch import (
    filter_successful,
    filter_errors,
    extract_results,
    get_results_by_custom_id,
)

all_results = processor.retrieve_results(batch_id)

# Filter by type
successful = filter_successful(all_results)  # List[BatchSuccess[T]]
errors = filter_errors(all_results)  # List[BatchError]
objects = extract_results(all_results)  # List[T]

# Access by custom_id
by_id = get_results_by_custom_id(all_results)
if "request-1" in by_id:
    result = by_id["request-1"]
    if result.success:
        print(f"Success: {result.result}")
    else:
        print(f"Error: {result.error_message}")
```

Responses that hit the token limit return `BatchError` with `error_type="incomplete_output"`.
This also applies when the partial JSON is valid or missing fields have defaults.
The error keeps the request's `custom_id` and raw response so you can retry it with a higher token limit.

### Custom IDs

By default, requests get the IDs `request-0`, `request-1`, and so on. Pass `custom_ids` to use your own IDs. This makes it easier to match results, including failed ones, back to your data:

```python
processor.create_batch_from_messages(
    messages_list,
    file_path="batch_requests.jsonl",
    custom_ids=["user_1", "user_2", "user_3"],  # one per conversation, same order
)
```

IDs must be unique and match the length of `messages_list`. Anthropic also requires IDs to match `^[a-zA-Z0-9_-]{1,64}$`. Invalid IDs raise a `ValueError` before any request is written.

## API Reference

| Method | Description |
|--------|-------------|
| `create_batch_from_messages(messages_list, file_path=None, ..., custom_ids=None)` | Create batch file or buffer |
| `submit_batch(file_path_or_buffer, metadata=None)` | Submit batch job, returns job ID |
| `get_batch_status(batch_id)` | Get job status |
| `retrieve_results(batch_id)` | Download and parse results |
| `parse_results(content)` | Parse raw results content |

## CLI Commands

```bash
# List batch jobs
instructor batch list --model "openai/gpt-6-luna"

# Create batch from file
instructor batch create-from-file --file-path batch.jsonl --model "openai/gpt-6-luna"

# Get batch results
instructor batch results --batch-id "batch_abc123" --output-file results.jsonl
```

## Best Practices

1. **Batch size**: Include at least 25,000 requests per job for optimal efficiency
2. **Cost optimization**: Use batch processing for non-urgent workloads
3. **Error handling**: Always check both successful and error results
4. **Timeouts**: Check the provider's batch completion window. OpenAI submissions
   use a 24-hour window by default.

## Troubleshooting

For OpenAI request failures, `BatchError.error_type` contains the provider type when present, then its code. `raw_data` preserves the complete record, including both fields. String errors retain their message. An unsuccessful HTTP response without an error body reports its status code.

| Issue | Solution |
|-------|----------|
| Unsupported provider: google | `BatchProcessor` supports only OpenAI, Anthropic, and Mistral |
| Invalid Model Name | Use format `provider/model-name` |
| Authentication Error | Verify API keys are set correctly |
| Delete not supported (Mistral) | Upgrade to `mistralai>=2.0.0` |
