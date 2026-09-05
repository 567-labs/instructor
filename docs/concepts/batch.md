---
title: Batch Processing
description: Process multiple LLM requests efficiently using batch processing for 50% cost savings.
---

# Batch Processing

Batch processing lets you send multiple requests in a single operation, saving up to 50% on costs. Instructor supports batch processing across multiple providers.

## Supported Providers

| Provider | Models | Cost Savings |
|----------|--------|--------------|
| OpenAI | gpt-4o, gpt-4.1-mini, gpt-5.4-mini | 50% |
| Anthropic | claude-3-5-sonnet, claude-3-opus, claude-3-haiku | 50% |
| Google GenAI | gemini-2.5-flash, gemini-2.0-flash, gemini-pro | 50% |

## Basic Usage

```python
from instructor.batch import BatchProcessor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


processor = BatchProcessor("openai/gpt-4.1-mini", User)

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
if status["status"] in ["completed", "ended", "JOB_STATE_SUCCEEDED"]:
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


processor = BatchProcessor("openai/gpt-4.1-mini", User)

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
processor = BatchProcessor("openai/gpt-4.1-mini", User)
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

```python
processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)
```

### Google GenAI

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GCS_BUCKET="your-gcs-bucket-name"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

```python
processor = BatchProcessor("google/gemini-2.5-flash", User)
```

Required permissions: `roles/aiplatform.user` and `roles/storage.objectUser`.

## Processing Results

Results use a Maybe/Result pattern for type-safe handling:

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

## API Reference

| Method | Description |
|--------|-------------|
| `create_batch_from_messages(messages_list, file_path=None, ...)` | Create batch file or buffer |
| `submit_batch(file_path_or_buffer, metadata=None)` | Submit batch job, returns job ID |
| `get_batch_status(batch_id)` | Get job status |
| `retrieve_results(batch_id)` | Download and parse results |
| `parse_results(content)` | Parse raw results content |

## CLI Commands

```bash
# List batch jobs
instructor batch list --model "openai/gpt-4.1-mini"

# Create batch from file
instructor batch create-from-file --file-path batch.jsonl --model "openai/gpt-4.1-mini"

# Get batch results
instructor batch results --batch-id "batch_abc123" --output-file results.jsonl
```

## Best Practices

1. **Batch size**: Include at least 25,000 requests per job for optimal efficiency
2. **Cost optimization**: Use batch processing for non-urgent workloads
3. **Error handling**: Always check both successful and error results
4. **Timeouts**: Batch jobs have execution limits (24 hours for Google)
5. **Storage**: For Google, ensure GCS bucket is in the same region as your batch job

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing GCS_BUCKET (Google) | Set the `GCS_BUCKET` environment variable |
| Permission Denied (Google) | Add `aiplatform.user` and `storage.objectUser` roles |
| Invalid Model Name | Use format `provider/model-name` |
| Authentication Error | Verify API keys are set correctly |
