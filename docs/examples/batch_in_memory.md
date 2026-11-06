---
title: In-Memory Batch Processing for Serverless Applications
description: Learn how to use Instructor's in-memory batch processing feature for serverless deployments without disk I/O.
---

## See Also

- [Batch Processing](./batch_job_oai.md) - File-based batch processing
- [Bulk Classification](./bulk_classification.md) - Process multiple classifications
- [from_provider Guide](../concepts/from_provider.md#async-clients) - Async client setup
- [Cost Optimization](./batch_job_oai.md) - Reduce API costs with batch processing

# In-Memory Batch Processing for Serverless

This guide demonstrates how to use Instructor's in-memory batch processing feature, which is perfect for serverless deployments and applications that need to avoid disk I/O.

## Overview

In-memory batch processing allows you to create and submit batch requests without writing to disk, using BytesIO buffers instead of files. This is ideal for:

- **Serverless environments** (AWS Lambda, Google Cloud Functions, Azure Functions)
- **Containerized applications** with read-only file systems
- **Security-sensitive applications** that avoid temporary files
- **High-performance applications** that minimize I/O overhead

## Quick Start

```python
import time
from pydantic import BaseModel
from instructor.batch.processor import BatchProcessor


class User(BaseModel):
    """User model for extraction."""

    name: str
    age: int
    email: str


def main():
    # Initialize batch processor
    processor = BatchProcessor("openai/gpt-5-nano", User)

    # Sample messages for batch processing
    messages_list = [
        [
            {"role": "system", "content": "Extract user information from the text."},
            {
                "role": "user",
                "content": "John Doe is 25 years old and his email is john@example.com",
            },
        ],
        [
            {"role": "system", "content": "Extract user information from the text."},
            {
                "role": "user",
                "content": "Jane Smith, age 30, can be reached at jane.smith@company.com",
            },
        ],
        [
            {"role": "system", "content": "Extract user information from the text."},
            {
                "role": "user",
                "content": "Bob Wilson (bob.wilson@email.com) is 28 years old",
            },
        ],
    ]

    # Create batch in memory (no file_path specified)
    batch_buffer = processor.create_batch_from_messages(
        messages_list,
        file_path=None,  # This triggers in-memory mode
        max_tokens=150,
        temperature=0.1,
    )

    print(f"Created batch buffer: {type(batch_buffer)}")
    print(f"Buffer size: {len(batch_buffer.getvalue())} bytes")

    # Submit the batch using the in-memory buffer
    batch_id = processor.submit_batch(
        batch_buffer, metadata={"description": "In-memory batch example"}
    )

    print(f"Batch submitted successfully! Batch ID: {batch_id}")

    # Poll for completion
    print("Waiting for batch to complete...")
    max_wait_time = 300  # 5 minutes max
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        status = processor.get_batch_status(batch_id)
        current_status = status.get("status", "unknown")

        print(f"Current status: {current_status}")

        if current_status in ["completed", "failed", "cancelled", "expired"]:
            break

        time.sleep(10)

    # Retrieve and process results
    if status.get("status") == "completed":
        print("Batch completed! Retrieving results...")

        results = processor.get_results(batch_id)

        successful_results = [r for r in results if hasattr(r, "result")]
        error_results = [r for r in results if hasattr(r, "error_message")]

        print(f"Total results: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Errors: {len(error_results)}")

        # Show successful extractions
        if successful_results:
            print("\nExtracted Users:")
            for result in successful_results:
                user = result.result
                print(f"   - {user.name}, {user.age} years old, {user.email}")

        # Show any errors
        if error_results:
            print("\nErrors encountered:")
            for error in error_results:
                print(f"   - {error.custom_id}: {error.error_message}")


if __name__ == "__main__":
    main()
```

## File vs In-Memory Comparison

### Traditional File-Based Approach

```python
# File-based approach
processor = BatchProcessor("openai/gpt-5-nano", User)

# Creates file on disk
file_path = processor.create_batch_from_messages(
    messages_list,
    file_path="temp_batch.jsonl",  # Specify file path
    max_tokens=150,
    temperature=0.1,
)

# Submit using file path
batch_id = processor.submit_batch(file_path)

# Remember to clean up
import os

if os.path.exists(file_path):
    os.remove(file_path)
```

### New In-Memory Approach

```python
# In-memory approach
processor = BatchProcessor("openai/gpt-5-nano", User)

# Creates BytesIO buffer in memory
buffer = processor.create_batch_from_messages(
    messages_list,
    file_path=None,  # No file path = in-memory
    max_tokens=150,
    temperature=0.1,
)

# Submit using buffer
batch_id = processor.submit_batch(buffer)

# No cleanup required - buffer is automatically garbage collected
```

## Benefits of In-Memory Processing

### ✅ Perfect for Serverless

```python
# AWS Lambda example
import json


def lambda_handler(event, _context):
    """AWS Lambda function using in-memory batch processing."""

    # Extract data from event
    messages_list = event.get("messages", [])

    # Process in memory - no disk I/O
    processor = BatchProcessor("openai/gpt-5-nano", User)
    buffer = processor.create_batch_from_messages(
        messages_list,
        file_path=None,  # Essential for Lambda
    )

    batch_id = processor.submit_batch(buffer)

    return {
        'statusCode': 200,
        'body': json.dumps(
            {'batch_id': batch_id, 'message': 'Batch submitted successfully'}
        ),
    }
```

### ✅ Memory Efficient

```python
# Check buffer size before submission
buffer = processor.create_batch_from_messages(messages_list, file_path=None)

print(f"Buffer size: {len(buffer.getvalue())} bytes")
print(f"Buffer type: {type(buffer)}")

# Buffer content is accessible
buffer.seek(0)
content_preview = buffer.read(200).decode("utf-8")
print(f"Preview: {content_preview}...")

# Reset for submission
buffer.seek(0)
batch_id = processor.submit_batch(buffer)
```

### ✅ Security Benefits

```python
# No temporary files on disk
# No file permissions to manage
# No cleanup required
# Buffer is automatically garbage collected

processor = BatchProcessor("openai/gpt-5-nano", User)

# This approach leaves no trace on the file system
buffer = processor.create_batch_from_messages(
    sensitive_messages,
    file_path=None,  # Keeps everything in memory
)

batch_id = processor.submit_batch(buffer)
# When buffer goes out of scope, it's automatically cleaned up
```

## Error Handling

```python
try:
    # Create batch buffer
    buffer = processor.create_batch_from_messages(
        messages_list,
        file_path=None,
    )

    # Submit batch
    batch_id = processor.submit_batch(buffer)

    # Process results
    results = processor.get_results(batch_id)

except Exception as e:
    print(f"Error during batch processing: {e}")
    #> Error during batch processing: name 'processor' is not defined
    # No file cleanup needed with in-memory approach
```

## Provider Support

All providers support in-memory batch processing:

### OpenAI
```python
processor = BatchProcessor("openai/gpt-5-nano", User)
buffer = processor.create_batch_from_messages(messages_list, file_path=None)
batch_id = processor.submit_batch(buffer)
```

### Anthropic
```python
processor = BatchProcessor("anthropic/claude-3-5-sonnet-20241022", User)
buffer = processor.create_batch_from_messages(messages_list, file_path=None)
batch_id = processor.submit_batch(buffer)
```

### Google GenAI
```python
processor = BatchProcessor("google/gemini-2.5-flash", User)
buffer = processor.create_batch_from_messages(messages_list, file_path=None)
batch_id = processor.submit_batch(buffer)
```

## Best Practices

1. **Always set `file_path=None`** to enable in-memory mode
2. **Monitor buffer size** for large batches to avoid memory issues
3. **Use appropriate models** that support JSON schema (e.g., gpt-4o-mini)
4. **Handle errors gracefully** - no file cleanup needed
5. **Consider memory limits** in serverless environments

## Limitations

- **Memory usage**: Large batches may consume significant memory
- **No debugging files**: Can't inspect batch files for troubleshooting
- **Temporary storage**: Buffer contents are lost if not submitted immediately

## Troubleshooting

### Buffer Size Issues
```python
# Check buffer size before submission
buffer = processor.create_batch_from_messages(messages_list, file_path=None)
size_mb = len(buffer.getvalue()) / (1024 * 1024)
print(f"Buffer size: {size_mb:.2f} MB")

if size_mb > 100:  # Adjust threshold as needed
    print("Warning: Large buffer size, consider splitting batch")
```

### Memory Monitoring
```python
import psutil
import os

# Check memory usage
process = psutil.Process(os.getpid())
memory_before = process.memory_info().rss / 1024 / 1024  # MB

buffer = processor.create_batch_from_messages(messages_list, file_path=None)

memory_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory increase: {memory_after - memory_before:.2f} MB")
```

This in-memory approach makes Instructor's batch processing perfect for modern serverless and containerized applications while maintaining the same powerful API and provider support.
