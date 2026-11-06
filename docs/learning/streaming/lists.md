---
title: Streaming Lists with Instructor
description: Learn how to stream lists of structured objects from LLMs, processing collection items as they are generated for better responsiveness.
---

# Streaming Lists

This guide explains how to stream lists of structured data with Instructor. Streaming lists allows you to process collection items as they're generated, improving responsiveness for larger outputs.

## Basic List Streaming

Here's how to stream a list of structured objects:

```python
from typing import Iterable
import instructor
from pydantic import BaseModel, Field
# Initialize the client
client = instructor.from_provider("openai/gpt-5-nano")

class Book(BaseModel):
    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Book author")
    year: int = Field(..., description="Publication year")

# Stream a list of books
for book in client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "List 5 classic science fiction books"}
    ],
    response_model=Iterable[Book],
):
    print(f"Received: {book.title} by {book.author} ({book.year})")
```

This example shows how to:
1. Define a Pydantic model for each list item
2. Use Python's typing system to specify a list
3. Process each item as it arrives in the stream

## Real-world Example: Task Generation

Here's a practical example of streaming a list of tasks with progress tracking:

```python
from typing import Iterable
import instructor
from pydantic import BaseModel, Field
import time
client = instructor.from_provider("openai/gpt-5-nano")


class Task(BaseModel):
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Detailed task description")
    priority: str = Field(..., description="Task priority (High/Medium/Low)")
    estimated_hours: float = Field(..., description="Estimated hours to complete")


print("Generating project tasks...")
start_time = time.time()
received_tasks = 0

for task in client.create(
    model="gpt-5.4-mini",
    messages=[
        {
            "role": "user",
            "content": "Generate a list of 5 tasks for building a personal website",
        }
    ],
    response_model=Iterable[Task],
    stream=True,
):
    received_tasks += 1
    print(f"\nTask {received_tasks}: {task.title} (Priority: {task.priority})")
    print(f"Description: {task.description[:100]}...")
    print(f"Estimated time: {task.estimated_hours} hours")

    # Calculate progress percentage based on expected items
    progress = (received_tasks / 5) * 100
    print(f"Progress: {progress:.0f}%")

elapsed_time = time.time() - start_time
print(f"\nAll {received_tasks} tasks generated in {elapsed_time:.2f} seconds")

```

## Related Resources

- [Streaming Basics](./basics.md) - Fundamentals of streaming structured outputs
- [List Extraction](../../learning/patterns/list_extraction.md) - Core concepts for working with lists
- [Validation Basics](../../learning/validation/basics.md) - Understanding validation for streaming
- [Streaming API](../../concepts/partial.md) - Technical details on the streaming implementation

## Next Steps

- Learn about [Validation](../../learning/validation/basics.md) to ensure your streamed data is valid
- Explore [Field Validation](../../learning/validation/field_level_validation.md) for more control
- See [Async Support](../../integrations/index.md) for integrating streaming with your specific provider when writing asynchronous code