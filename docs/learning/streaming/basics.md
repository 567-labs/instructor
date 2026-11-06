---
title: Streaming Basics with Instructor
description: Learn how to use streaming to receive partial structured responses from LLMs as they are generated.
---

# Streaming Basics

Streaming allows you to receive parts of a structured response as they're generated, rather than waiting for the complete response.

## Why Use Streaming?

Streaming offers several benefits:

1. **Faster Perceived Response**: Users see results immediately
2. **Progressive UI Updates**: Update your interface as data arrives
3. **Processing While Generating**: Start using data before the complete response is ready

```
Without Streaming:
┌─────────┐             ┌─────────────────────┐
│ Request │─── Wait ───>│ Complete Response   │
└─────────┘             └─────────────────────┘

With Streaming:
┌─────────┐    ┌───────┐    ┌───────┐    ┌───────┐
│ Request │───>│Part 1 │───>│Part 2 │───>│Part 3 │─── ...
└─────────┘    └───────┘    └───────┘    └───────┘
```

## Simple Example

Here's how to stream a structured response:

```python
import instructor
from pydantic import BaseModel
# Define your data structure
class UserProfile(BaseModel):
    name: str
    bio: str
    interests: list[str]

# Set up client
client = instructor.from_provider("openai/gpt-5-nano")

# Enable streaming
for partial in client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "Generate a profile for Alex Chen"}
    ],
    response_model=UserProfile,
    stream=True  # This enables streaming
):
    # Print each update as it arrives
    print("\nUpdate received:")

    # Access available fields
    if hasattr(partial, "name") and partial.name:
        print(f"Name: {partial.name}")
    if hasattr(partial, "bio") and partial.bio:
        print(f"Bio: {partial.bio[:30]}...")
    if hasattr(partial, "interests") and partial.interests:
        print(f"Interests: {', '.join(partial.interests)}")
```

## How Streaming Works

When streaming with Instructor:

1. Enable streaming with `stream=True`
2. The method returns an iterator of partial responses
3. Each partial contains fields that have been completed so far
4. You check for fields using `hasattr()` since they appear incrementally
5. The final iteration contains the complete response

## Progress Tracking Example

Here's a simple way to track progress:

```python
import instructor
from pydantic import BaseModel
client = instructor.from_provider("openai/gpt-5-nano")

class Report(BaseModel):
    title: str
    summary: str
    conclusion: str

# Track completed fields
completed = set()
total_fields = 3  # Number of fields in our model

for partial in client.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "Generate a report on climate change"}
    ],
    response_model=Report,
    stream=True
):
    # Check which fields are complete
    for field in ["title", "summary", "conclusion"]:
        if hasattr(partial, field) and getattr(partial, field) and field not in completed:
            completed.add(field)
            percent = (len(completed) / total_fields) * 100
            print(f"Received: {field} - {percent:.0f}% complete")
```

## Next Steps

- Explore [Streaming Lists](lists.md) for handling collections
- Learn about [Validation with Streaming](../validation/basics.md)