# Literal Types with Streaming

This example demonstrates how to use Literal types with streaming in Instructor.

## The Problem

Previously, when using `Literal` types with streaming mode and `Partial`, validation would fail for partial strings. For example, if the model is outputting "Alice" character by character, the intermediate values like "A", "Al", "Ali" would cause validation errors because they're not valid literal values.

## The Solution

Instructor now handles `Literal` types specially during streaming. When using `Partial`, literal fields accept any string value during streaming, allowing the model to output values character by character. The final validation ensures the value matches one of the allowed literal values.

## Usage

```python
from instructor import Partial
from pydantic import BaseModel
from typing import Literal

class User(BaseModel):
    name: Literal["Bob", "Alice", "John"]
    role: Literal["admin", "user", "guest"]
    age: int

# Use Partial[User] for streaming
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Create a user named Alice"}],
    response_model=Partial[User],
    stream=True,
)

for partial_user in stream:
    print(partial_user)  # Works even with partial literal values!
```

## Running the Example

```bash
python run.py
```

This will demonstrate:
1. Non-streaming mode (works normally)
2. Direct streaming without Partial (fails as expected)
3. Streaming with Partial (now works with Literal types!)
4. Complex models with multiple Literal fields