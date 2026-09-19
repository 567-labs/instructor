---
title: Using TypedDicts with OpenAI API
description: Learn how to utilize TypedDicts in Python with the OpenAI API for structured data responses.
---

---
title: TypedDict Support in Instructor - Dictionary Validation
description: Use Python TypedDict for type-safe dictionary structures with Instructor. Validate dictionary schemas without Pydantic models for lightweight structured outputs.
---

# TypedDicts

We also support typed dicts.

Fields can use `Required`, `NotRequired`, and `ReadOnly` from `typing_extensions`,
including combinations such as `ReadOnly[NotRequired[int]]`. Instructor preserves
whether each key is required and any `Annotated` field constraints when building
the response schema. `ReadOnly` describes static typing; the generated response
model follows the usual Pydantic mutation behavior.

```python
from typing_extensions import TypedDict
import instructor


class User(TypedDict):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")


response = client.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Timothy is a man from New York who is turning 32 this year",
        }
    ],
)
```
