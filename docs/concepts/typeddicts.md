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

Instructor converts a `TypedDict` into a Pydantic model. Keys starting with an
underscore, such as `_id`, use field aliases because Pydantic reserves those
attribute names. Use `response.model_dump(by_alias=True)` to export the original
keys. Required keys must still be present, and `NotRequired` keys may be omitted.
