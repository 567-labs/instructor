---
title: Maybe Types and Optional Handling in Instructor
description: Handle optional and nullable data with Maybe types in Instructor. Learn to work with potentially missing fields and optional responses from LLMs.
---

# Handling Missing Data

The `Maybe` pattern is a concept in functional programming used for error handling. Instead of raising exceptions or returning `None`, you can use a `Maybe` type to encapsulate both the result and potential errors.

This pattern is particularly useful when making LLM calls, as providing language models with an escape hatch can effectively reduce hallucinations.

## Defining the Model

Using Pydantic, we'll first define the `UserDetail` and `MaybeUser` classes.

```python
from pydantic import BaseModel, Field
from typing import Optional


class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)


class MaybeUser(BaseModel):
    result: Optional[UserDetail] = Field(default=None)
    error: bool = Field(default=False)
    message: Optional[str] = Field(default=None)

    def __bool__(self):
        return self.result is not None
```

Notice that `MaybeUser` has a `result` field that is an optional `UserDetail` instance where the extracted data will be stored. The `error` field is a boolean that indicates whether an error occurred, and the `message` field is an optional string that contains the error message.

## Defining the function

Once we have the model defined, we can create a function that uses the `Maybe` pattern to extract the data.

```python
import instructor
from pydantic import BaseModel, Field
from typing import Optional

# This enables the `response_model` keyword
client = instructor.from_provider("openai/gpt-4.1-mini")


class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)


class MaybeUser(BaseModel):
    result: Optional[UserDetail] = Field(default=None)
    error: bool = Field(default=False)
    message: Optional[str] = Field(default=None)

    def __bool__(self):
        return self.result is not None


def extract(content: str) -> MaybeUser:
    return client.create(
        response_model=MaybeUser,
        messages=[
            {"role": "user", "content": f"Extract `{content}`"},
        ],
    )


user1 = extract("Jason is a 25-year-old scientist")
print(user1.model_dump_json(indent=2))
"""
{
  "result": {
    "age": 25,
    "name": "Jason",
    "role": "scientist"
  },
  "error": false,
  "message": null
}
"""

user2 = extract("Unknown user")
print(user2.model_dump_json(indent=2))
"""
{
  "result": null,
  "error": false,
  "message": null
}
"""
```

As you can see, when the data is extracted successfully, the `result` field contains the `UserDetail` instance. When an error occurs, the `error` field is set to `True`, and the `message` field contains the error message.

If you want to learn more about pattern matching, check out Pydantic's docs on [Structural Pattern Matching](https://docs.pydantic.dev/latest/concepts/models/#structural-pattern-matching)
