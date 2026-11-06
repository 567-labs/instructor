---
title: Working with Types in Instructor
description: Learn how to use different data types with Instructor, from simple primitives to complex types.
---

# Working with Types in Instructor

Instructor supports a wide range of types for your structured outputs, from simple primitives to complex nested structures.

## Simple Types

In addition to `pydantic.BaseModel` (the recommended approach), Instructor also supports:

- Primitive types: `str`, `int`, `float`, `bool`
- Collection types: `List`, `Dict`
- Type composition: `Union`, `Literal`, `Optional`
- Specialized outputs: [Iterable](lists.md), [Partial](partial.md)

You can use these types directly in your `response_model` parameter without wrapping them in a Pydantic model.

For better documentation and control, use `typing.Annotated` to add more context to your types.

## What happens behind the scenes?

We will actually wrap the response model with a `pydantic.BaseModel` of the following form:

```python
from typing import Annotated
from pydantic import create_model, Field, BaseModel

typehint = Annotated[bool, Field(description="Sample Description")]

model = create_model("Response", content=(typehint, ...), __base__=BaseModel)

print(model.model_json_schema())
"""
{
    'properties': {
        'content': {
            'description': 'Sample Description',
            'title': 'Content',
            'type': 'boolean',
        }
    },
    'required': ['content'],
    'title': 'Response',
    'type': 'object',
}
"""
```

## Primitive Types (str, int, float, bool)

```python
import instructor

client = instructor.from_provider("openai/gpt-4.1-mini")

# Response model with simple types like str, int, float, bool
resp = client.create(
    response_model=bool,
    messages=[
        {
            "role": "user",
            "content": "Is it true that Paris is the capital of France?",
        },
    ],
)
assert resp is True, "Paris is the capital of France"
print(resp)
#> True
```

## Annotated

Annotations can be used to add more information about the type. This can be useful for adding descriptions to the type, along with more complex information like field names, and more.

```python
import instructor
from typing import Annotated
from pydantic import Field

client = instructor.from_provider("openai/gpt-4.1-mini")

UpperCaseStr = Annotated[str, Field(description="string must be upper case")]

# Response model with simple types like str, int, float, bool
resp = client.create(
    response_model=UpperCaseStr,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of france?",
        },
    ],
)
assert resp == "PARIS", "Paris is the capital of France"
print(resp)
#> PARIS
```

## Literal

When doing simple classification Literals go quite well, they support literal of string, int, bool.

```python
import instructor
from typing import Literal

client = instructor.from_provider("openai/gpt-4.1-mini")

resp = client.create(
    response_model=Literal["BILLING", "SHIPPING"],
    messages=[
        {
            "role": "user",
            "content": "Classify the following messages: 'I am having trouble with my billing'",
        },
    ],
)
assert resp == "BILLING"
print(resp)
#> BILLING
```

## Enum

Enums are harder to get right without some addition promping but are useful if these are values that are shared across the application.

```python
import instructor
from enum import Enum


class Label(str, Enum):
    BILLING = "BILLING"
    SHIPPING = "SHIPPING"


client = instructor.from_provider("openai/gpt-4.1-mini")

resp = client.create(
    response_model=Label,
    messages=[
        {
            "role": "user",
            "content": "Classify the following messages: 'I am having trouble with my billing'",
        },
    ],
)
assert resp == Label.BILLING
print(resp)
#> BILLING
```

## List

```python
import instructor
from typing import List

client = instructor.from_provider("openai/gpt-4.1-mini")

resp = client.create(
    response_model=List[int],
    messages=[
        {
            "role": "user",
            "content": "Give me the first 5 prime numbers",
        },
    ],
)

assert resp == [2, 3, 5, 7, 11]
print(resp)
#> [2, 3, 5, 7, 11]
```

## Union

Union is a great way to handle multiple types of responses, similar to multiple function calls but not limited to the function calling api, like in JSON_SCHEMA modes.

```python
import instructor
from pydantic import BaseModel
from typing import Union

client = instructor.from_provider("openai/gpt-4.1-mini")


class Add(BaseModel):
    a: int
    b: int


class Weather(BaseModel):
    location: str


resp = client.create(
    response_model=Union[Add, Weather],
    messages=[
        {
            "role": "user",
            "content": "What is 5 + 5?",
        },
    ],
)

assert resp == Add(a=5, b=5)
print(resp)
#> a=5 b=5
```

## See Also

- [Response Models](./models.md) - Using Pydantic models for structured outputs
- [Enums](./enums.md) - Working with enumerated types
- [Union Types](./unions.md) - Handling multiple possible types
- [Lists](./lists.md) - Working with collections
- [Optional Fields](../learning/patterns/optional_fields.md) - Handling missing data

## Complex Types

### Pandas DataFrame

This is a more complex example, where we use a custom type to convert markdown to a pandas DataFrame.

```python
from io import StringIO
from typing import Annotated, Any
from pydantic import BeforeValidator, PlainSerializer, InstanceOf, WithJsonSchema
import pandas as pd
import instructor


def md_to_df(data: Any) -> Any:
    # Convert markdown to DataFrame
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Process data
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .applymap(lambda x: x.strip())
        )
    return data


MarkdownDataFrame = Annotated[
    # Validates final type
    InstanceOf[pd.DataFrame],
    # Converts markdown to DataFrame
    BeforeValidator(md_to_df),
    # Converts DataFrame to markdown on model_dump_json
    PlainSerializer(lambda df: df.to_markdown()),
    # Adds a description to the type
    WithJsonSchema(
        {
            "type": "string",
            "description": """
            The markdown representation of the table,
            each one should be tidy, do not try to join
            tables that should be separate""",
        }
    ),
]


client = instructor.from_provider("openai/gpt-4.1-mini")

resp = client.create(
    response_model=MarkdownDataFrame,
    messages=[
        {
            "role": "user",
            "content": "Jason is 20, Sarah is 30, and John is 40",
        },
    ],
)

assert isinstance(resp, pd.DataFrame)
print(resp)
"""
        Age
 Name
Jason     20
Sarah     30
John      40
"""
```

### Lists of Unions

Just like Unions we can use List of Unions to represent multiple types of responses. This will feel similar to the parallel function calls but not limited to the function calling api, like in JSON_SCHEMA modes.

```python
import instructor
from pydantic import BaseModel
from typing import Union, List

client = instructor.from_provider("openai/gpt-4.1-mini")


class Weather(BaseModel, frozen=True):
    location: str


class Add(BaseModel, frozen=True):
    a: int
    b: int


resp = client.create(
    response_model=List[Union[Add, Weather]],
    messages=[
        {
            "role": "user",
            "content": "Add 5 and 5, and also whats the weather in Toronto?",
        },
    ],
)

assert resp == [Add(a=5, b=5), Weather(location="Toronto")]
print(resp)
#> [Add(a=5, b=5), Weather(location='Toronto')]
```
