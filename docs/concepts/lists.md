---
title: Streaming Lists with Instructor - Extract Multiple Objects
description: Learn how to extract multiple structured objects from a single LLM call using streaming lists. Stream collections of Pydantic models as they're generated.
---

# Multi-task and Streaming

A common use case of structured extraction is defining a single schema class and then making another schema to create a list to do multiple extraction

```python
from typing import List
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


class Users(BaseModel):
    users: List[User]


print(Users.model_json_schema())
"""
{
    '$defs': {
        'User': {
            'properties': {
                'name': {'title': 'Name', 'type': 'string'},
                'age': {'title': 'Age', 'type': 'integer'},
            },
            'required': ['name', 'age'],
            'title': 'User',
            'type': 'object',
        }
    },
    'properties': {
        'users': {'items': {'$ref': '#/$defs/User'}, 'title': 'Users', 'type': 'array'}
    },
    'required': ['users'],
    'title': 'Users',
    'type': 'object',
}
"""
```

Defining a task and creating a list of classes is a common enough pattern that we make this convenient by making use of `Iterable[T]`. This lets us dynamically create a new class that:

1. Has dynamic docstrings and class name based on the task
2. Support streaming by collecting tokens until a task is received back out.

## Extracting Tasks using Iterable

By using `Iterable` you get a very convenient class with prompts and names automatically defined:

```python
import instructor
from typing import Iterable
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "openai/gpt-4.1-mini-1106",
    mode=instructor.Mode.JSON,
)

users = client.create(
    temperature=0.1,
    response_model=Iterable[User],
    stream=False,
    messages=[
        {
            "role": "user",
            "content": (
                "Consider this data: Jason is 10 and John is 30. "
                "Correctly segment it into entities. "
                "Make sure the JSON is correct."
            ),
        },
    ],
)
for user in users:
    print(user)
    #> name='Jason' age=10
    #> name='John' age=30
```

## Streaming Tasks

We can also generate tasks as the tokens are streamed in by defining an `Iterable[T]` type.

Lets look at an example in action with the same class

```python hl_lines="6 26"
import instructor
from typing import Iterable
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "openai/gpt-4.1-mini",
    mode=instructor.Mode.TOOLS,
)

users = client.create(
    temperature=0.1,
    stream=True,
    response_model=Iterable[User],
    messages=[
        {"role": "system", "content": "You are a perfect entity extraction system"},
        {"role": "user", "content": "Extract `Jason is 10 and John is 10`"},
    ],
    max_tokens=1000,
)

for user in users:
    print(user)
    #> name='Jason' age=10
    #> name='John' age=10
```

## Asynchronous Streaming

I also just want to call out in this example that `instructor` also supports asynchronous streaming. This is useful when you want to stream a response model and process the results as they come in, but you'll need to use the `async for` syntax to iterate over the results.

```python
import instructor
from typing import Iterable
from pydantic import BaseModel


class UserExtract(BaseModel):
    name: str
    age: int


async def print_iterable_results():
    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        async_client=True,
        mode=instructor.Mode.TOOLS,
    )

    model = await client.create(
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    async for m in model:
        print(m)
        #> name='Alice' age=30
        #> name='Bob' age=25


import asyncio

asyncio.run(print_iterable_results())
```

## See Also

- [Streaming Partial](./partial.md) - Stream partially completed objects
- [Streaming Lists Tutorial](../learning/streaming/lists.md) - Step-by-step list streaming guide
- [Iterable Patterns](../learning/patterns/list_extraction.md) - List extraction patterns
- [Raw Response](./raw_response.md) - Access original LLM responses
