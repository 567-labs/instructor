---
title: Extract a list of Pydantic models with Instructor
description: Collect multiple records using Iterable[T] in client.create, or set stream=True to process each record as it arrives.
---

# Extract a list with Iterable[T] {#multi-task-and-streaming}

Pass `response_model=Iterable[User]` to `client.create` to extract several users in one call. With `stream=False` you get the list after the response finishes. Set `stream=True` to iterate over completed users, or use the [`create_iterable` shortcut](iterable.md).

If the collection needs a named field or other metadata, you can define the outer model yourself.

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

## Extracting Tasks using Iterable

For a collection of users alone, Instructor builds the wrapper schema for you.

```python
import instructor
from typing import Iterable
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "openai/gpt-6-luna",
    mode=instructor.Mode.JSON,
)

users = client.create(
    reasoning_effort="none",
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

Use `stream=True` to process items before the response finishes. An error can still occur after a valid item has been yielded. See [handling incomplete streams](iterable.md#stopping-iteration-early).

```python
import instructor
from typing import Iterable
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "openai/gpt-6-luna",
    mode=instructor.Mode.TOOLS,
)

users = client.create(
    stream=True,
    response_model=Iterable[User],
    messages=[
        {"role": "system", "content": "You are a perfect entity extraction system"},
        {"role": "user", "content": "Extract `Jason is 10 and John is 10`"},
    ],
    max_completion_tokens=1000,
)

for user in users:
    print(user)
    #> name='Jason' age=10
    #> name='John' age=10
```

## Asynchronous Streaming

With the async client, await `create` and consume the resulting stream with `async for`.

```python
import instructor
from typing import Iterable
from pydantic import BaseModel


class UserExtract(BaseModel):
    name: str
    age: int


async def print_iterable_results():
    client = instructor.from_provider(
        "openai/gpt-6-luna",
        async_client=True,
        mode=instructor.Mode.TOOLS,
    )

    model = await client.create(
        response_model=Iterable[UserExtract],
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Make up two people"},
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
