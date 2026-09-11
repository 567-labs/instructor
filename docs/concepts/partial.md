---
title: Streaming Partial Responses with Instructor and OpenAI
description: Learn to utilize field-level streaming with Instructor and OpenAI for incremental responses in Python.
---

# Streaming Partial Responses

!!! info "Literal"

    If the data structure you're using has literal values, you need to make sure to import the `PartialLiteralMixin` mixin.

    ```python
    from typing import Literal
    from pydantic import BaseModel
    from instructor.dsl.partial import PartialLiteralMixin


    class User(BaseModel, PartialLiteralMixin):
        name: str
        age: int
        category: Literal["admin", "user", "guest"]


    # The rest of your code below
    ```

    This is because `jiter` throws an error otherwise if it encounters a incomplete Literal value while it's being streamed in

Field level streaming provides incremental snapshots of the current state of the response model that are immediately usable. This approach is particularly relevant in contexts like rendering UI components.

Instructor supports this pattern by making use of `create_partial`. This lets us dynamically create a new class that treats all of the original model's fields as `Optional`.

## Understanding Partial Responses

Consider what happens whene we define a response model:

```python
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int
```

If we streamed json out from OpenAI, we would only be able to parse when the object is completed returned!

```
{"name": "Jo
{"name": "John", "ag
{"name": "John", "age:
{"name": "John", "age": 25} # Completed
```

When specifying a `create_partial` and setting `stream=True`, the response from `instructor` becomes a `Generator[T]`. As the generator yields results, you can iterate over these incremental updates. The last value yielded by the generator represents the completed extraction!

```
{"name": "Jo                 => User(name="Jo", age=None)
{"name": "John", "ag         => User(name="John", age=None)
{"name": "John", "age:       => User(name="John", age=None)
{"name": "John", "age": 25}  => User(name="John", age=25)
```

!!! warning "Limited Validator Support"

    Due to the streaming nature of the response model, we do not support validators since they would not be able to be applied to the streaming response.

Let's look at an example of streaming an extraction of conference information, that would be used to stream in an react component.

```python
import instructor
from pydantic import BaseModel
from typing import List
from rich.console import Console

client = instructor.from_provider("openai/gpt-4.1-mini")

text_block = """
In our recent online meeting, participants from various backgrounds joined to discuss the upcoming tech conference. The names and contact details of the participants were as follows:

- Name: John Doe, Email: johndoe@email.com, Twitter: @TechGuru44
- Name: Jane Smith, Email: janesmith@email.com, Twitter: @DigitalDiva88
- Name: Alex Johnson, Email: alexj@email.com, Twitter: @CodeMaster2023

During the meeting, we agreed on several key points. The conference will be held on March 15th, 2024, at the Grand Tech Arena located at 4521 Innovation Drive. Dr. Emily Johnson, a renowned AI researcher, will be our keynote speaker.

The budget for the event is set at $50,000, covering venue costs, speaker fees, and promotional activities. Each participant is expected to contribute an article to the conference blog by February 20th.

A follow-up meetingis scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
"""


class User(BaseModel):
    name: str
    email: str
    twitter: str


class MeetingInfo(BaseModel):
    users: List[User]
    date: str
    location: str
    budget: int
    deadline: str


extraction_stream = client.create_partial(
    response_model=MeetingInfo,
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    stream=True,
)


console = Console()

for extraction in extraction_stream:
    obj = extraction.model_dump()
    console.clear()
    console.print(obj)

print(extraction.model_dump_json(indent=2))
"""
{
  "users": [
    {
      "name": "John Doe",
      "email": "johndoe@email.com",
      "twitter": "@TechGuru44"
    },
    {
      "name": "Jane Smith",
      "email": "janesmith@email.com",
      "twitter": "@DigitalDiva88"
    },
    {
      "name": "Alex Johnson",
      "email": "alexj@email.com",
      "twitter": "@CodeMaster2023"
    }
  ],
  "date": "March 15th, 2024",
  "location": "Grand Tech Arena, 4521 Innovation Drive",
  "budget": 50000,
  "deadline": "February 20th"
}
"""
```

This will output the following:

![Partial Streaming Gif](../img/partial.gif)

## Asynchronous Streaming

I also just want to call out in this example that `instructor` also supports asynchronous streaming. This is useful when you want to stream a response model and process the results as they come in, but you'll need to use the `async for` syntax to iterate over the results.

```python
import instructor
from pydantic import BaseModel

client = instructor.from_provider(
    "openai/gpt-5-nano",
    async_client=True,
)


class User(BaseModel):
    name: str
    age: int


async def print_partial_results():
    user = client.create_partial(
        response_model=User,
        max_retries=2,
        stream=True,
        messages=[
            {"role": "user", "content": "Jason is 12 years old"},
        ],
    )
    async for m in user:
        print(m)
        #> name=None age=None
        #> name=None age=None
        #> name=None age=None
        #> name='' age=None
        #> name='Jason' age=None
        #> name='Jason' age=None
        #> name='Jason' age=None
        #> name='Jason' age=None
        #> name='Jason' age=12
        #> name='Jason' age=12


import asyncio

asyncio.run(print_partial_results())
```

## Concurrent requests and direct handlers

The core runtime passes each request's `stream` boolean explicitly. OpenAI-compatible,
Anthropic, Mistral, and xAI mode handlers honor that value even when another request
using the same model prepared a stream, failed, or was cancelled. This routing rule
does not change each provider's event format or output type.

When calling those handlers directly, pass `stream=True` or `stream=False` to
`parse_response` for overlapping requests. Omitting `stream` preserves the legacy
one-shot inference from `prepare_request`. That fallback is keyed by model class
and cannot distinguish overlapping calls using the same class. Explicit parsing
also retires a pending legacy marker; do not mix explicit and implicit parsing
for overlapping calls with the same model.

Sync partial parsing still materializes its results and may raise validation errors
inside `parse_response`; async partial validation occurs during iteration. Closing
a parsed generator early is not a promise that it closes the underlying SDK stream.
Retain and close the source stream according to its SDK's ownership contract.
Cancellation while awaiting a source propagates to that source.

The native xAI client has a separate SDK streaming path; the shared routing rule
here concerns its registry mode handlers. Providers without class-keyed stream
markers retain their existing routing. Async request preparation still performs
synchronous cache and media operations; this change does not offload them or alter
remote-media connection checks.

## See Also

- [Streaming Lists](./lists.md) - Stream collections of completed objects
- [Streaming Basics](../learning/streaming/basics.md) - Introduction to streaming concepts
- [Iterable Streaming](./iterable.md) - Stream multiple objects
- [Raw Response](./raw_response.md) - Access original LLM responses
