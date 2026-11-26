---
title: Stream validated records with create_iterable in Instructor
description: Use create_iterable to stream Pydantic models or unions from one LLM response, handle truncation, and close sync or async streams.
---

# Stream records with create_iterable {#multi-task-and-streaming}

Pass the type of **one item** to `client.create_iterable`. Instructor yields each validated item from a single response. Use a union when the stream can contain more than one type of record.

To get the whole collection at once, see [list extraction with `Iterable[T]`](lists.md). To watch fields fill in on one object, use [`create_partial`](partial.md).

Both examples below stream one `User` per person mentioned in the sentence.

=== "Using `create_iterable` (recommended)"
    ```python
    import instructor
    from pydantic import BaseModel

    client = instructor.from_provider("openai/gpt-6-luna")


    class User(BaseModel):
        name: str
        age: int


    resp = client.create_iterable(
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28, lives in Moscow and his friends are Alex, John and Mary who are 25, 30 and 27 respectively",
            }
        ],
        response_model=User,
    )

    for user in resp:
        print(user)
        #> name='Ivan' age=28
        #> name='Alex' age=25
        #> name='John' age=30
        #> name='Mary' age=27
    ```
    _Recommended for most use cases. Handles streaming and iteration for you._

=== "Using `create` with `Iterable[User]`"
    ```python
    import instructor
    from pydantic import BaseModel
    from typing import Iterable

    client = instructor.from_provider("openai/gpt-6-luna")


    class User(BaseModel):
        name: str
        age: int


    resp = client.create(
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28, lives in Moscow and his friends are Alex, John and Mary who are 25, 30 and 27 respectively",
            }
        ],
        response_model=Iterable[User],
        stream=True,
    )

    for user in resp:
        print(user)
        #> name='Ivan' age=28
        #> name='Alex' age=25
        #> name='John' age=30
        #> name='Mary' age=27
    ```
    _Use this if you need more manual control or compatibility with legacy code._

---


The next example accepts weather requests and search queries in the same response.

???+ warning

    Check that your provider and mode support the `anyOf` schema generated for unions.

## Synchronous Usage

=== "Using `create`"

    ```python
    import instructor
    from typing import Iterable, Union, Literal
    from pydantic import BaseModel


    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    client = instructor.from_provider("openai/gpt-6-luna", mode=instructor.Mode.TOOLS)

    results = client.create(
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
        stream=True,
    )

    for item in results:
        print(item)
        #> location='Toronto' units='metric'
        #> location='Dallas' units='imperial'
        #> query='Super Bowl winner'
    ```

=== "Using `create_iterable` (recommended)"

    ```python
    import instructor
    from typing import Union, Literal
    from pydantic import BaseModel


    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    client = instructor.from_provider("openai/gpt-6-luna", mode=instructor.Mode.TOOLS)

    results = client.create_iterable(
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Union[Weather, GoogleSearch],
    )

    for item in results:
        print(item)
        #> location='Toronto' units='metric'
        #> location='Dallas' units='imperial'
        #> query='Super Bowl winner'
    ```

---

## Stopping Iteration Early

On supported providers, including OpenAI, a token limit can raise `IncompleteOutputException` even after several valid items have arrived. The shared scanner also raises if the stream ends inside an item. Catch errors around the loop and decide whether to keep records you've already processed before retrying.

Once iteration has started, close the iterable if you stop before consuming all
results. Put synchronous iteration in a `try`/`finally` block and call
`resp.close()` in `finally`. For asynchronous iteration, use
`await resp.aclose()` instead. A `break` statement alone does not close a Python
generator.

The shared iterable parser releases a closable upstream source when parsing ends,
including validation failures. OpenAI chat-completion and Responses modes use this
path. This cleanup does not cover partial-model streaming or providers with their
own iterable parsers.

## See Also

- [Streaming Lists](./lists.md) - Collect records with `client.create`
- [Streaming Partial](./partial.md) - Stream partially completed objects
- [List Extraction Tutorial](../learning/patterns/list_extraction.md) - Step-by-step guide
- [Streaming Basics](../learning/streaming/basics.md) - Introduction to streaming

## Asynchronous Usage

=== "Using `create`"

    ```python
    import instructor
    from typing import Iterable, Union, Literal
    from pydantic import BaseModel
    import asyncio


    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    aclient = instructor.from_provider(
        "openai/gpt-6-luna", async_client=True, mode=instructor.Mode.TOOLS
    )


    async def main():
        results = await aclient.create(
            messages=[
                {"role": "system", "content": "You must always use tools"},
                {
                    "role": "user",
                    "content": "What is the weather in toronto and dallas and who won the super bowl?",
                },
            ],
            response_model=Iterable[Union[Weather, GoogleSearch]],
            stream=True,
        )
        async for item in results:
            print(item)
            #> location='Toronto' units='metric'
            #> location='Dallas' units='imperial'
            #> query='Super Bowl winner'


    asyncio.run(main())
    ```

=== "Using `create_iterable` (recommended)"

    ```python
    import asyncio
    from typing import Literal, Union

    import instructor
    from pydantic import BaseModel


    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    aclient = instructor.from_provider(
        "openai/gpt-6-luna", async_client=True, mode=instructor.Mode.TOOLS
    )


    async def iter_results():
        async for item in aclient.create_iterable(
            messages=[
                {"role": "system", "content": "You must always use tools"},
                {
                    "role": "user",
                    "content": "What is the weather in toronto and dallas and who won the super bowl?",
                },
            ],
            response_model=Union[Weather, GoogleSearch],
        ):
            yield item


    async def main():
        async for item in iter_results():
            print(item)
            #> location='Toronto' units='metric'
            #> location='Dallas' units='imperial'
            #> query='Super Bowl winner'


    asyncio.run(main())
    ```

### Markdown output and multiple tool calls

Native tool streams yield complete items as they arrive. Interleaved OpenAI tool calls are assembled by call index. In Markdown mode, a fenced JSON block takes precedence over JSON quoted in the surrounding prose. A fenced JSON value is available once its outer object or array closes. When the response has no fence, Instructor emits all bare JSON candidates in order at the end of the stream.

Code that calls `extract_json_from_stream` or its async counterpart directly with a known bare JSON stream can pass `prefer_fenced=False` to consume each completed value immediately.
