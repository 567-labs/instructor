from itertools import product
from collections.abc import Iterable
from pydantic import BaseModel
import pytest

import instructor
from .util import models, modes


class User(BaseModel):
    name: str
    age: int


Users = Iterable[User]


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multi_user(model, mode, client):
    client = instructor.from_openai(client, mode=mode)

    def stream_extract(input: str) -> Iterable[User]:
        return client.chat.completions.create(
            model=model,
            response_model=Users,
            messages=[
                {
                    "role": "system",
                    "content": "You are a perfect entity extraction system",
                },
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = [user for user in stream_extract(input="Jason is 20, Sarah is 30")]
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30


from typing import Any
from functools import partial


async def async_map_chat_completion_to_response(
    messages, client, *args, **kwargs
) -> Any:
    return await client.responses.create(
        *args,
        input=messages,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, modes))
async def test_multi_user_tools_mode_async(model, mode, aclient):
    from instructor.mode import Mode

    client = instructor.patch(
        aclient,
        create=(
            partial(async_map_chat_completion_to_response, client=aclient)
            if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}
            else aclient.chat.completions.create
        ),
        mode=mode,
    )

    async def stream_extract(input: str) -> Iterable[User]:
        return await client.chat.completions.create(
            model=model,
            response_model=Users,
            messages=[
                {
                    "role": "system",
                    "content": "You are a perfect entity extraction system",
                },
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}\n"
                        "Correctly segment it into entities. "
                        "Make sure the JSON is correct."
                    ),
                },
            ],
            max_tokens=1000,
            temperature=0,
        )

    resp = []
    for user in await stream_extract(input="Jason is 20, Sarah is 30"):
        resp.append(user)
    print(resp)
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_multi_user_stream(model, mode, client):
    client = instructor.from_openai(client, mode=mode)

    def stream_extract(input: str) -> Iterable[User]:
        return client.chat.completions.create(
            model=model,
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "system",
                    "content": "You are a perfect entity extraction system",
                },
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = [user for user in stream_extract(input="Jason is 20, Sarah is 30")]
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, modes))
async def test_multi_user_tools_mode_async_stream(model, mode, aclient):
    client = instructor.from_openai(aclient, mode=mode)

    async def stream_extract(input: str) -> Iterable[User]:
        return await client.chat.completions.create(
            model=model,
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "system",
                    "content": "You are a perfect entity extraction system",
                },
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}\n"
                        "Correctly segment it into entities. "
                        "Make sure the JSON is correct."
                    ),
                },
            ],
            max_tokens=1000,
            temperature=0,
        )

    resp = []
    async for user in await stream_extract(input="Jason is 20, Sarah is 30"):
        resp.append(user)
    print(resp)
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30
