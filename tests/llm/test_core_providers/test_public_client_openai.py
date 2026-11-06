import os
from operator import attrgetter

import pytest
import instructor
from .public_client_models import User

openai = pytest.importorskip("openai")

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is unavailable"
)


@pytest.mark.parametrize(
    "config", ["create", "messages.create", "chat.completions.create"]
)
def test_public_create_alias(config: str):
    client = instructor.from_openai(openai.OpenAI(), model="gpt-4.1-mini")

    user = attrgetter(config)(client)(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_chat_completions_create_with_response():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-4.1-mini")

    user, completion = client.chat.completions.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10

    from openai.types.chat import ChatCompletion

    assert isinstance(completion, ChatCompletion)


def test_client_chat_completions_create_partial():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-4.1-mini")

    last = None
    for user in client.chat.completions.create_partial(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    ):
        assert isinstance(user, User)
        last = user

    assert last is not None
    assert (last.name, last.age) == ("Jason", 10)


def test_client_chat_completions_create_iterable():
    client = instructor.from_openai(openai.OpenAI(), model="gpt-4.1-mini")

    users = [
        user
        for user in client.chat.completions.create_iterable(
            response_model=User,
            messages=[{"role": "user", "content": "Alice is 25, Bob is 30"}],
            temperature=0,
        )
    ]
    assert len(users) == 2


@pytest.mark.asyncio
async def test_async_client_chat_completions_create():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-4.1-mini")

    user = await instructor_client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_partial():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-4.1-mini")

    last = None
    async for user in instructor_client.chat.completions.create_partial(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    ):
        assert isinstance(user, User)
        last = user

    assert last is not None
    assert (last.name, last.age) == ("Jason", 10)


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_iterable():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-4.1-mini")

    async for user in instructor_client.chat.completions.create_iterable(
        response_model=User,
        messages=[{"role": "user", "content": "Alice is 25, Bob is 30"}],
        temperature=0,
    ):
        assert isinstance(user, User)


@pytest.mark.asyncio
async def test_async_client_chat_completions_create_with_response():
    client = openai.AsyncOpenAI()
    instructor_client = instructor.from_openai(client, model="gpt-4.1-mini")

    user, response = await instructor_client.chat.completions.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    from openai.types.chat import ChatCompletion

    assert user.name == "Jason"
    assert user.age == 10
    assert isinstance(response, ChatCompletion)
