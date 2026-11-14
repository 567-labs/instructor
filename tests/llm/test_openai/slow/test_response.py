import instructor
from tests.llm.test_openai.util import models
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import pytest
from collections.abc import AsyncIterable, Iterable
from itertools import product


class UserProfile(BaseModel):
    name: str
    age: int
    bio: str


response_modes = [
    instructor.Mode.RESPONSES_TOOLS,
    instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
]


@pytest.mark.parametrize("model, mode", product(models, response_modes))
def test_basic_response_methods(client: OpenAI, mode, model):
    instructor_client = instructor.from_openai(client, mode=mode)

    # Test create
    profile = instructor_client.responses.create(
        model=model,
        input="Generate a profile for a user named John who is 30 years old",
        response_model=UserProfile,
    )
    assert isinstance(profile, UserProfile)
    assert profile.name == "John"
    assert profile.age == 30


@pytest.mark.parametrize("model, mode", product(models, response_modes))
def test_create_iterable_from_create(client: OpenAI, mode, model):
    instructor_client = instructor.from_openai(client, mode=mode)

    # Test create
    profiles = instructor_client.responses.create(
        model=model,
        input="Generate three fake profiles",
        response_model=Iterable[UserProfile],
    )

    count = 0
    for profile in profiles:
        assert isinstance(profile, UserProfile)
        count += 1

    assert count >= 3


@pytest.mark.parametrize("model, mode", product(models, response_modes))
def test_create_with_completion(client: OpenAI, mode, model):
    from openai.types.responses import Response

    instructor_client = instructor.from_openai(client, mode=mode)

    # Test create
    response, completion = instructor_client.responses.create_with_completion(
        model=model,
        input="Generate a profile for a user named John who is 30 years old",
        response_model=UserProfile,
    )
    assert isinstance(response, UserProfile)
    assert response.name == "John"
    assert response.age == 30
    assert isinstance(completion, Response)


@pytest.mark.parametrize("model, mode", product(models, response_modes))
def test_create_iterable(client: OpenAI, mode, model):
    instructor_client = instructor.from_openai(client, mode=mode)

    # Test create
    users = instructor_client.responses.create_iterable(
        model=model,
        input="generate three fake profiles",
        response_model=UserProfile,
    )

    count = 0
    for user in users:
        assert isinstance(user, UserProfile)
        count += 1

    assert count == 3


@pytest.mark.parametrize("model, mode", product(models, response_modes))
def test_create_partial(client: OpenAI, mode, model):
    instructor_client = instructor.from_openai(client, mode=mode)

    # Test create
    resp = instructor_client.responses.create_partial(
        model=model,
        input="Generate a fake profile",
        response_model=UserProfile,
    )

    prev = None
    update_count = 0
    for user in resp:
        assert isinstance(user, UserProfile)
        if user != prev:
            update_count += 1
            prev = user

    assert update_count >= 1


# ASYNC TESTS


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, response_modes))
async def test_basic_response_methods_async(aclient: AsyncOpenAI, mode, model):
    instructor_client = instructor.from_openai(aclient, mode=mode)

    # Test create
    profile = await instructor_client.responses.create(
        model=model,
        input="Generate a profile for a user named John who is 30 years old",
        response_model=UserProfile,
    )
    assert isinstance(profile, UserProfile)
    assert profile.name == "John"
    assert profile.age == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, response_modes))
async def test_create_iterable_from_create_async(aclient: AsyncOpenAI, mode, model):
    instructor_client: instructor.AsyncInstructor = instructor.from_openai(
        aclient, mode=mode
    )

    # Test create
    profiles = await instructor_client.responses.create(
        model=model,
        input="Generate three fake profiles",
        response_model=Iterable[UserProfile],
    )

    count = 0
    assert isinstance(profiles, AsyncIterable)
    async for profile in profiles:
        assert isinstance(profile, UserProfile)
        count += 1

    assert count >= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, response_modes))
async def test_create_with_completion_async(aclient: AsyncOpenAI, mode, model):
    from openai.types.responses import Response

    instructor_client = instructor.from_openai(aclient, mode=mode)

    # Test create
    response, completion = await instructor_client.responses.create_with_completion(
        model=model,
        input="Generate a profile for a user named John who is 30 years old",
        response_model=UserProfile,
    )
    assert isinstance(response, UserProfile)
    assert response.name == "John"
    assert response.age == 30
    assert isinstance(completion, Response)


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, response_modes))
async def test_create_iterable_async(aclient: AsyncOpenAI, mode, model):
    instructor_client = instructor.from_openai(aclient, mode=mode)

    # Test create
    users = await instructor_client.responses.create_iterable(
        model=model,
        input="generate three fake profiles",
        response_model=UserProfile,
    )

    count = 0
    async for user in users:
        assert isinstance(user, UserProfile)
        count += 1

    assert count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("model, mode", product(models, response_modes))
async def test_create_partial_async(aclient: AsyncOpenAI, mode, model):
    instructor_client = instructor.from_openai(aclient, mode=mode)

    # Test create
    resp = instructor_client.responses.create_partial(
        model=model,
        input="Generate a fake profile",
        response_model=UserProfile,
    )

    prev = None
    update_count = 0
    async for user in resp:
        assert isinstance(user, UserProfile)
        if user != prev:
            update_count += 1
            prev = user

    assert update_count >= 1
