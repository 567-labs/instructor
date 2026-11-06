"""
Streaming tests that run across all core providers.

Tests streaming functionality including Partial and Iterable.
"""

from collections.abc import Iterable
from pydantic import BaseModel
from typing import Union, Literal
import pytest
import instructor
from instructor.dsl.partial import Partial
from .capabilities import skip_if_unsupported


class User(BaseModel):
    name: str
    age: int


class Weather(BaseModel):
    location: str
    temperature: int
    units: Literal["celsius", "fahrenheit"]


class SearchQuery(BaseModel):
    query: str
    category: str


@pytest.mark.asyncio
async def test_partial_streaming(provider_config):
    """Test partial streaming with incremental updates."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    updates = []
    async for partial_user in await client.create(
        response_model=Partial[User],
        messages=[{"role": "user", "content": "Jason Liu is 30 years old"}],
        stream=True,
    ):
        assert isinstance(partial_user, User)
        updates.append(partial_user)

    # Should receive at least one update
    assert len(updates) >= 1

    # Final update should have complete data
    final = updates[-1]
    assert final.name == "Jason Liu"
    assert final.age == 30


@pytest.mark.asyncio
async def test_iterable_streaming(provider_config):
    """Test streaming multiple objects with Iterable."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    users = []
    async for user in await client.create(
        response_model=Iterable[User],
        messages=[
            {
                "role": "user",
                "content": "Create 3 users: Alice (25), Bob (30), Carol (35)",
            }
        ],
    ):
        users.append(user)

    assert len(users) == 3
    assert all(isinstance(user, User) for user in users)
    assert {user.name for user in users} == {"Alice", "Bob", "Carol"}


@pytest.mark.asyncio
async def test_iterable_streaming_with_stream_flag(provider_config):
    """Test Iterable with explicit stream=True flag."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    users = []
    async for user in await client.create(
        response_model=Iterable[User],
        messages=[{"role": "user", "content": "Make 2 users: John (20), Jane (22)"}],
        stream=True,
    ):
        assert isinstance(user, User)
        users.append(user)

    assert len(users) == 2
    assert {user.name for user in users} == {"John", "Jane"}


@pytest.mark.asyncio
async def test_iterable_union_streaming(provider_config):
    """Test streaming union types with Iterable."""
    skip_if_unsupported(provider_config, "union_streaming")
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    results = []
    async for result in await client.create(
        response_model=Iterable[Union[Weather, SearchQuery]],
        messages=[
            {
                "role": "user",
                "content": "What's the weather in NYC and search for 'python tutorials'?",
            }
        ],
    ):
        results.append(result)

    assert len(results) >= 2
    assert any(isinstance(r, Weather) for r in results)
    assert any(isinstance(r, SearchQuery) for r in results)


@pytest.mark.asyncio
async def test_create_iterable_method(provider_config):
    """Test create_iterable convenience method."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    users = []
    async for user in client.chat.completions.create_iterable(
        response_model=User,
        messages=[
            {
                "role": "user",
                "content": "Generate 2 users: Tom (45), Jerry (40)",
            }
        ],
    ):
        users.append(user)

    assert len(users) == 2
    assert all(isinstance(user, User) for user in users)
