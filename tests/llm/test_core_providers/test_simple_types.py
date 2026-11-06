"""Test simple type extraction across all providers.

Tests that basic Python types (int, bool, str, Literal, Union, Enum) work
consistently across all providers using from_provider().
"""

import enum
from typing import Annotated, Literal, Union

import pytest
from pydantic import Field

import instructor
from .capabilities import skip_if_unsupported


@pytest.mark.asyncio
async def test_int(provider_config):
    """Test extracting int response."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.create(
        response_model=int,
        messages=[
            {
                "role": "user",
                "content": "Return the number 42",
            },
        ],
    )
    assert isinstance(response, int)


@pytest.mark.asyncio
async def test_bool(provider_config):
    """Test extracting bool response."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.create(
        response_model=bool,
        messages=[
            {
                "role": "user",
                "content": "Is the sky blue? Answer true or false",
            },
        ],
    )
    assert isinstance(response, bool)


@pytest.mark.asyncio
async def test_str(provider_config):
    """Test extracting str response."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.create(
        response_model=str,
        messages=[
            {
                "role": "user",
                "content": "Say 'hello world'",
            },
        ],
    )
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_literal(provider_config):
    """Test extracting Literal type response."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.create(
        response_model=Literal["red", "green", "blue"],
        messages=[
            {
                "role": "user",
                "content": "Pick one of these colors: red, green, or blue",
            },
        ],
    )
    assert response in ["red", "green", "blue"]


@pytest.mark.asyncio
async def test_union(provider_config):
    """Test extracting Union type response."""
    skip_if_unsupported(provider_config, "union_types")
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.create(
        response_model=Union[int, str],
        messages=[
            {
                "role": "user",
                "content": "Return either a number or a string",
            },
        ],
    )
    assert isinstance(response, (int, str))


@pytest.mark.asyncio
async def test_enum(provider_config):
    """Test extracting Enum type response."""
    skip_if_unsupported(provider_config, "enum_types")

    class Color(enum.Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.create(
        response_model=Color,
        messages=[
            {
                "role": "user",
                "content": "Pick one color: red, green, or blue",
            },
        ],
    )
    assert response in [Color.RED, Color.GREEN, Color.BLUE]


@pytest.mark.asyncio
async def test_annotated_int(provider_config):
    """Test extracting Annotated[int] with Field description."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    response = await client.create(
        response_model=Annotated[int, Field(description="A random number")],
        messages=[
            {
                "role": "user",
                "content": "Give me a random number",
            },
        ],
    )
    assert isinstance(response, int)
