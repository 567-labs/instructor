from instructor import from_writer
from writerai import Writer, AsyncWriter
from pydantic import BaseModel
from .util import models, modes


class User(BaseModel):
    first_name: str
    age: int


class UserList(BaseModel):
    items: list[User]


import pytest
from itertools import product

import instructor
import enum

from typing import Literal


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_format_literal(model: str, mode: instructor.Mode):
    client = instructor.from_writer(
        client=Writer(),
        mode=mode,
    )

    response = client.chat.completions.create(
        model=model,
        response_model=Literal["1231", "212", "331"],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in ["1231", "212", "331"]


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_format_enum(model: str, mode: instructor.Mode):
    class Options(enum.Enum):
        A = "A"
        B = "B"
        C = "C"

    client = instructor.from_writer(
        client=Writer(),
        mode=mode,
    )

    response = client.chat.completions.create(
        model=model,
        response_model=Options,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in [Options.A, Options.B, Options.C]


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_format_bool(model: str, mode: instructor.Mode):
    client = instructor.from_writer(
        client=Writer(),
        mode=mode,
    )

    response = client.chat.completions.create(
        model=model,
        response_model=bool,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) == bool


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_format_sync(model: str, mode: instructor.Mode):
    client = from_writer(
        client=Writer(),
        mode=mode,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract {{name}} is {{age}} years old.",
            }
        ],
        response_model=User,
        context={"name": "Jason", "age": 25},
    )

    assert isinstance(response, User)
    assert response.first_name == "Jason"
    assert response.age == 25


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_writer_format_async(mode: instructor.Mode, model: str):
    client = instructor.from_writer(
        client=AsyncWriter(),
        mode=mode,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : {{name}} is {{age}} and lives in Berlin",
            },
        ],
        context={
            "name": "Yan",
            "age": 27,
        },
        response_model=User,
    )

    assert response.first_name == "Yan"
    assert response.age == 27


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_format_list_of_strings(mode: instructor.Mode, model: str):
    client = instructor.from_writer(
        client=Writer(),
        mode=mode,
    )

    users = [
        {
            "name": "Jason",
            "age": 25,
        },
        {
            "name": "Elizabeth",
            "age": 12,
        },
        {
            "name": "Chris",
            "age": 27,
        },
    ]

    prompt = """
    Extract a list of users from the following text:

    {% for user in users %}
    - Name: {{ user.name }}, Age: {{ user.age }}
    {% endfor %}
    """
    response = client.chat.completions.create(
        model=model,
        response_model=UserList,
        messages=[
            {"role": "user", "content": prompt},
        ],
        context={"users": users},
    )

    assert isinstance(response, UserList), "Result should be an instance of UserList"
    assert isinstance(response.items, list), "items should be a list"
    assert len(response.items) == 3, "List should contain 3 items"

    names = [item.first_name for item in response.items]
    assert "Jason" in names, "'Jason' should be in the list"
    assert "Elizabeth" in names, "'Elizabeth' should be in the list"
    assert "Chris" in names, "'Chris' should be in the list"
