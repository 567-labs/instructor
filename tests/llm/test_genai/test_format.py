import pytest
from pydantic import BaseModel
import instructor
from .util import models, modes
from itertools import product
from google import genai
from google.genai import types


class User(BaseModel):
    name: str
    age: int


class Users(BaseModel):
    users: list[User]


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_simple_string_message(client, model, mode):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    response = client.chat.completions.create(
        model=model,
        messages=["Ivan is 28 years old"],  # type: ignore
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_system_prompt(client, model, mode):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ivan is 28 years old",
            },
            {
                "role": "user",
                "content": "Make sure that the response is a list of users",
            },
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_system_kwarg(client, model, mode):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    response = client.chat.completions.create(
        model=model,
        system="Ivan is 28 years old",
        messages=[
            {
                "role": "user",
                "content": "Make sure that the response is a list of users",
            },
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_system_kwarg_genai(client, model, mode):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    response = client.chat.completions.create(
        model=model,
        system="Ivan is 28 years old",
        messages=[
            genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part.from_text(
                        text="Make sure that the response is a list of users"
                    )
                ],
            ),
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_system_prompt_list(client, model, mode):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    "Ivan is",
                    " 28 years old",
                ],
            },  # type: ignore
            {
                "role": "user",
                "content": "Make sure that the response is a list of users",
            },
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_format_genai_typed(client, model, mode):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)
    response = client.chat.completions.create(
        model=model,
        response_model=User,
        messages=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="Extract {{name}} is {{age}} years old")
                ],
            ),  # type: ignore
        ],
        context={"name": "Jason", "age": 25},
    )
    assert isinstance(response, User)
    assert response.name == "Jason"
    assert response.age == 25


@pytest.mark.parametrize("model, mode, is_list", product(models, modes, [True, False]))
def test_format_string(client, model: str, mode: instructor.Mode, is_list: bool):
    client = instructor.from_provider(f"google/{model}", mode=mode, async_client=False)

    content = (
        ["Extract {{name}} is {{age}} years old."]
        if is_list
        else "Extract {{name}} is {{age}} years old."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        response_model=User,
        context={"name": "Jason", "age": 25},
    )

    assert isinstance(resp, User)
    assert resp.name == "Jason"
    assert resp.age == 25
