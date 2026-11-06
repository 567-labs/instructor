from itertools import product
from pydantic import BaseModel, field_validator
from openai.types.chat import ChatCompletion
from typing_extensions import TypedDict
import pytest
import instructor


from .util import models, modes


class UserExtract(BaseModel):
    name: str
    age: int


class UserExtractTypedDict(TypedDict):
    name: str
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_typed_dict(model, mode, client):
    if mode in {
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    }:
        pytest.skip("Avoiding testing responses tools with openai")

    client = instructor.patch(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        response_model=UserExtractTypedDict,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, BaseModel), "Should be instance of a pydantic model"
    assert model.name.lower() == "jason"
    assert model.age == 25
    assert hasattr(model, "_raw_response"), (
        "The raw response should be available from OpenAI"
    )


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_runmodel(model, mode, client):
    if mode in {
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    }:
        pytest.skip("Avoiding testing responses tools with openai")

    client = instructor.patch(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert model.age == 25
    assert hasattr(model, "_raw_response"), (
        "The raw response should be available from OpenAI"
    )

    ChatCompletion(**model._raw_response.model_dump())


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_runmodel_async(model, mode, aclient):
    if mode in {
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    }:
        pytest.skip("Avoiding testing responses tools with openai")

    aclient = instructor.patch(aclient, mode=mode)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert model.age == 25
    assert hasattr(model, "_raw_response"), (
        "The raw response should be available from OpenAI"
    )

    ChatCompletion(**model._raw_response.model_dump())


class UserExtractValidated(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError(
                "Name should have all letters in uppercase. Make sure to use the `uppercase` form of the name"
            )
        return v


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_runmodel_validator(model, mode, client):
    if mode in {
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    }:
        pytest.skip("Avoiding testing responses tools with openai")
    client = instructor.patch(client, mode=mode)
    model = client.chat.completions.create(
        model=model,
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtractValidated), "Should be instance of UserExtract"
    assert model.name == "JASON"
    assert hasattr(model, "_raw_response"), (
        "The raw response should be available from OpenAI"
    )

    ChatCompletion(**model._raw_response.model_dump())


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_runmodel_async_validator(model, mode, aclient):
    if mode in {
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    }:
        pytest.skip("Avoiding testing responses tools with openai")
    aclient = instructor.patch(aclient, mode=mode)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtractValidated), "Should be instance of UserExtract"
    assert model.name == "JASON"
    assert hasattr(model, "_raw_response"), (
        "The raw response should be available from OpenAI"
    )

    ChatCompletion(**model._raw_response.model_dump())
