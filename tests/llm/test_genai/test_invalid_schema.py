import pytest
from typing import Optional, Union

import instructor
from pydantic import BaseModel
from .util import models, modes
from itertools import product
from instructor.v2.providers.gemini.utils import map_to_gemini_function_schema

MODEL = f"google/{models[0]}"


@pytest.mark.parametrize("mode,model", product(modes, models))
def test_nested(mode, model):
    """Test that nested schemas are supported."""
    client = instructor.from_provider(f"google/{model}", mode=mode)

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Optional[Address] = None

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "John loves to go gardenning with his friends",
            }
        ],
        response_model=Person,
    )

    assert resp.name == "John"  # type: ignore
    assert resp.address is None  # type: ignore


@pytest.mark.parametrize("mode,model", product(modes, models))
def test_union(mode, model):
    """Test that union types are now supported with Gemini (issue #1964)."""
    client = instructor.from_provider(f"google/{model}", mode=mode)

    class UserData(BaseModel):
        name: str
        id_value: Union[str, int]

    # Union types are now supported by Google GenAI SDK
    # See: https://github.com/googleapis/python-genai/issues/447
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "User name is Alice with ID 12345"}],
        response_model=UserData,
    )

    assert response.name == "Alice"
    # The ID could be returned as either str or int
    assert response.id_value in ["12345", 12345]


def test_optional_types_allowed():
    """Test that Optional types are correctly mapped and don't throw errors."""

    class User(BaseModel):
        name: str
        age: Optional[int] = None
        email: Optional[str] = None

    schema = User.model_json_schema()
    # Should not raise an error
    result = map_to_gemini_function_schema(schema)

    assert result["properties"]["age"]["nullable"] is True
    assert result["properties"]["email"]["nullable"] is True
    assert result["required"] == ["name"]


def test_union_types_allowed_schema():
    """Test that Union types are now allowed in schema mapping (issue #1964)."""

    class UserWithUnion(BaseModel):
        name: str
        value: Union[int, str]

    schema = UserWithUnion.model_json_schema()

    # Union types are now supported - should not raise
    result = map_to_gemini_function_schema(schema)

    # The anyOf structure should be preserved
    assert "properties" in result
    assert "value" in result["properties"]
    assert "anyOf" in result["properties"]["value"]


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_STRUCTURED_OUTPUTS, instructor.Mode.GENAI_TOOLS]
)
def test_genai_api_call_with_different_types(mode):
    """Test actual API call with genai SDK using different types."""

    class UserProfile(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
        is_premium: bool
        score: float

    client = instructor.from_provider(MODEL, mode=mode)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a user profile for John Doe, 25 years old, premium user with score 85.5",
            }
        ],
        response_model=UserProfile,
    )

    assert isinstance(response, UserProfile)
    assert response.name == "John Doe"
    assert response.email is None


@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_STRUCTURED_OUTPUTS, instructor.Mode.GENAI_TOOLS]
)
def test_genai_api_call_with_nested_models(mode):
    """Test API call with nested models (multiple users)."""

    class User(BaseModel):
        name: str
        age: int
        department: Optional[str] = None

    class UserList(BaseModel):
        users: list[User]

    client = instructor.from_provider(MODEL, mode=mode)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a list of 3 employees: Alice (30, Engineering), Bob (25, Marketing), Charlie (35)",
            }
        ],
        response_model=UserList,
    )

    assert isinstance(response, UserList)
    assert len(response.users) == 3
    assert {user.name for user in response.users} == {"Alice", "Bob", "Charlie"}
    assert {user.age for user in response.users} == {25, 30, 35}
    assert {user.department for user in response.users} == {
        None,
        "Engineering",
        "Marketing",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_STRUCTURED_OUTPUTS, instructor.Mode.GENAI_TOOLS]
)
async def test_genai_api_call_with_different_types_async(mode):
    """Test actual async API call with genai SDK using different types."""

    class UserProfile(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
        is_premium: bool
        score: float

    client = instructor.from_provider(MODEL, mode=mode, async_client=True)

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a user profile for John Doe, 25 years old, premium user with score 85.5",
            }
        ],
        response_model=UserProfile,
    )

    assert isinstance(response, UserProfile)
    assert response.name == "John Doe"
    assert response.email is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", [instructor.Mode.GENAI_STRUCTURED_OUTPUTS, instructor.Mode.GENAI_TOOLS]
)
async def test_genai_api_call_with_nested_models_async(mode):
    """Test async API call with nested models (multiple users)."""

    class User(BaseModel):
        name: str
        age: int
        department: Optional[str] = None

    class UserList(BaseModel):
        users: list[User]

    client = instructor.from_provider(MODEL, mode=mode, async_client=True)

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Create a list of 3 employees: Alice (30, Engineering), Bob (25, Marketing), Charlie (35)",
            }
        ],
        response_model=UserList,
    )

    assert isinstance(response, UserList)
    assert len(response.users) == 3
    assert {user.name for user in response.users} == {"Alice", "Bob", "Charlie"}
    assert {user.age for user in response.users} == {25, 30, 35}
    assert {user.department for user in response.users} == {
        None,
        "Engineering",
        "Marketing",
    }
