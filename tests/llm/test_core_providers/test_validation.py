"""
Validation and retry tests that run across all core providers.

Tests validation logic, custom validators, and retry mechanisms.
"""

from pydantic import BaseModel, Field, field_validator
import pytest
import instructor


class UserWithValidation(BaseModel):
    name: str = Field(description="User's full name")
    age: int = Field(description="User's age in years", ge=0, le=150)

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v


class Email(BaseModel):
    email: str = Field(description="Valid email address")

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v: str) -> str:
        if "@" not in v or "." not in v:
            raise ValueError("Must be a valid email address")
        return v


class TemperatureReading(BaseModel):
    celsius: float = Field(description="Temperature in Celsius", ge=-273.15)
    location: str


@pytest.mark.asyncio
async def test_basic_validation(provider_config):
    """Test that basic field validation works."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    user = await client.create(
        response_model=UserWithValidation,
        messages=[{"role": "user", "content": "John Doe is 30 years old"}],
    )

    assert isinstance(user, UserWithValidation)
    assert user.name == "John Doe"
    assert user.age == 30
    assert 0 <= user.age <= 150


@pytest.mark.asyncio
async def test_list_with_validation(provider_config):
    """Test validation with lists of validated models."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    users = await client.create(
        response_model=list[UserWithValidation],
        messages=[
            {
                "role": "user",
                "content": "Extract: Alice is 25, Bob is 30, Carol is 35",
            }
        ],
    )

    assert isinstance(users, list)
    assert len(users) == 3
    for user in users:
        assert isinstance(user, UserWithValidation)
        assert 0 <= user.age <= 150


@pytest.mark.asyncio
async def test_custom_validator(provider_config):
    """Test custom field validators work correctly."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    email = await client.create(
        response_model=Email,
        messages=[{"role": "user", "content": "My email is john@example.com"}],
    )

    assert isinstance(email, Email)
    assert "@" in email.email
    assert "." in email.email


@pytest.mark.asyncio
async def test_field_constraints(provider_config):
    """Test Pydantic field constraints (ge, le, etc)."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    reading = await client.create(
        response_model=TemperatureReading,
        messages=[
            {
                "role": "user",
                "content": "The temperature in Paris is 20 degrees Celsius",
            }
        ],
    )

    assert isinstance(reading, TemperatureReading)
    assert reading.celsius >= -273.15  # Absolute zero constraint
    assert reading.location == "Paris"


@pytest.mark.asyncio
async def test_max_retries(provider_config):
    """Test that max_retries parameter is accepted."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    user = await client.create(
        response_model=UserWithValidation,
        messages=[{"role": "user", "content": "Jane Smith is 28 years old"}],
        max_retries=2,
    )

    assert isinstance(user, UserWithValidation)
    assert user.name == "Jane Smith"
    assert user.age == 28
