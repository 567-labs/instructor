"""
Retry and error handling tests that run across all core providers.
"""

from pydantic import BaseModel, Field, field_validator
import pytest
import instructor


class ValidatedUser(BaseModel):
    name: str
    age: int = Field(ge=0, le=120)

    @field_validator("name")
    @classmethod
    def name_must_have_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Name must not be empty")
        return v.strip()


@pytest.mark.asyncio
async def test_max_retries_parameter(provider_config):
    """Test that max_retries parameter is accepted and works."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    user = await client.create(
        response_model=ValidatedUser,
        messages=[{"role": "user", "content": "Create a user: John Smith, age 30"}],
        max_retries=3,
    )

    assert isinstance(user, ValidatedUser)
    assert user.name.strip() == "John Smith"
    assert 0 <= user.age <= 120


@pytest.mark.asyncio
async def test_validation_with_retries(provider_config):
    """Test that validation errors trigger retries (if supported)."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    # This should work after potential retries
    user = await client.create(
        response_model=ValidatedUser,
        messages=[
            {
                "role": "user",
                "content": "Extract: Sarah Johnson is 25 years old",
            }
        ],
        max_retries=2,
    )

    assert isinstance(user, ValidatedUser)
    assert user.age >= 0 and user.age <= 120
