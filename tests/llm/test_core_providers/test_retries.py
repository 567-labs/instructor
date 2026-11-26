"""
Retry and error handling tests that run across all core providers.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
import pytest
import instructor
from tests.llm.shared_config import PAIRED_PROVIDERS


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


@pytest.mark.paired
@pytest.mark.asyncio
async def test_validation_with_retries(provider_config):
    """Exercise the retry budget, forcing a reask on the paired routes."""
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    attempts = 0

    class RetriedUser(ValidatedUser):
        @model_validator(mode="after")
        def retry_once(self) -> "RetriedUser":
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError(
                    "Check both fields against the original text and submit them again."
                )
            return self

    paired = model.partition("/")[0] in PAIRED_PROVIDERS
    user = await client.create(
        response_model=RetriedUser if paired else ValidatedUser,
        messages=[
            {
                "role": "user",
                "content": "Extract: Sarah Johnson is 25 years old",
            }
        ],
        max_retries=2,
    )

    assert isinstance(user, ValidatedUser)
    assert 0 <= user.age <= 120
    if paired:
        assert attempts == 2
        assert (user.name, user.age) == ("Sarah Johnson", 25)
