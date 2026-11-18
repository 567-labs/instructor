import os

import pytest
import instructor
from .public_client_models import User

anthropic = pytest.importorskip("anthropic")

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is unavailable"
)


def test_client_from_anthropic_with_response():
    client = instructor.from_anthropic(
        anthropic.Anthropic(),
        max_tokens=1000,
        model="claude-haiku-5-5",
    )

    user, response = client.messages.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
    )
    assert user.name == "Jason"
    assert user.age == 10
    assert isinstance(response, anthropic.types.Message)


def test_client_anthropic_response():
    client = anthropic.Anthropic()
    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="claude-haiku-5-5",
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.asyncio
async def test_async_client_anthropic_response():
    client = anthropic.AsyncAnthropic()
    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="claude-haiku-5-5",
    )

    user = await instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
    )
    assert user.name == "Jason"
    assert user.age == 10
