import os

import pytest
import instructor
from .public_client_models import User

pytestmark = pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY is unavailable"
)


def test_client_from_mistral_with_response():
    from instructor.v2.providers.mistral.client import Mistral

    assert Mistral is not None

    client = instructor.from_mistral(
        Mistral(api_key=os.environ.get("MISTRAL_API_KEY")),
        max_tokens=1000,
        model="mistral-large-latest",
    )

    user, response = client.messages.create_with_completion(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


def test_client_mistral_response():
    from instructor.v2.providers.mistral.client import Mistral

    assert Mistral is not None

    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    instructor_client = instructor.from_mistral(
        client, max_tokens=1000, model="mistral-large-latest"
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10
