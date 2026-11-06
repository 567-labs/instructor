import os

import pytest
import instructor

from tests.llm.test_core_providers.public_client_models import User

anthropic = pytest.importorskip("anthropic")


@pytest.fixture(autouse=True)
def bedrock_credentials():
    boto3 = pytest.importorskip("boto3")
    if (
        boto3.Session(region_name=os.getenv("AWS_REGION_NAME")).get_credentials()
        is None
    ):
        pytest.skip("AWS Bedrock credentials are unavailable")


@pytest.mark.skip(reason="Skip for now")
def test_client_anthropic_bedrock_response():
    client = anthropic.AnthropicBedrock(
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_region=os.getenv("AWS_REGION_NAME"),
    )

    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    user = instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10


@pytest.mark.skip(reason="Skip for now")
@pytest.mark.asyncio
async def test_async_client_anthropic_bedrock_response():
    client = anthropic.AsyncAnthropicBedrock(
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_region=os.getenv("AWS_REGION_NAME"),
    )

    instructor_client = instructor.from_anthropic(
        client,
        max_tokens=1000,
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    user = await instructor_client.messages.create(
        response_model=User,
        messages=[{"role": "user", "content": "Jason is 10"}],
        temperature=0,
    )
    assert user.name == "Jason"
    assert user.age == 10
