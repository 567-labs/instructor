"""Integration tests for the Azure OpenAI / AI Foundry provider.

Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to be set in the
environment. These tests make real API calls and are skipped automatically
when the key or endpoint is absent.
"""

import os

import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic import BaseModel

import instructor
from instructor import Mode


pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"),
        reason="Azure deployment credentials are required for live tests",
    ),
]


class User(BaseModel):
    name: str
    age: int


@pytest.fixture(scope="module")
def sync_tools_client():
    raw = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    return instructor.from_azure(raw, mode=Mode.TOOLS)


@pytest.fixture(scope="module")
def sync_json_client():
    raw = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    return instructor.from_azure(raw, mode=Mode.JSON)


@pytest.fixture(scope="module")
def async_tools_client():
    raw = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    return instructor.from_azure(raw, mode=Mode.TOOLS)


def test_azure_tools_basic_extraction(sync_tools_client):
    user = sync_tools_client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        response_model=User,
    )
    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25


def test_azure_json_mode(sync_json_client):
    user = sync_json_client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Extract: Bob is 42 years old"}],
        response_model=User,
    )
    assert isinstance(user, User)
    assert user.name == "Bob"
    assert user.age == 42


@pytest.mark.asyncio
async def test_azure_async_tools(async_tools_client):
    user = await async_tools_client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Extract: Alice is 30 years old"}],
        response_model=User,
    )
    assert isinstance(user, User)
    assert user.name == "Alice"
    assert user.age == 30
