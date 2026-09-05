"""Integration tests for the MiniMax provider.

Requires MINIMAX_API_KEY to be set in the environment.
These tests make real API calls and are skipped automatically when the key
is absent (see conftest.py).
"""

import os
import pytest
import openai
from pydantic import BaseModel

import instructor
from instructor import Mode


MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M2.7"


class User(BaseModel):
    name: str
    age: int


@pytest.fixture(scope="module")
def sync_tools_client():
    raw = openai.OpenAI(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url=MINIMAX_BASE_URL,
    )
    return instructor.from_minimax(raw, mode=Mode.MINIMAX_TOOLS)


@pytest.fixture(scope="module")
def sync_json_client():
    raw = openai.OpenAI(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url=MINIMAX_BASE_URL,
    )
    return instructor.from_minimax(raw, mode=Mode.MINIMAX_JSON)


@pytest.fixture(scope="module")
def async_tools_client():
    raw = openai.AsyncOpenAI(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url=MINIMAX_BASE_URL,
    )
    return instructor.from_minimax(raw, mode=Mode.MINIMAX_TOOLS)


def test_tools_mode_extraction(sync_tools_client):
    user = sync_tools_client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        response_model=User,
    )
    assert isinstance(user, User)
    assert user.name == "Jason"
    assert user.age == 25


def test_json_mode_extraction(sync_json_client):
    user = sync_json_client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=[{"role": "user", "content": "Extract: Sarah is 30 years old"}],
        response_model=User,
    )
    assert isinstance(user, User)
    assert user.name == "Sarah"
    assert user.age == 30


@pytest.mark.asyncio
async def test_async_tools_mode_extraction(async_tools_client):
    user = await async_tools_client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=[{"role": "user", "content": "Extract: Bob is 42 years old"}],
        response_model=User,
    )
    assert isinstance(user, User)
    assert user.name == "Bob"
    assert user.age == 42


def test_from_provider(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", os.environ["MINIMAX_API_KEY"])
    client = instructor.from_provider("minimax/MiniMax-M2.7")
    user = client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=[{"role": "user", "content": "Extract: Alice is 35 years old"}],
        response_model=User,
    )
    assert isinstance(user, User)
    assert user.name == "Alice"
    assert user.age == 35
