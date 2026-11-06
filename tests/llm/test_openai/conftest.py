# conftest.py
import os
import pytest

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(
        "OPENAI_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("openai package is not installed", allow_module_level=True)


@pytest.fixture(scope="function")
def client():
    yield OpenAI()


@pytest.fixture(scope="function")
def aclient():
    yield AsyncOpenAI()
