# conftest.py
import os
import pytest

import instructor
from .util import models

if not os.getenv("GOOGLE_API_KEY"):
    pytest.skip(
        "GOOGLE_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from google.genai import Client
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("google-genai package is not installed", allow_module_level=True)


@pytest.fixture(scope="function")
def client():
    yield Client()


@pytest.fixture(scope="function")
def aclient():
    yield Client()


@pytest.fixture(scope="function")
def genai_client():
    # Use the recommended model for sync client, let the test set the mode
    return instructor.from_provider(
        f"google/{models[0]}",
    )
