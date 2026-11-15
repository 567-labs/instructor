from __future__ import annotations

import inspect
from typing import Any

import pytest
from instructor.auto_client import from_provider
from instructor.core.exceptions import (
    ClientError,
    ConfigurationError,
    InstructorRetryException,
)
from openai import OpenAIError
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel, ValidationError


# --- User model and prompt (from main.py) ---
class User(BaseModel):
    name: str
    age: int


USER_EXTRACTION_PROMPT: ChatCompletionUserMessageParam = {
    "role": "user",
    "content": "Ivan is 28 and strays in Singapore. Extract it as a user object",
}

# --- Providers to test (from main.py) ---
PROVIDERS = [
    "anthropic/claude-sonnet-4-6",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "azure_openai/gpt-4o-mini",
    "mistral/ministral-8b-latest",
    "cohere/command-a-03-2025",
    "perplexity/sonar-pro",
    "groq/llama-3.1-8b-instant",
    "writer/palmyra-x5",
    "cerebras/llama-4-scout-17b-16e-instruct",
    "deepseek/deepseek-chat",
    "fireworks/accounts/fireworks/models/llama4-maverick-instruct-basic",
    "vertexai/gemini-3-flash",
]


def close_sync_provider_client(client: Any) -> None:
    for name, args in (("close", ()), ("__exit__", (None, None, None))):
        close = getattr(client, name, None)
        if callable(close):
            close(*args)
            return


async def close_async_provider_client(client: Any) -> None:
    for name, args in (
        ("aclose", ()),
        ("close", ()),
        ("__aexit__", (None, None, None)),
    ):
        close = getattr(client, name, None)
        if not callable(close):
            continue
        result = close(*args)
        if inspect.isawaitable(result):
            await result
        return


def should_skip_provider(provider_string: str) -> bool:
    import os

    if os.getenv("INSTRUCTOR_ENV") == "CI":
        return provider_string not in [
            "cohere/command-a-03-2025",
            "google/gemini-2.5-flash",
            "openai/gpt-4o-mini",
        ]
    return False


def should_skip_provider_exception(exc: Exception) -> bool:
    """Return True for provider failures caused by local environment setup."""
    if isinstance(exc, (ClientError, ConfigurationError, ImportError)):
        return True
    if isinstance(exc, OpenAIError):
        return "api_key client option must be set" in str(exc).lower()
    if type(exc) is ValueError or (
        type(exc).__module__.startswith("cohere.") and type(exc).__name__ == "ApiError"
    ):
        message = str(exc).lower()
        return any(
            marker in message
            for marker in ("api key", "api_key", "project id is required")
        )
    if isinstance(exc, InstructorRetryException):
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "api key",
                "authentication",
                "connection",
                "connect",
                "quota",
                "rate limit",
                "resource_exhausted",
            )
        )
    return False


def skip_or_raise_provider_exception(provider_string: str, exc: Exception) -> None:
    if should_skip_provider_exception(exc):
        raise pytest.skip.Exception(
            f"Provider {provider_string} not available in this environment: {exc}"
        )
    raise exc


@pytest.mark.parametrize(
    "exception_type, message, should_skip",
    [
        ("value", "Missing key inputs argument: provide api_key", True),
        (
            "cohere",
            "The client must be instantiated with CO_API_KEY",
            True,
        ),
        (
            "openai",
            "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable",
            True,
        ),
        ("value", "Invalid response format", False),
        (
            "cohere",
            "The requested model is unavailable",
            False,
        ),
        ("openai", "The requested model is unavailable", False),
    ],
    ids=[
        "google-credentials",
        "cohere-credentials",
        "openai-credentials",
        "value-error",
        "api-error",
        "openai-error",
    ],
)
def test_skip_or_raise_provider_exception(exception_type, message, should_skip):
    if exception_type == "cohere":
        api_error = pytest.importorskip("cohere.core.api_error").ApiError
        exc = api_error(body=message)
    elif exception_type == "openai":
        exc = OpenAIError(message)
    else:
        exc = ValueError(message)
    expected = pytest.skip.Exception if should_skip else type(exc)

    with pytest.raises(expected):
        skip_or_raise_provider_exception("provider/model", exc)


def test_validation_error_with_api_key_is_not_skipped() -> None:
    with pytest.raises(ValidationError) as error:
        User.model_validate({"api_key": "present-but-not-a-user"})

    assert not should_skip_provider_exception(error.value)
    with pytest.raises(ValidationError):
        skip_or_raise_provider_exception("provider/model", error.value)


@pytest.mark.parametrize("provider_string", PROVIDERS)
@pytest.mark.llm
def test_user_extraction_sync(provider_string):
    """Test user extraction for each provider (sync)."""

    if should_skip_provider(provider_string):
        raise pytest.skip.Exception(f"Skipping provider {provider_string} on CI")

    try:
        client = from_provider(provider_string)  # type: ignore[arg-type]
    except Exception as e:
        skip_or_raise_provider_exception(provider_string, e)

    try:
        response = client.chat.completions.create(
            messages=[USER_EXTRACTION_PROMPT],  # type: ignore[arg-type]
            response_model=User,
        )
        assert isinstance(response, User)
        assert response.name.lower() == "ivan"
        assert response.age == 28
    except Exception as e:
        skip_or_raise_provider_exception(provider_string, e)
    finally:
        close_sync_provider_client(client.client)


@pytest.mark.parametrize("provider_string", PROVIDERS)
@pytest.mark.asyncio
@pytest.mark.llm
async def test_user_extraction_async(provider_string):
    """Test user extraction for each provider (async)."""

    if should_skip_provider(provider_string):
        raise pytest.skip.Exception(f"Skipping provider {provider_string} on CI")

    try:
        client = from_provider(provider_string, async_client=True)  # type: ignore[arg-type]
    except Exception as e:
        skip_or_raise_provider_exception(provider_string, e)

    try:
        response = await client.chat.completions.create(
            messages=[USER_EXTRACTION_PROMPT],  # type: ignore[arg-type]
            response_model=User,
        )
        assert isinstance(response, User)
        assert response.name.lower() == "ivan"
        assert response.age == 28
    except Exception as e:
        skip_or_raise_provider_exception(provider_string, e)
    finally:
        await close_async_provider_client(client.client)


@pytest.mark.llm
def test_additional_kwargs_passed():
    """Test that additional kwargs are passed to provider."""
    import instructor
    from instructor.core.exceptions import InstructorRetryException
    import os

    if os.getenv("INSTRUCTOR_ENV") == "CI" or not os.getenv("ANTHROPIC_API_KEY"):
        raise pytest.skip.Exception("Skipping live Anthropic test without credentials")

    client = instructor.from_provider("anthropic/claude-sonnet-4-6", max_tokens=10)

    try:
        with pytest.raises(InstructorRetryException) as excinfo:
            client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": "Generate a sentence with 20 characters",
                    }
                ],
                response_model=str,
            )

        assert "The output is incomplete due to a max_tokens length limit" in str(
            excinfo.value
        )
    finally:
        close_sync_provider_client(client.client)
