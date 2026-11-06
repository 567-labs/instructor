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


def test_invalid_provider_format():
    """Test that error is raised for invalid provider format."""
    from instructor.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        from_provider("invalid-format")
    assert "Model string must be in format" in str(excinfo.value)


def test_unsupported_provider():
    """Test that error is raised for unsupported provider."""
    from instructor.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        from_provider("unsupported/model")
    assert "Unsupported provider" in str(excinfo.value)


def test_provider_dispatch_uses_registered_builder(monkeypatch):
    """Provider dispatch delegates parsed arguments to the registered builder."""
    import instructor.v2.auto_client as auto_client

    result = object()
    calls = []

    def builder(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setitem(auto_client._PROVIDER_BUILDERS, "openai", builder)

    assert from_provider("openai/gpt-4", api_key="test-key", extra="value") is result
    assert calls == [
        {
            "provider": "openai",
            "model_name": "gpt-4",
            "async_client": False,
            "mode": None,
            "api_key": "test-key",
            "kwargs": {"extra": "value"},
            "provider_info": {"provider": "openai", "operation": "initialize"},
        }
    ]


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


@pytest.mark.parametrize(
    "async_client, base_url, expected_base_url",
    [
        (False, "https://api.example.com/v1", "https://api.example.com/v1"),
        (True, "https://api.example.com/v1", "https://api.example.com/v1"),
        (False, None, None),
    ],
)
def test_openai_provider_base_url_handling(async_client, base_url, expected_base_url):
    """Ensure OpenAI provider passes base_url to client constructor when provided."""
    from unittest.mock import patch, MagicMock

    openai_class = "openai.AsyncOpenAI" if async_client else "openai.OpenAI"
    with patch(openai_class) as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            if async_client:
                client = from_provider(
                    "openai/gpt-4",
                    api_key="test-key",
                    base_url=base_url,
                    async_client=True,
                )
            else:
                client = from_provider(
                    "openai/gpt-4",
                    api_key="test-key",
                    base_url=base_url,
                )

            mock_openai_class.assert_called_once()
            _, kwargs = mock_openai_class.call_args
            if expected_base_url is None:
                assert kwargs.get("base_url") in (None, "")
            else:
                assert kwargs["base_url"] == expected_base_url
            assert kwargs["api_key"] == "test-key"
            mock_from_openai.assert_called_once()
            assert client is mock_instructor


def test_databricks_provider_uses_environment_configuration():
    """Ensure Databricks provider pulls host and token from the environment."""
    from unittest.mock import patch, MagicMock
    import os

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            with patch.dict(
                os.environ,
                {
                    "DATABRICKS_HOST": "https://example.cloud.databricks.com",
                    "DATABRICKS_TOKEN": "secret-token",
                },
                clear=True,
            ):
                client = from_provider("databricks/dbrx-instruct")

            mock_openai_class.assert_called_once()
            _, kwargs = mock_openai_class.call_args
            assert kwargs["api_key"] == "secret-token"
            assert (
                kwargs["base_url"]
                == "https://example.cloud.databricks.com/serving-endpoints"
            )
            mock_from_openai.assert_called_once()
            assert client is mock_instructor


def test_databricks_provider_respects_custom_base_url():
    """Ensure Databricks provider does not duplicate serving-endpoints suffix."""
    from unittest.mock import patch, MagicMock
    import os

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            with patch.dict(
                os.environ,
                {
                    "DATABRICKS_TOKEN": "secret-token",
                },
                clear=True,
            ):
                client = from_provider(
                    "databricks/dbrx-instruct",
                    base_url="https://example.cloud.databricks.com/serving-endpoints",
                )

            _, kwargs = mock_openai_class.call_args
            assert (
                kwargs["base_url"]
                == "https://example.cloud.databricks.com/serving-endpoints"
            )
            mock_from_openai.assert_called_once()
            assert client is mock_instructor


def test_databricks_provider_async_client():
    """Ensure Databricks provider returns async client when requested."""
    from unittest.mock import patch, MagicMock
    import os

    with patch("openai.AsyncOpenAI") as mock_async_openai_class:
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client

        with patch("instructor.from_openai") as mock_from_openai:
            mock_instructor = MagicMock()
            mock_from_openai.return_value = mock_instructor

            with patch.dict(
                os.environ,
                {
                    "DATABRICKS_HOST": "https://example.cloud.databricks.com",
                    "DATABRICKS_TOKEN": "secret-token",
                },
                clear=True,
            ):
                client = from_provider("databricks/dbrx-instruct", async_client=True)

            mock_async_openai_class.assert_called_once()
            _, kwargs = mock_async_openai_class.call_args
            assert (
                kwargs["base_url"]
                == "https://example.cloud.databricks.com/serving-endpoints"
            )
            assert kwargs["api_key"] == "secret-token"
            mock_from_openai.assert_called_once()
            assert client is mock_instructor


def test_databricks_provider_requires_token():
    """Ensure Databricks provider raises when no token is available."""
    from instructor.core.exceptions import ConfigurationError
    from unittest.mock import patch, MagicMock
    import os

    with patch("openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with patch("instructor.from_openai") as mock_from_openai:
            mock_from_openai.return_value = MagicMock()
            with patch.dict(
                os.environ,
                {
                    "DATABRICKS_HOST": "https://example.cloud.databricks.com",
                },
                clear=True,
            ):
                with pytest.raises(ConfigurationError):
                    from_provider("databricks/dbrx-instruct")


def test_databricks_provider_requires_host():
    """Ensure Databricks provider raises when no host is available."""
    from instructor.core.exceptions import ConfigurationError
    from unittest.mock import patch, MagicMock
    import os

    with patch("openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with patch("instructor.from_openai") as mock_from_openai:
            mock_from_openai.return_value = MagicMock()
            with patch.dict(
                os.environ,
                {
                    "DATABRICKS_TOKEN": "secret-token",
                },
                clear=True,
            ):
                with pytest.raises(ConfigurationError):
                    from_provider("databricks/dbrx-instruct")


def test_genai_mode_parameter_passed_to_provider():
    """Test that mode parameter is correctly passed to provider functions."""
    from unittest.mock import patch, MagicMock
    from types import ModuleType
    import sys
    import instructor

    mock_google = ModuleType("google")
    mock_genai = ModuleType("google.genai")
    mock_google.__dict__["genai"] = mock_genai

    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        with patch("google.genai.Client", create=True) as mock_genai_class:
            mock_client = MagicMock()
            mock_genai_class.return_value = mock_client

            with patch("instructor.from_genai", create=True) as mock_from_genai:
                mock_instructor = MagicMock()
                mock_from_genai.return_value = mock_instructor

                from_provider(
                    "google/gemini-2.5-flash",
                    mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
                )

                mock_from_genai.assert_called_once()
                _, kwargs = mock_from_genai.call_args
                assert "mode" in kwargs
                assert kwargs["mode"] == instructor.Mode.GENAI_STRUCTURED_OUTPUTS


def test_genai_mode_defaults_when_not_provided():
    """Test that GenAI provider uses generic TOOLS mode when mode is not provided."""
    from unittest.mock import patch, MagicMock
    from types import ModuleType
    import sys
    import instructor

    mock_google = ModuleType("google")
    mock_genai = ModuleType("google.genai")
    mock_google.__dict__["genai"] = mock_genai

    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        with patch("google.genai.Client", create=True) as mock_genai_class:
            mock_client = MagicMock()
            mock_genai_class.return_value = mock_client

            with patch("instructor.from_genai", create=True) as mock_from_genai:
                mock_instructor = MagicMock()
                mock_from_genai.return_value = mock_instructor

                from_provider("google/gemini-2.5-flash")

                mock_from_genai.assert_called_once()
                _, kwargs = mock_from_genai.call_args
                assert "mode" in kwargs
                assert kwargs["mode"] == instructor.Mode.TOOLS


def test_google_provider_runtime_import_error_propagates():
    """Test that ImportError during client initialization is NOT masked.

    This is a regression test for issue #1940 - when using SOCKS proxy without
    socksio installed, httpx raises ImportError during genai.Client() initialization.
    This error should propagate instead of being caught and converted to
    ConfigurationError about missing google-genai package.
    """
    from unittest.mock import patch, MagicMock
    import sys

    # Create mock module for google.genai
    mock_genai_module = MagicMock()

    # Simulate socksio ImportError during Client() initialization
    def client_init_raises(*_args, **_kwargs):
        raise ImportError(
            "Using SOCKS proxy, but the 'socksio' package is not installed. "
            "Make sure to install httpx using `pip install httpx[socks]`."
        )

    mock_genai_module.Client = client_init_raises

    # Create a mock google module
    mock_google = MagicMock()
    mock_google.genai = mock_genai_module

    # Patch sys.modules to use our mock modules
    with patch.dict(
        sys.modules,
        {"google": mock_google, "google.genai": mock_genai_module},
    ):
        mock_from_genai = MagicMock()
        with patch.object(
            __import__("instructor"), "from_genai", mock_from_genai, create=True
        ):
            with pytest.raises(ImportError) as excinfo:
                from_provider("google/gemini-2.5-flash")

            # Should be the socksio error, NOT a ConfigurationError about google-genai
            assert "socksio" in str(excinfo.value)
            assert "google-genai" not in str(excinfo.value)


def test_vertexai_provider_uses_vertexai_sdk_path():
    """The deprecated vertexai provider still routes through the Vertex AI SDK."""
    from unittest.mock import MagicMock, patch
    from types import ModuleType
    import sys

    mock_vertexai = ModuleType("vertexai")
    mock_gener_models = ModuleType("vertexai.generative_models")
    mock_vertexai.__dict__["generative_models"] = mock_gener_models

    with patch.dict(
        sys.modules,
        {
            "vertexai": mock_vertexai,
            "vertexai.generative_models": mock_gener_models,
        },
    ):
        with patch("vertexai.init", create=True) as mock_init:
            with patch(
                "vertexai.generative_models.GenerativeModel", create=True
            ) as mock_model:
                mock_model.return_value = MagicMock()
                with patch("instructor.from_vertexai") as mock_from_vertexai:
                    with pytest.warns(
                        DeprecationWarning,
                        match="The 'vertexai' provider is deprecated",
                    ):
                        from_provider(
                            "vertexai/gemini-pro",
                            project="demo-project",
                            location="us-central1",
                        )

    mock_init.assert_called_once()
    mock_model.assert_called_once_with("gemini-pro")
    mock_from_vertexai.assert_called_once()


def test_generative_ai_provider_runtime_import_error_propagates():
    """Test that ImportError during generative-ai client initialization is NOT masked.

    Similar to test_google_provider_runtime_import_error_propagates but for
    the deprecated generative-ai provider.
    """
    from unittest.mock import patch, MagicMock

    # Create mock module for google.genai
    mock_genai_module = MagicMock()

    # Simulate socksio ImportError during Client() initialization
    def client_init_raises(*_args, **_kwargs):
        raise ImportError(
            "Using SOCKS proxy, but the 'socksio' package is not installed."
        )

    mock_genai_module.Client = client_init_raises

    # Create a mock google module with genai attribute
    mock_google = MagicMock()
    mock_google.genai = mock_genai_module

    with patch.dict(
        "sys.modules",
        {"google": mock_google, "google.genai": mock_genai_module},
    ):
        mock_from_genai = MagicMock()
        with patch.object(
            __import__("instructor"), "from_genai", mock_from_genai, create=True
        ):
            with pytest.warns(
                DeprecationWarning,
                match="The 'generative-ai' provider is deprecated",
            ):
                with pytest.raises(ImportError) as excinfo:
                    from_provider("generative-ai/gemini-pro")

            # Should be the socksio error, NOT a ConfigurationError
            assert "socksio" in str(excinfo.value)
