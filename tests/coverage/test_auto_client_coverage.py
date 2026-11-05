from __future__ import annotations

import builtins
import importlib
import logging
from collections.abc import Iterator
from functools import partial
from typing import Any, cast

import pytest

from instructor import AsyncInstructor, Instructor, Mode
from instructor.v2 import auto_client
from instructor.v2.core.errors import ConfigurationError
from tests.coverage.client_cleanup import (
    clear_proxy_environment,
    close_idle_event_loop,
    close_provider_client,
)


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "ANYSCALE_API_KEY",
        "TOGETHER_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "DATABRICKS_TOKEN",
        "DATABRICKS_API_KEY",
        "DATABRICKS_HOST",
        "DATABRICKS_BASE_URL",
        "DATABRICKS_WORKSPACE_URL",
        "GOOGLE_API_KEY",
        "MISTRAL_API_KEY",
        "PERPLEXITY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_proxy_environment(monkeypatch)

    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    yield

    close_idle_event_loop()


REAL_PROVIDER_CASES = [
    ("openai/gpt-4", {"api_key": "test-key"}, "openai"),
    (
        "azure_openai/gpt-4",
        {"api_key": "test-key", "azure_endpoint": "https://azure.invalid"},
        "openai.lib.azure",
    ),
    ("anyscale/llama", {"api_key": "test-key"}, "openai"),
    ("together/llama", {"api_key": "test-key"}, "openai"),
    ("anthropic/claude", {"api_key": "test-key"}, "anthropic"),
    ("google/gemini", {"api_key": "test-key"}, "google.genai"),
    ("mistral/mistral-small", {"api_key": "test-key"}, "mistralai"),
    ("cohere/command", {"api_key": "test-key"}, "cohere"),
    ("perplexity/sonar", {"api_key": "test-key"}, "openai"),
    ("groq/llama", {"api_key": "test-key"}, "groq"),
    ("writer/palmyra", {"api_key": "test-key"}, "writerai"),
    ("bedrock/anthropic.claude", {}, "botocore"),
]


@pytest.mark.parametrize("model,kwargs,inner_module", REAL_PROVIDER_CASES)
@pytest.mark.parametrize("async_client", [False, True])
def test_real_provider_clients_build_without_network(
    request: pytest.FixtureRequest,
    model: str,
    kwargs: dict[str, Any],
    inner_module: str,
    async_client: bool,
) -> None:
    client = auto_client.from_provider(model, async_client=async_client, **dict(kwargs))
    request.addfinalizer(
        partial(close_provider_client, client.client, async_client=async_client)
    )

    assert type(client) is (AsyncInstructor if async_client else Instructor)
    assert type(client.client).__module__.startswith(inner_module)
    assert callable(client.create_fn)


def test_openai_forwards_all_constructor_settings_and_custom_mode(
    request: pytest.FixtureRequest,
) -> None:
    client = auto_client.from_provider(
        "openai/gpt-4",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        organization="org-test",
        timeout=12,
        max_retries="4",
        default_headers={"X-Test": "yes"},
        default_query={"trace": "on"},
        _strict_response_validation=True,
        mode=Mode.JSON,
    )
    request.addfinalizer(partial(close_provider_client, client.client))

    assert type(client) is Instructor
    assert client.mode is Mode.JSON
    assert client.client is not None
    assert str(client.client.base_url) == "https://example.invalid/v1/"
    assert client.client.organization == "org-test"
    assert client.client.max_retries == 4
    assert client.client.default_headers["X-Test"] == "yes"
    assert client.client.default_query == {"trace": "on"}


def test_azure_reads_environment_and_reports_missing_settings(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_API_KEY is not set"):
        auto_client.from_provider("azure_openai/gpt-4")

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT is not set"):
        auto_client.from_provider("azure_openai/gpt-4")

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.invalid")
    client = auto_client.from_provider("azure_openai/gpt-4", api_version="2025-01-01")
    request.addfinalizer(partial(close_provider_client, client.client))

    assert type(client) is Instructor
    assert client.client is not None
    assert client.client.api_key == "test-key"
    assert client.client._api_version == "2025-01-01"


@pytest.mark.parametrize(
    "model,environment_key,default_url",
    [
        (
            "anyscale/llama",
            "ANYSCALE_API_KEY",
            "https://api.endpoints.anyscale.com/v1/",
        ),
        ("together/llama", "TOGETHER_API_KEY", "https://api.together.xyz/v1/"),
    ],
)
def test_openai_compatible_providers_use_environment_and_custom_url(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    model: str,
    environment_key: str,
    default_url: str,
) -> None:
    monkeypatch.setenv(environment_key, "test-key")
    default_client = auto_client.from_provider(model)
    request.addfinalizer(partial(close_provider_client, default_client.client))
    custom_client = auto_client.from_provider(
        model, base_url="https://compatible.invalid/v1", async_client=True
    )
    request.addfinalizer(
        partial(close_provider_client, custom_client.client, async_client=True)
    )

    assert default_client.client is not None
    assert custom_client.client is not None
    assert str(default_client.client.base_url) == default_url
    assert str(custom_client.client.base_url) == "https://compatible.invalid/v1/"
    assert custom_client.mode is Mode.TOOLS


def test_anthropic_sets_user_agent_and_default_token_limit(
    request: pytest.FixtureRequest,
) -> None:
    default_client = auto_client.from_provider("anthropic/claude", api_key="test-key")
    request.addfinalizer(partial(close_provider_client, default_client.client))
    limited_client = auto_client.from_provider(
        "anthropic/claude", api_key="test-key", max_tokens=32, mode=Mode.JSON
    )
    request.addfinalizer(partial(close_provider_client, limited_client.client))

    assert default_client.client is not None
    assert default_client.client.default_headers["User-Agent"].startswith("instructor/")
    assert default_client.kwargs["max_tokens"] == 4096
    assert limited_client.kwargs["max_tokens"] == 32
    assert limited_client.mode is Mode.JSON


def test_anthropic_reports_unavailable_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    import instructor.v2.providers.anthropic.client as anthropic_client

    monkeypatch.setattr(anthropic_client, "from_anthropic", None)
    with pytest.raises(ConfigurationError, match="Failed to import Anthropic provider"):
        auto_client.from_provider("anthropic/claude", api_key="test-key")


def test_google_extracts_client_options_and_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import instructor

    seen: dict[str, Any] = {}

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        seen["client"] = client
        seen["kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(instructor, "from_genai", capture)
    result = auto_client._build_google(
        provider="google",
        model_name="gemini",
        async_client=False,
        mode=Mode.JSON,
        api_key="test-key",
        kwargs={
            "http_options": {"api_version": "v1beta"},
            "model": "gemini-override",
            "max_tokens": 23,
        },
        provider_info={"provider": "google", "operation": "initialize"},
    )

    assert type(seen["client"]).__module__ == "google.genai.client"
    assert result == {
        "mode": Mode.JSON,
        "use_async": False,
        "model": "gemini-override",
        "max_tokens": 23,
    }


class LegacyGemini:
    configured: list[str] = []

    @classmethod
    def configure(cls, *, api_key: str) -> None:
        cls.configured.append(api_key)

    class GenerativeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name


@pytest.mark.parametrize("async_client", [False, True])
def test_legacy_gemini_configures_key_and_forwards_mode(
    monkeypatch: pytest.MonkeyPatch, async_client: bool
) -> None:
    import instructor.v2.providers.gemini.client as gemini_client

    calls: list[tuple[object, dict[str, Any]]] = []

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        calls.append((client, kwargs))
        return kwargs

    LegacyGemini.configured = []
    monkeypatch.setattr(importlib, "import_module", lambda _name: LegacyGemini)
    monkeypatch.setattr(gemini_client, "from_gemini", capture)
    result = auto_client.from_provider(
        "gemini/gemini-pro", api_key="legacy-key", async_client=async_client
    )

    assert LegacyGemini.configured == ["legacy-key"]
    assert isinstance(calls[0][0], LegacyGemini.GenerativeModel)
    assert calls[0][0].model_name == "gemini-pro"
    assert result == {"mode": Mode.MD_JSON, "use_async": async_client}


def test_legacy_gemini_does_not_configure_an_empty_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import instructor.v2.providers.gemini.client as gemini_client

    calls: list[tuple[object, dict[str, Any]]] = []

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        calls.append((client, kwargs))
        return kwargs

    LegacyGemini.configured = []
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(importlib, "import_module", lambda _name: LegacyGemini)
    monkeypatch.setattr(gemini_client, "from_gemini", capture)
    with caplog.at_level(logging.DEBUG, logger="instructor.auto_client"):
        result = auto_client.from_provider("gemini/gemini-pro", api_key="")

    assert LegacyGemini.configured == []
    assert isinstance(calls[0][0], LegacyGemini.GenerativeModel)
    assert calls[0][0].model_name == "gemini-pro"
    assert result == {"mode": Mode.MD_JSON, "use_async": False}
    assert "API key provided for gemini provider" not in caplog.text


def test_mistral_requires_key_and_accepts_environment_key(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    with pytest.raises(ValueError, match="MISTRAL_API_KEY is not set"):
        auto_client.from_provider("mistral/mistral-small")

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = auto_client.from_provider("mistral/mistral-small", async_client=True)
    request.addfinalizer(
        partial(close_provider_client, client.client, async_client=True)
    )

    assert type(client) is AsyncInstructor
    assert client.client is not None
    assert callable(client.create_fn)


def test_mistral_builder_supports_sdk_v1_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import instructor.v2.providers.mistral.client as mistral_client

    class LegacyMistral:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    class LegacySDK:
        Mistral = LegacyMistral

    real_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> Any:
        if name == "mistralai.client":
            return object()
        if name == "mistralai":
            return LegacySDK
        return real_import_module(name, package)

    captured: dict[str, Any] = {}

    def from_mistral(client: object, **kwargs: Any) -> dict[str, Any]:
        captured["client"] = client
        captured["kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(mistral_client, "from_mistral", from_mistral)

    result = cast(
        Any,
        auto_client._build_mistral(
            provider="mistral",
            model_name="mistral-small",
            async_client=True,
            mode=None,
            api_key="test-key",
            kwargs={"temperature": 0.2},
            provider_info={"provider": "mistral", "operation": "initialize"},
        ),
    )

    assert isinstance(captured["client"], LegacyMistral)
    assert captured["client"].api_key == "test-key"
    assert captured["kwargs"] == {
        "model": "mistral-small",
        "use_async": True,
        "temperature": 0.2,
    }
    assert result == captured["kwargs"]


@pytest.mark.parametrize(
    "model,environment_key",
    [
        ("perplexity/sonar", "PERPLEXITY_API_KEY"),
    ],
)
def test_compatible_providers_require_keys_and_accept_environment_keys(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    model: str,
    environment_key: str,
) -> None:
    with pytest.raises(ConfigurationError, match=f"{environment_key} is not set"):
        auto_client.from_provider(model)

    monkeypatch.setenv(environment_key, "test-key")
    client = auto_client.from_provider(model, async_client=True)
    request.addfinalizer(
        partial(close_provider_client, client.client, async_client=True)
    )

    assert type(client) is AsyncInstructor
    assert client.client is not None
    assert client.client.api_key == "test-key"


def test_bedrock_forwards_explicit_region_credentials_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boto3
    import instructor.v2.providers.bedrock.client as bedrock_client

    seen: dict[str, Any] = {}

    def create_client(service_name: str, **kwargs: Any) -> object:
        seen["service"] = service_name
        seen["client_kwargs"] = kwargs
        return object()

    def create_instructor(client: object, **kwargs: Any) -> dict[str, Any]:
        seen["client"] = client
        seen["factory_kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(boto3, "client", create_client)
    monkeypatch.setattr(bedrock_client, "from_bedrock", create_instructor)
    result = auto_client.from_provider(
        "bedrock/amazon.titan",
        region="eu-west-1",
        aws_access_key_id="explicit-access",
        aws_secret_access_key="explicit-secret",
        aws_session_token="explicit-token",
        mode=Mode.JSON,
        async_client=True,
        max_tokens=19,
    )

    assert seen["service"] == "bedrock-runtime"
    assert seen["client_kwargs"] == {
        "aws_access_key_id": "explicit-access",
        "aws_secret_access_key": "explicit-secret",
        "aws_session_token": "explicit-token",
        "region_name": "eu-west-1",
    }
    assert result == {
        "mode": Mode.JSON,
        "async_client": True,
        "model": "amazon.titan",
        "max_tokens": 19,
    }


def test_bedrock_reads_all_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import boto3
    import instructor.v2.providers.bedrock.client as bedrock_client

    seen: dict[str, Any] = {}

    def create_client(_service_name: str, **kwargs: Any) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "environment-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "environment-token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setattr(boto3, "client", create_client)
    monkeypatch.setattr(
        bedrock_client, "from_bedrock", lambda _client, **kwargs: kwargs
    )
    result = cast(dict[str, Any], auto_client.from_provider("bedrock/amazon.titan"))

    assert seen == {
        "aws_access_key_id": "environment-access",
        "aws_secret_access_key": "environment-secret",
        "aws_session_token": "environment-token",
        "region_name": "ap-south-1",
    }
    assert result["mode"] is Mode.MD_JSON


MISSING_IMPORT_CASES = [
    ("openai/gpt-4", "openai", "openai package is required"),
    ("azure_openai/gpt-4", "openai", "Azure OpenAI provider"),
    ("anyscale/llama", "openai", "anyscale provider"),
    ("together/llama", "openai", "together provider"),
    ("databricks/model", "openai", "Databricks provider"),
    ("anthropic/claude", "anthropic", "Anthropic provider"),
    ("google/gemini", "google.genai", "Google provider"),
    ("mistral/model", "mistralai", "Mistral provider"),
    ("cohere/command", "cohere", "Cohere provider"),
    ("perplexity/sonar", "openai", "Perplexity provider"),
    ("groq/llama", "groq", "Groq provider"),
    ("writer/palmyra", "writerai", "Writer provider"),
    ("bedrock/model", "boto3", "AWS Bedrock provider"),
]


@pytest.mark.parametrize("model,blocked_import,message", MISSING_IMPORT_CASES)
def test_missing_provider_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    blocked_import: str,
    message: str,
) -> None:
    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == blocked_import or name.startswith(f"{blocked_import}."):
            raise ModuleNotFoundError(
                f"No module named '{blocked_import}'", name=blocked_import
            )
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    def guarded_import_module(name: str, package: str | None = None) -> Any:
        if name == blocked_import or name.startswith(f"{blocked_import}."):
            raise ModuleNotFoundError(
                f"No module named '{blocked_import}'", name=blocked_import
            )
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    with pytest.raises(ConfigurationError, match=message):
        auto_client.from_provider(model, api_key="test-key")


def test_legacy_gemini_missing_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> None:
        raise ModuleNotFoundError(
            "No module named 'google.generativeai'", name="google.generativeai"
        )

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ConfigurationError, match="Gemini provider"):
        auto_client.from_provider("gemini/gemini-pro")


def test_openai_does_not_hide_unrelated_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "openai":
            raise ModuleNotFoundError("No module named 'unrelated'", name="unrelated")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ModuleNotFoundError, match="unrelated"):
        auto_client.from_provider("openai/gpt-4", api_key="test-key")


FACTORY_FAILURE_CASES = [
    ("openai/gpt-4", "instructor", "from_openai", {"api_key": "test-key"}),
    (
        "azure_openai/gpt-4",
        "instructor.v2.providers.azure.client",
        "from_azure",
        {"api_key": "test-key", "azure_endpoint": "https://azure.invalid"},
    ),
    (
        "anyscale/llama",
        "instructor.v2.providers.openai.client",
        "from_anyscale",
        {"api_key": "test-key"},
    ),
    (
        "databricks/model",
        "instructor",
        "from_openai",
        {"api_key": "test-key", "base_url": "https://workspace.invalid"},
    ),
    (
        "anthropic/claude",
        "instructor.v2.providers.anthropic.client",
        "from_anthropic",
        {"api_key": "test-key"},
    ),
    ("google/gemini", "instructor", "from_genai", {"api_key": "test-key"}),
    (
        "mistral/model",
        "instructor.v2.providers.mistral.client",
        "from_mistral",
        {"api_key": "test-key"},
    ),
    (
        "cohere/command",
        "instructor.v2.providers.cohere.client",
        "from_cohere",
        {"api_key": "test-key"},
    ),
    (
        "perplexity/sonar",
        "instructor.v2.providers.perplexity.client",
        "from_perplexity",
        {"api_key": "test-key"},
    ),
    (
        "groq/llama",
        "instructor.v2.providers.groq.client",
        "from_groq",
        {"api_key": "test-key"},
    ),
    (
        "writer/palmyra",
        "instructor.v2.providers.writer.client",
        "from_writer",
        {"api_key": "test-key"},
    ),
    (
        "bedrock/model",
        "instructor.v2.providers.bedrock.client",
        "from_bedrock",
        {},
    ),
]


@pytest.mark.parametrize("model,module_name,factory_name,kwargs", FACTORY_FAILURE_CASES)
def test_provider_factory_failures_are_logged_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    model: str,
    module_name: str,
    factory_name: str,
    kwargs: dict[str, Any],
) -> None:
    module = importlib.import_module(module_name)

    def fail(client: object, **_kwargs: Any) -> None:
        close_provider_client(client)
        raise RuntimeError("provider factory failed")

    monkeypatch.setattr(module, factory_name, fail)
    with caplog.at_level(logging.ERROR, logger="instructor.auto_client"):
        with pytest.raises(RuntimeError, match="provider factory failed"):
            auto_client.from_provider(model, **dict(kwargs))

    assert f"Error initializing {model.split('/', 1)[0]} client" in caplog.text


def test_legacy_gemini_factory_failure_is_logged_and_propagated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import instructor.v2.providers.gemini.client as gemini_client

    def fail(_client: object, **_kwargs: Any) -> None:
        raise RuntimeError("legacy Gemini factory failed")

    monkeypatch.setattr(importlib, "import_module", lambda _name: LegacyGemini)
    monkeypatch.setattr(gemini_client, "from_gemini", fail)
    with caplog.at_level(logging.ERROR, logger="instructor.auto_client"):
        with pytest.raises(RuntimeError, match="legacy Gemini factory failed"):
            auto_client.from_provider("gemini/gemini-pro", api_key="test-key")

    assert "Error initializing gemini client" in caplog.text


def test_openai_constructor_failure_is_logged_and_propagated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="instructor.auto_client"):
        with pytest.raises(ValueError, match="invalid literal"):
            auto_client.from_provider(
                "openai/gpt-4", api_key="test-key", max_retries="invalid"
            )

    assert "Error initializing openai client" in caplog.text
