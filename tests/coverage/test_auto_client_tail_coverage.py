from __future__ import annotations

import builtins
import importlib
import logging
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any

import pytest

from instructor import Mode
from instructor.v2 import auto_client
from instructor.v2.core.errors import ConfigurationError
from tests.coverage.client_cleanup import (
    clear_proxy_environment,
    close_idle_event_loop,
    close_provider_client,
    ignore_fireworks_pydantic_warning,
)


def provider_info(provider: str) -> dict[str, str]:
    return {"provider": provider, "operation": "initialize"}


def expected_provider_deprecation(provider: str) -> AbstractContextManager[Any]:
    if provider in {"vertexai", "generative-ai"}:
        return pytest.warns(
            DeprecationWarning, match=rf"The '{provider}' provider is deprecated\."
        )
    return nullcontext()


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    clear_proxy_environment(monkeypatch)

    yield

    close_idle_event_loop()


@pytest.mark.parametrize(
    "provider,builder,sdk_module,sync_name,async_name,factory_module,factory_name",
    [
        (
            "cerebras",
            auto_client._build_cerebras,
            "cerebras.cloud.sdk",
            "Cerebras",
            "AsyncCerebras",
            "instructor.v2.providers.cerebras.client",
            "from_cerebras",
        ),
        (
            "fireworks",
            auto_client._build_fireworks,
            "fireworks.client",
            "Fireworks",
            "AsyncFireworks",
            "instructor.v2.providers.fireworks.client",
            "from_fireworks",
        ),
    ],
)
@pytest.mark.parametrize("async_client", [False, True])
def test_cloud_sdk_builders_forward_real_client_model_and_options(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    provider: str,
    builder: Any,
    sdk_module: str,
    sync_name: str,
    async_name: str,
    factory_module: str,
    factory_name: str,
    async_client: bool,
) -> None:
    with ignore_fireworks_pydantic_warning():
        sdk = importlib.import_module(sdk_module)
        factory = importlib.import_module(factory_module)
    seen: dict[str, Any] = {}

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        request.addfinalizer(
            partial(close_provider_client, client, async_client=async_client)
        )
        seen["client"] = client
        seen["kwargs"] = kwargs
        return {"provider": provider, **kwargs}

    monkeypatch.setattr(factory, factory_name, capture)
    result = builder(
        provider=provider,
        model_name="llama-test",
        async_client=async_client,
        mode=Mode.JSON,
        api_key="test-key",
        kwargs={"max_tokens": 17},
        provider_info=provider_info(provider),
    )

    expected_type = getattr(sdk, async_name if async_client else sync_name)
    assert isinstance(seen["client"], expected_type)
    if provider == "fireworks":
        assert seen["client"]._client_v1.api_key == "test-key"
    else:
        assert seen["client"].api_key == "test-key"
    assert seen["kwargs"] == {"model": "llama-test", "max_tokens": 17}
    assert result == {"provider": provider, "model": "llama-test", "max_tokens": 17}


@pytest.mark.parametrize("async_client", [False, True])
def test_vertexai_builder_initializes_project_and_forwards_mode(
    monkeypatch: pytest.MonkeyPatch, async_client: bool
) -> None:
    import instructor
    import vertexai
    import vertexai.generative_models as generative_models
    from google.auth.credentials import AnonymousCredentials

    credentials = AnonymousCredentials()
    initialized: dict[str, Any] = {}
    seen: dict[str, Any] = {}
    vertex_model = object()
    model_names: list[str] = []

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        seen["client"] = client
        seen["kwargs"] = kwargs
        return kwargs

    def initialize(**kwargs: Any) -> None:
        initialized.update(kwargs)

    def create_model(model_name: str) -> object:
        model_names.append(model_name)
        return vertex_model

    monkeypatch.setattr(vertexai, "init", initialize)
    monkeypatch.setattr(generative_models, "GenerativeModel", create_model)
    monkeypatch.setattr(instructor, "from_vertexai", capture)
    with expected_provider_deprecation("vertexai"):
        result = auto_client._build_vertexai(
            provider="vertexai",
            model_name="gemini-test",
            async_client=async_client,
            mode=Mode.JSON,
            api_key=None,
            kwargs={
                "project": "project-test",
                "location": "europe-west1",
                "credentials": credentials,
                "max_tokens": 23,
            },
            provider_info=provider_info("vertexai"),
        )

    assert initialized == {
        "project": "project-test",
        "location": "europe-west1",
        "credentials": credentials,
    }
    assert model_names == ["gemini-test"]
    assert seen["client"] is vertex_model
    assert result == {"use_async": async_client, "mode": Mode.JSON, "max_tokens": 23}


def test_vertexai_builder_requires_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    with expected_provider_deprecation("vertexai"):
        with pytest.raises(ValueError, match="Project ID is required for Vertex AI"):
            auto_client._build_vertexai(
                provider="vertexai",
                model_name="gemini-test",
                async_client=False,
                mode=None,
                api_key=None,
                kwargs={},
                provider_info=provider_info("vertexai"),
            )


@pytest.mark.parametrize("async_client", [False, True])
def test_generative_ai_builder_uses_environment_key_and_forwards_options(
    monkeypatch: pytest.MonkeyPatch, async_client: bool
) -> None:
    from google import genai
    import instructor.v2.providers.genai.client as genai_client

    seen: dict[str, Any] = {}

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        seen["client"] = client
        seen["kwargs"] = kwargs
        return kwargs

    monkeypatch.setenv("GOOGLE_API_KEY", "environment-key")
    monkeypatch.setattr(genai_client, "from_genai", capture)
    with expected_provider_deprecation("generative-ai"):
        result = auto_client._build_generative_ai(
            provider="generative-ai",
            model_name="gemini-test",
            async_client=async_client,
            mode=Mode.JSON,
            api_key=None,
            kwargs={"max_tokens": 31},
            provider_info=provider_info("generative-ai"),
        )

    assert isinstance(seen["client"], genai.Client)
    assert seen["client"]._api_client.api_key == "environment-key"
    expected = {"model": "gemini-test", "mode": Mode.JSON, "max_tokens": 31}
    if async_client:
        expected["use_async"] = True
    assert result == expected


@pytest.mark.parametrize(
    "provider,builder,factory_module,factory_name,environment_key,default_url",
    [
        (
            "deepseek",
            auto_client._build_deepseek,
            "instructor.v2.providers.openai.client",
            "from_deepseek",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com",
        ),
        (
            "openrouter",
            auto_client._build_openrouter,
            "instructor.v2.providers.openrouter.client",
            "from_openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1/",
        ),
        (
            "requesty",
            auto_client._build_requesty,
            "instructor.v2.providers.requesty.client",
            "from_requesty",
            "REQUESTY_API_KEY",
            "https://router.requesty.ai/v1/",
        ),
    ],
)
@pytest.mark.parametrize("async_client", [False, True])
def test_openai_compatible_tail_builders_use_environment_and_custom_url(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    provider: str,
    builder: Any,
    factory_module: str,
    factory_name: str,
    environment_key: str,
    default_url: str,
    async_client: bool,
) -> None:
    import openai

    factory = importlib.import_module(factory_module)
    calls: list[tuple[object, dict[str, Any]]] = []

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        request.addfinalizer(
            partial(close_provider_client, client, async_client=async_client)
        )
        calls.append((client, kwargs))
        return kwargs

    monkeypatch.setenv(environment_key, "environment-key")
    monkeypatch.setattr(factory, factory_name, capture)
    default_result = builder(
        provider=provider,
        model_name="model-test",
        async_client=async_client,
        mode=None,
        api_key=None,
        kwargs={"max_tokens": 41},
        provider_info=provider_info(provider),
    )
    custom_result = builder(
        provider=provider,
        model_name="model-test",
        async_client=async_client,
        mode=Mode.JSON,
        api_key="explicit-key",
        kwargs={"base_url": "https://compatible.invalid/v1", "max_tokens": 43},
        provider_info=provider_info(provider),
    )

    expected_type = openai.AsyncOpenAI if async_client else openai.OpenAI
    assert isinstance(calls[0][0], expected_type)
    assert calls[0][0].api_key == "environment-key"
    assert str(calls[0][0].base_url) == default_url
    assert default_result == {
        "model": "model-test",
        "mode": Mode.TOOLS,
        "max_tokens": 41,
    }
    assert isinstance(calls[1][0], expected_type)
    assert calls[1][0].api_key == "explicit-key"
    assert str(calls[1][0].base_url) == "https://compatible.invalid/v1/"
    assert custom_result == {
        "model": "model-test",
        "mode": Mode.JSON,
        "max_tokens": 43,
    }


@pytest.mark.parametrize(
    "provider,builder,environment_key",
    [
        ("deepseek", auto_client._build_deepseek, "DEEPSEEK_API_KEY"),
        ("openrouter", auto_client._build_openrouter, "OPENROUTER_API_KEY"),
        ("requesty", auto_client._build_requesty, "REQUESTY_API_KEY"),
    ],
)
def test_openai_compatible_tail_builders_require_api_key(
    monkeypatch: pytest.MonkeyPatch, provider: str, builder: Any, environment_key: str
) -> None:
    monkeypatch.delenv(environment_key, raising=False)

    with pytest.raises(ConfigurationError, match=f"{environment_key} is not set"):
        builder(
            provider=provider,
            model_name="model-test",
            async_client=False,
            mode=None,
            api_key=None,
            kwargs={},
            provider_info=provider_info(provider),
        )


@pytest.mark.parametrize("async_client", [False, True])
def test_ollama_builder_forwards_mode_url_and_real_client(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    async_client: bool,
) -> None:
    import openai
    import instructor.v2.providers.openai.client as openai_client

    calls: list[tuple[object, dict[str, Any]]] = []

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        request.addfinalizer(
            partial(close_provider_client, client, async_client=async_client)
        )
        calls.append((client, kwargs))
        return kwargs

    monkeypatch.setattr(openai_client, "from_openai", capture)
    tools_result = auto_client._build_ollama(
        provider="ollama",
        model_name="llama3.1:8b",
        async_client=async_client,
        mode=None,
        api_key=None,
        kwargs={"base_url": "http://localhost:22444/v1", "max_tokens": 47},
        provider_info=provider_info("ollama"),
    )
    json_result = auto_client._build_ollama(
        provider="ollama",
        model_name="phi-mini",
        async_client=async_client,
        mode=None,
        api_key=None,
        kwargs={},
        provider_info=provider_info("ollama"),
    )

    expected_type = openai.AsyncOpenAI if async_client else openai.OpenAI
    assert isinstance(calls[0][0], expected_type)
    assert str(calls[0][0].base_url) == "http://localhost:22444/v1/"
    assert calls[0][0].api_key == "ollama"
    assert tools_result == {
        "model": "llama3.1:8b",
        "mode": Mode.TOOLS,
        "max_tokens": 47,
    }
    assert isinstance(calls[1][0], expected_type)
    assert str(calls[1][0].base_url) == "http://localhost:11434/v1/"
    assert json_result == {"model": "phi-mini", "mode": Mode.JSON}


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 10), reason="xai-sdk requires Python 3.10+")
async def test_xai_builder_forwards_real_client_and_mode(
    monkeypatch: pytest.MonkeyPatch, async_client: bool
) -> None:
    import grpc
    from xai_sdk.aio.client import Client as AsyncClient
    from xai_sdk.sync.client import Client as SyncClient
    import instructor.v2.providers.xai.client as xai_client

    seen: dict[str, Any] = {}
    channels: list[Any] = []

    if async_client:
        create_channel = grpc.aio.secure_channel

        def track_channel(*args: Any, **kwargs: Any) -> Any:
            channel = create_channel(*args, **kwargs)
            channels.append(channel)
            return channel

        monkeypatch.setattr(grpc.aio, "secure_channel", track_channel)
    else:
        create_channel = grpc.secure_channel

        def track_channel(*args: Any, **kwargs: Any) -> Any:
            channel = create_channel(*args, **kwargs)
            channels.append(channel)
            return channel

        monkeypatch.setattr(grpc, "secure_channel", track_channel)

    def capture(client: object, **kwargs: Any) -> dict[str, Any]:
        seen["client"] = client
        return kwargs

    monkeypatch.setattr(xai_client, "from_xai", capture)
    try:
        result = auto_client._build_xai(
            provider="xai",
            model_name="grok-test",
            async_client=async_client,
            mode=Mode.JSON,
            api_key="test-key",
            kwargs={"max_tokens": 53},
            provider_info=provider_info("xai"),
        )

        assert isinstance(seen["client"], AsyncClient if async_client else SyncClient)
        assert result == {"model": "grok-test", "mode": Mode.JSON, "max_tokens": 53}
        assert len(channels) == 1
    finally:
        for channel in channels:
            if async_client:
                await channel.close()
            else:
                channel.close()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="xai-sdk requires Python 3.10+")
def test_xai_builder_reports_unavailable_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import instructor.v2.providers.xai.client as xai_client

    monkeypatch.setattr(xai_client, "from_xai", None)
    with pytest.raises(ConfigurationError, match="Failed to import xAI provider"):
        auto_client._build_xai(
            provider="xai",
            model_name="grok-test",
            async_client=False,
            mode=None,
            api_key="test-key",
            kwargs={},
            provider_info=provider_info("xai"),
        )


@pytest.mark.parametrize("async_client", [False, True])
def test_litellm_builder_selects_completion_and_forwards_mode(
    monkeypatch: pytest.MonkeyPatch, async_client: bool
) -> None:
    import litellm
    import instructor.v2.providers.litellm.client as litellm_client

    seen: dict[str, Any] = {}

    def capture(completion: object, **kwargs: Any) -> dict[str, Any]:
        seen["completion"] = completion
        return kwargs

    monkeypatch.setattr(litellm_client, "from_litellm", capture)
    result = auto_client._build_litellm(
        provider="litellm",
        model_name="ignored-model",
        async_client=async_client,
        mode=Mode.JSON,
        api_key=None,
        kwargs={"max_tokens": 59},
        provider_info=provider_info("litellm"),
    )

    assert seen["completion"] is (
        litellm.acompletion if async_client else litellm.completion
    )
    assert result == {"mode": Mode.JSON, "max_tokens": 59}


MISSING_IMPORTS = [
    (
        "cerebras",
        auto_client._build_cerebras,
        "cerebras.cloud.sdk",
        "Cerebras provider",
    ),
    (
        "fireworks",
        auto_client._build_fireworks,
        "fireworks.client",
        "Fireworks provider",
    ),
    ("vertexai", auto_client._build_vertexai, "vertexai", "VertexAI provider"),
    (
        "generative-ai",
        auto_client._build_generative_ai,
        "google",
        "Google GenAI provider",
    ),
    ("ollama", auto_client._build_ollama, "openai", "Ollama provider"),
    ("deepseek", auto_client._build_deepseek, "openai", "DeepSeek provider"),
    (
        "xai",
        auto_client._build_xai,
        "xai_sdk.sync.client",
        "optional dependency `xai-sdk`",
    ),
    ("openrouter", auto_client._build_openrouter, "openai", "OpenRouter provider"),
    ("requesty", auto_client._build_requesty, "openai", "Requesty provider"),
    ("litellm", auto_client._build_litellm, "litellm", "LiteLLM provider"),
]


@pytest.mark.parametrize("provider,builder,blocked_import,message", MISSING_IMPORTS)
def test_tail_builder_missing_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    builder: Any,
    blocked_import: str,
    message: str,
) -> None:
    real_import = builtins.__import__

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
    with expected_provider_deprecation(provider):
        with pytest.raises(ConfigurationError, match=message):
            builder(
                provider=provider,
                model_name="model-test",
                async_client=False,
                mode=None,
                api_key="test-key",
                kwargs={},
                provider_info=provider_info(provider),
            )


FACTORY_FAILURES = [
    (
        "cerebras",
        auto_client._build_cerebras,
        "instructor.v2.providers.cerebras.client",
        "from_cerebras",
    ),
    (
        "fireworks",
        auto_client._build_fireworks,
        "instructor.v2.providers.fireworks.client",
        "from_fireworks",
    ),
    ("vertexai", auto_client._build_vertexai, "instructor", "from_vertexai"),
    (
        "generative-ai",
        auto_client._build_generative_ai,
        "instructor.v2.providers.genai.client",
        "from_genai",
    ),
    (
        "ollama",
        auto_client._build_ollama,
        "instructor.v2.providers.openai.client",
        "from_openai",
    ),
    (
        "deepseek",
        auto_client._build_deepseek,
        "instructor.v2.providers.openai.client",
        "from_deepseek",
    ),
    pytest.param(
        "xai",
        auto_client._build_xai,
        "instructor.v2.providers.xai.client",
        "from_xai",
        marks=pytest.mark.skipif(
            sys.version_info < (3, 10), reason="xai-sdk requires Python 3.10+"
        ),
    ),
    (
        "openrouter",
        auto_client._build_openrouter,
        "instructor.v2.providers.openrouter.client",
        "from_openrouter",
    ),
    (
        "requesty",
        auto_client._build_requesty,
        "instructor.v2.providers.requesty.client",
        "from_requesty",
    ),
    (
        "litellm",
        auto_client._build_litellm,
        "instructor.v2.providers.litellm.client",
        "from_litellm",
    ),
]


@pytest.mark.parametrize("provider,builder,module_name,factory_name", FACTORY_FAILURES)
def test_tail_builder_factory_failure_is_logged_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    builder: Any,
    module_name: str,
    factory_name: str,
) -> None:
    with ignore_fireworks_pydantic_warning():
        module = importlib.import_module(module_name)

    if provider == "vertexai":
        import vertexai
        import vertexai.generative_models as generative_models

        monkeypatch.setattr(vertexai, "init", lambda **_kwargs: None)
        monkeypatch.setattr(
            generative_models, "GenerativeModel", lambda _name: object()
        )

    def fail(client: object, **_kwargs: Any) -> None:
        close_provider_client(client)
        raise RuntimeError("provider factory failed")

    monkeypatch.setattr(module, factory_name, fail)
    with expected_provider_deprecation(provider):
        with caplog.at_level(logging.ERROR, logger="instructor.auto_client"):
            with pytest.raises(RuntimeError, match="provider factory failed"):
                builder(
                    provider=provider,
                    model_name="model-test",
                    async_client=False,
                    mode=None,
                    api_key="test-key",
                    kwargs={"project": "project-test"}
                    if provider == "vertexai"
                    else {},
                    provider_info=provider_info(provider),
                )

    assert f"Error initializing {provider} client" in caplog.text
