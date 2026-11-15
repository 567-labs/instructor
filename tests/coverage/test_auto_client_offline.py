"""Offline from_provider contract, driven by one provider table.

Every provider is checked for a missing SDK, a failing instructor factory and,
where the SDK builds offline, a real client. Builder-specific behavior follows
as explicit cases.
"""

from __future__ import annotations

import builtins
import importlib
import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from functools import partial, partialmethod
from typing import Any, NamedTuple, cast

import pytest

from instructor import AsyncInstructor, Instructor, Mode
from instructor.v2 import auto_client
from instructor.v2.core.errors import ConfigurationError, ModeError
from tests.coverage.client_cleanup import (
    clear_proxy_environment,
    close_idle_event_loop,
    close_provider_client,
    ignore_fireworks_pydantic_warning,
)

KEY = {"api_key": "test-key"}
# Tests pass runtime bools and option dicts the Literal overloads cannot match.
from_provider = cast(Callable[..., Any], auto_client.from_provider)


class LegacyGemini:
    """Stand-in for the retired google.generativeai module."""

    configured: list[str] = []

    @classmethod
    def configure(cls, *, api_key: str) -> None:
        cls.configured.append(api_key)

    class GenerativeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name


def offline_vertexai(monkeypatch: pytest.MonkeyPatch) -> None:
    import vertexai
    import vertexai.generative_models as generative_models

    monkeypatch.setattr(vertexai, "init", lambda **_kwargs: None)
    monkeypatch.setattr(generative_models, "GenerativeModel", lambda _name: object())


def legacy_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    LegacyGemini.configured = []
    monkeypatch.setattr(importlib, "import_module", lambda _name: LegacyGemini)


class ProviderCase(NamedTuple):
    provider: str
    sdk: str  # import blocked to simulate a missing SDK
    missing: str  # fragment of the actionable missing-SDK message
    factory: tuple[str, str]  # instructor factory (module, attribute)
    options: dict[str, Any] = KEY
    client_module: str | None = None  # module of the SDK client built offline
    setup: Callable[[pytest.MonkeyPatch], None] | None = None


OPENAI_CLIENT = "instructor.v2.providers.openai.client"
PROVIDER_TABLE = [
    ProviderCase(
        "openai",
        "openai",
        "openai package is required",
        ("instructor", "from_openai"),
        client_module="openai",
    ),
    ProviderCase(
        "azure_openai",
        "openai",
        "Azure OpenAI provider",
        ("instructor.v2.providers.azure.client", "from_azure"),
        {**KEY, "azure_endpoint": "https://azure.invalid"},
        "openai.lib.azure",
    ),
    ProviderCase(
        "anyscale",
        "openai",
        "anyscale provider",
        (OPENAI_CLIENT, "from_anyscale"),
        client_module="openai",
    ),
    ProviderCase(
        "together",
        "openai",
        "together provider",
        (OPENAI_CLIENT, "from_together"),
        client_module="openai",
    ),
    ProviderCase(
        "databricks",
        "openai",
        "Databricks provider",
        ("instructor", "from_openai"),
        {**KEY, "base_url": "https://workspace.invalid"},
        "openai",
    ),
    ProviderCase(
        "anthropic",
        "anthropic",
        "Anthropic provider",
        ("instructor.v2.providers.anthropic.client", "from_anthropic"),
        client_module="anthropic",
    ),
    ProviderCase(
        "google",
        "google.genai",
        "Google provider",
        ("instructor", "from_genai"),
        client_module="google.genai",
    ),
    ProviderCase(
        "gemini",
        "google.generativeai",
        "Gemini provider",
        ("instructor.v2.providers.gemini.client", "from_gemini"),
        setup=legacy_gemini,
    ),
    ProviderCase(
        "mistral",
        "mistralai",
        "Mistral provider",
        ("instructor.v2.providers.mistral.client", "from_mistral"),
        client_module="mistralai",
    ),
    ProviderCase(
        "cohere",
        "cohere",
        "Cohere provider",
        ("instructor.v2.providers.cohere.client", "from_cohere"),
        client_module="cohere",
    ),
    ProviderCase(
        "perplexity",
        "openai",
        "Perplexity provider",
        ("instructor.v2.providers.perplexity.client", "from_perplexity"),
        client_module="openai",
    ),
    ProviderCase(
        "groq",
        "groq",
        "Groq provider",
        ("instructor.v2.providers.groq.client", "from_groq"),
        client_module="groq",
    ),
    ProviderCase(
        "writer",
        "writerai",
        "Writer provider",
        ("instructor.v2.providers.writer.client", "from_writer"),
        client_module="writerai",
    ),
    ProviderCase(
        "bedrock",
        "boto3",
        "AWS Bedrock provider",
        ("instructor.v2.providers.bedrock.client", "from_bedrock"),
        {},
        "botocore",
    ),
    ProviderCase(
        "cerebras",
        "cerebras.cloud.sdk",
        "Cerebras provider",
        ("instructor.v2.providers.cerebras.client", "from_cerebras"),
    ),
    ProviderCase(
        "fireworks",
        "fireworks",
        "Fireworks provider",
        ("instructor.v2.providers.fireworks.client", "from_fireworks"),
    ),
    ProviderCase(
        "vertexai",
        "vertexai",
        "VertexAI provider",
        ("instructor", "from_vertexai"),
        {**KEY, "project": "project-test"},
        setup=offline_vertexai,
    ),
    ProviderCase(
        "generative-ai",
        "google",
        "Google GenAI provider",
        ("instructor.v2.providers.genai.client", "from_genai"),
    ),
    ProviderCase("ollama", "openai", "Ollama provider", (OPENAI_CLIENT, "from_openai")),
    ProviderCase(
        "deepseek", "openai", "DeepSeek provider", (OPENAI_CLIENT, "from_deepseek")
    ),
    ProviderCase(
        "xai",
        "xai_sdk.sync.client",
        "optional dependency `xai-sdk`",
        ("instructor.v2.providers.xai.client", "from_xai"),
    ),
    ProviderCase(
        "openrouter",
        "openai",
        "OpenRouter provider",
        ("instructor.v2.providers.openrouter.client", "from_openrouter"),
    ),
    ProviderCase(
        "litellm",
        "litellm",
        "LiteLLM provider",
        ("instructor.v2.providers.litellm.client", "from_litellm"),
    ),
]
BUILT_OFFLINE = [case for case in PROVIDER_TABLE if case.client_module]


def ids(cases: list[ProviderCase]) -> list[str]:
    return [case.provider for case in cases]


def provider_info(provider: str) -> dict[str, str]:
    return {"provider": provider, "operation": "initialize"}


def expected_deprecation(provider: str) -> AbstractContextManager[Any]:
    if provider in {"vertexai", "generative-ai"}:
        return pytest.warns(
            DeprecationWarning, match=rf"The '{provider}' provider is deprecated\."
        )
    return nullcontext()


def capture(calls: list[tuple[Any, dict[str, Any]]]) -> Callable[..., Any]:
    """Return a stand-in instructor factory that records its client and options."""

    def factory(client: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((client, kwargs))
        return kwargs

    return factory


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
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "MISTRAL_API_KEY",
        "OPENROUTER_API_KEY",
        "PERPLEXITY_API_KEY",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_proxy_environment(monkeypatch)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    from cerebras.cloud.sdk import AsyncCerebras, Cerebras

    for sdk in (Cerebras, AsyncCerebras):
        monkeypatch.setattr(
            sdk, "__init__", partialmethod(sdk.__init__, warm_tcp_connection=False)
        )

    yield

    close_idle_event_loop()


def block_imports(monkeypatch: pytest.MonkeyPatch, blocked: str) -> None:
    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def missing(name: str) -> bool:
        return name == blocked or name.startswith(f"{blocked}.")

    def error() -> ModuleNotFoundError:
        return ModuleNotFoundError(f"No module named '{blocked}'", name=blocked)

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if missing(name):
            raise error()
        return real_import(name, *args, **kwargs)

    def guarded_import_module(name: str, package: str | None = None) -> Any:
        if missing(name):
            raise error()
        return real_import_module(name, package)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)


# --- table-driven contract ----------------------------------------------------


@pytest.mark.parametrize("case", PROVIDER_TABLE, ids=ids(PROVIDER_TABLE))
def test_missing_sdk_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, case: ProviderCase
) -> None:
    block_imports(monkeypatch, case.sdk)

    with expected_deprecation(case.provider):
        with pytest.raises(ConfigurationError, match=case.missing):
            from_provider(f"{case.provider}/model", **case.options)


@pytest.mark.parametrize("case", PROVIDER_TABLE, ids=ids(PROVIDER_TABLE))
def test_factory_failure_is_logged_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: ProviderCase,
) -> None:
    module_name, factory_name = case.factory
    with ignore_fireworks_pydantic_warning():
        module = importlib.import_module(module_name)
    if case.setup is not None:
        case.setup(monkeypatch)

    def fail(client: object, **_kwargs: Any) -> None:
        close_provider_client(client)
        raise RuntimeError("provider factory failed")

    monkeypatch.setattr(module, factory_name, fail)
    with expected_deprecation(case.provider):
        with caplog.at_level(logging.ERROR, logger="instructor.auto_client"):
            with pytest.raises(RuntimeError, match="provider factory failed"):
                from_provider(f"{case.provider}/model", **case.options)

    assert f"Error initializing {case.provider} client" in caplog.text
    if case.provider != "bedrock":  # Bedrock logs from its own builder.
        record = next(r for r in caplog.records if r.name == "instructor.auto_client")
        assert record.exc_info is None
        assert record.funcName == "from_provider"
        assert "provider factory failed" not in record.getMessage()


@pytest.mark.parametrize("case", BUILT_OFFLINE, ids=ids(BUILT_OFFLINE))
@pytest.mark.parametrize("async_client", [False, True])
def test_provider_builds_real_sdk_client_offline(
    request: pytest.FixtureRequest, case: ProviderCase, async_client: bool
) -> None:
    client = from_provider(
        f"{case.provider}/model", async_client=async_client, **case.options
    )
    request.addfinalizer(
        partial(close_provider_client, client.client, async_client=async_client)
    )

    assert type(client) is (AsyncInstructor if async_client else Instructor)
    assert type(client.client).__module__.startswith(cast(str, case.client_module))
    assert callable(client.create_fn)


# --- dispatch -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("gpt-5", "Model string must be in format"),
        ("openai/", "Model string must be in format"),
        ("/gpt-4", "Model string must be in format"),
        ("mystery/model", "Unsupported provider: mystery"),
    ],
)
def test_from_provider_rejects_malformed_or_unknown_models(
    model: str, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        from_provider(model)


def test_from_provider_passes_parsed_options_to_registered_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    cache = object()

    def builder(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "client"

    monkeypatch.setitem(auto_client._PROVIDER_BUILDERS, "openai", builder)

    result = from_provider(
        "openai/gpt-5-nano",
        cache=cache,
        api_key="secret",
        mode=Mode.JSON_SCHEMA,
        timeout=30,
    )

    assert result == "client"
    assert calls == [
        {
            "provider": "openai",
            "model_name": "gpt-5-nano",
            "async_client": False,
            "mode": Mode.JSON_SCHEMA,
            "api_key": "secret",
            "kwargs": {"cache": cache, "timeout": 30},
            "provider_info": provider_info("openai"),
        }
    ]


@pytest.mark.parametrize(
    ("provider", "mode"),
    [
        ("mistral", Mode.JSON_SCHEMA),
        ("groq", Mode.MD_JSON),
        ("writer", Mode.JSON_SCHEMA),
        ("cerebras", Mode.JSON_SCHEMA),
        ("fireworks", Mode.JSON_SCHEMA),
    ],
)
def test_from_provider_preserves_a_registered_mode(
    provider: str, mode: Mode, request: pytest.FixtureRequest
) -> None:
    with ignore_fireworks_pydantic_warning():
        client = from_provider(f"{provider}/unit", mode=mode, **KEY)
    request.addfinalizer(partial(close_provider_client, client.client))

    assert client.mode == mode


def test_from_provider_rejects_an_unregistered_perplexity_mode() -> None:
    with pytest.raises(ModeError):
        from_provider("perplexity/unit", mode=Mode.TOOLS, **KEY)


# --- OpenAI and compatible endpoints -------------------------------------------


def test_openai_forwards_constructor_settings_and_custom_mode(
    request: pytest.FixtureRequest,
) -> None:
    client = from_provider(
        "openai/gpt-4",
        base_url="https://example.invalid/v1",
        organization="org-test",
        timeout=12,
        max_retries="4",
        default_headers={"X-Test": "yes"},
        default_query={"trace": "on"},
        _strict_response_validation=True,
        mode=Mode.JSON,
        **KEY,
    )
    request.addfinalizer(partial(close_provider_client, client.client))
    sdk = client.client
    assert sdk is not None

    assert client.mode is Mode.JSON
    assert str(sdk.base_url) == "https://example.invalid/v1/"
    assert sdk.organization == "org-test"
    assert sdk.max_retries == 4
    assert sdk.default_headers["X-Test"] == "yes"
    assert sdk.default_query == {"trace": "on"}


def test_openai_constructor_failure_is_logged_and_propagated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="instructor.auto_client"):
        with pytest.raises(ValueError, match="invalid literal"):
            from_provider("openai/gpt-4", max_retries="invalid", **KEY)

    assert "Error initializing openai client" in caplog.text


def test_openai_does_not_hide_unrelated_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ModuleNotFoundError("No module named 'unrelated'", name="unrelated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ModuleNotFoundError, match="unrelated"):
        from_provider("openai/gpt-4", **KEY)


@pytest.mark.parametrize(
    ("model", "async_client", "sdk_module", "sdk_class"),
    [
        ("openai/gpt-5", False, "openai", "OpenAI"),
        ("openai/gpt-5", True, "openai", "AsyncOpenAI"),
        ("google/gemini-2.5-flash", False, "google.genai", "Client"),
        ("generative-ai/gemini-pro", False, "google.genai", "Client"),
    ],
)
def test_sdk_constructor_import_error_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    async_client: bool,
    sdk_module: str,
    sdk_class: str,
) -> None:
    """A SOCKS proxy without socksio fails inside the SDK constructor (#1940)."""

    def socks_failure(*_args: Any, **_kwargs: Any) -> None:
        raise ImportError("Using SOCKS proxy, but socksio is not installed.")

    real_import = builtins.__import__

    def reject_legacy_httpx(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "httpx" and sdk_module == "openai":
            raise AssertionError("_build_openai must not import legacy httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib.import_module(sdk_module), sdk_class, socks_failure)
    monkeypatch.setattr(builtins, "__import__", reject_legacy_httpx)

    with expected_deprecation(model.split("/")[0]):
        with pytest.raises(ImportError, match="socksio") as caught:
            from_provider(model, async_client=async_client, **KEY)

    assert not isinstance(caught.value, ConfigurationError)


def test_azure_reads_environment_and_reports_missing_settings(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_API_KEY is not set"):
        from_provider("azure_openai/gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT is not set"):
        from_provider("azure_openai/gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.invalid")

    client = from_provider("azure_openai/gpt-4", api_version="2025-01-01")
    request.addfinalizer(partial(close_provider_client, client.client))

    assert client.client is not None
    assert client.client.api_key == "test-key"
    assert client.client._api_version == "2025-01-01"


@pytest.mark.parametrize(
    ("provider", "environment_key", "default_url", "factory"),
    [
        (
            "anyscale",
            "ANYSCALE_API_KEY",
            "https://api.endpoints.anyscale.com/v1/",
            (OPENAI_CLIENT, "from_anyscale"),
        ),
        (
            "together",
            "TOGETHER_API_KEY",
            "https://api.together.xyz/v1/",
            (OPENAI_CLIENT, "from_together"),
        ),
        (
            "deepseek",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com",
            (OPENAI_CLIENT, "from_deepseek"),
        ),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1/",
            ("instructor.v2.providers.openrouter.client", "from_openrouter"),
        ),
    ],
)
@pytest.mark.parametrize("async_client", [False, True])
def test_compatible_endpoints_use_environment_key_and_custom_url(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    provider: str,
    environment_key: str,
    default_url: str,
    factory: tuple[str, str],
    async_client: bool,
) -> None:
    import openai

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(importlib.import_module(factory[0]), factory[1], capture(calls))
    monkeypatch.setenv(environment_key, "environment-key")
    model = f"{provider}/model-test"

    from_provider(model, async_client=async_client, max_tokens=41)
    from_provider(
        model,
        async_client=async_client,
        api_key="explicit-key",
        base_url="https://compatible.invalid/v1",
        mode=Mode.JSON,
        max_tokens=43,
    )

    for client, _ in calls:
        request.addfinalizer(
            partial(close_provider_client, client, async_client=async_client)
        )
        assert isinstance(client, openai.AsyncOpenAI if async_client else openai.OpenAI)
    (default, default_options), (custom, custom_options) = calls
    assert (default.api_key, str(default.base_url)) == ("environment-key", default_url)
    assert default_options == {
        "model": "model-test",
        "mode": Mode.TOOLS,
        "max_tokens": 41,
    }
    assert (custom.api_key, str(custom.base_url)) == (
        "explicit-key",
        "https://compatible.invalid/v1/",
    )
    assert custom_options == {
        "model": "model-test",
        "mode": Mode.JSON,
        "max_tokens": 43,
    }


@pytest.mark.parametrize(
    ("provider", "environment_key"),
    [
        ("anyscale", "ANYSCALE_API_KEY"),
        ("perplexity", "PERPLEXITY_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
    ],
)
def test_missing_api_key_names_its_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    provider: str,
    environment_key: str,
) -> None:
    with pytest.raises(
        (ConfigurationError, ValueError), match=f"{environment_key} is not set"
    ):
        from_provider(f"{provider}/model")

    monkeypatch.setenv(environment_key, "test-key")
    client = from_provider(f"{provider}/model", async_client=True)
    request.addfinalizer(
        partial(close_provider_client, client.client, async_client=True)
    )

    assert type(client) is AsyncInstructor


@pytest.mark.parametrize(
    ("environment", "options", "expected"),
    [
        (
            {"DATABRICKS_TOKEN": "db-token", "DATABRICKS_HOST": "https://ws.invalid/"},
            {},
            "https://ws.invalid/serving-endpoints/",
        ),
        (
            {
                "DATABRICKS_API_KEY": "db-token",
                "DATABRICKS_WORKSPACE_URL": "https://ws",
            },
            {},
            "https://ws/serving-endpoints/",
        ),
        (
            {"DATABRICKS_TOKEN": "db-token"},
            {"base_url": "https://ws.invalid/serving-endpoints"},
            "https://ws.invalid/serving-endpoints/",
        ),
        ({"DATABRICKS_HOST": "https://ws.invalid"}, {}, "DATABRICKS_TOKEN is not set"),
        ({"DATABRICKS_TOKEN": "db-token"}, {}, "DATABRICKS_HOST is not set"),
    ],
    ids=["host", "workspace-url", "custom-base-url", "missing-token", "missing-host"],
)
@pytest.mark.parametrize("async_client", [False, True])
def test_databricks_reads_workspace_settings(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    environment: dict[str, str],
    options: dict[str, Any],
    expected: str,
    async_client: bool,
) -> None:
    import instructor
    import openai

    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(instructor, "from_openai", capture(calls))
    build = partial(
        from_provider,
        "databricks/meta-llama",
        async_client=async_client,
        timeout=10,
        max_retries=4,
        custom="value",
        **options,
    )

    if not expected.startswith("https"):
        with pytest.raises(ConfigurationError, match=expected):
            build()
        return
    build()

    ((client, forwarded),) = calls
    request.addfinalizer(
        partial(close_provider_client, client, async_client=async_client)
    )
    assert isinstance(client, openai.AsyncOpenAI if async_client else openai.OpenAI)
    assert (client.api_key, str(client.base_url)) == ("db-token", expected)
    assert (client.timeout, client.max_retries) == (10, 4)
    assert forwarded == {"model": "meta-llama", "mode": Mode.TOOLS, "custom": "value"}


# --- other builders -------------------------------------------------------------


def test_anthropic_sets_user_agent_and_default_token_limit(
    request: pytest.FixtureRequest,
) -> None:
    default = from_provider("anthropic/claude", **KEY)
    request.addfinalizer(partial(close_provider_client, default.client))
    limited = from_provider("anthropic/claude", max_tokens=32, mode=Mode.JSON, **KEY)
    request.addfinalizer(partial(close_provider_client, limited.client))

    assert default.client is not None
    assert default.client.default_headers["User-Agent"].startswith("instructor/")
    assert default.kwargs["max_tokens"] == 4096
    assert (limited.kwargs["max_tokens"], limited.mode) == (32, Mode.JSON)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, Mode.TOOLS),
        (Mode.GENAI_STRUCTURED_OUTPUTS, Mode.GENAI_STRUCTURED_OUTPUTS),
    ],
)
def test_google_extracts_client_options_and_model_override(
    monkeypatch: pytest.MonkeyPatch, mode: Mode | None, expected: Mode
) -> None:
    import instructor

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(instructor, "from_genai", capture(calls))

    result = auto_client._build_google(
        provider="google",
        model_name="gemini",
        async_client=False,
        mode=mode,
        api_key="test-key",
        kwargs={
            "http_options": {"api_version": "v1beta"},
            "model": "gemini-override",
            "max_tokens": 23,
        },
        provider_info=provider_info("google"),
    )

    assert type(calls[0][0]).__module__ == "google.genai.client"
    assert result == {
        "mode": expected,
        "use_async": False,
        "model": "gemini-override",
        "max_tokens": 23,
    }


@pytest.mark.parametrize(
    ("api_key", "async_client"),
    [("legacy-key", False), ("legacy-key", True), ("", False)],
)
def test_legacy_gemini_configures_key_and_forwards_mode(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    api_key: str,
    async_client: bool,
) -> None:
    import instructor.v2.providers.gemini.client as gemini_client

    legacy_gemini(monkeypatch)
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(gemini_client, "from_gemini", capture(calls))

    with caplog.at_level(logging.DEBUG, logger="instructor.auto_client"):
        result = from_provider(
            "gemini/gemini-pro", api_key=api_key, async_client=async_client
        )

    assert LegacyGemini.configured == ([api_key] if api_key else [])
    assert ("API key provided for gemini provider" in caplog.text) is bool(api_key)
    assert isinstance(calls[0][0], LegacyGemini.GenerativeModel)
    assert calls[0][0].model_name == "gemini-pro"
    assert result == {"mode": Mode.MD_JSON, "use_async": async_client}


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

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(mistral_client, "from_mistral", capture(calls))

    auto_client._build_mistral(
        provider="mistral",
        model_name="mistral-small",
        async_client=True,
        mode=Mode.JSON_SCHEMA,
        api_key="test-key",
        kwargs={"temperature": 0.2},
        provider_info=provider_info("mistral"),
    )

    ((client, forwarded),) = calls
    assert isinstance(client, LegacyMistral) and client.api_key == "test-key"
    assert forwarded == {
        "model": "mistral-small",
        "mode": Mode.JSON_SCHEMA,
        "use_async": True,
        "temperature": 0.2,
    }


@pytest.mark.parametrize(
    ("model", "options", "credentials", "expected"),
    [
        (
            "anthropic.claude-3-7-sonnet",
            {},
            {"region_name": "ap-south-1", "aws_session_token": "environment-token"},
            {"mode": Mode.TOOLS, "async_client": False},
        ),
        (
            "amazon.titan",
            {},
            {"region_name": "ap-south-1", "aws_session_token": "environment-token"},
            {"mode": Mode.MD_JSON, "async_client": False},
        ),
        (
            "amazon.titan",
            {
                "region": "eu-west-1",
                "aws_access_key_id": "explicit-access",
                "aws_secret_access_key": "explicit-secret",
                "aws_session_token": "explicit-token",
                "mode": Mode.JSON,
                "async_client": True,
                "max_tokens": 19,
            },
            {
                "aws_access_key_id": "explicit-access",
                "aws_secret_access_key": "explicit-secret",
                "aws_session_token": "explicit-token",
                "region_name": "eu-west-1",
            },
            {"mode": Mode.JSON, "async_client": True, "max_tokens": 19},
        ),
    ],
    ids=["environment-anthropic-model", "environment-other-model", "explicit"],
)
def test_bedrock_reads_credentials_and_chooses_mode_from_model(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    options: dict[str, Any],
    credentials: dict[str, str],
    expected: dict[str, Any],
) -> None:
    import boto3
    import instructor.v2.providers.bedrock.client as bedrock_client

    monkeypatch.setenv("AWS_SESSION_TOKEN", "environment-token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    sessions: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        boto3,
        "client",
        lambda service, **kwargs: sessions.append((service, kwargs)) or object(),
    )
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(bedrock_client, "from_bedrock", capture(calls))

    result = from_provider(f"bedrock/{model}", **options)

    assert sessions == [
        (
            "bedrock-runtime",
            {
                "aws_access_key_id": "test-access-key",
                "aws_secret_access_key": "test-secret-key",
                **credentials,
            },
        )
    ]
    assert result == {**expected, "model": model}


@pytest.mark.parametrize(
    "provider,sdk_module,sync_name,async_name,factory_module,factory_name",
    [
        (
            "cerebras",
            "cerebras.cloud.sdk",
            "Cerebras",
            "AsyncCerebras",
            "instructor.v2.providers.cerebras.client",
            "from_cerebras",
        ),
        (
            "fireworks",
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
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(factory, factory_name, capture(calls))

    result = from_provider(
        f"{provider}/llama-test",
        async_client=async_client,
        mode=Mode.JSON_SCHEMA,
        max_tokens=17,
        **KEY,
    )

    ((client, forwarded),) = calls
    request.addfinalizer(
        partial(close_provider_client, client, async_client=async_client)
    )
    assert isinstance(client, getattr(sdk, async_name if async_client else sync_name))
    assert getattr(client, "_client_v1", client).api_key == "test-key"
    assert (
        forwarded
        == result
        == {
            "model": "llama-test",
            "max_tokens": 17,
            "mode": Mode.JSON_SCHEMA,
        }
    )


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
    model_names: list[str] = []
    vertex_model = object()
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(vertexai, "init", lambda **kwargs: initialized.update(kwargs))
    monkeypatch.setattr(
        generative_models,
        "GenerativeModel",
        lambda name: model_names.append(name) or vertex_model,
    )
    monkeypatch.setattr(instructor, "from_vertexai", capture(calls))

    with expected_deprecation("vertexai"):
        result = from_provider(
            "vertexai/gemini-test",
            async_client=async_client,
            mode=Mode.JSON,
            project="project-test",
            location="europe-west1",
            credentials=credentials,
            max_tokens=23,
        )

    assert initialized == {
        "project": "project-test",
        "location": "europe-west1",
        "credentials": credentials,
    }
    assert model_names == ["gemini-test"]
    assert calls[0][0] is vertex_model
    assert result == {"use_async": async_client, "mode": Mode.JSON, "max_tokens": 23}
    with expected_deprecation("vertexai"):
        with pytest.raises(ValueError, match="Project ID is required for Vertex AI"):
            from_provider("vertexai/gemini-test")


@pytest.mark.parametrize("async_client", [False, True])
def test_generative_ai_builder_uses_environment_key_and_forwards_options(
    monkeypatch: pytest.MonkeyPatch, async_client: bool
) -> None:
    from google import genai
    import instructor.v2.providers.genai.client as genai_client

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setenv("GOOGLE_API_KEY", "environment-key")
    monkeypatch.setattr(genai_client, "from_genai", capture(calls))

    with expected_deprecation("generative-ai"):
        result = from_provider(
            "generative-ai/gemini-test",
            async_client=async_client,
            mode=Mode.JSON,
            max_tokens=31,
        )

    assert isinstance(calls[0][0], genai.Client)
    assert calls[0][0]._api_client.api_key == "environment-key"
    assert result == {
        "model": "gemini-test",
        "mode": Mode.JSON,
        "max_tokens": 31,
        "use_async": async_client,
    }


@pytest.mark.parametrize(
    ("model", "options", "base_url", "api_key", "expected"),
    [
        (
            "llama3.1:8b",
            {"base_url": "http://localhost:22444/v1", "max_tokens": 47},
            "http://localhost:22444/v1/",
            "ollama",
            {"mode": Mode.TOOLS, "max_tokens": 47},
        ),
        ("phi4-mini", {}, "http://localhost:11434/v1/", "ollama", {"mode": Mode.JSON}),
        (
            "llama3.2",
            {"api_key": "sec"},
            "http://localhost:11434/v1/",
            "sec",
            {"mode": Mode.TOOLS},
        ),
    ],
    ids=["tool-model-custom-url", "json-model", "explicit-key"],
)
@pytest.mark.parametrize("async_client", [False, True])
def test_ollama_builder_picks_mode_from_model_and_forwards_url(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    model: str,
    options: dict[str, Any],
    base_url: str,
    api_key: str,
    expected: dict[str, Any],
    async_client: bool,
) -> None:
    import openai
    import instructor.v2.providers.openai.client as openai_client

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(openai_client, "from_openai", capture(calls))

    result = from_provider(f"ollama/{model}", async_client=async_client, **options)

    ((client, _),) = calls
    request.addfinalizer(
        partial(close_provider_client, client, async_client=async_client)
    )
    assert isinstance(client, openai.AsyncOpenAI if async_client else openai.OpenAI)
    assert (str(client.base_url), client.api_key) == (base_url, api_key)
    assert result == {"model": model, **expected}


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.asyncio
async def test_xai_builder_forwards_real_client_and_mode(
    monkeypatch: pytest.MonkeyPatch, async_client: bool
) -> None:
    import grpc
    from xai_sdk.aio.client import Client as AsyncClient
    from xai_sdk.sync.client import Client as SyncClient
    import instructor.v2.providers.xai.client as xai_client

    channels: list[Any] = []
    channel_module = grpc.aio if async_client else grpc
    create_channel = channel_module.secure_channel

    def track_channel(*args: Any, **kwargs: Any) -> Any:
        channel = create_channel(*args, **kwargs)
        channels.append(channel)
        return channel

    monkeypatch.setattr(channel_module, "secure_channel", track_channel)
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(xai_client, "from_xai", capture(calls))
    try:
        result = from_provider(
            "xai/grok-test",
            async_client=async_client,
            mode=Mode.JSON,
            max_tokens=53,
            **KEY,
        )

        assert isinstance(calls[0][0], AsyncClient if async_client else SyncClient)
        assert result == {"model": "grok-test", "mode": Mode.JSON, "max_tokens": 53}
        assert len(channels) == 1
    finally:
        for channel in channels:
            if async_client:
                await channel.close()
            else:
                channel.close()


@pytest.mark.parametrize("async_client", [False, True])
def test_litellm_builder_selects_completion_and_forwards_mode(
    monkeypatch: pytest.MonkeyPatch, async_client: bool
) -> None:
    import litellm
    import instructor.v2.providers.litellm.client as litellm_client

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(litellm_client, "from_litellm", capture(calls))

    result = from_provider(
        "litellm/ignored-model",
        async_client=async_client,
        mode=Mode.JSON,
        max_tokens=59,
    )

    assert calls[0][0] is (litellm.acompletion if async_client else litellm.completion)
    assert result == {"mode": Mode.JSON, "max_tokens": 59}
