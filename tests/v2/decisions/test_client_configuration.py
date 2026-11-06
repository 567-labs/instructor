"""Decision routing contracts independent of provider SDKs and services."""

from __future__ import annotations

from typing import Any, get_type_hints

import httpx
import pytest

import instructor
from instructor.cache import AutoCache
from instructor.decisions.client import AsyncDecisionsClient, DecisionsClient


@pytest.mark.parametrize(
    "provider,model,endpoint,key_name",
    [
        (
            "typesafe",
            "jev-latest",
            "https://api.typesafe.ai/v1/systemone",
            "TYPESAFE_API_KEY",
        ),
        (
            "openrouter",
            "typesafe/jev-1.13",
            "https://openrouter.ai/api/alpha/decisions",
            "OPENROUTER_API_KEY",
        ),
        (
            "openai",
            "gpt-6-luna",
            "https://api.openai.com/v1/decisions",
            "OPENAI_API_KEY",
        ),
    ],
)
@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_provider_configuration_uses_its_own_credentials(
    monkeypatch, provider, model, endpoint, key_name, async_client
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(key_name, "decision-only-key")
    if async_client:
        async with instructor.from_provider(
            f"{provider}/{model}", mode=instructor.Mode.DECISIONS, async_client=True
        ) as client:
            assert isinstance(client, AsyncDecisionsClient)
            assert client.model == model
            assert client.endpoint == endpoint
            assert client._client.timeout == httpx.Timeout(60.0)
    else:
        with instructor.from_provider(
            f"{provider}/{model}", mode=instructor.Mode.DECISIONS
        ) as client:
            assert isinstance(client, DecisionsClient)
            assert client.model == model
            assert client.endpoint == endpoint
            assert client._client.timeout == httpx.Timeout(60.0)


@pytest.mark.parametrize(
    "provider,key_name,other_key",
    [
        ("typesafe", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY"),
        ("openai", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    ],
)
@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
def test_missing_provider_key_does_not_fall_back_to_another_provider(
    monkeypatch, provider, key_name, other_key, async_client
):
    monkeypatch.delenv(key_name, raising=False)
    monkeypatch.setenv(other_key, "wrong-provider-key")
    if key_name != "OPENAI_API_KEY":
        monkeypatch.setenv("OPENAI_API_KEY", "wrong-sdk-key")
    with pytest.raises(ValueError, match=f"Set {key_name} or pass api_key"):
        instructor.from_provider(
            f"{provider}/test-model",
            mode=instructor.Mode.DECISIONS,
            async_client=async_client,
        )


@pytest.mark.parametrize("provider", ["typesafe", "openrouter", "openai"])
@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_wrong_http_client_type_is_rejected(provider, async_client):
    async with httpx.AsyncClient(trust_env=False) as async_http:
        with httpx.Client(trust_env=False) as sync_http:
            with pytest.raises(TypeError, match="require.*httpx"):
                instructor.from_provider(
                    f"{provider}/test-model",
                    mode=instructor.Mode.DECISIONS,
                    api_key="test-key",
                    async_client=async_client,
                    http_client=sync_http if async_client else async_http,
                )


@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("timeout", [1.75, None])
@pytest.mark.asyncio
async def test_owned_http_client_receives_timeout(async_client, timeout):
    if async_client:
        async with instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            async_client=True,
            api_key="test-key",
            timeout=timeout,
        ) as client:
            assert client._client.timeout == httpx.Timeout(timeout)
    else:
        with instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="test-key",
            timeout=timeout,
        ) as client:
            assert client._client.timeout == httpx.Timeout(timeout)


@pytest.mark.parametrize("provider", ["anthropic", "unknown"])
def test_unsupported_decisions_provider_is_rejected(provider):
    with pytest.raises(ValueError, match="Decisions mode does not support provider"):
        instructor.from_provider(
            f"{provider}/test-model", mode=instructor.Mode.DECISIONS, api_key="key"
        )


@pytest.mark.parametrize("option", ["stream", "max_retries", "hooks", "base_url"])
def test_unsupported_decisions_options_are_not_silently_ignored(option):
    options: dict[str, Any] = {option: True}
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{option}'"):
        instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            **options,
        )


def test_decisions_cache_is_explicitly_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument 'cache'"):
        instructor.from_provider(
            "typesafe/jev-latest",
            mode=instructor.Mode.DECISIONS,
            api_key="key",
            cache=AutoCache(),
        )


def test_public_factory_return_types_are_runtime_resolvable():
    hints = get_type_hints(instructor.from_provider)
    assert DecisionsClient in hints["return"].__args__
    assert AsyncDecisionsClient in hints["return"].__args__
