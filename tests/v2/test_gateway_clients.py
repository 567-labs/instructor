from __future__ import annotations

from typing import Any

import openai
import pytest
from pydantic import BaseModel

import instructor
from instructor import Mode, Provider
from instructor.v2.core.errors import ConfigurationError, ModeError
from instructor.v2.core.providers import get_provider
from instructor.v2.core.registry import mode_registry
from tests._http import JSONHandler, serve


CASES = [
    ("zhipuai", "open.bigmodel.cn", "ZHIPUAI_API_KEY", Mode.TOOLS),
    ("zai", "api.z.ai", "ZAI_API_KEY", Mode.TOOLS),
    ("hubris", "api.hubris.pw", "HUBRIS_API_KEY", Mode.TOOLS),
    ("atlascloud", "api.atlascloud.ai", "ATLASCLOUD_API_KEY", Mode.TOOLS),
    ("requesty", "router.requesty.ai", "REQUESTY_API_KEY", Mode.TOOLS),
    ("sarvam", "api.sarvam.ai", "SARVAM_API_KEY", Mode.MD_JSON),
    ("orcarouter", "api.orcarouter.ai", "ORCAROUTER_API_KEY", Mode.TOOLS),
    ("minimax", "api.minimax.io", "MINIMAX_API_KEY", Mode.TOOLS),
    ("manifest", "app.manifest.build", "MANIFEST_API_KEY", Mode.TOOLS),
]


class Contact(BaseModel):
    name: str


@pytest.fixture(scope="module")
def wire_server():
    received: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    class Handler(JSONHandler):
        def do_POST(self) -> None:
            payload = self.read_json()
            received.append(
                (self.path, {k.lower(): v for k, v in self.headers.items()}, payload)
            )
            arguments = '{"name":"Ada"}'
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "```json\n" + arguments + "\n```",
            }
            if "tools" in payload:
                message["tool_calls"] = [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "Contact", "arguments": arguments},
                    }
                ]
            self.send_json(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": payload["model"],
                    "created": 1,
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": "stop"}
                    ],
                }
            )

    with serve(Handler) as url:
        yield f"{url}/v1", received


@pytest.mark.parametrize(("name", "host", "key", "default"), CASES)
def test_provider_default_and_exact_host(
    monkeypatch: pytest.MonkeyPatch, name: str, host: str, key: str, default: Mode
) -> None:
    monkeypatch.setenv(key, "gateway-test")
    client = instructor.from_provider(name + "/org/model")
    assert isinstance(client.client, openai.OpenAI)
    try:
        assert client.provider == Provider(name)
        assert client.mode == default
        assert client.client.base_url.host == host
        assert get_provider(str(client.client.base_url)) == Provider(name)
        assert (
            get_provider("https://" + host + ".example.invalid/v1") == Provider.UNKNOWN
        )
        assert (
            get_provider("https://example.invalid/v1?endpoint=" + host)
            == Provider.UNKNOWN
        )
    finally:
        client.client.close()


@pytest.mark.parametrize(("name", "_host", "key", "_default"), CASES)
def test_unrelated_key_cannot_authenticate(
    monkeypatch: pytest.MonkeyPatch, name: str, _host: str, key: str, _default: Mode
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "other-service")
    monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    if name == "zhipuai":
        monkeypatch.setenv("ZAI_API_KEY", "different-region")
    if name == "zai":
        monkeypatch.setenv("ZHIPUAI_API_KEY", "different-region")
    with pytest.raises(ConfigurationError, match=key):
        instructor.from_provider(name + "/model")


@pytest.mark.parametrize(("name", "_host", "key", "default"), CASES)
@pytest.mark.asyncio
async def test_gateway_sync_and_async_wire(
    wire_server,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    _host: str,
    key: str,
    default: Mode,
) -> None:
    base_url, received = wire_server
    monkeypatch.setenv(key, "key-for-" + name)
    factory = getattr(instructor, "from_" + name)
    request = {"messages": [{"role": "user", "content": "Ada"}]}
    with openai.OpenAI(
        base_url=base_url,
        api_key="explicit-key",
        http_client=openai.DefaultHttpxClient(trust_env=False),
    ) as sdk:
        sync = factory(sdk, model="org/model")
        assert sync.provider == Provider(name)
        assert sync.create(response_model=Contact, **request).name == "Ada"
        path, headers, data = received[-1]
        assert (
            path == "/v1/chat/completions"
            and headers["authorization"] == "Bearer explicit-key"
        )
        assert data["model"] == "org/model"
        assert ("tools" in data) == (default == Mode.TOOLS)
    async_client = factory(
        "org/model",
        async_client=True,
        base_url=base_url,
        http_client=openai.DefaultAsyncHttpxClient(trust_env=False),
    )
    try:
        result = await async_client.create(response_model=Contact, **request)
        assert result.name == "Ada"
        assert received[-1][1]["authorization"] == "Bearer key-for-" + name
        assert received[-1][2]["model"] == "org/model"
    finally:
        await async_client.client.close()


def test_regional_key_alias_and_supported_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.setenv("BIGMODEL_API_KEY", "regional-test")
    client = instructor.from_zhipuai("model")
    assert isinstance(client.client, openai.OpenAI)
    try:
        assert client.client.api_key == "regional-test"
    finally:
        client.client.close()
    assert mode_registry.is_registered(Provider.HUBRIS, Mode.PARALLEL_TOOLS)
    assert mode_registry.is_registered(Provider.REQUESTY, Mode.JSON_SCHEMA)
    assert not mode_registry.is_registered(Provider.SARVAM, Mode.PARALLEL_TOOLS)
    assert not mode_registry.is_registered(Provider.MINIMAX, Mode.JSON_SCHEMA)
    with pytest.raises(ModeError):
        instructor.from_minimax("model", api_key="test", mode=Mode.RESPONSES_TOOLS)


def test_legacy_provider_modes_and_invalid_url() -> None:
    with openai.OpenAI(api_key="test") as sdk:
        assert instructor.from_sarvam(sdk, mode=Mode.SARVAM_JSON).mode is Mode.MD_JSON
        assert instructor.from_minimax(sdk, mode=Mode.MINIMAX_TOOLS).mode is Mode.TOOLS
    assert get_provider("https://[broken") == Provider.UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_gateway_model_factories_keep_project_and_organization_on_the_sdk(
    wire_server, is_async: bool
):
    base_url, received = wire_server
    client: Any = instructor.from_zai(
        "model",
        async_client=is_async,
        base_url=base_url,
        api_key="fixture",
        organization="org-fixture",
        project="proj-fixture",
    )
    sdk_type = openai.AsyncOpenAI if is_async else openai.OpenAI
    assert isinstance(client.client, sdk_type)
    request = dict(
        response_model=Contact, messages=[{"role": "user", "content": "Ada"}]
    )
    try:
        result = client.create(**request)
        assert (await result if is_async else result).name == "Ada"
    finally:
        await client.client.close() if is_async else client.client.close()
    _path, headers, body = received[-1]
    assert headers["openai-organization"] == "org-fixture"
    assert headers["openai-project"] == "proj-fixture"
    assert "organization" not in body and "project" not in body
