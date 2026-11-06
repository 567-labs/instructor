"""Security boundaries exercised with real models, sockets and local HTTP."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Union

import pytest
from requests.adapters import HTTPAdapter
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

import instructor
from instructor.cache import AutoCache, load_cached_response, make_request_cache_key
from instructor.v2.core import remote
from instructor.v2.core.json import extract_json_from_codeblock
from instructor.v2.core.multimodal import PDF
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.providers.anthropic.multimodal import pdf_to_anthropic
from instructor.v2.validation.async_validators import (
    async_field_validator,
    async_model_validator,
)


class Answer(BaseModel):
    value: int


class Marked(BaseModel):
    value: int

    @async_field_validator("value")
    async def check(cls, _value: int) -> int:
        raise ValueError("must never silently accept")


class Nested(BaseModel):
    answer: list[Marked]


class ModelMarked(BaseModel):
    value: int

    @async_model_validator()
    async def check(self):
        raise ValueError("must never silently accept")


@pytest.fixture
def local_provider():
    calls: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            body = json.dumps(
                {
                    "id": "local",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"value":1}'},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request_key(request, **overrides):
    params: dict[str, Any] = dict(
        request=request,
        args=(),
        response_model=Answer,
        provider="genai",
        mode="json",
        namespace="tenant",
        context=None,
        strict=True,
    )
    params.update(overrides)
    return make_request_cache_key(**params)


@pytest.mark.parametrize(
    "left,right",
    [
        ({"message": "alice"}, {"message": "bob"}),
        (
            {"config": {"system_instruction": "alice"}},
            {"config": {"system_instruction": "bob"}},
        ),
        ({"temperature": 0}, {"temperature": 1}),
        (
            {"extra_headers": {"X-Tenant": "alice"}},
            {"extra_headers": {"X-Tenant": "bob"}},
        ),
    ],
)
def test_full_request_cache_identity(left, right):
    assert request_key(left) != request_key(right)
    assert request_key(left) == request_key(left)


def test_genai_config_cache_identity():
    from google.genai.types import GenerateContentConfig

    assert request_key(
        {"config": GenerateContentConfig(system_instruction="alice")}
    ) != request_key({"config": GenerateContentConfig(system_instruction="bob")})


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "cohere"},
        {"namespace": "other-tenant"},
        {"strict": False},
        {"context": {"tenant": "other"}},
        {"args": ("other",)},
    ],
)
def test_cache_policy_identity(change):
    assert request_key({}) != request_key({}, **change)


def test_unknown_identity_disables_cache():
    assert request_key({"opaque": object()}) is None


def test_cached_response_uses_context_and_strict():
    class Policy(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def allowed(cls, value: int, info: ValidationInfo):
            if info.context and value > info.context["limit"]:
                raise ValueError("current policy rejects")
            return value

    cache = AutoCache()
    cache.set("numeric", '{"value":1}')
    assert (
        load_cached_response(cache, "numeric", Policy, context={"limit": 2}).value == 1
    )
    with pytest.raises(ValidationError, match="current policy rejects"):
        load_cached_response(cache, "numeric", Policy, context={"limit": 0})
    cache.set("coercible", '{"value":"1"}')
    with pytest.raises(ValidationError):
        load_cached_response(cache, "coercible", Policy, strict=True)
    assert load_cached_response(cache, "coercible", Policy, strict=False).value == 1


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.asyncio
async def test_real_client_cache_isolation(local_provider, async_client):
    url, calls = local_provider
    cache = AutoCache()
    clients = [
        instructor.from_provider(
            "openai/local",
            base_url=url,
            api_key="local",
            mode=instructor.Mode.JSON,
            async_client=async_client,
        )
        for _ in range(2)
    ]

    async def ask(client, **kwargs):
        response = client.create(
            response_model=Answer,
            cache=cache,
            messages=[{"role": "user", "content": "answer"}],
            **kwargs,
        )
        return await response if async_client else response

    assert (await ask(clients[0])).value == 1
    assert (await ask(clients[0])).value == 1
    assert len(calls) == 1
    await ask(clients[1])
    assert len(calls) == 2
    await ask(clients[0], context={"tenant": "bob"})
    await ask(clients[0], strict=False)
    await ask(clients[0], temperature=0.7)
    assert len(calls) == 5
    await ask(clients[0], cache_namespace="intentional-shared-scope")
    await ask(clients[1], cache_namespace="intentional-shared-scope")
    assert len(calls) == 6
    assert "cache_namespace" not in calls[-1]


@pytest.mark.parametrize(
    "model",
    [Marked, Nested, ModelMarked, list[Marked], Iterable[Union[Marked, Answer]]],
)
def test_markers_fail_closed(model):
    with pytest.raises(ValueError, match="async validators are not supported"):
        prepare_response_model(model)


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.asyncio
async def test_markers_rejected_before_provider(local_provider, async_client):
    url, calls = local_provider
    client = instructor.from_provider(
        "openai/local",
        base_url=url,
        api_key="local",
        mode=instructor.Mode.JSON,
        async_client=async_client,
    )
    with pytest.raises(ValueError, match="async validators are not supported"):
        result = client.create(
            response_model=Nested,
            stream=True,
            messages=[{"role": "user", "content": "answer"}],
        )
        if async_client:
            await result
    assert calls == []


def test_cache_rejects_marked_model():
    cache = AutoCache()
    cache.set("key", '{"value":1}')
    with pytest.raises(ValueError, match="async validators are not supported"):
        load_cached_response(cache, "key", Marked)


@pytest.mark.parametrize(
    "connection", [remote._PublicHTTPConnection, remote._PublicHTTPSConnection]
)
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_transport_rejects_private_before_any_http(connection, host):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(0.01)
        conn = connection(host, port=listener.getsockname()[1], timeout=1)
        with pytest.raises(remote.RemoteFetchError, match="non-public"):
            conn.request("GET", "/sensitive")
        with pytest.raises(socket.timeout):
            listener.accept()
        conn.close()


def test_session_mounts_public_transport():
    with remote._new_session() as session:
        for scheme, cls in (
            ("http", remote._PublicHTTPConnection),
            ("https", remote._PublicHTTPSConnection),
        ):
            adapter = session.get_adapter(f"{scheme}://example.com")
            assert isinstance(adapter, HTTPAdapter)
            pool = adapter.poolmanager.connection_from_url(f"{scheme}://example.com")
            assert pool.ConnectionCls is cls
        assert not session.trust_env


@pytest.mark.parametrize(
    "source", ["HTTP://127.0.0.1/private.pdf", "HTTP://[::1]/private.pdf"]
)
def test_anthropic_pdf_fallback_rejects_private(source):
    with pytest.raises(remote.RemoteFetchError, match="non-public"):
        pdf_to_anthropic(PDF.from_url(source))


def test_anthropic_pdf_url_control():
    source = "https://example.com/document.pdf"
    result = pdf_to_anthropic(PDF.from_url(source))
    assert result["source"] == {"type": "url", "url": source}


def test_json_extraction_bounds_and_recovery():
    at_limit = "[" * 128 + "]" * 128
    assert extract_json_from_codeblock(at_limit) == at_limit
    with pytest.raises(ValueError, match="nesting"):
        extract_json_from_codeblock("[" * 129 + "]" * 129)
    # Decoder recursion limits vary by Python version; malformed input must
    # still be rejected by either the nesting or bounded-work guard.
    with pytest.raises(ValueError, match="nesting|malformed-input work limit"):
        extract_json_from_codeblock("[" * 2000)
    with pytest.raises(ValueError, match="character limit"):
        extract_json_from_codeblock("x" * (1024 * 1024 + 1))
    assert extract_json_from_codeblock('prose [broken {"value":1}') == '{"value":1}'
    assert (
        extract_json_from_codeblock('{"nested":{"value":1}}')
        == '{"nested":{"value":1}}'
    )
    assert extract_json_from_codeblock('{"value":0} then {"value":1}') == '{"value":1}'


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.asyncio
async def test_cache_tracks_mutable_sdk_settings(local_provider, async_client):
    url, calls = local_provider
    client = instructor.from_provider(
        "openai/local",
        base_url=url,
        api_key="tenant-a",
        mode=instructor.Mode.JSON,
        async_client=async_client,
    )
    cache = AutoCache()

    async def ask():
        response = client.create(
            response_model=Answer,
            cache=cache,
            messages=[{"role": "user", "content": "answer"}],
        )
        return await response if async_client else response

    await ask()
    await ask()
    assert len(calls) == 1
    client.client.api_key = "tenant-b"
    await ask()
    assert len(calls) == 2
    client.client.base_url = url + "/other-endpoint"
    await ask()
    assert len(calls) == 3


def test_named_alias_markers_rejected():
    from typing_extensions import TypeAliasType
    from pydantic import create_model

    Alias = TypeAliasType("Alias", Marked)
    outer = create_model("AliasedOuter", answer=(Alias, ...))
    with pytest.raises(ValueError, match="async validators are not supported"):
        prepare_response_model(outer)


@pytest.mark.parametrize("async_parser", [False, True])
@pytest.mark.asyncio
async def test_direct_response_parser_rejects_markers(async_parser):
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from instructor.v2.core.response import process_response, process_response_async

    completion = ChatCompletion(
        id="local",
        model="local",
        created=0,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content='{"value":1}'),
            )
        ],
    )
    with pytest.raises(ValueError, match="async validators are not supported"):
        if async_parser:
            await process_response_async(
                completion,
                response_model=Marked,
                stream=False,
                mode=instructor.Mode.JSON,
            )
        else:
            process_response(
                completion,
                response_model=Marked,
                stream=False,
                mode=instructor.Mode.JSON,
            )


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.asyncio
async def test_xai_json_schema_rejects_markers(async_client):
    pytest.importorskip("xai_sdk")
    client = instructor.from_provider(
        "xai/grok-3",
        api_key="local",
        async_client=async_client,
        mode=instructor.Mode.JSON_SCHEMA,
    )
    with pytest.raises(ValueError, match="async validators are not supported"):
        response = client.create(
            response_model=Marked, messages=[{"role": "user", "content": "answer"}]
        )
        if async_client:
            await response


def test_json_recovers_final_object_after_unterminated_prose_string():
    text = '{"value":0} then [ "unclosed prose {"value":1}'
    assert extract_json_from_codeblock(text) == '{"value":1}'


def test_json_failed_decode_work_is_bounded():
    with pytest.raises(ValueError, match="work limit"):
        extract_json_from_codeblock("[broken " * 2000)
    # Multiple complete answers remain supported; only failed suffix scans
    # consume the defensive work allowance.
    assert extract_json_from_codeblock('{"value":1}' * 2000) == '{"value":1}'


def test_generic_named_alias_markers_rejected():
    from typing_extensions import TypeAliasType, TypeVar
    from pydantic import create_model

    T = TypeVar("T")
    GenericAlias = TypeAliasType("GenericAlias", tuple[Marked, T], type_params=(T,))
    outer = create_model("GenericAliasedOuter", answer=(GenericAlias[int], ...))
    with pytest.raises(ValueError, match="async validators are not supported"):
        prepare_response_model(outer)


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.asyncio
async def test_cache_does_not_reuse_response_for_stringified_context_keys(
    local_provider, async_client
):
    url, calls = local_provider
    client = instructor.from_provider(
        "openai/local",
        base_url=url,
        api_key="local",
        mode=instructor.Mode.JSON,
        async_client=async_client,
    )
    cache = AutoCache()

    async def ask(policy):
        response = client.create(
            response_model=Answer,
            cache=cache,
            context={"policy": policy},
            messages=[{"role": "user", "content": "answer"}],
        )
        return await response if async_client else response

    try:
        assert (await ask({1: "allowed"})).value == 1
        assert (await ask({"1": "allowed"})).value == 1
        assert len(calls) == 2
        assert (await ask({"1": "allowed"})).value == 1
        assert len(calls) == 2
        assert "context" not in calls[-1]
        assert "cache" not in calls[-1]
    finally:
        if async_client:
            await client.client.close()
        else:
            client.client.close()
