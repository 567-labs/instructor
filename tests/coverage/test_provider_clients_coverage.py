from __future__ import annotations

import builtins
import importlib
import json
import runpy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

# These tests pass an `httpx.Client`/transport into provider SDKs, which
# perform an `isinstance(http_client, httpx.Client)` guard. We pin the real
# httpx here so the guards pass; the migration to httpx2 is handled in the
# production code path (`instructor.v2.auto_client`).
import httpx
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import (
    ClientError,
    IncompleteOutputException,
    ModeError,
    ResponseParsingError,
)
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.providers.writer.handlers import (
    WriterJSONSchemaHandler,
    WriterMDJSONHandler,
    WriterToolsHandler,
    handle_writer_tools,
    reask_writer_json,
    reask_writer_tools,
)
from tests.coverage._openai import chat_completion, tool_call
from tests.coverage.client_cleanup import (
    clear_proxy_environment,
    close_async_provider_client,
    close_provider_client,
    ignore_fireworks_pydantic_warning,
)


class User(BaseModel):
    name: str
    age: int


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MESSAGES = [{"role": "user", "content": "Return Ada, age 37"}]
PROVIDERS = [
    pytest.param("cerebras", Provider.CEREBRAS, id="cerebras"),
    pytest.param("fireworks", Provider.FIREWORKS, id="fireworks"),
    pytest.param("groq", Provider.GROQ, id="groq"),
    pytest.param("writer", Provider.WRITER, id="writer"),
]


def tool_response() -> ChatCompletion:
    return chat_completion(
        tool_calls=[tool_call("User", {"name": "Ada", "age": 37}, "call-user")],
        finish_reason="tool_calls",
        usage=True,
    )


def text_response(content: str) -> ChatCompletion:
    return chat_completion(content=content, usage=True)


def run_with_blocked_import(
    monkeypatch: pytest.MonkeyPatch, path: Path, blocked_module: str
) -> dict[str, Any]:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked_module:
            raise ImportError(
                f"No module named '{blocked_module}'", name=blocked_module
            )
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", guarded_import)
        return runpy.run_path(str(path))


@pytest.mark.parametrize(
    ("provider", "sdk_module", "sync_name", "async_name", "install_hint"),
    [
        pytest.param(
            "cerebras",
            "cerebras.cloud.sdk",
            "Cerebras",
            "AsyncCerebras",
            "cerebras-cloud-sdk",
            id="cerebras",
        ),
        pytest.param(
            "fireworks",
            "fireworks.client",
            "Fireworks",
            "AsyncFireworks",
            "fireworks-ai",
            id="fireworks",
        ),
        pytest.param("groq", "groq", "groq", "groq", "groq", id="groq"),
        pytest.param(
            "writer", "writerai", "Writer", "AsyncWriter", "writer-sdk", id="writer"
        ),
    ],
)
def test_client_factories_give_clear_error_when_optional_sdk_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    sdk_module: str,
    sync_name: str,
    async_name: str,
    install_hint: str,
) -> None:
    namespace = run_with_blocked_import(
        monkeypatch,
        PROJECT_ROOT / f"instructor/v2/providers/{provider}/client.py",
        sdk_module,
    )

    assert namespace[sync_name] is None
    assert namespace[async_name] is None
    with pytest.raises(ClientError, match=f"not installed.*{install_hint}"):
        namespace[f"from_{provider}"](object())


@pytest.mark.parametrize(("provider", "_provider_value"), PROVIDERS)
def test_provider_package_stays_importable_when_its_client_import_fails(
    monkeypatch: pytest.MonkeyPatch, provider: str, _provider_value: Provider
) -> None:
    namespace = run_with_blocked_import(
        monkeypatch,
        PROJECT_ROOT / f"instructor/v2/providers/{provider}/__init__.py",
        f"instructor.v2.providers.{provider}.client",
    )

    assert namespace[f"from_{provider}"] is None
    assert namespace["__all__"] == [f"from_{provider}"]


def load_provider_module(provider: str) -> ModuleType:
    with ignore_fireworks_pydantic_warning():
        module = importlib.import_module(f"instructor.v2.providers.{provider}.client")
    return module


def response_payload(provider: str, model: str) -> dict[str, Any]:
    payload = tool_response().model_dump(mode="json", exclude_none=True)
    payload.update(
        model=model,
        system_fingerprint="coverage-fingerprint",
        time_info={},
    )
    if provider == "writer":
        payload["choices"][0]["message"]["content"] = ""
    return payload


def sync_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    http_client: httpx.Client,
) -> tuple[ModuleType, Any]:
    clear_proxy_environment(monkeypatch)
    module = load_provider_module(provider)
    kwargs = {
        "api_key": "test-key",
        "base_url": f"https://{provider}.invalid"
        + ("/v1" if provider == "fireworks" else ""),
    }

    if provider == "cerebras":
        return module, module.Cerebras(
            **kwargs,
            http_client=http_client,
            warm_tcp_connection=False,
            _strict_response_validation=True,
        )
    if provider == "groq":
        return module, module.groq.Groq(
            **kwargs, http_client=http_client, _strict_response_validation=True
        )
    if provider == "writer":
        return module, module.Writer(
            **kwargs, http_client=http_client, _strict_response_validation=True
        )

    client = module.Fireworks(**kwargs)
    client._client_v1._client.close()
    client._client_v1._client = http_client
    return module, client


async def async_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    http_client: httpx.AsyncClient,
) -> tuple[ModuleType, Any]:
    clear_proxy_environment(monkeypatch)
    module = load_provider_module(provider)
    kwargs = {
        "api_key": "test-key",
        "base_url": f"https://{provider}.invalid"
        + ("/v1" if provider == "fireworks" else ""),
    }

    if provider == "cerebras":
        return module, module.AsyncCerebras(
            **kwargs,
            http_client=http_client,
            warm_tcp_connection=False,
            _strict_response_validation=True,
        )
    if provider == "groq":
        return module, module.groq.AsyncGroq(
            **kwargs, http_client=http_client, _strict_response_validation=True
        )
    if provider == "writer":
        return module, module.AsyncWriter(
            **kwargs, http_client=http_client, _strict_response_validation=True
        )

    client = module.AsyncFireworks(**kwargs)
    await client._client_v1._async_client.aclose()
    client._client_v1._async_client = http_client
    return module, client


def assert_tools_request(call: dict[str, Any], provider: Provider, model: str) -> None:
    assert call["messages"] == MESSAGES
    assert call["model"] == model
    assert call["tools"][0]["type"] == "function"
    assert call["tools"][0]["function"]["name"] == "User"
    if provider is Provider.WRITER:
        assert call["tool_choice"] == "auto"
    else:
        assert call["tool_choice"] == {
            "type": "function",
            "function": {"name": "User"},
        }


@pytest.mark.parametrize(("provider", "provider_value"), PROVIDERS)
def test_sync_provider_factory_patches_realistic_chat_completion(
    monkeypatch: pytest.MonkeyPatch, provider: str, provider_value: Provider
) -> None:
    seen: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.host == f"{provider}.invalid"
        assert (
            request.url.path
            == {
                "cerebras": "/v1/chat/completions",
                "fireworks": "/v1/chat/completions",
                "groq": "/openai/v1/chat/completions",
                "writer": "/v1/chat",
            }[provider]
        )
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=response_payload(provider, body["model"]))

    http_client = httpx.Client(transport=httpx.MockTransport(handle), trust_env=False)
    module, sdk_client = sync_sdk_client(monkeypatch, provider, http_client)

    try:
        client = getattr(module, f"from_{provider}")(
            sdk_client, mode=Mode.TOOLS, model="sync-model"
        )
        parsed = client.create(User, MESSAGES, max_retries=0)

        assert isinstance(client, Instructor)
        assert client.client is sdk_client
        assert client.provider is provider_value
        assert client.mode is Mode.TOOLS
        assert isinstance(parsed, User)
        assert parsed.model_dump() == {"name": "Ada", "age": 37}
        assert len(seen) == 1
        assert_tools_request(seen[0], provider_value, "sync-model")
    finally:
        close_provider_client(sdk_client)


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "provider_value"), PROVIDERS)
async def test_async_provider_factory_patches_realistic_chat_completion(
    monkeypatch: pytest.MonkeyPatch, provider: str, provider_value: Provider
) -> None:
    seen: list[dict[str, Any]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.host == f"{provider}.invalid"
        assert (
            request.url.path
            == {
                "cerebras": "/v1/chat/completions",
                "fireworks": "/v1/chat/completions",
                "groq": "/openai/v1/chat/completions",
                "writer": "/v1/chat",
            }[provider]
        )
        body = json.loads(request.content)
        seen.append(body)
        if body.get("stream"):
            chunk = {
                "id": "chatcmpl-stream-coverage",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "delta": {"role": "assistant", "content": "streamed"},
                    }
                ],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(200, json=response_payload(provider, body["model"]))

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), trust_env=False
    )
    module, sdk_client = await async_sdk_client(monkeypatch, provider, http_client)

    try:
        client = getattr(module, f"from_{provider}")(
            sdk_client, mode=Mode.TOOLS, model="async-model"
        )
        parsed = await client.create(User, MESSAGES, max_retries=0)

        assert isinstance(client, AsyncInstructor)
        assert client.client is sdk_client
        assert client.provider is provider_value
        assert client.mode is Mode.TOOLS
        assert isinstance(parsed, User)
        assert parsed.model_dump() == {"name": "Ada", "age": 37}
        assert_tools_request(seen[0], provider_value, "async-model")

        if provider == "fireworks":
            stream = await client.create(None, MESSAGES, max_retries=0, stream=True)
            chunks = [chunk async for chunk in stream]
            assert len(chunks) == 1
            assert chunks[0].choices[0].delta.content == "streamed"
            assert seen[1]["stream"] is True
            assert seen[1]["model"] == "async-model"
        else:
            assert len(seen) == 1
    finally:
        await close_async_provider_client(sdk_client)


@pytest.mark.parametrize(("provider", "provider_value"), PROVIDERS)
def test_provider_factories_reject_wrong_client_and_unsupported_mode(
    monkeypatch: pytest.MonkeyPatch, provider: str, provider_value: Provider
) -> None:
    module, sdk_client = sync_sdk_client(
        monkeypatch, provider, httpx.Client(trust_env=False)
    )
    factory = getattr(module, f"from_{provider}")

    try:
        with pytest.raises(
            ClientError, match="Client must be an instance.*Got: object"
        ):
            factory(object(), mode=Mode.TOOLS)
        with pytest.raises(ModeError) as exc:
            factory(sdk_client, mode=Mode.RESPONSES_TOOLS)

        assert exc.value.mode == Mode.RESPONSES_TOOLS.value
        assert exc.value.provider == provider_value.value
        assert Mode.TOOLS.value in exc.value.valid_modes
    finally:
        close_provider_client(sdk_client)


def test_writer_reask_recovers_message_shapes_without_openai_model_dump() -> None:
    choice_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant", content='{"age":"old"}', tool_calls=[]
                )
            )
        ]
    )
    tools_kwargs = reask_writer_tools(
        {"messages": list(MESSAGES)}, choice_response, ValueError("age must be an int")
    )
    text_kwargs = reask_writer_json(
        {"messages": list(MESSAGES)},
        SimpleNamespace(text='{"name":"Ada"}'),
        ValueError("age is required"),
    )
    content_kwargs = reask_writer_json(
        {"messages": list(MESSAGES)},
        SimpleNamespace(content='{"age":37}'),
        ValueError("name is required"),
    )

    assert tools_kwargs["messages"][1] == {
        "role": "assistant",
        "content": '{"age":"old"}',
    }
    assert "age must be an int" in tools_kwargs["messages"][2]["content"]
    assert (
        "fill tool call arguments/name correctly"
        in tools_kwargs["messages"][2]["content"]
    )
    assert text_kwargs["messages"][1]["content"] == '{"name":"Ada"}'
    assert "age is required" in text_kwargs["messages"][2]["content"]
    assert content_kwargs["messages"][1]["content"] == '{"age":37}'
    assert "name is required" in content_kwargs["messages"][2]["content"]


def test_writer_tools_helper_and_length_error_keep_writer_contract() -> None:
    prepared_model, kwargs = handle_writer_tools(User, {"messages": list(MESSAGES)})
    incomplete = tool_response()
    incomplete.choices[0].finish_reason = "length"

    assert prepared_model is User
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"][0]["function"]["name"] == "User"
    assert set(kwargs["tools"][0]["function"]["parameters"]["required"]) == {
        "name",
        "age",
    }
    with pytest.raises(IncompleteOutputException) as exc:
        WriterToolsHandler().parse_response(incomplete, User)
    assert exc.value.last_completion is incomplete


@pytest.mark.parametrize(
    ("response", "message"),
    [
        pytest.param(object(), "No choices in Writer response", id="missing-choices"),
        pytest.param(
            SimpleNamespace(choices=[]),
            "No choices in Writer response",
            id="empty-choices",
        ),
        pytest.param(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop", message=SimpleNamespace(tool_calls=[])
                    )
                ]
            ),
            "No tool calls in Writer response",
            id="empty-tool-calls",
        ),
    ],
)
def test_writer_tools_rejects_malformed_responses(response: Any, message: str) -> None:
    with pytest.raises(ResponseParsingError, match=message) as error:
        WriterToolsHandler().parse_response(response, User)

    assert error.value.mode == Mode.TOOLS.value
    assert error.value.raw_response is response


def test_writer_md_json_handles_multimodal_system_prompt_and_empty_messages() -> None:
    handler = WriterMDJSONHandler()
    multimodal = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Be concise"}]},
            {"role": "user", "content": "Return Ada"},
        ]
    }

    model, multimodal_kwargs = handler.prepare_request(User, multimodal)
    _, empty_kwargs = handler.prepare_request(User, {"messages": []})

    assert model is not None
    assert issubclass(model, User)
    assert multimodal_kwargs["messages"][0]["role"] == "system"
    assert multimodal_kwargs["messages"][0]["content"][0]["text"].startswith(
        "Be concise\n\n"
    )
    assert '"age"' in multimodal_kwargs["messages"][0]["content"][0]["text"]
    assert multimodal_kwargs["messages"][-1]["role"] == "user"
    assert any(
        "```json codeblock" in part["text"]
        for part in multimodal_kwargs["messages"][-1]["content"]
    )
    assert empty_kwargs["messages"][0]["role"] == "system"
    assert '"name"' in empty_kwargs["messages"][0]["content"]
    assert "```json codeblock" in empty_kwargs["messages"][1]["content"]


@pytest.mark.parametrize(
    ("handler_type", "response"),
    [
        pytest.param(WriterToolsHandler, tool_response(), id="tools"),
        pytest.param(
            WriterMDJSONHandler,
            text_response('```json\n{"name":"Ada","age":37}\n```'),
            id="md-json",
        ),
        pytest.param(
            WriterJSONSchemaHandler,
            text_response('{"name":"Ada","age":37}'),
            id="json-schema",
        ),
    ],
)
def test_writer_handler_modes_prepare_parse_and_reask(
    handler_type: type[
        WriterToolsHandler | WriterMDJSONHandler | WriterJSONSchemaHandler
    ],
    response: ChatCompletion,
) -> None:
    handler = handler_type()
    empty_request = {"messages": list(MESSAGES)}

    assert handler.prepare_request(None, empty_request) == (None, empty_request)
    model, kwargs = handler.prepare_request(
        User,
        {
            "messages": [
                {"role": "system", "content": "Keep the reply short"},
                *MESSAGES,
            ]
        },
    )
    assert model is not None
    parsed = handler.parse_response(response, model, strict=True)
    reasked = handler.handle_reask(
        {"messages": list(MESSAGES)}, response, ValueError("age must be present")
    )

    assert issubclass(model, User)
    assert isinstance(parsed, User)
    assert parsed.model_dump() == {"name": "Ada", "age": 37}
    assert reasked["messages"][1]["role"] == "assistant"
    assert "age must be present" in reasked["messages"][2]["content"]
    if isinstance(handler, WriterToolsHandler):
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["tools"][0]["function"]["name"] == "User"
    elif isinstance(handler, WriterMDJSONHandler):
        assert kwargs["messages"][0]["content"].startswith("Keep the reply short\n\n")
        assert '"age"' in kwargs["messages"][0]["content"]
    else:
        assert kwargs["response_format"]["type"] == "json_schema"
        assert set(kwargs["response_format"]["json_schema"]["schema"]["required"]) == {
            "name",
            "age",
        }


@pytest.mark.parametrize(
    ("handler", "delta"),
    [
        pytest.param(
            WriterToolsHandler(),
            {
                "tool_calls": [
                    SimpleNamespace(
                        function=SimpleNamespace(
                            arguments='{"tasks":[{"name":"Ada","age":37}]}'
                        )
                    )
                ]
            },
            id="tools",
        ),
        pytest.param(
            WriterMDJSONHandler(),
            {"content": '```json\n{"tasks":[{"name":"Ada","age":37}]}\n```'},
            id="md-json",
        ),
        pytest.param(
            WriterJSONSchemaHandler(),
            {"content": '{"tasks":[{"name":"Ada","age":37}]}'},
            id="json-schema",
        ),
    ],
)
def test_writer_handlers_parse_streaming_structured_output(
    handler: WriterToolsHandler | WriterMDJSONHandler | WriterJSONSchemaHandler,
    delta: dict[str, Any],
) -> None:
    iterable_user = IterableModel(User)
    prepared_model, kwargs = handler.prepare_request(
        iterable_user, {"messages": list(MESSAGES), "stream": True}
    )
    assert prepared_model is not None
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(**delta))])

    parsed = handler.parse_response([chunk], prepared_model)

    assert kwargs["stream"] is True
    assert list(parsed) == [User(name="Ada", age=37)]
    assert handler._consume_streaming_flag(prepared_model) is False
