"""Offline xAI client factory tests, each run against the sync and async SDK."""

from __future__ import annotations

import builtins
import inspect
import runpy
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union, cast

import pytest

pytest.importorskip("xai_sdk")

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from xai_sdk import chat as xchat
from xai_sdk.aio.client import Client as XAIAsyncClient
from xai_sdk.sync.client import Client as XAISyncClient

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import ClientError, ModeError
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.providers.xai import client as xai_client


class Answer(BaseModel):
    """An answer returned by Grok."""

    answer: int


class Reason(BaseModel):
    """A reason returned by Grok."""

    reason: str


MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "Be precise"},
    {"role": "user", "content": "Return an answer"},
]


def tool_call(name: str, arguments: str, tool_id: str = "call-1") -> Any:
    return xchat.chat_pb2.ToolCall(
        id=tool_id,
        function={"name": name, "arguments": arguments},
    )


def tool_chunk(name: str, arguments: str) -> tuple[Any, None]:
    return SimpleNamespace(tool_calls=[tool_call(name, arguments)]), None


def content_chunk(content: str) -> tuple[None, Any]:
    return None, SimpleNamespace(content=content)


class SyncChat:
    def __init__(
        self,
        *,
        sampled: Any = None,
        parsed: tuple[Any, BaseModel] | None = None,
        stream: list[tuple[Any, Any]] | None = None,
    ) -> None:
        self.sampled = sampled
        self.parsed = parsed
        self.stream_items = stream or []
        self.parse_shapes: list[type[BaseModel]] = []
        self.proto = xchat.chat_pb2.GetCompletionsRequest()

    def sample(self) -> Any:
        return self.sampled

    def parse(self, shape: type[BaseModel]) -> Any:
        self.parse_shapes.append(shape)
        assert self.parsed is not None
        return self.parsed

    def stream(self) -> Any:
        return iter(self.stream_items)


class AsyncChat(SyncChat):
    async def sample(self) -> Any:
        return SyncChat.sample(self)

    async def parse(self, shape: type[BaseModel]) -> Any:
        return SyncChat.parse(self, shape)

    async def stream(self) -> Any:
        for item in self.stream_items:
            yield item


class ChatFactory:
    def __init__(self, chat: SyncChat) -> None:
        self.chat = chat
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SyncChat:
        self.calls.append(kwargs)
        self.chat.proto = xchat.chat_pb2.GetCompletionsRequest(
            model=kwargs["model"],
            messages=kwargs["messages"],
            tools=kwargs.get("tools", []),
        )
        return self.chat


class OfflineSyncClient(XAISyncClient):
    def __init__(self, factory: ChatFactory) -> None:
        object.__setattr__(self, "chat", factory)


class OfflineAsyncClient(XAIAsyncClient):
    def __init__(self, factory: ChatFactory) -> None:
        object.__setattr__(self, "chat", factory)


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def is_async(request: pytest.FixtureRequest) -> bool:
    return request.param


def make_client(
    is_async: bool, mode: Mode = Mode.TOOLS, **chat_options: Any
) -> tuple[Any, SyncChat, ChatFactory]:
    """Wrap an offline SDK client; ``chat_options`` configure the fake chat."""
    defaults = chat_options.pop("defaults", {})
    chat = (AsyncChat if is_async else SyncChat)(**chat_options)
    factory = ChatFactory(chat)
    native = (OfflineAsyncClient if is_async else OfflineSyncClient)(factory)
    wrapped = xai_client.from_xai(cast(Any, native), mode=mode, **defaults)
    assert isinstance(wrapped, AsyncInstructor) is is_async
    assert isinstance(wrapped, Instructor)
    return wrapped, chat, factory


async def resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def collect(stream: Any) -> list[Any]:
    stream = await resolve(stream)
    if hasattr(stream, "__aiter__"):
        return [item async for item in stream]
    return list(stream)


def test_optional_import_fallbacks_are_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def without_xai(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "xai_sdk.sync.client":
            raise ImportError("xai-sdk is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_xai)
    isolated = runpy.run_path(str(Path(xai_client.__file__)))

    assert isolated["SyncClient"] is None
    assert isolated["AsyncClient"] is None
    assert isolated["xchat"] is None


def test_package_import_falls_back_when_client_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def without_client(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "instructor.v2.providers.xai.client":
            raise ImportError("optional client dependency is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_client)
    isolated = runpy.run_path(str(Path(xai_client.__file__).with_name("__init__.py")))

    assert isolated["from_xai"] is None
    assert isolated["__all__"] == ["from_xai"]


def test_schema_name_and_markdown_helpers() -> None:
    class NoSchema:
        pass

    schema = xai_client._get_model_schema(Answer)
    assert schema["properties"]["answer"]["type"] == "integer"
    assert xai_client._get_model_schema(NoSchema) == {}
    assert xai_client._get_model_name(Answer) == "Answer"
    assert xai_client._get_model_name(NoSchema) == "NoSchema"
    assert xai_client._get_model_name(object()) == "Model"
    messages = [dict(message) for message in MESSAGES]
    assert xai_client._add_md_json_instructions(messages, object()) == messages

    user_only = [{"role": "user", "content": "hello"}]
    prepended = xai_client._add_md_json_instructions(user_only, Answer)
    assert prepended[0]["role"] == "system"
    assert "Schema:" in prepended[0]["content"]
    assert prepended[1:] == user_only
    extended = xai_client._add_md_json_instructions(messages, Answer)
    assert extended[0]["content"].startswith("Be precise\n\nReturn your answer as JSON")


def test_message_conversion_and_finalize_helpers() -> None:
    converted = xai_client._convert_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "rules"},
            {"role": "tool", "content": "7"},
        ]
    )
    assert converted == [
        xchat.user(xchat.text("hi")),
        xchat.assistant(xchat.text("hello")),
        xchat.system(xchat.text("rules")),
        xchat.tool_result("7"),
    ]
    with pytest.raises(ValueError, match="Unsupported role: developer"):
        xai_client._convert_messages([{"role": "developer", "content": "rules"}])

    raw = object()
    parsed = Answer(answer=42)
    assert xai_client._finalize_parsed_response(parsed, raw) is parsed
    assert cast(Any, parsed)._raw_response is raw

    iterable_model = cast(type[BaseModel], prepare_response_model(list[Answer]))
    iterable = iterable_model.model_validate({"tasks": [{"answer": 1}]})
    finalized = xai_client._finalize_parsed_response(iterable, raw)
    assert [item.model_dump() for item in finalized] == [{"answer": 1}]

    adapter_model = cast(type[BaseModel], prepare_response_model(int))
    adapter = adapter_model.model_validate({"content": 7})
    assert xai_client._finalize_parsed_response(adapter, raw) == 7
    assert xai_client._finalize_parsed_response({"answer": 8}, raw) == {"answer": 8}


@pytest.mark.asyncio
async def test_tool_argument_deltas_emit_only_new_suffixes(is_async: bool) -> None:
    def call(arguments: Any, tool_id: str | None = None) -> Any:
        function = SimpleNamespace(arguments=arguments)
        return SimpleNamespace(
            tool_calls=[SimpleNamespace(id=tool_id, function=function)]
        )

    chunks = [
        (call(None), None),
        (call('{"answer":', "first"), None),
        (call('{"answer":7}', "second"), None),
        (call('{"answer":7}', "second"), None),
        (call({"answer": 9}), None),
    ]
    if is_async:

        async def stream() -> Any:
            for chunk in chunks:
                yield chunk

        deltas = [d async for d in xai_client._aiter_tool_call_arg_deltas(stream())]
    else:
        deltas = list(xai_client._iter_tool_call_arg_deltas(iter(chunks)))

    assert deltas == ['{"answer":', "7}", '{"answer": 9}']


def test_factory_rejects_unsupported_modes_and_invalid_client_types() -> None:
    native = OfflineSyncClient(ChatFactory(SyncChat()))

    with pytest.raises(ModeError, match="xai") as unsupported:
        xai_client.from_xai(native, mode=Mode.ANTHROPIC_JSON)
    assert "tool_call" in str(unsupported.value)

    with pytest.raises(ClientError, match="Got: object"):
        xai_client.from_xai(cast(Any, object()), mode=Mode.TOOLS)


@pytest.mark.asyncio
async def test_unstructured_request_converts_messages_and_filters_instructor_args(
    is_async: bool,
) -> None:
    raw = SimpleNamespace(content="plain answer")
    wrapped, _, factory = make_client(
        is_async, sampled=raw, defaults={"temperature": 0.25}
    )

    result = await resolve(
        wrapped.create(
            response_model=None,
            messages=MESSAGES,
            model="grok-test",
            max_retries=2,
            context={"request": "coverage"},
            validation_context={"ignored": True},
        )
    )

    assert result is raw
    assert wrapped.mode is Mode.TOOLS
    assert wrapped.provider is Provider.XAI
    assert factory.calls == [
        {
            "model": "grok-test",
            "messages": [
                xchat.system(xchat.text("Be precise")),
                xchat.user(xchat.text("Return an answer")),
            ],
            "temperature": 0.25,
        }
    ]

    with pytest.raises(ValueError, match="not supported for xAI"):
        await resolve(
            wrapped.create(
                response_model=Answer,
                messages=MESSAGES,
                model="grok-test",
                token_budget=100,
            )
        )
    assert len(factory.calls) == 1


@pytest.mark.asyncio
async def test_json_schema_parse_attaches_raw_response(is_async: bool) -> None:
    raw = SimpleNamespace(id="raw")
    wrapped, chat, factory = make_client(
        is_async, Mode.JSON_SCHEMA, parsed=(raw, Answer(answer=7))
    )

    result = await resolve(
        wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")
    )

    assert result.model_dump() == {"answer": 7}
    assert result._raw_response is raw
    assert chat.parse_shapes == [Answer]
    assert factory.calls[0]["model"] == "grok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "iterable_stream", "partial_stream"),
    [
        (
            Mode.JSON_SCHEMA,
            [content_chunk('{"tasks":[{"answer":1},'), content_chunk('{"answer":2}]}')],
            [content_chunk('{"answer":'), content_chunk("7}")],
        ),
        (
            Mode.TOOLS,
            [
                tool_chunk("IterableAnswer", '{"tasks":[{"answer":1},'),
                tool_chunk("IterableAnswer", '{"tasks":[{"answer":1},{"answer":2}]}'),
            ],
            [
                tool_chunk("PartialAnswer", '{"answer":'),
                tool_chunk("PartialAnswer", '{"answer":7}'),
            ],
        ),
    ],
    ids=["json_schema", "tools"],
)
async def test_streams_iterable_and_partial_models_and_reject_plain_models(
    is_async: bool,
    mode: Mode,
    iterable_stream: list[tuple[Any, Any]],
    partial_stream: list[tuple[Any, Any]],
) -> None:
    wrapped, iterable_chat, _ = make_client(is_async, mode, stream=iterable_stream)
    answers = await collect(
        wrapped.create_iterable(response_model=Answer, messages=MESSAGES, model="grok")
    )

    partial_client, partial_chat, _ = make_client(is_async, mode, stream=partial_stream)
    partials = await collect(
        partial_client.create_partial(
            response_model=Answer, messages=MESSAGES, model="grok"
        )
    )

    assert [item.answer for item in answers] == [1, 2]
    assert partials[0].answer is None
    assert partials[-1].answer == 7
    if mode is Mode.JSON_SCHEMA:
        assert iterable_chat.proto.response_format.schema
        assert partial_chat.proto.response_format.schema

    plain_client, _, _ = make_client(is_async, mode)
    with pytest.raises(ValueError, match="Unsupported response model.*Answer"):
        await resolve(
            plain_client.create_fn(
                response_model=Answer, messages=MESSAGES, model="grok", stream=True
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_model", "arguments", "expected"),
    [
        (Answer, '{"answer":7}', {"answer": 7}),
        (list[Answer], '{"tasks":[{"answer":1},{"answer":2}]}', [1, 2]),
        (int, '{"content":7}', 7),
    ],
)
async def test_tools_validate_structured_iterable_and_simple_results(
    is_async: bool, response_model: Any, arguments: str, expected: Any
) -> None:
    response = SimpleNamespace(tool_calls=[tool_call("Answer", arguments)])
    wrapped, chat, _ = make_client(is_async, sampled=response)

    result = await resolve(
        wrapped.create(
            response_model=response_model,
            messages=MESSAGES,
            model="grok",
            strict=False,
        )
    )

    if isinstance(expected, dict):
        assert result.model_dump() == expected
        assert result._raw_response is response
    elif isinstance(expected, list):
        assert [item.answer for item in result] == expected
    else:
        assert result == expected
    assert len(chat.proto.tools) == 1
    assert chat.proto.tool_choice.function_name == chat.proto.tools[0].function.name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(tool_calls=[], text='```json\n{"answer":7}\n```'),
        SimpleNamespace(tool_calls=[], content='{"answer":7}'),
        SimpleNamespace(tool_calls=[], content=['{"answer":7}']),
    ],
)
async def test_tools_fall_back_to_text_or_content(
    is_async: bool, response: Any
) -> None:
    wrapped, _, _ = make_client(is_async, sampled=response)

    result = await resolve(
        wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")
    )

    assert result.model_dump() == {"answer": 7}
    assert result._raw_response is response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "response", "match"),
    [
        (
            Mode.TOOLS,
            SimpleNamespace(tool_calls=[], content=[]),
            "No tool calls returned from xAI",
        ),
        (
            Mode.TOOLS,
            SimpleNamespace(tool_calls=[], content={"answer": 7}),
            "No tool calls returned from xAI",
        ),
        (
            Mode.MD_JSON,
            SimpleNamespace(content=[]),
            "Could not extract JSON from xAI response",
        ),
        (
            Mode.MD_JSON,
            SimpleNamespace(content={"answer": 7}),
            "Could not extract JSON from xAI response",
        ),
    ],
    ids=["tools-empty", "tools-object", "md-empty", "md-object"],
)
async def test_responses_without_usable_content_raise(
    is_async: bool, mode: Mode, response: Any, match: str
) -> None:
    wrapped, _, _ = make_client(is_async, mode, sampled=response)

    with pytest.raises(ValueError, match=match):
        await resolve(
            wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")
        )


@pytest.mark.asyncio
async def test_parallel_tools_register_each_schema_and_parse_known_calls(
    is_async: bool,
) -> None:
    response = SimpleNamespace(
        tool_calls=[
            tool_call("Answer", '{"answer":7}'),
            tool_call("Reason", '{"reason":"checked"}', "call-3"),
        ]
    )
    wrapped, chat, _ = make_client(is_async, Mode.PARALLEL_TOOLS, sampled=response)

    results = await collect(
        wrapped.create_fn(
            response_model=Iterable[Union[Answer, Reason]],
            messages=MESSAGES,
            model="grok",
        )
    )

    assert [item.model_dump() for item in results] == [
        {"answer": 7},
        {"reason": "checked"},
    ]
    assert [tool.function.name for tool in chat.proto.tools] == ["Answer", "Reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "messages"),
    [
        (SimpleNamespace(text='```json\n{"answer":7}\n```'), MESSAGES),
        (SimpleNamespace(content='{"answer":7}'), [{"role": "user", "content": "hi"}]),
        (SimpleNamespace(content=['{"answer":7}']), MESSAGES),
    ],
)
async def test_md_json_extracts_all_supported_content_shapes(
    is_async: bool, response: Any, messages: list[ChatCompletionMessageParam]
) -> None:
    wrapped, _, factory = make_client(is_async, Mode.MD_JSON, sampled=response)

    result = await resolve(
        wrapped.create(response_model=Answer, messages=messages, model="grok")
    )

    assert result.model_dump() == {"answer": 7}
    assert result._raw_response is response
    assert wrapped.mode is Mode.MD_JSON
    system = factory.calls[0]["messages"][0]
    assert system.role == xchat.system("instructions").role
    assert "Return your answer as JSON" in system.content[0].text
    assert '"answer"' in system.content[0].text


@pytest.mark.parametrize(
    ("deprecated_mode", "replacement"),
    [(Mode.XAI_TOOLS, Mode.TOOLS), (Mode.XAI_JSON, Mode.MD_JSON)],
)
def test_deprecated_xai_modes_warn_and_normalize(
    deprecated_mode: Mode, replacement: Mode
) -> None:
    from instructor.v2.core.mode import reset_deprecated_mode_warnings

    reset_deprecated_mode_warnings()
    with pytest.warns(
        DeprecationWarning, match=f"Use Mode\\.{replacement.name} instead"
    ):
        wrapped, _, _ = make_client(False, deprecated_mode)

    assert wrapped.mode is replacement
