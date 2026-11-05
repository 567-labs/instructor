from __future__ import annotations

import builtins
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
        self.sample_calls = 0
        self.parse_shapes: list[type[BaseModel]] = []
        self.proto = xchat.chat_pb2.GetCompletionsRequest()

    def sample(self) -> Any:
        self.sample_calls += 1
        return self.sampled

    def parse(self, shape: type[BaseModel]) -> tuple[Any, BaseModel]:
        self.parse_shapes.append(shape)
        assert self.parsed is not None
        return self.parsed

    def stream(self) -> Iterable[tuple[Any, Any]]:
        return iter(self.stream_items)


class AsyncChat:
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
        self.sample_calls = 0
        self.parse_shapes: list[type[BaseModel]] = []
        self.proto = xchat.chat_pb2.GetCompletionsRequest()

    async def sample(self) -> Any:
        self.sample_calls += 1
        return self.sampled

    async def parse(self, shape: type[BaseModel]) -> tuple[Any, BaseModel]:
        self.parse_shapes.append(shape)
        assert self.parsed is not None
        return self.parsed

    async def stream(self) -> Any:
        for item in self.stream_items:
            yield item


class ChatFactory:
    def __init__(self, chat: SyncChat | AsyncChat) -> None:
        self.chat = chat
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SyncChat | AsyncChat:
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


def sync_client(
    chat: SyncChat, mode: Mode = Mode.TOOLS, **defaults: Any
) -> tuple[Instructor, ChatFactory]:
    factory = ChatFactory(chat)
    wrapped = xai_client.from_xai(OfflineSyncClient(factory), mode=mode, **defaults)
    assert isinstance(wrapped, Instructor)
    assert not isinstance(wrapped, AsyncInstructor)
    return wrapped, factory


def async_client(
    chat: AsyncChat, mode: Mode = Mode.TOOLS, **defaults: Any
) -> tuple[AsyncInstructor, ChatFactory]:
    factory = ChatFactory(chat)
    wrapped = xai_client.from_xai(OfflineAsyncClient(factory), mode=mode, **defaults)
    assert isinstance(wrapped, AsyncInstructor)
    return wrapped, factory


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
    with pytest.raises(ImportError, match="xai_sdk is required"):
        isolated["_convert_messages"]([{"role": "user", "content": "hello"}])


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


def test_message_schema_and_finalize_helpers_cover_real_runtime_models() -> None:
    assert xai_client._get_model_schema(object()) == {}
    assert xai_client._get_model_name(object()) == "Model"
    raw_messages = [
        {"role": "system", "content": "Be precise"},
        {"role": "user", "content": "Return an answer"},
    ]
    assert xai_client._add_md_json_instructions(raw_messages, object()) == raw_messages

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
    iterable_model = cast(type[BaseModel], prepare_response_model(list[Answer]))
    assert iterable_model is not None
    iterable = iterable_model.model_validate({"tasks": [{"answer": 1}]})
    finalized = xai_client._finalize_parsed_response(iterable, raw)
    assert [item.model_dump() for item in finalized] == [{"answer": 1}]

    adapter_model = cast(type[BaseModel], prepare_response_model(int))
    assert adapter_model is not None
    adapter = adapter_model.model_validate({"content": 7})
    assert xai_client._finalize_parsed_response(adapter, raw) == 7
    assert xai_client._finalize_parsed_response({"answer": 8}, raw) == {"answer": 8}


def test_tool_argument_deltas_skip_missing_and_repeated_values() -> None:
    def call(arguments: Any, tool_id: str | None = None) -> Any:
        return SimpleNamespace(
            id=tool_id, function=SimpleNamespace(arguments=arguments)
        )

    stream = [
        (SimpleNamespace(tool_calls=[call(None)]), None),
        (SimpleNamespace(tool_calls=[call('{"answer":', "first")]), None),
        (SimpleNamespace(tool_calls=[call('{"answer":7}', "second")]), None),
        (SimpleNamespace(tool_calls=[call('{"answer":7}', "second")]), None),
        (SimpleNamespace(tool_calls=[call({"answer": 9})]), None),
    ]

    assert list(xai_client._iter_tool_call_arg_deltas(stream)) == [
        '{"answer":',
        "7}",
        '{"answer": 9}',
    ]


@pytest.mark.asyncio
async def test_async_tool_argument_deltas_skip_missing_and_repeated_values() -> None:
    def call(arguments: Any, tool_id: str | None = None) -> Any:
        return SimpleNamespace(
            id=tool_id, function=SimpleNamespace(arguments=arguments)
        )

    async def stream() -> Any:
        yield SimpleNamespace(tool_calls=[call(None)]), None
        yield SimpleNamespace(tool_calls=[call('{"answer":', "first")]), None
        yield SimpleNamespace(tool_calls=[call('{"answer":7}', "second")]), None
        yield SimpleNamespace(tool_calls=[call('{"answer":7}', "second")]), None
        yield SimpleNamespace(tool_calls=[call({"answer": 9})]), None

    assert [
        delta async for delta in xai_client._aiter_tool_call_arg_deltas(stream())
    ] == ['{"answer":', "7}", '{"answer": 9}']


def test_factory_rejects_unsupported_modes_and_invalid_client_types() -> None:
    native = OfflineSyncClient(ChatFactory(SyncChat()))

    with pytest.raises(ModeError, match="xai") as unsupported:
        xai_client.from_xai(native, mode=Mode.ANTHROPIC_JSON)
    assert "tool_call" in str(unsupported.value)

    with pytest.raises(ClientError, match="Got: object"):
        xai_client.from_xai(cast(Any, object()), mode=Mode.TOOLS)


def test_sync_unstructured_request_converts_messages_and_filters_instructor_args() -> (
    None
):
    raw = SimpleNamespace(content="plain answer")
    wrapped, factory = sync_client(
        SyncChat(sampled=raw), mode=Mode.TOOLS, temperature=0.25
    )

    result = wrapped.create(
        response_model=None,
        messages=MESSAGES,
        model="grok-test",
        max_retries=2,
        context={"request": "coverage"},
        validation_context={"ignored": True},
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
        wrapped.create(
            response_model=Answer,
            messages=MESSAGES,
            model="grok-test",
            token_budget=100,
        )
    assert len(factory.calls) == 1


@pytest.mark.asyncio
async def test_async_unstructured_request_converts_messages_and_filters_instructor_args() -> (
    None
):
    raw = SimpleNamespace(content="plain answer")
    wrapped, factory = async_client(
        AsyncChat(sampled=raw), mode=Mode.TOOLS, temperature=0.25
    )

    result = await wrapped.create(
        response_model=None,
        messages=MESSAGES,
        model="grok-test",
        max_retries=2,
        context={"request": "coverage"},
        validation_context={"ignored": True},
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
        await wrapped.create(
            response_model=Answer,
            messages=MESSAGES,
            model="grok-test",
            token_budget=100,
        )
    assert len(factory.calls) == 1


def test_sync_json_schema_parse_attaches_raw_response() -> None:
    raw = SimpleNamespace(id="raw-sync")
    chat = SyncChat(parsed=(raw, Answer(answer=7)))
    wrapped, factory = sync_client(chat, mode=Mode.JSON_SCHEMA)

    result = wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")

    assert result.model_dump() == {"answer": 7}
    assert cast(Any, result)._raw_response is raw
    assert chat.parse_shapes == [Answer]
    assert factory.calls[0]["model"] == "grok"


@pytest.mark.asyncio
async def test_async_json_schema_parse_attaches_raw_response() -> None:
    raw = SimpleNamespace(id="raw-async")
    chat = AsyncChat(parsed=(raw, Answer(answer=7)))
    wrapped, factory = async_client(chat, mode=Mode.JSON_SCHEMA)

    result = await wrapped.create(
        response_model=Answer, messages=MESSAGES, model="grok"
    )

    assert result.model_dump() == {"answer": 7}
    assert cast(Any, result)._raw_response is raw
    assert chat.parse_shapes == [Answer]
    assert factory.calls[0]["model"] == "grok"


def test_sync_json_schema_streams_iterable_and_partial_models() -> None:
    iterable_chat = SyncChat(
        stream=[
            (None, SimpleNamespace(content='{"tasks":[{"answer":1},')),
            (None, SimpleNamespace(content='{"answer":2}]}')),
        ]
    )
    wrapped, _ = sync_client(iterable_chat, mode=Mode.JSON_SCHEMA)

    answers = list(
        wrapped.create_iterable(response_model=Answer, messages=MESSAGES, model="grok")
    )

    assert [item.model_dump() for item in answers] == [
        {"answer": 1},
        {"answer": 2},
    ]
    assert iterable_chat.proto.response_format.schema

    partial_chat = SyncChat(
        stream=[
            (None, SimpleNamespace(content='{"answer":')),
            (None, SimpleNamespace(content="7}")),
        ]
    )
    partial_client, _ = sync_client(partial_chat, mode=Mode.JSON_SCHEMA)
    partials = list(
        partial_client.create_partial(
            response_model=Answer, messages=MESSAGES, model="grok"
        )
    )

    assert partials[0].answer is None
    assert partials[-1].model_dump() == {"answer": 7}
    assert partial_chat.proto.response_format.schema


@pytest.mark.asyncio
async def test_async_json_schema_streams_iterable_and_partial_models() -> None:
    iterable_chat = AsyncChat(
        stream=[
            (None, SimpleNamespace(content='{"tasks":[{"answer":1},')),
            (None, SimpleNamespace(content='{"answer":2}]}')),
        ]
    )
    wrapped, _ = async_client(iterable_chat, mode=Mode.JSON_SCHEMA)
    answers = [
        item
        async for item in wrapped.create_iterable(
            response_model=Answer, messages=MESSAGES, model="grok"
        )
    ]

    assert [item.model_dump() for item in answers] == [
        {"answer": 1},
        {"answer": 2},
    ]
    assert iterable_chat.proto.response_format.schema

    partial_chat = AsyncChat(
        stream=[
            (None, SimpleNamespace(content='{"answer":')),
            (None, SimpleNamespace(content="7}")),
        ]
    )
    partial_client, _ = async_client(partial_chat, mode=Mode.JSON_SCHEMA)
    partials = [
        item
        async for item in partial_client.create_partial(
            response_model=Answer, messages=MESSAGES, model="grok"
        )
    ]

    assert partials[0].answer is None
    assert partials[-1].model_dump() == {"answer": 7}
    assert partial_chat.proto.response_format.schema


def test_sync_json_schema_rejects_non_streamable_response_models() -> None:
    wrapped, _ = sync_client(SyncChat(), mode=Mode.JSON_SCHEMA)

    with pytest.raises(ValueError, match="Unsupported response model.*Answer"):
        wrapped.create_fn(
            response_model=Answer, messages=MESSAGES, model="grok", stream=True
        )


@pytest.mark.asyncio
async def test_async_json_schema_rejects_non_streamable_response_models() -> None:
    wrapped, _ = async_client(AsyncChat(), mode=Mode.JSON_SCHEMA)

    with pytest.raises(ValueError, match="Unsupported response model.*Answer"):
        await wrapped.create_fn(
            response_model=Answer, messages=MESSAGES, model="grok", stream=True
        )


@pytest.mark.parametrize(
    ("response_model", "arguments", "expected"),
    [
        (Answer, '{"answer":7}', {"answer": 7}),
        (list[Answer], '{"tasks":[{"answer":1},{"answer":2}]}', [1, 2]),
        (int, '{"content":7}', 7),
    ],
)
def test_sync_tools_validate_structured_iterable_and_simple_results(
    response_model: Any, arguments: str, expected: Any
) -> None:
    response = SimpleNamespace(tool_calls=[tool_call("Answer", arguments)])
    chat = SyncChat(sampled=response)
    wrapped, _ = sync_client(chat, mode=Mode.TOOLS)

    result = wrapped.create(
        response_model=response_model,
        messages=MESSAGES,
        model="grok",
        strict=False,
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
    ("response_model", "arguments", "expected"),
    [
        (Answer, '{"answer":7}', {"answer": 7}),
        (list[Answer], '{"tasks":[{"answer":1},{"answer":2}]}', [1, 2]),
        (int, '{"content":7}', 7),
    ],
)
async def test_async_tools_validate_structured_iterable_and_simple_results(
    response_model: Any, arguments: str, expected: Any
) -> None:
    response = SimpleNamespace(tool_calls=[tool_call("Answer", arguments)])
    chat = AsyncChat(sampled=response)
    wrapped, _ = async_client(chat, mode=Mode.TOOLS)

    result = await wrapped.create(
        response_model=response_model,
        messages=MESSAGES,
        model="grok",
        strict=False,
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


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(tool_calls=[], text='```json\n{"answer":7}\n```'),
        SimpleNamespace(tool_calls=[], content='{"answer":7}'),
        SimpleNamespace(tool_calls=[], content=['{"answer":7}']),
    ],
)
def test_sync_tools_fall_back_to_text_or_content(response: Any) -> None:
    wrapped, _ = sync_client(SyncChat(sampled=response), mode=Mode.TOOLS)

    result = wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")

    assert result.model_dump() == {"answer": 7}
    assert cast(Any, result)._raw_response is response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(tool_calls=[], text='```json\n{"answer":7}\n```'),
        SimpleNamespace(tool_calls=[], content='{"answer":7}'),
        SimpleNamespace(tool_calls=[], content=['{"answer":7}']),
    ],
)
async def test_async_tools_fall_back_to_text_or_content(response: Any) -> None:
    wrapped, _ = async_client(AsyncChat(sampled=response), mode=Mode.TOOLS)

    result = await wrapped.create(
        response_model=Answer, messages=MESSAGES, model="grok"
    )

    assert result.model_dump() == {"answer": 7}
    assert cast(Any, result)._raw_response is response


def test_sync_tools_raise_when_no_arguments_or_text_are_available() -> None:
    wrapped, _ = sync_client(
        SyncChat(sampled=SimpleNamespace(tool_calls=[], content=[])), mode=Mode.TOOLS
    )

    with pytest.raises(ValueError, match="No tool calls returned from xAI"):
        wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")


@pytest.mark.asyncio
async def test_async_tools_raise_when_no_arguments_or_text_are_available() -> None:
    wrapped, _ = async_client(
        AsyncChat(sampled=SimpleNamespace(tool_calls=[], content=[])), mode=Mode.TOOLS
    )

    with pytest.raises(ValueError, match="No tool calls returned from xAI"):
        await wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")


@pytest.mark.parametrize(
    ("chat", "client_factory"),
    [
        (
            SyncChat(sampled=SimpleNamespace(tool_calls=[], content={"answer": 7})),
            sync_client,
        ),
        (
            AsyncChat(sampled=SimpleNamespace(tool_calls=[], content={"answer": 7})),
            async_client,
        ),
    ],
)
@pytest.mark.asyncio
async def test_tools_reject_unsupported_content_objects(
    chat: SyncChat | AsyncChat, client_factory: Any
) -> None:
    wrapped, _ = client_factory(chat, mode=Mode.TOOLS)

    with pytest.raises(ValueError, match="No tool calls returned from xAI"):
        result = wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")
        if isinstance(wrapped, AsyncInstructor):
            await result


def test_sync_tools_stream_iterable_partial_and_reject_plain_models() -> None:
    iterable_chat = SyncChat(
        stream=[
            (
                SimpleNamespace(
                    tool_calls=[tool_call("IterableAnswer", '{"tasks":[{"answer":1},')]
                ),
                None,
            ),
            (
                SimpleNamespace(
                    tool_calls=[
                        tool_call(
                            "IterableAnswer", '{"tasks":[{"answer":1},{"answer":2}]}'
                        )
                    ]
                ),
                None,
            ),
        ]
    )
    wrapped, _ = sync_client(iterable_chat, mode=Mode.TOOLS)
    answers = list(
        wrapped.create_iterable(response_model=Answer, messages=MESSAGES, model="grok")
    )

    assert [item.answer for item in answers] == [1, 2]

    partial_chat = SyncChat(
        stream=[
            (
                SimpleNamespace(tool_calls=[tool_call("PartialAnswer", '{"answer":')]),
                None,
            ),
            (
                SimpleNamespace(
                    tool_calls=[tool_call("PartialAnswer", '{"answer":7}')]
                ),
                None,
            ),
        ]
    )
    partial_client, _ = sync_client(partial_chat, mode=Mode.TOOLS)
    partials = list(
        partial_client.create_partial(
            response_model=Answer, messages=MESSAGES, model="grok"
        )
    )

    assert partials[0].answer is None
    assert partials[-1].answer == 7

    plain_client, _ = sync_client(SyncChat(), mode=Mode.TOOLS)
    with pytest.raises(ValueError, match="Unsupported response model.*Answer"):
        plain_client.create_fn(
            response_model=Answer, messages=MESSAGES, model="grok", stream=True
        )


@pytest.mark.asyncio
async def test_async_tools_stream_iterable_partial_and_reject_plain_models() -> None:
    iterable_chat = AsyncChat(
        stream=[
            (
                SimpleNamespace(
                    tool_calls=[tool_call("IterableAnswer", '{"tasks":[{"answer":1},')]
                ),
                None,
            ),
            (
                SimpleNamespace(
                    tool_calls=[
                        tool_call(
                            "IterableAnswer", '{"tasks":[{"answer":1},{"answer":2}]}'
                        )
                    ]
                ),
                None,
            ),
        ]
    )
    wrapped, _ = async_client(iterable_chat, mode=Mode.TOOLS)
    answers = [
        item
        async for item in wrapped.create_iterable(
            response_model=Answer, messages=MESSAGES, model="grok"
        )
    ]

    assert [item.answer for item in answers] == [1, 2]

    partial_chat = AsyncChat(
        stream=[
            (
                SimpleNamespace(tool_calls=[tool_call("PartialAnswer", '{"answer":')]),
                None,
            ),
            (
                SimpleNamespace(
                    tool_calls=[tool_call("PartialAnswer", '{"answer":7}')]
                ),
                None,
            ),
        ]
    )
    partial_client, _ = async_client(partial_chat, mode=Mode.TOOLS)
    partials = [
        item
        async for item in partial_client.create_partial(
            response_model=Answer, messages=MESSAGES, model="grok"
        )
    ]

    assert partials[0].answer is None
    assert partials[-1].answer == 7

    plain_client, _ = async_client(AsyncChat(), mode=Mode.TOOLS)
    with pytest.raises(ValueError, match="Unsupported response model.*Answer"):
        await plain_client.create_fn(
            response_model=Answer, messages=MESSAGES, model="grok", stream=True
        )


def test_sync_parallel_tools_register_each_schema_and_parse_known_calls() -> None:
    response = SimpleNamespace(
        tool_calls=[
            tool_call("Answer", '{"answer":7}'),
            tool_call("Reason", '{"reason":"checked"}', "call-3"),
        ]
    )
    chat = SyncChat(sampled=response)
    wrapped, _ = sync_client(chat, mode=Mode.PARALLEL_TOOLS)

    results = list(
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
async def test_async_parallel_tools_register_each_schema_and_parse_known_calls() -> (
    None
):
    response = SimpleNamespace(
        tool_calls=[
            tool_call("Answer", '{"answer":7}'),
            tool_call("Reason", '{"reason":"checked"}', "call-3"),
        ]
    )
    chat = AsyncChat(sampled=response)
    wrapped, _ = async_client(chat, mode=Mode.PARALLEL_TOOLS)

    iterator = await wrapped.create_fn(
        response_model=Iterable[Union[Answer, Reason]], messages=MESSAGES, model="grok"
    )
    results = list(iterator)

    assert [item.model_dump() for item in results] == [
        {"answer": 7},
        {"reason": "checked"},
    ]
    assert [tool.function.name for tool in chat.proto.tools] == ["Answer", "Reason"]


@pytest.mark.parametrize(
    ("response", "messages"),
    [
        (SimpleNamespace(text='```json\n{"answer":7}\n```'), MESSAGES),
        (SimpleNamespace(content='{"answer":7}'), [{"role": "user", "content": "hi"}]),
        (SimpleNamespace(content=['{"answer":7}']), MESSAGES),
    ],
)
def test_sync_md_json_extracts_all_supported_content_shapes(
    response: Any, messages: list[ChatCompletionMessageParam]
) -> None:
    chat = SyncChat(sampled=response)
    wrapped, factory = sync_client(chat, mode=Mode.MD_JSON)

    result = wrapped.create(response_model=Answer, messages=messages, model="grok")

    assert result.model_dump() == {"answer": 7}
    assert cast(Any, result)._raw_response is response
    assert wrapped.mode is Mode.MD_JSON
    system = factory.calls[0]["messages"][0]
    assert system.role == xchat.system("instructions").role
    assert "Return your answer as JSON" in system.content[0].text
    assert '"answer"' in system.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "messages"),
    [
        (SimpleNamespace(text='```json\n{"answer":7}\n```'), MESSAGES),
        (SimpleNamespace(content='{"answer":7}'), [{"role": "user", "content": "hi"}]),
        (SimpleNamespace(content=['{"answer":7}']), MESSAGES),
    ],
)
async def test_async_md_json_extracts_all_supported_content_shapes(
    response: Any, messages: list[ChatCompletionMessageParam]
) -> None:
    chat = AsyncChat(sampled=response)
    wrapped, factory = async_client(chat, mode=Mode.MD_JSON)

    result = await wrapped.create(
        response_model=Answer, messages=messages, model="grok"
    )

    assert result.model_dump() == {"answer": 7}
    assert cast(Any, result)._raw_response is response
    assert wrapped.mode is Mode.MD_JSON
    system = factory.calls[0]["messages"][0]
    assert system.role == xchat.system("instructions").role
    assert "Return your answer as JSON" in system.content[0].text
    assert '"answer"' in system.content[0].text


def test_sync_md_json_raises_when_the_response_has_no_json_content() -> None:
    wrapped, _ = sync_client(
        SyncChat(sampled=SimpleNamespace(content=[])), mode=Mode.MD_JSON
    )

    with pytest.raises(ValueError, match="Could not extract JSON from xAI response"):
        wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")


@pytest.mark.asyncio
async def test_async_md_json_raises_when_the_response_has_no_json_content() -> None:
    wrapped, _ = async_client(
        AsyncChat(sampled=SimpleNamespace(content=[])), mode=Mode.MD_JSON
    )

    with pytest.raises(ValueError, match="Could not extract JSON from xAI response"):
        await wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")


@pytest.mark.parametrize(
    ("chat", "client_factory"),
    [
        (SyncChat(sampled=SimpleNamespace(content={"answer": 7})), sync_client),
        (AsyncChat(sampled=SimpleNamespace(content={"answer": 7})), async_client),
    ],
)
@pytest.mark.asyncio
async def test_md_json_rejects_unsupported_content_objects(
    chat: SyncChat | AsyncChat, client_factory: Any
) -> None:
    wrapped, _ = client_factory(chat, mode=Mode.MD_JSON)

    with pytest.raises(ValueError, match="Could not extract JSON from xAI response"):
        result = wrapped.create(response_model=Answer, messages=MESSAGES, model="grok")
        if isinstance(wrapped, AsyncInstructor):
            await result


@pytest.mark.parametrize(
    ("deprecated_mode", "replacement"),
    [(Mode.XAI_TOOLS, Mode.TOOLS), (Mode.XAI_JSON, Mode.MD_JSON)],
)
def test_deprecated_xai_modes_warn_and_normalize(
    deprecated_mode: Mode, replacement: Mode
) -> None:
    with pytest.warns(
        DeprecationWarning, match=f"Use Mode\\.{replacement.name} instead"
    ):
        wrapped, _ = sync_client(SyncChat(), mode=deprecated_mode)

    assert wrapped.mode is replacement
