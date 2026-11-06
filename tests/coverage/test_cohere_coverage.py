"""Focused coverage for the v2 Cohere clients and mode handlers."""

from __future__ import annotations

import builtins
import importlib
import runpy
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import cohere
import httpx
import pytest
import pytest_asyncio
from pydantic import BaseModel

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import (
    ClientError,
    ConfigurationError,
    ModeError,
    ResponseParsingError,
)
from instructor.v2.core.mode import Mode, reset_deprecated_mode_warnings
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.providers.cohere.client import from_cohere
from instructor.v2.providers.cohere.handlers import (
    CohereJSONSchemaHandler,
    CohereMDJSONHandler,
    CohereToolsHandler,
    _convert_messages_to_cohere_v1,
    _extract_text_from_response,
    _extract_text_from_stream_chunk,
)
from instructor.v2.providers.cohere.templating import process_message
from tests.coverage._streams import async_items


class Answer(BaseModel):
    answer: int


@pytest.fixture
def http_client() -> Iterator[httpx.Client]:
    with httpx.Client(trust_env=False) as client:
        yield client


@pytest_asyncio.fixture
async def async_http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(trust_env=False) as client:
        yield client


def _v1_response(text: str) -> cohere.NonStreamedChatResponse:
    return cohere.NonStreamedChatResponse(text=text, finish_reason="COMPLETE")


def _v2_response(text: str) -> cohere.ChatResponse:
    return cohere.ChatResponse(
        id="response-1",
        finish_reason="COMPLETE",
        message=cohere.AssistantMessageResponse(
            content=[cohere.TextAssistantMessageResponseContentItem(text=text)]
        ),
    )


def _v2_delta(text: str | None) -> cohere.ContentDeltaV2ChatStreamResponse:
    return cohere.ContentDeltaV2ChatStreamResponse(
        delta=cohere.ChatContentDeltaEventDelta(
            message=cohere.ChatContentDeltaEventDeltaMessage(
                content=cohere.ChatContentDeltaEventDeltaMessageContent(text=text)
            )
        )
    )


def _stream_events(answer: int) -> list[Any]:
    return [
        _v2_delta('{"tasks":['),
        cohere.ContentEndV2ChatStreamResponse(),
        _v2_delta(f'{{"answer":{answer}}}]}}'),
    ]


@pytest.mark.parametrize(
    ("client_type", "response_factory", "version"),
    [
        (cohere.Client, _v1_response, "v1"),
        (cohere.ClientV2, _v2_response, "v2"),
    ],
)
def test_sync_cohere_clients_retry_and_stream(
    client_type: type[Any],
    response_factory: Any,
    version: str,
    http_client: httpx.Client,
) -> None:
    raw = client_type(
        api_key="test-key",
        base_url="http://127.0.0.1:9",
        httpx_client=http_client,
    )
    raw.chat = Mock(
        side_effect=[
            response_factory('{"answer":"wrong"}'),
            response_factory('{"answer":7}'),
        ]
    )
    raw.chat_stream = Mock(return_value=iter(_stream_events(8)))

    wrapped = from_cohere(raw, mode=Mode.TOOLS)

    assert isinstance(wrapped, Instructor)
    assert wrapped.provider is Provider.COHERE
    assert wrapped.mode is Mode.TOOLS
    assert wrapped.kwargs["_cohere_client_version"] == version

    result = wrapped.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "Give the answer"}],
        model_name="command-r",
        max_retries=1,
    )
    streamed = list(
        wrapped.create_iterable(
            response_model=Answer,
            messages=[{"role": "user", "content": "Stream the answer"}],
            model_name="command-r",
            max_retries=0,
        )
    )

    assert result.model_dump() == {"answer": 7}
    assert streamed == [Answer(answer=8)]
    assert raw.chat.call_count == 2
    assert raw.chat_stream.call_count == 1
    retry_kwargs = raw.chat.call_args_list[1].kwargs
    stream_kwargs = raw.chat_stream.call_args.kwargs
    assert retry_kwargs["model"] == stream_kwargs["model"] == "command-r"
    assert "model_name" not in retry_kwargs
    assert "stream" not in stream_kwargs
    if version == "v1":
        assert "messages" not in retry_kwargs
        assert "Correct the following JSON response" in retry_kwargs["message"]
        assert retry_kwargs["chat_history"][-1]["message"] == "Give the answer"
        assert stream_kwargs["message"] == "Stream the answer"
    else:
        assert (
            "Correct the following JSON response"
            in retry_kwargs["messages"][-1]["content"]
        )
        assert stream_kwargs["messages"][-1]["content"] == "Stream the answer"


def test_cohere_legacy_tools_mode_warns_and_normalizes(
    http_client: httpx.Client,
) -> None:
    raw = cohere.ClientV2(
        api_key="test-key",
        base_url="http://127.0.0.1:9",
        httpx_client=http_client,
    )
    reset_deprecated_mode_warnings()

    with pytest.warns(DeprecationWarning, match=r"Mode\.COHERE_TOOLS.*Mode\.TOOLS"):
        wrapped = from_cohere(raw, mode=Mode.COHERE_TOOLS)

    assert wrapped.mode is Mode.TOOLS
    reset_deprecated_mode_warnings()


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "{{ name }}"},
        {"role": "user", "message": [{"type": "text", "text": "{{ name }}"}]},
        {"role": "user", "message": None},
    ],
)
def test_cohere_templating_leaves_non_string_messages_unchanged(
    message: dict[str, Any],
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def apply_template(value: str, context: dict[str, Any]) -> str:
        calls.append((value, context))
        return value

    result = process_message(message, {"name": "Ada"}, apply_template)

    assert result is message
    assert calls == []


def test_cohere_templating_applies_context_to_a_string_message() -> None:
    message = {"role": "user", "message": "Hello, {{ name }}"}

    result = process_message(
        message,
        {"name": "Ada"},
        lambda value, context: value.replace("{{ name }}", context["name"]),
    )

    assert result == {"role": "user", "message": "Hello, Ada"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_type", "response_factory", "version"),
    [
        (cohere.AsyncClient, _v1_response, "v1"),
        (cohere.AsyncClientV2, _v2_response, "v2"),
    ],
)
async def test_async_cohere_clients_retry_and_stream(
    client_type: type[Any],
    response_factory: Any,
    version: str,
    async_http_client: httpx.AsyncClient,
) -> None:
    raw = client_type(
        api_key="test-key",
        base_url="http://127.0.0.1:9",
        httpx_client=async_http_client,
    )
    responses = [
        response_factory('{"answer":"wrong"}'),
        response_factory('{"answer":9}'),
    ]
    raw.chat = AsyncMock(side_effect=responses)
    raw.chat_stream = Mock(return_value=async_items(_stream_events(10)))

    wrapped = from_cohere(raw, mode=Mode.TOOLS)

    assert isinstance(wrapped, AsyncInstructor)
    assert wrapped.provider is Provider.COHERE
    assert wrapped.mode is Mode.TOOLS
    assert wrapped.kwargs["_cohere_client_version"] == version

    result = await wrapped.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "Give the answer"}],
        model_name="command-r",
        max_retries=1,
    )
    streamed = [
        item
        async for item in wrapped.create_iterable(
            response_model=Answer,
            messages=[{"role": "user", "content": "Stream the answer"}],
            model_name="command-r",
            max_retries=0,
        )
    ]

    assert result.model_dump() == {"answer": 9}
    assert streamed == [Answer(answer=10)]
    assert raw.chat.await_count == 2
    assert raw.chat_stream.call_count == 1
    retry_kwargs = raw.chat.call_args_list[1].kwargs
    stream_kwargs = raw.chat_stream.call_args.kwargs
    assert retry_kwargs["model"] == stream_kwargs["model"] == "command-r"
    assert "stream" not in stream_kwargs
    if version == "v1":
        assert "Correct the following JSON response" in retry_kwargs["message"]
        assert stream_kwargs["message"] == "Stream the answer"
    else:
        assert (
            "Correct the following JSON response"
            in retry_kwargs["messages"][-1]["content"]
        )
        assert stream_kwargs["messages"][-1]["content"] == "Stream the answer"


@pytest.mark.asyncio
async def test_async_cohere_client_accepts_a_non_awaitable_chat_fallback(
    monkeypatch: pytest.MonkeyPatch,
    async_http_client: httpx.AsyncClient,
) -> None:
    raw = cohere.AsyncClientV2(
        api_key="test-key",
        base_url="http://127.0.0.1:9",
        httpx_client=async_http_client,
    )
    chat = Mock(return_value=_v2_response('{"answer":13}'))
    monkeypatch.setattr(raw, "chat", chat)

    wrapped = from_cohere(raw, mode=Mode.TOOLS)
    result = await wrapped.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "Give the answer"}],
        model_name="command-r",
        max_retries=0,
    )

    assert result.model_dump() == {"answer": 13}
    chat.assert_called_once()


def test_cohere_factory_rejects_bad_mode_and_client(
    http_client: httpx.Client,
) -> None:
    raw = cohere.ClientV2(
        api_key="test-key",
        base_url="http://127.0.0.1:9",
        httpx_client=http_client,
    )

    with pytest.raises(ModeError, match="Invalid mode 'json_mode'") as mode_error:
        from_cohere(raw, mode=Mode.JSON)

    assert mode_error.value.provider == "cohere"
    assert "tool_call" in mode_error.value.valid_modes
    with pytest.raises(ClientError, match="Got: object"):
        from_cohere(cast(cohere.ClientV2, object()), mode=Mode.TOOLS)


def test_cohere_imports_fail_cleanly_without_optional_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__
    client_path = Path(
        importlib.import_module("instructor.v2.providers.cohere.client").__file__
    )
    package_path = client_path.with_name("__init__.py")

    def block_cohere(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cohere":
            raise ImportError("cohere is unavailable")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", block_cohere)
        client_globals = runpy.run_path(str(client_path))

    assert client_globals["cohere"] is None
    with pytest.raises(ClientError, match="cohere is not installed"):
        client_globals["from_cohere"](object())

    def block_client(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "instructor.v2.providers.cohere.client":
            raise ImportError("cohere client is unavailable")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", block_client)
        package_globals = runpy.run_path(str(package_path))

    assert package_globals["from_cohere"] is None


def test_cohere_v1_conversion_and_invalid_response_shapes() -> None:
    converted = _convert_messages_to_cohere_v1(
        {
            "_cohere_client_version": "v1",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
                {"role": "user", "content": "Third"},
            ],
            "model_name": "command-r",
            "strict": True,
        }
    )

    assert converted == {
        "message": "Third",
        "chat_history": [
            {"role": "user", "message": "First"},
            {"role": "assistant", "message": "Second"},
        ],
        "model": "command-r",
    }
    with pytest.raises(ResponseParsingError, match="Could not extract text"):
        _extract_text_from_response(object())
    with pytest.raises(ResponseParsingError, match="Could not extract text"):
        _extract_text_from_response(
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        SimpleNamespace(type="thinking", text="ignored"),
                        SimpleNamespace(type="image"),
                    ]
                )
            )
        )


def test_cohere_stream_chunk_shapes_and_sync_extractor() -> None:
    v1_text = cohere.TextGenerationStreamedChatResponse(text="v1 text")
    v2_text = _v2_delta("v2 text")
    empty_delta = _v2_delta(None)
    content_end = cohere.ContentEndV2ChatStreamResponse()

    assert _extract_text_from_stream_chunk(v1_text) == "v1 text"
    assert _extract_text_from_stream_chunk(v2_text) == "v2 text"
    assert _extract_text_from_stream_chunk(empty_delta) is None
    assert _extract_text_from_stream_chunk(content_end) is None
    assert list(
        CohereToolsHandler().extract_streaming_json(
            [content_end, v1_text, empty_delta, v2_text]
        )
    ) == ["v1 text", "v2 text"]


@pytest.mark.asyncio
async def test_cohere_tools_stream_parser_forwards_context_and_strictness() -> None:
    handler = CohereToolsHandler()
    iterable_answer = IterableModel(Answer)
    sync_values = list(
        handler.parse_response(
            _stream_events(11),
            iterable_answer,
            validation_context=None,
            strict=None,
            stream=True,
            is_async=False,
        )
    )
    async_values = [
        item
        async for item in handler.parse_response(
            async_items(_stream_events(12)),
            iterable_answer,
            validation_context={"request_id": "req-1"},
            strict=True,
            stream=True,
            is_async=True,
        )
    ]

    assert sync_values == [Answer(answer=11)]
    assert async_values == [Answer(answer=12)]
    with pytest.raises(ConfigurationError, match="Iterable or Partial"):
        handler.parse_response(iter(_stream_events(13)), Answer, stream=True)


def test_cohere_tools_falls_back_when_tool_call_has_no_parameters() -> None:
    response = SimpleNamespace(
        text='{"answer":14}', tool_calls=[SimpleNamespace(name="answer")]
    )

    assert CohereToolsHandler().parse_response(response, Answer) == Answer(answer=14)


@pytest.mark.parametrize("handler", [CohereJSONSchemaHandler(), CohereMDJSONHandler()])
def test_cohere_json_handlers_reask_v1_and_reject_streaming(handler: Any) -> None:
    without_history = handler.handle_reask(
        {"message": "Original question"}, object(), ValueError("invalid field")
    )
    with_history = handler.handle_reask(
        {
            "message": "Original question",
            "chat_history": [{"role": "assistant", "message": "Earlier"}],
        },
        _v1_response('{"answer":"wrong"}'),
        ValueError("invalid field"),
    )

    assert without_history["chat_history"] == [
        {"role": "user", "message": "Original question"}
    ]
    assert "invalid field" in without_history["message"]
    assert "object object" in without_history["message"]
    assert with_history["chat_history"][-1] == {
        "role": "user",
        "message": "Original question",
    }
    assert 'JSON:\n{"answer":"wrong"}' in with_history["message"]
    with pytest.raises(ConfigurationError, match="Streaming is not supported"):
        handler.parse_response(iter(_stream_events(15)), Answer, stream=True)
