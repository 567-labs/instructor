"""Offline coverage for the xAI v2 request, retry, and response handlers."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, Union, cast

import pytest

pytest.importorskip("xai_sdk")

from pydantic import BaseModel, ValidationError

import instructor.v2.core.decorators as decorators
from instructor.v2.core.errors import ConfigurationError
from instructor.v2.dsl.iterable import IterableBase, IterableModel
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial
from instructor.v2.dsl.simple_type import ModelAdapter
from instructor.v2.providers.xai import handlers
from tests.coverage._streams import async_items


class Answer(BaseModel):
    """A numeric answer returned by Grok."""

    answer: float


class User(BaseModel):
    """A user returned by Grok."""

    name: str
    age: int


def _tool_call(
    name: str,
    arguments: dict[str, Any] | str | None,
    tool_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    wrapped: bool = True,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    if wrapped:
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
    return SimpleNamespace(delta=delta)


def test_optional_xai_sdk_import_has_a_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the provider without its optional SDK must fail only on use."""
    monkeypatch.setitem(sys.modules, "xai_sdk", None)
    monkeypatch.setattr(
        decorators,
        "register_mode_handler",
        lambda *_args: lambda handler: handler,
    )

    isolated = runpy.run_path(handlers.__file__, run_name="xai_without_sdk")

    assert isolated["xchat"] is None
    with pytest.raises(ImportError, match="xai_sdk is required"):
        isolated["_convert_messages"]([{"role": "user", "content": "hello"}])


def test_convert_messages_maps_all_xai_roles_and_rejects_bad_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_chat = SimpleNamespace(
        text=lambda value: ("text", value),
        user=lambda value: ("user", value),
        assistant=lambda value: ("assistant", value),
        system=lambda value: ("system", value),
        tool_result=lambda value: ("tool_result", value),
    )
    monkeypatch.setattr(handlers, "xchat", fake_chat)

    converted = handlers._convert_messages(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "thinking"},
            {"role": "system", "content": "be exact"},
            {"role": "tool", "content": '{"answer": 4}'},
        ]
    )

    assert converted == [
        ("user", ("text", "question")),
        ("assistant", ("text", "thinking")),
        ("system", ("text", "be exact")),
        ("tool_result", '{"answer": 4}'),
    ]
    with pytest.raises(ValueError, match="Only string content"):
        handlers._convert_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is here?"},
                        {"type": "image_url", "image_url": "data:image/png;base64,AA"},
                    ],
                }
            ]
        )
    with pytest.raises(ValueError, match="Unsupported role: developer"):
        handlers._convert_messages([{"role": "developer", "content": "rules"}])


def test_legacy_request_helpers_convert_messages_strip_retry_fields_and_make_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_chat = SimpleNamespace(
        text=lambda value: value,
        user=lambda value: {"xai_role": "user", "text": value},
        assistant=lambda value: {"xai_role": "assistant", "text": value},
        system=lambda value: {"xai_role": "system", "text": value},
        tool_result=lambda value: {"xai_role": "tool", "text": value},
        tool=lambda **value: value,
    )
    monkeypatch.setattr(handlers, "xchat", fake_chat)
    request = {
        "messages": [{"role": "user", "content": "2 + 2?"}],
        "max_retries": 3,
        "context": {"tenant": "test"},
        "hooks": object(),
        "temperature": 0,
    }

    json_model, json_kwargs = handlers.handle_xai_json(Answer, request.copy())
    tool_model, tool_kwargs = handlers.handle_xai_tools(Answer, request.copy())
    no_model, no_model_kwargs = handlers.handle_xai_tools(None, request.copy())

    assert json_model is Answer
    assert json_kwargs["x_messages"] == [{"xai_role": "user", "text": "2 + 2?"}]
    assert {"max_retries", "context", "hooks"}.isdisjoint(json_kwargs)
    assert json_kwargs["temperature"] == 0
    assert tool_model is Answer
    assert tool_kwargs["tool"]["name"] == "Answer"
    assert tool_kwargs["tool"]["description"] == "A numeric answer returned by Grok."
    assert tool_kwargs["tool"]["parameters"]["properties"]["answer"]["type"] == "number"
    assert no_model is None
    assert "tool" not in no_model_kwargs


def test_legacy_reask_helpers_preserve_the_failed_response_and_validation_error() -> (
    None
):
    response = SimpleNamespace(text="not a number")
    original = {"messages": [{"role": "user", "content": "2 + 2?"}]}
    error = ValueError("answer must be a number")

    json_retry = handlers.reask_xai_json(original, response, error)
    tools_retry = handlers.reask_xai_tools(original, response, error)

    assert json_retry is not original
    assert json_retry["messages"][-3]["role"] == "user"
    assert "Validation Errors found" in json_retry["messages"][-3]["content"]
    assert "answer must be a number" in json_retry["messages"][-3]["content"]
    assert "not a number" in json_retry["messages"][-3]["content"]
    assert tools_retry is not original
    assert tools_retry["messages"][-2] == {
        "role": "assistant",
        "content": str(response),
    }
    assert tools_retry["messages"][-1]["role"] == "user"
    assert "Validation Error found" in tools_retry["messages"][-1]["content"]
    assert "answer must be a number" in tools_retry["messages"][-1]["content"]


def test_streaming_flags_only_track_streamable_models_and_are_consumed_once() -> None:
    handler = handlers.XAIToolsHandler()
    users = IterableModel(User)
    partial_answer = Partial[Answer]

    handler._register_streaming_from_kwargs(None, {"stream": True})
    handler.mark_streaming_model(None, True)
    handler.mark_streaming_model(Answer, False)
    handler.mark_streaming_model(Answer, True)
    assert handler._consume_streaming_flag(None) is False
    assert handler._consume_streaming_flag(ParallelBase(Answer)) is False
    assert handler._consume_streaming_flag(Answer) is False

    handler._register_streaming_from_kwargs(users, {"stream": True})
    handler._register_streaming_from_kwargs(partial_answer, {"stream": True})

    assert handler._consume_streaming_flag(users) is True
    assert handler._consume_streaming_flag(users) is False
    assert handler._consume_streaming_flag(partial_answer) is True


def test_sync_stream_extractors_handle_content_fences_and_cumulative_tool_deltas() -> (
    None
):
    tool_handler = handlers.XAIToolsHandler()
    schema_handler = handlers.XAIJSONSchemaHandler()
    markdown_handler = handlers.XAIMDJSONHandler()
    tool_stream = [
        SimpleNamespace(choices=[SimpleNamespace(delta=None)]),
        _chunk(tool_calls=[SimpleNamespace(id="ignored", function=None)]),
        _chunk(tool_calls=[_tool_call("Answer", '{"answer":', "call-1")]),
        _chunk(tool_calls=[_tool_call("Answer", '{"answer": 4', "call-1")]),
        _chunk(tool_calls=[_tool_call("Answer", '{"answer": 4.0}', "call-2")]),
        _chunk(tool_calls=[_tool_call("Answer", '{"answer": 4.0}', "call-2")]),
    ]
    schema_stream = [
        SimpleNamespace(delta=None),
        _chunk(content=None, wrapped=False),
        _chunk(content='{"answer": ', wrapped=False),
        _chunk(content="5.0}", wrapped=False),
    ]
    markdown_stream = [
        _chunk(content="Here is the result:\n```json\n"),
        _chunk(content='{"answer": 6.0}'),
        _chunk(content="\n```"),
    ]

    assert (
        "".join(tool_handler.extract_streaming_json(tool_stream)) == '{"answer": 4.0}'
    )
    assert (
        "".join(schema_handler.extract_streaming_json(schema_stream))
        == '{"answer": 5.0}'
    )
    assert (
        "".join(markdown_handler.extract_streaming_json(markdown_stream))
        == '{"answer": 6.0}'
    )
    assert (
        list(
            handlers.XAIParallelToolsHandler().extract_streaming_json(
                [_chunk(content='{"answer": 6.0}')]
            )
        )
        == []
    )


@pytest.mark.asyncio
async def test_async_stream_extractors_handle_content_fences_and_cumulative_tool_deltas() -> (
    None
):
    tool_handler = handlers.XAIToolsHandler()
    schema_handler = handlers.XAIJSONSchemaHandler()
    markdown_handler = handlers.XAIMDJSONHandler()
    tool_stream = [
        SimpleNamespace(choices=[SimpleNamespace(delta=None)]),
        _chunk(tool_calls=[SimpleNamespace(id="ignored", function=None)]),
        _chunk(tool_calls=[_tool_call("Answer", '{"answer":', None)]),
        _chunk(tool_calls=[_tool_call("Answer", '{"answer": 7', None)]),
        _chunk(tool_calls=[_tool_call("Answer", '{"answer": 7.0}', "call-2")]),
        _chunk(tool_calls=[_tool_call("Answer", '{"answer": 7.0}', "call-2")]),
    ]
    schema_stream = [
        SimpleNamespace(delta=None),
        _chunk(content=None, wrapped=False),
        _chunk(content='{"answer": ', wrapped=False),
        _chunk(content="8.0}", wrapped=False),
    ]
    markdown_stream = [
        _chunk(content="```json\n"),
        _chunk(content='{"answer": 9.0}'),
        _chunk(content="\n```"),
    ]

    tool_chunks = [
        chunk
        async for chunk in tool_handler.extract_streaming_json_async(
            async_items(tool_stream)
        )
    ]
    schema_chunks = [
        chunk
        async for chunk in schema_handler.extract_streaming_json_async(
            async_items(schema_stream)
        )
    ]
    markdown_chunks = [
        chunk
        async for chunk in markdown_handler.extract_streaming_json_async(
            async_items(markdown_stream)
        )
    ]

    assert "".join(tool_chunks) == '{"answer": 7.0}'
    assert "".join(schema_chunks) == '{"answer": 8.0}'
    assert "".join(markdown_chunks) == '{"answer": 9.0}'
    assert [
        chunk
        async for chunk in handlers.XAIParallelToolsHandler().extract_streaming_json_async(
            async_items([_chunk(content='{"answer": 9.0}')])
        )
    ] == []


def test_tools_streaming_parser_returns_complete_iterable_items_with_validation_options() -> (
    None
):
    handler = handlers.XAIToolsHandler()
    users = IterableModel(User)
    prepared, request = handler.prepare_request(users, {"stream": True})
    assert prepared is not None
    stream = [
        _chunk(
            tool_calls=[
                _tool_call("IterableUser", '{"tasks": [{"name": "Ada"', "users")
            ]
        ),
        _chunk(
            tool_calls=[
                _tool_call(
                    "IterableUser",
                    '{"tasks": [{"name": "Ada", "age": 31}, {"name": "Lin", "age": 29}]}',
                    "users",
                )
            ]
        ),
    ]

    parsed = handler.parse_response(
        stream,
        prepared,
        validation_context={"request_id": "xai-sync"},
        strict=True,
    )

    assert request["stream"] is True
    assert [(item.name, item.age) for item in parsed] == [("Ada", 31), ("Lin", 29)]


@pytest.mark.asyncio
async def test_json_schema_streaming_parser_returns_complete_async_iterable_items() -> (
    None
):
    handler = handlers.XAIJSONSchemaHandler()
    users = IterableModel(User)
    prepared, request = handler.prepare_request(users, {"stream": True})
    assert prepared is not None
    stream = [
        _chunk(content='{"tasks": [{"name": "Ada"'),
        _chunk(content=', "age": 31}, {"name": "Lin", "age": 29}]}'),
    ]

    parsed = handler.parse_response(
        async_items(stream),
        prepared,
        validation_context={"request_id": "xai-async"},
        strict=True,
    )
    items = [item async for item in parsed]

    assert request["_xai_json_schema"]["name"] == prepared.__name__
    assert [(item.name, item.age) for item in items] == [("Ada", 31), ("Lin", 29)]


def test_markdown_streaming_parser_returns_partial_updates_and_a_complete_answer() -> (
    None
):
    handler = handlers.XAIMDJSONHandler()
    partial_answer = Partial[Answer]
    prepared, request = handler.prepare_request(
        partial_answer,
        {"messages": [{"role": "user", "content": "2 + 2?"}], "stream": True},
    )
    assert prepared is not None
    stream = [
        _chunk(content="```json\n"),
        _chunk(content='{"answer": 4'),
        _chunk(content=".0}\n```"),
    ]

    parsed = handler.parse_response(
        stream,
        prepared,
        validation_context={"request_id": "xai-partial"},
        strict=True,
    )

    assert request["messages"][0]["role"] == "system"
    assert isinstance(parsed, list)
    assert parsed[-1].answer == 4.0


@pytest.mark.asyncio
async def test_async_partial_parser_and_custom_sync_stream_model_use_the_stream_extractor() -> (
    None
):
    handler = handlers.XAIJSONSchemaHandler()
    partial_answer = Partial[Answer]
    async_result = handler._parse_streaming_response(
        partial_answer,
        async_items([_chunk(content='{"answer": 1'), _chunk(content="2.0}")]),
        validation_context={"request_id": "partial"},
        strict=True,
    )
    async_updates = [item async for item in async_result]

    class StreamingAnswer(BaseModel):
        answer: float

        @classmethod
        def from_streaming_response(
            cls,
            completion: Iterable[Any],
            stream_extractor: Any,
            **kwargs: Any,
        ) -> Iterable[StreamingAnswer]:
            assert kwargs == {"context": {"request_id": "custom"}, "strict": False}
            payload = "".join(stream_extractor(completion))
            yield cls.model_validate_json(payload, **kwargs)

    sync_result = handler._parse_streaming_response(
        StreamingAnswer,
        [_chunk(content='{"answer": 13.0}')],
        validation_context={"request_id": "custom"},
        strict=False,
    )

    assert async_updates[-1].answer == 12.0
    assert [item.answer for item in sync_result] == [13.0]


@pytest.mark.asyncio
async def test_streaming_iterables_allow_empty_streams_without_validation_options() -> (
    None
):
    handler = handlers.XAIJSONSchemaHandler()
    users = IterableModel(User)

    sync_result = handler._parse_streaming_response(
        users, [], validation_context=None, strict=None
    )
    async_result = handler._parse_streaming_response(
        users, async_items([]), validation_context=None, strict=False
    )

    assert list(sync_result) == []
    assert [item async for item in async_result] == []


def test_finalize_unwraps_iterable_parallel_and_simple_type_results() -> None:
    handler = handlers.XAIToolsHandler()
    users = IterableModel(User)
    parsed_users = users(tasks=[User(name="Ada", age=31), User(name="Lin", age=29)])
    parallel = ParallelBase(Answer, User)
    adapter_model = cast(type[BaseModel], ModelAdapter[int])
    adapted = adapter_model(content=42)

    iterable_result = handler._finalize_parsed_result(users, object(), parsed_users)
    parallel_result = handler._finalize_parsed_result(parallel, object(), ["kept"])
    adapted_result = handler._finalize_parsed_result(adapter_model, object(), adapted)

    assert isinstance(parsed_users, IterableBase)
    assert [(item.name, item.age) for item in iterable_result] == [
        ("Ada", 31),
        ("Lin", 29),
    ]
    assert parallel_result == ["kept"]
    assert adapted_result == 42
    assert handler._finalize_parsed_result(Answer, object(), {"answer": 4.0}) == {
        "answer": 4.0
    }


def test_parallel_tools_build_schemas_and_parse_known_calls() -> None:
    handler = handlers.XAIParallelToolsHandler()
    model = cast(type[BaseModel], Iterable[Union[Answer, User]])
    request = {"messages": [{"role": "user", "content": "answer and user"}]}

    no_model, unchanged = handler.prepare_request(None, request)
    prepared, prepared_request = handler.prepare_request(model, request)
    parsed = list(
        handler.parse_response(
            SimpleNamespace(
                tool_calls=[
                    _tool_call("Answer", {"answer": 4.0}),
                    _tool_call("User", '{"name": "Ada", "age": 31}'),
                ]
            ),
            model,
            validation_context={"request_id": "parallel"},
            strict=True,
        )
    )
    empty = list(handler.parse_response(SimpleNamespace(tool_calls=None), model))

    assert no_model is None
    assert unchanged is request
    assert isinstance(prepared, ParallelBase)
    assert {tool["function"]["name"] for tool in prepared_request["_xai_tools"]} == {
        "Answer",
        "User",
    }
    assert [type(item) for item in parsed] == [Answer, User]
    assert parsed[0].answer == 4.0
    assert (parsed[1].name, parsed[1].age) == ("Ada", 31)
    assert empty == []


def test_parallel_tools_reject_streaming_and_reask_with_the_failed_response() -> None:
    handler = handlers.XAIParallelToolsHandler()
    model = cast(type[BaseModel], Iterable[Union[Answer, User]])
    response = SimpleNamespace(tool_calls=[_tool_call("Answer", '{"answer": "bad"}')])

    with pytest.raises(ConfigurationError, match="stream=True is not supported"):
        handler.prepare_request(model, {"stream": True})
    retry = handler.handle_reask(
        {"messages": [{"role": "user", "content": "answer"}]},
        response,
        ValueError("answer must be a number"),
    )

    assert retry["messages"][-2] == {"role": "assistant", "content": str(response)}
    assert retry["messages"][-1]["role"] == "user"
    assert "answer must be a number" in retry["messages"][-1]["content"]


def test_schema_and_markdown_parsers_accept_remaining_content_shapes_and_reject_refusal() -> (
    None
):
    schema_handler = handlers.XAIJSONSchemaHandler()
    markdown_handler = handlers.XAIMDJSONHandler()

    schema = schema_handler.parse_response(
        SimpleNamespace(text=None, content=['{"answer": 16.0}']),
        Answer,
        strict=True,
    )
    markdown = markdown_handler.parse_response(
        SimpleNamespace(text=None, content='```json\n{"answer": 17.0}\n```'),
        Answer,
        strict=True,
    )

    assert schema.answer == 16.0
    assert markdown.answer == 17.0
    with pytest.raises(ValueError, match="Could not parse xAI response"):
        schema_handler.parse_response(
            SimpleNamespace(text=None, content=None, refusal="I cannot help with that"),
            Answer,
        )
    with pytest.raises(ValidationError, match="answer"):
        markdown_handler.parse_response(
            SimpleNamespace(text='```json\n{"answer": "not-a-number"}\n```'),
            Answer,
            strict=True,
        )


@pytest.mark.parametrize(
    ("handler", "match"),
    [
        (handlers.XAIToolsHandler(), "No tool calls returned from xAI"),
        (handlers.XAIJSONSchemaHandler(), "Could not parse xAI response"),
        (handlers.XAIMDJSONHandler(), "Could not extract JSON from xAI response"),
    ],
)
def test_response_handlers_reject_unsupported_content_objects(
    handler: Any, match: str
) -> None:
    response = SimpleNamespace(tool_calls=[], text=None, content={"answer": 4.0})

    with pytest.raises(ValueError, match=match):
        handler.parse_response(response, Answer)


def test_json_schema_handler_rejects_an_unparsed_sdk_tuple() -> None:
    raw = SimpleNamespace(id="raw-xai-response")

    with pytest.raises(ValueError, match="Could not parse xAI response"):
        handlers.XAIJSONSchemaHandler().parse_response((raw, {"answer": 4.0}), Answer)


def test_markdown_request_preserves_multimodal_system_text_and_handles_empty_messages() -> (
    None
):
    handler = handlers.XAIMDJSONHandler()
    multimodal_messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Keep this instruction."},
                {"type": "image_url", "image_url": "data:image/png;base64,AA"},
            ],
        },
        {"role": "user", "content": "What is in the image?"},
    ]

    _, multimodal = handler.prepare_request(Answer, {"messages": multimodal_messages})
    _, empty = handler.prepare_request(Answer, {"messages": []})
    _, empty_system = handler.prepare_request(
        Answer,
        {"messages": [{"role": "system", "content": []}]},
    )

    first_part = multimodal["messages"][0]["content"][0]
    assert first_part["text"].startswith("Keep this instruction.")
    assert "json_schema" in first_part["text"]
    assert multimodal["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": "data:image/png;base64,AA",
    }
    assert multimodal["messages"][-1]["role"] == "user"
    assert "```json codeblock" in multimodal["messages"][-1]["content"]
    assert empty["messages"][0]["role"] == "system"
    assert empty_system["messages"][0]["role"] == "system"
    assert "json_schema" in empty_system["messages"][0]["content"]
