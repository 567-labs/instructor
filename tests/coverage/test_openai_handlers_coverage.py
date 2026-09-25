from __future__ import annotations

import json
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, Union, cast

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from pydantic import BaseModel

from instructor.v2.core.errors import (
    ConfigurationError,
    IncompleteOutputException,
    ResponseParsingError,
)
from instructor.v2.core.mode import Mode
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial
from instructor.v2.dsl.simple_type import ModelAdapter
from instructor.v2.providers.openai import handlers
from instructor.v2.providers.openai.handlers import (
    OpenAIHandlerBase,
    OpenAIJSONHandler,
    OpenAIJSONSchemaHandler,
    OpenAIMDJSONHandler,
    OpenAIParallelToolsHandler,
    OpenAIResponsesToolsHandler,
    OpenAIToolsHandler,
    reask_default,
    reask_responses_tools,
    reask_tools,
)
from tests.coverage._openai import chat_chunk, chat_completion, tool_call
from tests.coverage._streams import async_items


class User(BaseModel):
    name: str


class Search(BaseModel):
    query: str


def base_handler(mode: Mode) -> OpenAIHandlerBase:
    handler = OpenAIToolsHandler()
    handler.mode = mode
    return handler


def response_deltas(
    item_id: str, *deltas: str
) -> list[ResponseFunctionCallArgumentsDeltaEvent]:
    return [
        ResponseFunctionCallArgumentsDeltaEvent(
            delta=delta,
            item_id=item_id,
            output_index=0,
            sequence_number=index,
            type="response.function_call_arguments.delta",
        )
        for index, delta in enumerate(deltas, start=1)
    ]


def test_responses_tool_filter_accepts_legacy_items_and_formats_missing_details() -> (
    None
):
    legacy_call = SimpleNamespace(arguments='{"name":"Ada"}')
    ignored_message = SimpleNamespace(type="message", content="hello")

    assert handlers._filter_responses_tool_calls([ignored_message, legacy_call]) == [
        legacy_call
    ]
    assert handlers._format_responses_tool_call_details(legacy_call) == ""


def test_reask_default_keeps_assistant_message_before_correction() -> None:
    response = chat_completion(content='{"name": 2}')

    result = reask_default(
        {"messages": [{"role": "user", "content": "extract a user"}]},
        response,
        ValueError("name must be a string"),
    )

    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == '{"name": 2}'
    assert result["messages"][2] == {
        "role": "user",
        "content": (
            "Recall the function correctly, fix the errors, exceptions found\n"
            "name must be a string"
        ),
    }


def test_tools_reask_preserves_assistant_calls_and_adds_one_tool_error_per_call() -> (
    None
):
    response = chat_completion(
        tool_calls=[
            tool_call("User", '{"name":3}', "call_user"),
            tool_call("Search", '{"query":4}', "call_search"),
        ]
    )

    result = OpenAIToolsHandler().handle_reask(
        {"messages": [{"role": "user", "content": "extract both values"}]},
        response,
        ValueError("invalid fields"),
    )

    assert result["messages"] == [
        {"role": "user", "content": "extract both values"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                tool_call("User", '{"name":3}', "call_user").model_dump(),
                tool_call("Search", '{"query":4}', "call_search").model_dump(),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_user",
            "name": "User",
            "content": (
                "Validation Error found:\ninvalid fields\n"
                "Recall the function correctly, fix the errors"
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "name": "Search",
            "content": (
                "Validation Error found:\ninvalid fields\n"
                "Recall the function correctly, fix the errors"
            ),
        },
    ]


def test_tools_reask_without_tool_call_adds_a_user_correction() -> None:
    # OpenAI-compatible providers that don't honor forced tool_choice (or a
    # model that answers in prose) can return a plain assistant message whose
    # tool_calls is None. reask_tools must fall back to a user correction rather
    # than iterating None and raising TypeError.
    response = chat_completion(content="The answer is many.")

    result = reask_tools(
        {"messages": [{"role": "user", "content": "How many?"}]},
        response,
        ValueError("answer must be an int"),
    )

    assert result["messages"][1]["role"] == "assistant"
    correction = result["messages"][-1]
    assert correction["role"] == "user"
    assert "answer must be an int" in correction["content"]
    assert "Recall the function correctly" in correction["content"]


def test_responses_reask_uses_legacy_arguments_without_inventing_call_details() -> None:
    response = SimpleNamespace(output=[SimpleNamespace(arguments='{"name": 3}')])

    result = reask_responses_tools(
        {"messages": []}, response, ValueError("name must be a string")
    )

    assert result["messages"] == [
        {
            "role": "user",
            "content": (
                "Validation Error found:\nname must be a string\n"
                'Recall the function correctly, fix the errors with {"name": 3}'
            ),
        }
    ]


def test_streaming_flags_ignore_none_non_classes_and_non_streaming_models() -> None:
    handler = OpenAIToolsHandler()
    iterable_user = IterableModel(User)

    handler.mark_streaming_model(None, True)
    handler.mark_streaming_model(iterable_user, False)
    handler.mark_streaming_model(User, True)

    assert handler._consume_streaming_flag(None) is False
    assert handler._consume_streaming_flag(ParallelBase(User)) is False
    assert handler._consume_streaming_flag(iterable_user) is False


def test_responses_tools_replaces_non_dict_text_format_and_preserves_options() -> None:
    original_text = {"format": "json_object", "verbosity": "low"}

    _, kwargs = OpenAIResponsesToolsHandler().prepare_request(
        User, {"text": original_text}
    )

    assert kwargs["text"] is not original_text
    assert kwargs["text"]["verbosity"] == "low"
    assert kwargs["text"]["format"] == {
        "type": "json_schema",
        "name": "User",
        "strict": True,
        "schema": kwargs["tools"][0]["parameters"],
    }


@pytest.mark.parametrize(
    ("mode", "valid_chunk", "expected"),
    [
        (
            Mode.FUNCTIONS,
            chat_chunk(
                {"function_call": {"name": "User", "arguments": '{"name":"Ada"}'}}
            ),
            ['{"name":"Ada"}'],
        ),
        (Mode.JSON, chat_chunk({"content": '{"name":"Ada"}'}), ['{"name":"Ada"}']),
        (
            Mode.JSON_SCHEMA,
            chat_chunk({"content": '{"name":"Ada"}'}),
            ['{"name":"Ada"}'],
        ),
        (
            Mode.TOOLS,
            chat_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "User", "arguments": '{"name":"Ada"}'},
                        }
                    ]
                }
            ),
            ['{"name":"Ada"}'],
        ),
        (
            Mode.MD_JSON,
            chat_chunk(
                {"content": 'Here is the result:\n```json\n{"name":"Ada"}\n```'}
            ),
            ['{"name":"Ada"}\n'],
        ),
    ],
)
def test_sync_stream_extractor_handles_chat_delta_modes_and_bad_chunks(
    mode: Mode,
    valid_chunk: ChatCompletionChunk,
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Mode, "warn_mode_functions_deprecation", staticmethod(lambda: None)
    )
    chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace())]),
        valid_chunk,
    ]

    result = list(base_handler(mode).extract_streaming_json(chunks))
    if mode is Mode.MD_JSON:
        assert json.loads("".join(result)) == {"name": "Ada"}
    else:
        assert result == expected


def test_sync_responses_stream_extractor_only_yields_argument_delta_events() -> None:
    event = ResponseFunctionCallArgumentsDeltaEvent(
        delta='{"name":"Ada"}',
        item_id="fc_1",
        output_index=0,
        sequence_number=1,
        type="response.function_call_arguments.delta",
    )

    assert list(
        base_handler(Mode.RESPONSES_TOOLS).extract_streaming_json(
            [SimpleNamespace(type="response.output_text.delta"), event]
        )
    ) == ['{"name":"Ada"}']


@pytest.mark.parametrize(
    ("mode", "delta"),
    [
        (Mode.FUNCTIONS, {"function_call": {"name": "User", "arguments": ""}}),
        (Mode.JSON, {"content": ""}),
        (Mode.TOOLS, {"tool_calls": []}),
        (
            Mode.TOOLS,
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "User", "arguments": None},
                    }
                ]
            },
        ),
        (Mode.COHERE_JSON_SCHEMA, {"content": '{"name":"ignored"}'}),
    ],
)
def test_sync_stream_extractor_ignores_empty_and_unsupported_deltas(
    mode: Mode, delta: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Mode, "warn_mode_functions_deprecation", staticmethod(lambda: None)
    )

    assert list(base_handler(mode).extract_streaming_json([chat_chunk(delta)])) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "valid_chunk", "expected"),
    [
        (
            Mode.FUNCTIONS,
            chat_chunk(
                {"function_call": {"name": "User", "arguments": '{"name":"Ada"}'}}
            ),
            ['{"name":"Ada"}'],
        ),
        (Mode.JSON, chat_chunk({"content": '{"name":"Ada"}'}), ['{"name":"Ada"}']),
        (
            Mode.TOOLS,
            chat_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "User", "arguments": '{"name":"Ada"}'},
                        }
                    ]
                }
            ),
            ['{"name":"Ada"}'],
        ),
        (
            Mode.MD_JSON,
            chat_chunk({"content": '```json\n{"name":"Ada"}\n```'}),
            ['{"name":"Ada"}\n'],
        ),
    ],
)
async def test_async_stream_extractor_handles_chat_delta_modes_and_bad_chunks(
    mode: Mode,
    valid_chunk: ChatCompletionChunk,
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Mode, "warn_mode_functions_deprecation", staticmethod(lambda: None)
    )
    chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace())]),
        valid_chunk,
    ]

    result = [
        part
        async for part in base_handler(mode).extract_streaming_json_async(
            async_items(chunks)
        )
    ]
    if mode is Mode.MD_JSON:
        assert json.loads("".join(result)) == {"name": "Ada"}
    else:
        assert result == expected


@pytest.mark.asyncio
async def test_async_responses_stream_extractor_only_yields_argument_delta_events() -> (
    None
):
    event = ResponseFunctionCallArgumentsDeltaEvent(
        delta='{"name":"Ada"}',
        item_id="fc_1",
        output_index=0,
        sequence_number=1,
        type="response.function_call_arguments.delta",
    )

    assert [
        part
        async for part in base_handler(
            Mode.RESPONSES_TOOLS
        ).extract_streaming_json_async(
            async_items([SimpleNamespace(type="response.output_text.delta"), event])
        )
    ] == ['{"name":"Ada"}']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "delta"),
    [
        (Mode.FUNCTIONS, {"function_call": {"name": "User", "arguments": ""}}),
        (Mode.JSON, {"content": ""}),
        (Mode.TOOLS, {"tool_calls": []}),
        (
            Mode.TOOLS,
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "User", "arguments": None},
                    }
                ]
            },
        ),
        (Mode.COHERE_JSON_SCHEMA, {"content": '{"name":"ignored"}'}),
    ],
)
async def test_async_stream_extractor_ignores_empty_and_unsupported_deltas(
    mode: Mode, delta: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Mode, "warn_mode_functions_deprecation", staticmethod(lambda: None)
    )

    assert [
        part
        async for part in base_handler(mode).extract_streaming_json_async(
            async_items([chat_chunk(delta)])
        )
    ] == []


def test_parse_streaming_response_forwards_context_strict_and_mode() -> None:
    received: dict[str, Any] = {}

    class StreamingUser(BaseModel):
        @classmethod
        def from_streaming_response(
            cls,
            response: list[Any],
            *,
            stream_extractor: Any,
            mode: Mode,
            **kwargs: Any,
        ) -> Iterable[User]:
            received.update(
                response=response, extractor=stream_extractor, mode=mode, **kwargs
            )
            return iter([User(name="Ada")])

    response = [chat_chunk({"content": '{"name":"Ada"}'})]
    result = base_handler(Mode.JSON)._parse_streaming_response(
        StreamingUser,
        response,
        validation_context={"source": "chat"},
        strict=True,
    )

    assert result == [User(name="Ada")]
    assert received["response"] is response
    assert received["mode"] is Mode.JSON
    assert received["context"] == {"source": "chat"}
    assert received["strict"] is True
    assert callable(received["extractor"])


def test_parse_streaming_response_falls_back_when_signature_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StreamingUser(BaseModel):
        @classmethod
        def from_streaming_response(
            cls, _response: list[Any], *, stream_extractor: Any, **kwargs: Any
        ) -> Iterable[User]:
            assert "mode" not in kwargs
            assert callable(stream_extractor)
            return iter([User(name="Ada")])

    original_signature = handlers.inspect.signature

    def unavailable_signature(value: Any) -> Any:
        if getattr(value, "__name__", "") == "from_streaming_response":
            raise ValueError("extension method has no signature")
        return original_signature(value)

    monkeypatch.setattr(handlers.inspect, "signature", unavailable_signature)

    assert base_handler(Mode.JSON)._parse_streaming_response(
        StreamingUser, [], validation_context=None, strict=None
    ) == [User(name="Ada")]


@pytest.mark.asyncio
async def test_parse_streaming_response_returns_async_iterable_results() -> None:
    iterable_user = IterableModel(User)
    chunks = async_items([chat_chunk({"content": '{"tasks":[{"name":"Ada"}]}'})])

    result = base_handler(Mode.JSON)._parse_streaming_response(
        iterable_user, chunks, validation_context=None, strict=None
    )

    assert [item async for item in result] == [User(name="Ada")]


def test_finalize_parsed_result_handles_parallel_iterable_adapter_and_base_model() -> (
    None
):
    handler = OpenAIToolsHandler()
    response = chat_completion(content="unused")
    iterable_user = IterableModel(User)
    adapter = cast(type[BaseModel], ModelAdapter[str])
    parallel = ParallelBase(User, Search)

    assert handler._finalize_parsed_result(
        iterable_user,
        response,
        iterable_user(tasks=[User(name="Ada")]),
    ) == [User(name="Ada")]
    assert handler._finalize_parsed_result(parallel, response, ("kept",)) == ("kept",)
    assert (
        handler._finalize_parsed_result(
            adapter, response, adapter.model_validate({"content": "ok"})
        )
        == "ok"
    )
    parsed_user = User(name="Ada")
    assert handler._finalize_parsed_result(User, response, parsed_user) is parsed_user
    assert parsed_user._raw_response is response
    assert handler._finalize_parsed_result(User, response, "already parsed") == (
        "already parsed"
    )


def test_extract_tool_call_json_supports_legacy_and_serializable_arguments() -> None:
    handler = OpenAIToolsHandler()
    legacy = chat_completion(function_call=("User", '{"name":"Ada"}'))
    dict_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    refusal=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(arguments={"name": "Ada"})
                        )
                    ],
                )
            )
        ]
    )
    list_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    refusal=None,
                    tool_calls=[
                        SimpleNamespace(function=SimpleNamespace(arguments=["Ada"]))
                    ],
                )
            )
        ]
    )

    assert handler._extract_tool_call_json(legacy) == '{"name":"Ada"}'
    assert json.loads(handler._extract_tool_call_json(dict_response)) == {"name": "Ada"}
    assert json.loads(handler._extract_tool_call_json(list_response)) == ["Ada"]


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        (
            chat_completion(refusal="policy"),
            AssertionError,
            "Unable to generate a response due to policy",
        ),
        (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            refusal=None,
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(arguments=None)
                                )
                            ],
                        )
                    )
                ]
            ),
            ResponseParsingError,
            "Tool call arguments missing in response",
        ),
        (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            refusal=None,
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(arguments=object())
                                )
                            ],
                        )
                    )
                ]
            ),
            ResponseParsingError,
            "Tool call arguments must be JSON-serializable",
        ),
        (
            chat_completion(content="no tool call", tool_calls=[]),
            ResponseParsingError,
            "No tool calls or function call found in response",
        ),
    ],
)
def test_extract_tool_call_json_reports_refusals_and_malformed_responses(
    response: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        OpenAIToolsHandler()._extract_tool_call_json(response)


def test_tools_prepare_and_parse_parallel_calls_and_reject_unknown_tool() -> None:
    handler = OpenAIToolsHandler()
    response_model = Iterable[Union[User, Search]]

    prepared_model, kwargs = handler.prepare_request(response_model, {"messages": []})
    response = chat_completion(
        tool_calls=[
            tool_call("User", '{"name":"Ada"}', "call_user"),
            tool_call("Search", '{"query":"python"}', "call_search"),
        ]
    )

    assert prepared_model is response_model
    assert kwargs["tool_choice"] == "auto"
    assert {tool["function"]["name"] for tool in kwargs["tools"]} == {"User", "Search"}
    assert list(
        handler.parse_response(
            response,
            prepared_model,
            validation_context={"source": "parallel"},
            strict=True,
        )
    ) == [User(name="Ada"), Search(query="python")]
    unknown = chat_completion(
        tool_calls=[tool_call("Unrelated", "{}", "call_unknown")]
    )
    with pytest.raises(ResponseParsingError, match="Unrelated"):
        list(handler.parse_response(unknown, prepared_model))


def test_tools_prepare_strict_schema_and_parse_incomplete_output() -> None:
    handler = OpenAIToolsHandler()

    prepared_model, kwargs = handler.prepare_request(User, {"strict": True})

    assert prepared_model.__name__ == "User"
    assert prepared_model.model_fields["name"].annotation is str
    assert kwargs["tools"][0]["function"]["strict"] is True
    assert "strict" not in kwargs
    with pytest.raises(IncompleteOutputException):
        handler.parse_response(
            chat_completion(
                tool_calls=[tool_call("User", '{"name":"Ada"}')], finish_reason="length"
            ),
            User,
        )


def test_tools_handler_reports_an_empty_choices_response() -> None:
    response = chat_completion(
        tool_calls=[tool_call("User", '{"name":"Ada"}')]
    ).model_copy(update={"choices": []})

    with pytest.raises(ResponseParsingError, match="No choices in OpenAI response") as (
        error
    ):
        OpenAIToolsHandler().parse_response(response, User)

    assert error.value.mode == Mode.TOOLS.value
    assert error.value.raw_response is response


@pytest.mark.parametrize(
    "handler",
    [OpenAIJSONSchemaHandler(), OpenAIJSONHandler(), OpenAIMDJSONHandler()],
)
def test_json_handlers_reject_incomplete_output(handler: OpenAIHandlerBase) -> None:
    with pytest.raises(IncompleteOutputException):
        handler.parse_response(
            chat_completion(content="{", finish_reason="length"), User
        )


@pytest.mark.parametrize(
    "handler",
    [OpenAIJSONSchemaHandler(), OpenAIJSONHandler(), OpenAIMDJSONHandler()],
)
def test_json_handlers_report_an_empty_choices_response(
    handler: OpenAIHandlerBase,
) -> None:
    response = chat_completion(content='{"name":"Ada"}').model_copy(
        update={"choices": []}
    )

    with pytest.raises(ResponseParsingError, match="No choices in OpenAI response") as (
        error
    ):
        handler.parse_response(response, User)

    assert error.value.mode == handler.mode.value
    assert error.value.raw_response is response


def test_json_schema_stream_parses_iterable_and_partial_models() -> None:
    handler = OpenAIJSONSchemaHandler()
    iterable_user = IterableModel(User)
    partial_user = Partial[User]

    iterable_result = handler.parse_response(
        [chat_chunk({"content": '{"tasks":[{"name":"Ada"}]}'})],
        iterable_user,
        stream=True,
    )
    partial_result = handler.parse_response(
        [chat_chunk({"content": '{"name":"Ada"}'})], partial_user, stream=True
    )

    assert list(iterable_result) == [User(name="Ada")]
    assert partial_result[-1] == User(name="Ada")


@pytest.mark.parametrize("handler", [OpenAIJSONHandler(), OpenAIMDJSONHandler()])
def test_json_prompt_handlers_extend_list_system_content_and_create_system_message(
    handler: OpenAIHandlerBase,
) -> None:
    list_messages = [
        {"role": "system", "content": [{"type": "text", "text": "Extract users."}]}
    ]
    string_messages = [{"role": "system", "content": "Extract users."}]

    _, with_list = handler.prepare_request(User, {"messages": list_messages})
    _, with_string = handler.prepare_request(User, {"messages": string_messages})
    _, without_messages = handler.prepare_request(User, {"messages": []})

    assert "Extract users." in with_list["messages"][0]["content"][0]["text"]
    assert "json_schema" in with_list["messages"][0]["content"][0]["text"]
    assert "Extract users." in with_string["messages"][0]["content"]
    assert "json_schema" in with_string["messages"][0]["content"]
    assert without_messages["messages"][0]["role"] == "system"
    assert "User" in without_messages["messages"][0]["content"]
    if isinstance(handler, OpenAIMDJSONHandler):
        assert "```json codeblock" in with_list["messages"][-1]["content"][0]["text"]


@pytest.mark.parametrize(
    ("handler", "content"),
    [
        (OpenAIJSONHandler(), '{"tasks":[{"name":"Ada"}]}'),
        (OpenAIMDJSONHandler(), '```json\n{"tasks":[{"name":"Ada"}]}\n```'),
    ],
)
def test_json_prompt_handlers_consume_registered_streaming_model(
    handler: OpenAIHandlerBase, content: str
) -> None:
    iterable_user = IterableModel(User)
    _, kwargs = handler.prepare_request(iterable_user, {"stream": True})

    parsed = handler.parse_response([chat_chunk({"content": content})], iterable_user)

    assert kwargs["stream"] is True
    assert list(parsed) == [User(name="Ada")]
    assert handler._consume_streaming_flag(iterable_user) is False


def test_parallel_tools_prepare_handles_none_streaming_and_model_union() -> None:
    handler = OpenAIParallelToolsHandler()
    response_model = cast(type[BaseModel], Iterable[Union[User, Search]])

    assert handler.prepare_request(None, {"messages": []}) == (None, {"messages": []})
    with pytest.raises(ConfigurationError, match="stream=True is not supported"):
        handler.prepare_request(response_model, {"stream": True})
    prepared, kwargs = handler.prepare_request(response_model, {"messages": []})

    assert isinstance(prepared, ParallelBase)
    assert prepared.registry == {"User": User, "Search": Search}
    assert kwargs["tool_choice"] == "auto"
    assert {tool["function"]["name"] for tool in kwargs["tools"]} == {"User", "Search"}


def test_parallel_tools_parse_valid_calls_and_report_empty_or_incomplete_output() -> (
    None
):
    handler = OpenAIParallelToolsHandler()
    response_model = cast(type[BaseModel], Iterable[Union[User, Search]])
    valid = chat_completion(
        tool_calls=[
            tool_call("User", '{"name":"Ada"}', "call_user"),
            tool_call("Search", '{"query":"python"}', "call_search"),
        ]
    )

    assert list(
        handler.parse_response(
            valid,
            response_model,
            validation_context={"source": "parallel"},
            strict=True,
        )
    ) == [User(name="Ada"), Search(query="python")]
    unknown = chat_completion(
        tool_calls=[tool_call("Unrelated", "{}", "call_unknown")]
    )
    with pytest.raises(ResponseParsingError, match="Unrelated"):
        list(handler.parse_response(unknown, response_model))
    with pytest.raises(ResponseParsingError, match="No tool calls in response"):
        handler.parse_response(chat_completion(tool_calls=[]), response_model)
    empty_response = chat_completion(tool_calls=[]).model_copy(update={"choices": []})
    with pytest.raises(ResponseParsingError, match="No choices in OpenAI response") as (
        error
    ):
        handler.parse_response(empty_response, response_model)
    assert error.value.mode == Mode.PARALLEL_TOOLS.value
    assert error.value.raw_response is empty_response
    with pytest.raises(IncompleteOutputException):
        handler.parse_response(
            chat_completion(tool_calls=[], finish_reason="length"), response_model
        )


def test_responses_tools_converts_max_tokens_and_falls_back_to_chat_tool_call() -> None:
    handler = OpenAIResponsesToolsHandler()

    prepared, kwargs = handler.prepare_request(User, {"max_tokens": 64})
    assert prepared is not None
    parsed = handler.parse_response(
        chat_completion(tool_calls=[tool_call("User", '{"name":"Ada"}')]),
        prepared,
        validation_context={"source": "fallback"},
        strict=True,
    )

    assert prepared.__name__ == "User"
    assert prepared.model_fields["name"].annotation is str
    assert kwargs["max_output_tokens"] == 64
    assert "max_tokens" not in kwargs
    assert parsed.model_dump() == {"name": "Ada"}
    assert parsed._raw_response.choices[0].message.tool_calls[0].function.name == "User"


def test_responses_tools_falls_back_when_output_is_empty_or_has_no_tool_call() -> None:
    handler = OpenAIResponsesToolsHandler()
    message = ResponseOutputMessage(
        id="msg_1",
        content=[
            ResponseOutputText(
                annotations=[],
                text="No structured output was emitted.",
                type="output_text",
            )
        ],
        role="assistant",
        status="completed",
        type="message",
    )

    for output in ([], [message]):
        response = chat_completion(
            tool_calls=[tool_call("User", '{"name":"Ada"}')]
        ).model_copy(update={"output": output})

        parsed = handler.parse_response(response, User)

        assert parsed == User(name="Ada")
        assert parsed._raw_response is response


def test_responses_tools_skips_an_empty_call_and_parses_the_next_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    empty = ResponseFunctionToolCall(
        arguments="",
        call_id="call_empty",
        name="User",
        type="function_call",
    )
    valid = ResponseFunctionToolCall(
        arguments='{"name":"Ada"}',
        call_id="call_valid",
        name="User",
        type="function_call",
    )
    response = SimpleNamespace(output=[empty, valid])

    with caplog.at_level("WARNING", logger="instructor"):
        parsed = OpenAIResponsesToolsHandler().parse_response(response, User)

    assert parsed == User(name="Ada")
    assert parsed._raw_response is response
    assert len(caplog.records) == 1
    assert "tool 'User' returned empty arguments" in caplog.records[0].message


def test_responses_tools_streams_iterable_and_partial_models() -> None:
    iterable_user = IterableModel(User)
    partial_user = Partial[User]

    iterable_events = response_deltas("fc_iterable", '{"tasks":[', '{"name":"Ada"}]}')
    partial_events = response_deltas("fc_partial", '{"name":', '"Ada"}')

    handler = OpenAIResponsesToolsHandler()
    iterable_result = handler.parse_response(
        iterable_events, iterable_user, stream=True
    )
    partial_result = handler.parse_response(partial_events, partial_user, stream=True)

    assert list(iterable_result) == [User(name="Ada")]
    assert partial_result[-1] == User(name="Ada")


@pytest.mark.asyncio
async def test_responses_tools_streams_async_iterable_and_partial_models() -> None:
    iterable_user = IterableModel(User)
    partial_user = Partial[User]

    iterable_events = response_deltas("fc_iterable", '{"tasks":[', '{"name":"Ada"}]}')
    partial_events = response_deltas("fc_partial", '{"name":', '"Ada"}')

    handler = OpenAIResponsesToolsHandler()
    iterable_result = handler.parse_response(
        async_items(iterable_events), iterable_user, stream=True
    )
    partial_result = handler.parse_response(
        async_items(partial_events), partial_user, stream=True
    )

    assert [item async for item in iterable_result] == [User(name="Ada")]
    assert [item async for item in partial_result][-1] == User(name="Ada")
