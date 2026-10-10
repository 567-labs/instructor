from __future__ import annotations

import builtins
import inspect
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, Literal, Union, cast

import anthropic
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from pydantic import BaseModel, PrivateAttr, ValidationError

from instructor.v2.core.errors import (
    ConfigurationError,
    IncompleteOutputException,
    ResponseParsingError,
)
from instructor.v2.core.mode import Mode
from instructor.v2.core.multimodal import Audio, Image, PDF
from instructor.v2.dsl.iterable import IterableModel
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial
from instructor.v2.dsl.simple_type import ModelAdapter
from instructor.v2.providers.anthropic import handlers
from instructor.v2.providers.anthropic.handlers import (
    AnthropicJSONHandler,
    AnthropicParallelToolsHandler,
    AnthropicStructuredOutputsHandler,
    AnthropicToolsHandler,
    SystemMessage,
    combine_system_messages,
    extract_system_messages,
    process_messages_for_anthropic,
    serialize_message_content,
)
from tests.coverage._openai import chat_completion
from tests.coverage._streams import async_items


class User(BaseModel):
    name: str
    _raw_response: Any = PrivateAttr(default=None)


class Job(BaseModel):
    title: str


def message(
    *content: Any,
    stop_reason: Literal[
        "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
    ] = "end_turn",
) -> Message:
    return Message(
        id="msg_local",
        content=list(content),
        model="claude-sonnet-4-20250514",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=11, output_tokens=7),
    )


def tool(
    name: str, value: dict[str, Any], tool_id: str = "toolu_local"
) -> ToolUseBlock:
    return ToolUseBlock(id=tool_id, input=value, name=name, type="tool_use")


def chunk(**delta: Any) -> SimpleNamespace:
    return SimpleNamespace(delta=SimpleNamespace(**delta))


def test_system_messages_combine_all_supported_shapes_and_reject_invalid() -> None:
    text: SystemMessage = {"type": "text", "text": "second"}

    assert combine_system_messages(None, "first") == "first"
    assert combine_system_messages("first", "second") == "first\n\nsecond"
    assert combine_system_messages([text], [text]) == [text, text]
    assert combine_system_messages("first", [text]) == [
        {"type": "text", "text": "first"},
        text,
    ]
    assert combine_system_messages([text], "last") == [
        text,
        {"type": "text", "text": "last"},
    ]

    with pytest.raises(ValueError, match="System messages must be strings or lists"):
        combine_system_messages(cast(Any, 1), "valid")
    with pytest.raises(ValueError, match="System messages must be strings or lists"):
        combine_system_messages(None, cast(Any, 1))


def test_extract_system_messages_handles_empty_blocks_and_reports_bad_content() -> None:
    assert extract_system_messages([]) == []
    assert extract_system_messages([{"role": "user", "content": "hi"}]) == []
    assert extract_system_messages(
        [
            {"role": "system", "content": None},
            {"role": "system", "content": "one"},
            {
                "role": "system",
                "content": [None, {"type": "text", "text": "two"}, "three"],
            },
        ]
    ) == [
        {"type": "text", "text": "one"},
        {"type": "text", "text": "two"},
        {"type": "text", "text": "three"},
    ]

    with pytest.raises(ValueError, match="Unsupported content type"):
        extract_system_messages([{"role": "system", "content": 4}])


def test_message_serialization_preserves_anthropic_blocks_and_multimodal_content() -> (
    None
):
    image = Image(source="raw-image", media_type="image/png", data="aW1hZ2U=")
    pdf = PDF(source="raw-pdf", data="cGRm")
    remote_audio = Audio(
        source="https://example.test/sample.wav", media_type="audio/wav"
    )
    local_audio = Audio(source="encoded-audio", media_type="audio/wav", data="YXVkaW8=")
    user = User(name="Ada")
    already_serialized = {"type": "text", "text": "leave me alone"}

    serialized = serialize_message_content(
        [
            image,
            pdf,
            remote_audio,
            local_audio,
            "hello",
            already_serialized,
            {"user": user},
            3,
        ]
    )

    assert serialized[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="},
    }
    assert serialized[1] == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "cGRm"},
    }
    assert serialized[2] == {
        "type": "audio",
        "source": {"type": "url", "url": "https://example.test/sample.wav"},
    }
    assert serialized[3] == {
        "type": "audio",
        "source": {"type": "base64", "media_type": "audio/wav", "data": "YXVkaW8="},
    }
    assert serialized[4:] == [
        {"type": "text", "text": "hello"},
        already_serialized,
        {"user": {"name": "Ada"}},
        3,
    ]

    source = [
        {"role": "user", "content": ["hello", user]},
        {"role": "assistant", "content": user},
        {"role": "user", "content": "plain"},
        {"role": "assistant"},
    ]
    processed = process_messages_for_anthropic(source)
    assert processed == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}, {"name": "Ada"}],
        },
        {"role": "assistant", "content": {"name": "Ada"}},
        {"role": "user", "content": "plain"},
        {"role": "assistant"},
    ]
    assert source[0]["content"][0] == "hello"


def test_output_format_detection_handles_old_new_and_uninspectable_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthropic.resources.messages import Messages

    assert handlers._anthropic_supports_output_format() is False

    def create_with_output_format(self: Any, *, output_format: Any = None) -> Any:
        return self, output_format

    monkeypatch.setattr(Messages, "create", create_with_output_format)
    assert handlers._anthropic_supports_output_format() is True

    monkeypatch.setattr(
        handlers.inspect,
        "signature",
        lambda _value: (_ for _ in ()).throw(ValueError("signature unavailable")),
    )
    assert handlers._anthropic_supports_output_format() is False

    real_import = builtins.__import__

    def missing_messages(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic.resources.messages":
            raise ImportError("anthropic messages are unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_messages)
    assert handlers._anthropic_supports_output_format() is False


def test_stream_extractors_accept_sdk_shaped_deltas_and_skip_unrelated_events() -> None:
    tool_handler = AnthropicToolsHandler()
    json_handler = AnthropicJSONHandler()

    assert list(
        tool_handler.extract_streaming_json(
            [chunk(partial_json='{"name":'), object(), chunk(partial_json='"Ada"}')]
        )
    ) == ['{"name":', '"Ada"}']
    assert list(
        json_handler.extract_streaming_json(
            [chunk(text='{"name":'), chunk(text=""), object(), chunk(text='"Ada"}')]
        )
    ) == ['{"name":', '"Ada"}']

    handler = AnthropicJSONHandler()
    handler.mode = Mode.MD_JSON
    assert list(handler.extract_streaming_json([chunk(text="ignored")])) == []


@pytest.mark.asyncio
async def test_async_stream_extractors_accept_sdk_shaped_deltas_and_skip_events() -> (
    None
):
    tools = [
        value
        async for value in AnthropicParallelToolsHandler().extract_streaming_json_async(
            async_items([chunk(partial_json="["), object(), chunk(partial_json="]")])
        )
    ]
    text = [
        value
        async for value in AnthropicStructuredOutputsHandler().extract_streaming_json_async(
            async_items([chunk(text="{"), chunk(text=""), object(), chunk(text="}")])
        )
    ]

    assert tools == ["[", "]"]
    assert text == ["{", "}"]

    ignored_handler = AnthropicJSONHandler()
    ignored_handler.mode = Mode.MD_JSON
    assert [
        value
        async for value in ignored_handler.extract_streaming_json_async(
            async_items([chunk(text="ignored")])
        )
    ] == []


def test_convert_messages_uses_the_correct_anthropic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[dict[str, Any]], Mode, bool]] = []

    def convert(
        messages: list[dict[str, Any]], mode: Mode, autodetect_images: bool = False
    ) -> list[dict[str, Any]]:
        calls.append((messages, mode, autodetect_images))
        return [{"role": "user", "content": f"converted:{mode.value}"}]

    monkeypatch.setattr(handlers, "convert_messages_v1", convert)
    source = [{"role": "user", "content": "hello"}]

    assert (
        AnthropicToolsHandler()
        .convert_messages(source, True)[0]["content"]
        .endswith(Mode.ANTHROPIC_TOOLS.value)
    )
    assert (
        AnthropicJSONHandler()
        .convert_messages(source)[0]["content"]
        .endswith(Mode.ANTHROPIC_JSON.value)
    )
    assert calls == [
        (source, Mode.ANTHROPIC_TOOLS, True),
        (source, Mode.ANTHROPIC_JSON, False),
    ]


def test_streaming_flag_is_consumed_once_and_sync_models_parse_real_chunks() -> None:
    tools = AnthropicToolsHandler()
    iterable_model = IterableModel(User)
    tools.mark_streaming_model(iterable_model, True)
    parsed_iterable = tools.parse_response(
        [chunk(partial_json='{"tasks":[{"name":"Ada"},{"name":"Grace"}]}')],
        iterable_model,
        validation_context={"request": "one"},
        strict=False,
    )

    assert inspect.isgenerator(parsed_iterable)
    assert list(parsed_iterable) == [User(name="Ada"), User(name="Grace")]
    assert tools._consume_streaming_flag(iterable_model) is False
    assert tools._consume_streaming_flag(None) is False
    assert tools._consume_streaming_flag(ParallelBase(User)) is False

    json_handler = AnthropicJSONHandler()
    partial_model = Partial[User]
    json_handler._register_streaming_from_kwargs(partial_model, {"stream": True})
    parsed_partial = json_handler.parse_response(
        [chunk(text='{"name":"Ad'), chunk(text='a"}')], partial_model
    )

    assert [item.name for item in parsed_partial] == ["Ad", "Ada"]
    json_handler.mark_streaming_model(User, True)
    json_handler.mark_streaming_model(partial_model, False)
    json_handler._register_streaming_from_kwargs(None, {"stream": True})
    assert json_handler._consume_streaming_flag(User) is False


@pytest.mark.asyncio
async def test_async_streaming_parse_returns_the_model_async_generator() -> None:
    iterable_model = IterableModel(User)
    handler = AnthropicToolsHandler()
    handler.mark_streaming_model(iterable_model, True)

    result = handler.parse_response(
        async_items([chunk(partial_json='{"tasks":[{"name":"Ada"}]}')]),
        iterable_model,
        validation_context={"request": "async"},
        strict=True,
    )

    assert inspect.isasyncgen(result)
    assert [item async for item in result] == [User(name="Ada")]


def test_streaming_protocol_model_and_result_finalization_keep_expected_values() -> (
    None
):
    class StreamingProtocol(BaseModel):
        @classmethod
        def from_streaming_response(cls, response: Any, **kwargs: Any) -> Iterable[str]:
            assert response == ["chunk"]
            assert callable(kwargs["stream_extractor"])
            return iter(["one", "two"])

    handler = AnthropicJSONHandler()
    assert handler._parse_streaming_response(
        StreamingProtocol, ["chunk"], None, None
    ) == [
        "one",
        "two",
    ]

    raw = object()
    iterable_model = IterableModel(User)
    parsed_iterable = iterable_model(tasks=[User(name="Ada")])
    assert handler._finalize_parsed_result(iterable_model, raw, parsed_iterable) == [
        User(name="Ada")
    ]
    adapter_model = cast(type[BaseModel], ModelAdapter[str])
    adapter = adapter_model(content="plain value")
    assert handler._finalize_parsed_result(adapter_model, raw, adapter) == "plain value"
    parsed = User(name="Ada")
    assert handler._finalize_parsed_result(User, raw, parsed) is parsed
    assert parsed._raw_response is raw
    parallel = ParallelBase(User)
    sentinel = object()
    assert handler._finalize_parsed_result(parallel, raw, sentinel) is sentinel


def test_tools_prepare_request_serializes_messages_selects_tools_and_respects_choice() -> (
    None
):
    handler = AnthropicToolsHandler()
    request_model, request = handler.prepare_request(
        User,
        {
            "system": "existing",
            "messages": [
                {"role": "system", "content": "extract a user"},
                {"role": "user", "content": ["hello", User(name="input")]},
            ],
        },
    )

    assert request_model is not None
    assert request_model.__name__ == "User"
    assert request_model.model_fields.keys() == User.model_fields.keys()
    assert request["system"] == [
        {"type": "text", "text": "existing"},
        {"type": "text", "text": "extract a user"},
    ]
    assert request["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}, {"name": "input"}],
        }
    ]
    assert request["tools"][0]["name"] == "User"
    assert request["tool_choice"] == {
        "type": "tool",
        "name": "User",
        "disable_parallel_tool_use": True,
    }

    no_model, passthrough = handler.prepare_request(
        None,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {"type": "auto"},
        },
    )
    assert no_model is None
    assert passthrough["tool_choice"] == {"type": "auto"}

    adapted, simple = handler.prepare_request(
        cast(Any, str),
        {
            "messages": [{"role": "user", "content": "say hello"}],
            "tool_choice": {"type": "any"},
        },
    )
    assert adapted is not None
    assert adapted.__name__ == "Response"
    assert simple["tools"][0]["name"] == "Response"
    assert simple["tool_choice"] == {"type": "any"}


def test_tools_prepare_parallel_and_thinking_requests_use_auto_choice() -> None:
    parallel_type = Iterable[Union[User, Job]]
    handler = AnthropicToolsHandler()
    returned, request = handler.prepare_request(
        cast(Any, parallel_type),
        {"messages": [{"role": "user", "content": "find both"}]},
    )

    assert returned is parallel_type
    assert {schema["name"] for schema in request["tools"]} == {"User", "Job"}
    assert request["tool_choice"] == {"type": "auto"}

    _, thinking = handler.prepare_request(
        User,
        {
            "messages": [{"role": "user", "content": "think first"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )
    assert thinking["tool_choice"] == {"type": "auto"}
    assert thinking["system"] == [
        {"type": "text", "text": "Return only the tool call and no additional text."}
    ]


def test_tools_reask_handles_missing_response_text_only_and_old_sdk_blocks() -> None:
    handler = AnthropicToolsHandler()
    exception = ValueError("name is required")

    missing = handler.handle_reask(
        {"messages": [{"role": "user", "content": "extract"}]},
        cast(Any, None),
        exception,
    )
    assert missing["messages"][-1] == {
        "role": "user",
        "content": "Validation Error found:\nname is required\nRecall the function correctly, fix the errors",
    }
    no_content = handler.handle_reask({"messages": []}, cast(Any, object()), exception)
    assert "name is required" in no_content["messages"][-1]["content"]

    text_only = handler.handle_reask(
        {"messages": []}, message(TextBlock(type="text", text="not a tool")), exception
    )
    assert text_only["messages"][0]["role"] == "assistant"
    assert text_only["messages"][1]["content"].startswith(
        "Validation Error due to no tool invocation"
    )

    class OlderToolBlock:
        type = "tool_use"
        id = "toolu_old"

        def model_dump(self) -> dict[str, Any]:
            return {"type": "tool_use", "id": self.id, "name": "User", "input": {}}

    old_response = SimpleNamespace(content=[OlderToolBlock()])
    old = handler.handle_reask({"messages": []}, cast(Any, old_response), exception)
    result_block = old["messages"][-1]["content"][0]
    assert old["messages"][0]["content"][0]["id"] == "toolu_old"
    assert result_block["tool_use_id"] == "toolu_old"
    assert result_block["is_error"] is True
    assert "name is required" in result_block["content"]


def test_tools_parse_single_parallel_invalid_and_incomplete_responses() -> None:
    handler = AnthropicToolsHandler()
    response = message(
        TextBlock(type="text", text="calling tool"),
        tool("User", {"name": "Ada"}),
        stop_reason="tool_use",
    )
    parsed = handler.parse_response(
        response, User, validation_context={"source": "test"}
    )
    assert parsed.model_dump() == {"name": "Ada"}
    assert parsed._raw_response is response
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7

    parallel_response = message(
        TextBlock(type="text", text="calling multiple tools"),
        tool("User", {"name": "Ada"}, "toolu_one"),
        tool("Unknown", {"ignored": True}, "toolu_two"),
        tool("Job", {"title": "Engineer"}, "toolu_three"),
        stop_reason="tool_use",
    )
    with pytest.raises(ResponseParsingError, match="Unknown"):
        list(
            handler.parse_response(
                parallel_response, cast(Any, Iterable[Union[User, Job]])
            )
        )

    with pytest.raises(ValidationError):
        handler.parse_response(message(TextBlock(type="text", text="no tool")), User)
    with pytest.raises(IncompleteOutputException) as incomplete:
        handler.parse_response(
            message(tool("User", {"name": "Ada"}), stop_reason="max_tokens"), User
        )
    assert isinstance(incomplete.value.last_completion, Message)
    assert incomplete.value.last_completion.stop_reason == "max_tokens"


def test_parallel_tools_prepare_reask_and_parse_real_tool_blocks() -> None:
    handler = AnthropicParallelToolsHandler()
    parallel_type = Iterable[Union[User, Job]]
    returned, request = handler.prepare_request(
        cast(Any, parallel_type),
        {
            "system": "existing",
            "messages": [
                {"role": "system", "content": "extract all results"},
                {"role": "user", "content": "find both"},
            ],
        },
    )

    assert returned is parallel_type
    assert request["system"] == [
        {"type": "text", "text": "existing"},
        {"type": "text", "text": "extract all results"},
    ]
    assert request["messages"] == [{"role": "user", "content": "find both"}]
    assert {schema["name"] for schema in request["tools"]} == {"User", "Job"}
    assert request["tool_choice"] == {"type": "auto"}
    assert handler.prepare_request(None, {"messages": []}) == (None, {"messages": []})

    with pytest.raises(ConfigurationError, match="stream=True is not supported"):
        handler.prepare_request(
            cast(Any, parallel_type), {"messages": [], "stream": True}
        )

    response = message(
        TextBlock(type="text", text="two calls"),
        tool("User", {"name": "Ada"}, "toolu_one"),
        tool("Unknown", {"value": 1}, "toolu_two"),
        tool("Job", {"title": "Engineer"}, "toolu_three"),
        stop_reason="tool_use",
    )
    with pytest.raises(ResponseParsingError, match="Unknown"):
        list(handler.parse_response(response, parallel_type, strict=True))
    assert list(handler.parse_response(None, parallel_type)) == []
    assert list(handler.parse_response(object(), parallel_type)) == []
    reask = handler.handle_reask({"messages": []}, response, ValueError("bad job"))
    # Anthropic requires a tool_result for *every* tool_use block (#2485).
    reask_results = reask["messages"][-1]["content"]
    assert [block["tool_use_id"] for block in reask_results] == [
        "toolu_one",
        "toolu_two",
        "toolu_three",
    ]
    assert all(block.get("is_error") is True for block in reask_results)


def test_json_prepare_reask_and_parse_anthropic_and_openai_shaped_responses() -> None:
    handler = AnthropicJSONHandler()
    returned, request = handler.prepare_request(
        User,
        {
            "system": "existing",
            "messages": [
                {"role": "system", "content": "extract a user"},
                {"role": "user", "content": ["hello", User(name="input")]},
            ],
        },
    )
    assert returned is User
    assert request["messages"][0]["content"][1] == {"name": "input"}
    assert request["system"][0:2] == [
        {"type": "text", "text": "existing"},
        {"type": "text", "text": "extract a user"},
    ]
    assert "json_schema" in request["system"][-1]["text"]
    assert '"name"' in request["system"][-1]["text"]
    assert handler.prepare_request(None, {"messages": []}) == (None, {"messages": []})

    response = message(
        TextBlock(type="text", text='result: ```json\n{"name":"Ada"}\n```')
    )
    parsed = handler.parse_response(response, User, strict=False)
    assert parsed.model_dump() == {"name": "Ada"}
    assert parsed._raw_response is response
    assert handler.parse_response(response, User, strict=True).model_dump() == {
        "name": "Ada"
    }

    openai_response = chat_completion(content='{"name":"Grace"}')
    assert handler.parse_response(openai_response, User, strict=True).model_dump() == {
        "name": "Grace"
    }
    exhausted = chat_completion(content="{", finish_reason="length")
    with pytest.raises(IncompleteOutputException) as incomplete:
        handler.parse_response(exhausted, User)
    assert incomplete.value.last_completion is exhausted.choices[0]

    reask = handler.handle_reask({"messages": []}, response, ValueError("bad name"))
    assert "bad name" in reask["messages"][-1]["content"]
    assert 'result: ```json\n{"name":"Ada"}\n```' in reask["messages"][-1]["content"]
    no_text = handler.handle_reask(
        {"messages": []}, message(tool("User", {"name": "Ada"})), ValueError("bad name")
    )
    assert no_text["messages"][-1]["content"].endswith(
        "No text content found in response"
    )


def test_json_parse_reports_invalid_incomplete_and_textless_anthropic_responses() -> (
    None
):
    handler = AnthropicJSONHandler()

    with pytest.raises(
        ResponseParsingError, match="Response must be an Anthropic Message"
    ):
        handler.parse_response(object(), User)
    with pytest.raises(IncompleteOutputException) as incomplete:
        handler.parse_response(
            message(
                TextBlock(type="text", text='{"name":"Ada"}'), stop_reason="max_tokens"
            ),
            User,
        )
    assert isinstance(incomplete.value.last_completion, Message)
    assert incomplete.value.last_completion.stop_reason == "max_tokens"
    with pytest.raises(ResponseParsingError, match="No text content in response"):
        handler.parse_response(message(tool("User", {"name": "Ada"})), User)


def test_structured_prepare_falls_back_for_old_sdk_and_requires_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = AnthropicStructuredOutputsHandler()

    with pytest.raises(ConfigurationError, match="requires a `response_model`"):
        handler.prepare_request(None, {"messages": []})

    monkeypatch.setattr(handlers, "_anthropic_supports_output_format", lambda: False)
    with pytest.warns(UserWarning, match="falling back to JSON mode instructions"):
        returned, request = handler.prepare_request(
            User, {"messages": [{"role": "user", "content": "extract"}]}
        )
    assert returned is User
    assert "output_format" not in request
    assert "json_schema" in request["system"][0]["text"]


@pytest.mark.parametrize(
    ("betas", "expected"),
    [
        (None, ["structured-outputs-2025-11-13"]),
        ("other-beta", ["other-beta", "structured-outputs-2025-11-13"]),
        (("other-beta",), ["other-beta", "structured-outputs-2025-11-13"]),
        (
            ["structured-outputs-2025-11-13"],
            ["structured-outputs-2025-11-13"],
        ),
    ],
)
def test_structured_prepare_builds_schema_normalizes_betas_and_clears_tools(
    monkeypatch: pytest.MonkeyPatch, betas: Any, expected: list[str]
) -> None:
    monkeypatch.setattr(handlers, "_anthropic_supports_output_format", lambda: True)
    monkeypatch.setattr(
        anthropic,
        "transform_schema",
        lambda model: {
            "title": model.__name__,
            "type": "object",
            "additionalProperties": False,
        },
        raising=False,
    )
    kwargs: dict[str, Any] = {
        "system": "existing",
        "messages": [
            {"role": "system", "content": "extract a user"},
            {"role": "user", "content": User(name="input")},
        ],
        "tools": [{"name": "legacy"}],
        "tool_choice": {"type": "auto"},
    }
    if betas is not None:
        kwargs["betas"] = betas

    returned, request = AnthropicStructuredOutputsHandler().prepare_request(
        User, kwargs
    )

    assert returned is User
    assert request["messages"] == [{"role": "user", "content": {"name": "input"}}]
    assert request["system"] == [
        {"type": "text", "text": "existing"},
        {"type": "text", "text": "extract a user"},
    ]
    assert request["output_format"] == {
        "type": "json_schema",
        "schema": {"title": "User", "type": "object", "additionalProperties": False},
    }
    assert request["betas"] == expected
    assert "tools" not in request
    assert "tool_choice" not in request


def test_structured_prepare_uses_pydantic_schema_when_transform_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handlers, "_anthropic_supports_output_format", lambda: True)
    monkeypatch.delattr(anthropic, "transform_schema", raising=False)

    with pytest.warns(
        UserWarning, match="Falling back to response_model.model_json_schema"
    ):
        returned, request = AnthropicStructuredOutputsHandler().prepare_request(
            User, {"messages": [{"role": "user", "content": "extract"}]}
        )

    assert returned is User
    assert request["output_format"]["schema"] == User.model_json_schema()
    assert request["betas"] == ["structured-outputs-2025-11-13"]


def test_structured_reask_and_parse_validates_text_and_reports_refusals() -> None:
    handler = AnthropicStructuredOutputsHandler()
    response = message(
        TextBlock(type="text", text="earlier"),
        TextBlock(type="text", text='{"name":"Ada"}'),
    )

    parsed = handler.parse_response(response, User, strict=False)
    assert parsed.model_dump() == {"name": "Ada"}
    assert parsed._raw_response is response
    assert handler.parse_response(response, User, strict=True).model_dump() == {
        "name": "Ada"
    }
    reask = handler.handle_reask({"messages": []}, response, ValueError("bad name"))
    assert reask["messages"][-1]["content"].endswith('{"name":"Ada"}')

    refusal = message(tool("User", {"name": "ignored"}), stop_reason="end_turn")
    refusal_reask = handler.handle_reask(
        {"messages": []}, refusal, ValueError("refused")
    )
    assert refusal_reask["messages"][-1]["content"].endswith(
        "No text content found in response"
    )
    with pytest.raises(
        ResponseParsingError, match="Response must be an Anthropic Message"
    ):
        handler.parse_response(object(), User)
    with pytest.raises(IncompleteOutputException) as incomplete:
        handler.parse_response(
            message(
                TextBlock(type="text", text='{"name":"Ada"}'), stop_reason="max_tokens"
            ),
            User,
        )
    assert isinstance(incomplete.value.last_completion, Message)
    assert incomplete.value.last_completion.stop_reason == "max_tokens"
    with pytest.raises(
        ResponseParsingError,
        match="No text content found in structured output response",
    ):
        handler.parse_response(refusal, User)
