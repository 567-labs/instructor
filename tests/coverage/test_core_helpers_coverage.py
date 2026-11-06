"""Coverage for shared v2 JSON, message, mode, template, and usage helpers."""

from __future__ import annotations

import builtins
import json
import logging
import threading
import warnings
from typing import Any, cast

import pytest
import vertexai.generative_models as gm
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message import FunctionCall
from pydantic import BaseModel, ValidationError

from instructor.v2.core import mode as mode_module
from instructor.v2.core import usage, utils
from instructor.v2.core.json import (
    extract_json_from_codeblock,
    extract_json_from_stream,
    extract_json_from_stream_async,
)
from instructor.v2.core.messages import dump_message, merge_consecutive_messages
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.templating import handle_templating, process_message

pytestmark = pytest.mark.unit


def test_extract_json_from_codeblock_recovers_after_unclosed_object() -> None:
    content = 'broken {"incomplete": true then {"name":"Ada"} suffix'

    assert extract_json_from_codeblock(content) == '{"name":"Ada"}'


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["prose with `one tick ", '{"value":1}'], '{"value":1}'),
        (["```json\n`not a fence\n", '{"value":1}'], '{"value":1}'),
        (["```json\n", '{"value":1', "\n```"], '{"value":1\n'),
        (["```json\n", '{"value":1', "`continued", "}"], '{"value":1continued}'),
        (["prefix ", '{"value":'], '{"value":'),
    ],
)
def test_extract_json_from_stream_handles_partial_fences_and_incomplete_json(
    chunks: list[str], expected: str
) -> None:
    assert "".join(extract_json_from_stream(chunks)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (["prose with `one tick ", '{"value":1}'], '{"value":1}'),
        (["```json\n`not a fence\n", '{"value":1}'], '{"value":1}'),
        (["```json\n", '{"value":1', "\n```"], '{"value":1\n'),
        (["```json\n", '{"value":1', "`continued", "}"], '{"value":1continued}'),
        (["prefix ", '{"value":'], '{"value":'),
    ],
)
async def test_extract_json_from_stream_async_handles_partial_fences_and_incomplete_json(
    parts: list[str], expected: str
) -> None:
    async def chunks():
        for part in parts:
            yield part

    assert "".join(
        [part async for part in extract_json_from_stream_async(chunks())]
    ) == (expected)


def test_dump_message_joins_text_and_refusal_content_before_function_call() -> None:
    message = ChatCompletionMessage.model_construct(
        role="assistant",
        content=[
            {"type": "text", "text": "Looking up the account. "},
            {"type": "refusal", "refusal": "Private fields are hidden. "},
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            "ignored",
        ],
        function_call=FunctionCall(name="lookup", arguments='{"id":7}'),
        tool_calls=None,
    )

    with pytest.warns(UserWarning, match="Pydantic serializer warnings:"):
        result = dump_message(message)

    assert result["role"] == "assistant"
    assert result["content"] == (
        "Looking up the account. Private fields are hidden. "
        + json.dumps({"arguments": '{"id":7}', "name": "lookup"})
    )
    assert "tool_calls" not in result


def test_dump_message_appends_function_call_to_text_content() -> None:
    message = ChatCompletionMessage(
        role="assistant",
        content="Looking up the account. ",
        function_call=FunctionCall(name="lookup", arguments='{"id":7}'),
    )

    result = dump_message(message)

    assert result["content"] == (
        "Looking up the account. "
        + json.dumps({"arguments": '{"id":7}', "name": "lookup"})
    )


def test_dump_message_retains_function_call_when_content_is_empty() -> None:
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        function_call=FunctionCall(name="lookup", arguments='{"id":7}'),
    )

    result = dump_message(message)

    assert result["content"] == json.dumps({"arguments": '{"id":7}', "name": "lookup"})


def test_merge_consecutive_messages_checks_tail_for_non_string_content() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"line {index}"} for index in range(10)
    ]
    messages.append({"role": "user", "content": [{"type": "text", "text": "tail"}]})

    assert merge_consecutive_messages(messages) == [
        {
            "role": "user",
            "content": [
                *[{"type": "text", "text": f"line {index}"} for index in range(10)],
                {"type": "text", "text": "tail"},
            ],
        }
    ]


def test_merge_consecutive_messages_keeps_single_non_list_content_piece() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "describe this"}]},
        {"role": "user", "content": {"type": "image", "source": "image-1"}},
    ]

    assert merge_consecutive_messages(messages) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image", "source": "image-1"},
            ],
        }
    ]


def test_mode_groups_keep_tool_and_json_modes_distinct() -> None:
    tool_modes = Mode.tool_modes()
    json_modes = Mode.json_modes()

    assert {Mode.TOOLS, Mode.RESPONSES_TOOLS, Mode.ANTHROPIC_PARALLEL_TOOLS} <= (
        tool_modes
    )
    assert {Mode.JSON, Mode.JSON_SCHEMA, Mode.ANTHROPIC_JSON} <= json_modes
    assert Mode.JSON not in tool_modes
    assert Mode.TOOLS not in json_modes
    assert Mode.OPENROUTER_STRUCTURED_OUTPUTS in tool_modes & json_modes


def test_legacy_mode_warnings_are_emitted_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mode_module, "_functions_deprecation_shown", False)
    monkeypatch.setattr(mode_module, "_reasoning_tools_deprecation_shown", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Mode.warn_mode_functions_deprecation()
        Mode.warn_mode_functions_deprecation()
        Mode.warn_anthropic_reasoning_tools_deprecation()
        Mode.warn_anthropic_reasoning_tools_deprecation()

    assert [item.category for item in caught] == [
        DeprecationWarning,
        DeprecationWarning,
    ]
    assert "FUNCTIONS mode is deprecated" in str(caught[0].message)
    assert "Use Mode.ANTHROPIC_TOOLS with thinking" in str(caught[1].message)


def test_provider_mode_warning_ignores_core_mode_and_resets() -> None:
    mode_module.reset_deprecated_mode_warnings()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Mode.warn_deprecated_mode(Mode.TOOLS)
            Mode.warn_deprecated_mode(Mode.CEREBRAS_JSON)
            Mode.warn_deprecated_mode(Mode.CEREBRAS_JSON)
            mode_module.reset_deprecated_mode_warnings()
            Mode.warn_deprecated_mode(Mode.CEREBRAS_JSON)

        assert len(caught) == 2
        assert all(item.category is DeprecationWarning for item in caught)
        assert all(
            "Mode.CEREBRAS_JSON is deprecated" in str(item.message) for item in caught
        )
        assert all("Use Mode.MD_JSON instead" in str(item.message) for item in caught)
    finally:
        mode_module.reset_deprecated_mode_warnings()


def test_process_message_templates_vertex_text_and_preserves_binary_part() -> None:
    message = gm.Content(
        role="user",
        parts=[
            gm.Part.from_text("Hello {{ name }}"),
            gm.Part.from_data(b"raw-image", mime_type="application/octet-stream"),
        ],
    )

    with pytest.warns(
        DeprecationWarning,
        match="The argument `including_default_value_fields` has been removed",
    ):
        result = process_message(
            cast(dict[str, Any], message), {"name": "Ada"}, Provider.VERTEXAI
        )

    assert isinstance(result, gm.Content)
    assert result.role == "user"
    assert result.parts[0].text == "Hello Ada"
    assert result.parts[1].mime_type == "application/octet-stream"
    assert result.parts[1].inline_data.data == b"raw-image"


def test_process_message_leaves_unknown_message_shape_unchanged() -> None:
    message = {"role": "user", "attachments": [{"id": "file-1"}]}

    assert process_message(message, {"name": "Ada"}, Provider.OPENAI) is message


@pytest.mark.parametrize("kwargs", [[], {}, {"messages": []}, {"contents": []}])
def test_handle_templating_accepts_empty_message_shapes(kwargs: Any) -> None:
    result = handle_templating(
        kwargs,
        Mode.TOOLS,
        provider=Provider.OPENAI,
        context={"name": "Ada"},
    )

    assert result == kwargs
    assert result is not kwargs


def test_handle_templating_uses_contents_when_messages_are_empty() -> None:
    kwargs = {
        "messages": [],
        "contents": [{"role": "user", "content": "Hello {{ name }}"}],
    }

    result = handle_templating(
        kwargs,
        Mode.TOOLS,
        provider=Provider.OPENAI,
        context={"name": "Ada"},
    )

    assert result == {
        "messages": [],
        "contents": [{"role": "user", "content": "Hello Ada"}],
    }
    assert kwargs["contents"][0]["content"] == "Hello {{ name }}"


def test_handle_templating_preserves_uncopyable_metadata_and_nested_input() -> None:
    guard = threading.Lock()
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": "Hello {{ name }}",
                "metadata": {"guard": guard},
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Nested {{ name }}",
                        "metadata": {"guard": guard},
                    },
                    guard,
                ],
            },
            {"role": "user", "parts": ["Part {{ name }}", guard]},
        ]
    }

    result = handle_templating(
        kwargs,
        Mode.TOOLS,
        provider=Provider.OPENAI,
        context={"name": "Ada"},
    )

    assert result["messages"][0]["content"] == "Hello Ada"
    assert result["messages"][0]["metadata"]["guard"] is guard
    assert result["messages"][1]["content"][0]["text"] == "Nested Ada"
    assert result["messages"][1]["content"][0]["metadata"]["guard"] is guard
    assert result["messages"][1]["content"][1] is guard
    assert result["messages"][2]["parts"] == ["Part Ada", guard]
    assert kwargs["messages"][0]["content"] == "Hello {{ name }}"
    original_content = kwargs["messages"][1]["content"]
    assert isinstance(original_content[0], dict)
    assert original_content[0]["text"] == "Nested {{ name }}"
    assert kwargs["messages"][2]["parts"][0] == "Part {{ name }}"


def test_handle_templating_does_not_mutate_cohere_chat_history() -> None:
    guard = threading.Lock()
    kwargs = {
        "message": "Hello {{ name }}",
        "chat_history": [
            {"message": "Previous {{ name }}", "metadata": {"guard": guard}}
        ],
    }

    result = handle_templating(
        kwargs,
        Mode.TOOLS,
        provider=Provider.COHERE,
        context={"name": "Ada"},
    )

    assert result["message"] == "Hello Ada"
    assert result["chat_history"][0]["message"] == "Previous Ada"
    assert result["chat_history"][0]["metadata"]["guard"] is guard
    assert kwargs["message"] == "Hello {{ name }}"
    assert kwargs["chat_history"][0]["message"] == "Previous {{ name }}"


def test_update_total_usage_accepts_missing_response() -> None:
    total_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    assert usage.update_total_usage(None, total_usage) is None


def test_update_total_usage_tolerates_missing_optional_anthropic_support(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Response:
        usage = {"input_tokens": 2, "output_tokens": 3}

    real_import = builtins.__import__

    def import_without_anthropic_usage(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "instructor.v2.providers.anthropic.usage":
            raise ImportError("anthropic usage support is unavailable")
        return real_import(name, *args, **kwargs)

    response = Response()
    monkeypatch.setattr(builtins, "__import__", import_without_anthropic_usage)
    caplog.set_level(logging.DEBUG, logger="instructor")

    total_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    assert usage.update_total_usage(response, total_usage) is response
    assert "No compatible response.usage found" in caplog.text


def test_disable_pydantic_error_url_removes_only_help_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RequiresCount(BaseModel):
        count: int

    original_str = utils._validation_error_original_str or ValidationError.__str__
    monkeypatch.setattr(ValidationError, "__str__", original_str)
    monkeypatch.setattr(utils, "_validation_error_original_str", None)

    with pytest.raises(ValidationError) as caught:
        RequiresCount.model_validate({"count": "not-a-number"})

    before = str(caught.value)
    assert "https://errors.pydantic.dev" in before
    assert "count" in before

    utils.disable_pydantic_error_url()
    utils.disable_pydantic_error_url()

    after = str(caught.value)
    assert "https://errors.pydantic.dev" not in after
    assert "count" in after
    assert "valid integer" in after
