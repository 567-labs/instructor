from unittest.mock import MagicMock

from pydantic import BaseModel

from openai import pydantic_function_tool

from instructor.v2.providers.openai.handlers import (
    OpenAIResponsesToolsHandler,
    reask_responses_tools,
)


def _tool_parameters_schema(model: type[BaseModel]) -> dict:
    return pydantic_function_tool(model)["function"]["parameters"]


class ResponseToolModel(BaseModel):
    """Extract a structured response for the user."""

    name: str


class AlternateModel(BaseModel):
    title: str
    count: int


def test_responses_tools_preserves_function_description() -> None:
    expected_description = pydantic_function_tool(ResponseToolModel)["function"][
        "description"
    ]

    _, kwargs = OpenAIResponsesToolsHandler().prepare_request(ResponseToolModel, {})

    assert kwargs["tools"][0]["description"] == expected_description


def test_responses_tools_sets_text_format() -> None:
    _, kwargs = OpenAIResponsesToolsHandler().prepare_request(ResponseToolModel, {})

    fmt = kwargs["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "ResponseToolModel"
    assert fmt["strict"] is True
    assert fmt["schema"] == _tool_parameters_schema(ResponseToolModel)
    assert fmt["schema"].get("additionalProperties") is False


def test_responses_tools_overrides_conflicting_text_format() -> None:
    conflicting_text = {
        "format": {
            "type": "json_schema",
            "name": "AlternateModel",
            "strict": True,
            "schema": AlternateModel.model_json_schema(),
        },
        "verbosity": "low",
    }

    _, kwargs = OpenAIResponsesToolsHandler().prepare_request(
        ResponseToolModel,
        {"text": conflicting_text},
    )

    fmt = kwargs["text"]["format"]
    assert fmt["name"] == "ResponseToolModel"
    assert fmt["schema"] == _tool_parameters_schema(ResponseToolModel)
    assert kwargs["text"]["verbosity"] == "low"
    assert kwargs["text"] is not conflicting_text


def test_responses_tools_preserves_matching_text_format() -> None:
    matching_text = {
        "format": {
            "type": "json_schema",
            "name": "ResponseToolModel",
            "strict": True,
            "schema": _tool_parameters_schema(ResponseToolModel),
        }
    }

    _, kwargs = OpenAIResponsesToolsHandler().prepare_request(
        ResponseToolModel,
        {"text": matching_text},
    )

    assert kwargs["text"] is matching_text


def test_responses_tools_none_model_no_text() -> None:
    _, kwargs = OpenAIResponsesToolsHandler().prepare_request(None, {})

    assert "text" not in kwargs


def _make_mock_response(arguments: str | None) -> MagicMock:
    tool_call = MagicMock()
    tool_call.type = "function_call"
    tool_call.arguments = arguments
    tool_call.name = "ResponseToolModel"
    tool_call.id = "call_123"

    response = MagicMock()
    response.output = [tool_call]
    return response


def test_reask_responses_tools_empty_args_message() -> None:
    response = _make_mock_response("{}")
    error = ValueError(
        "1 validation error for ResponseToolModel\nname\n  Field required"
    )

    result = reask_responses_tools({"messages": []}, response, error)

    msg = result["messages"][0]["content"]
    assert "empty arguments" in msg
    assert "MUST populate ALL required fields" in msg
    assert "fix the errors with" not in msg


def test_reask_responses_tools_nonempty_args_message() -> None:
    response = _make_mock_response('{"name": 123}')
    error = ValueError(
        "1 validation error for ResponseToolModel\nname\n  Input should be a valid string"
    )

    result = reask_responses_tools({"messages": []}, response, error)

    msg = result["messages"][0]["content"]
    assert "fix the errors with" in msg
    assert '{"name": 123}' in msg


def test_reask_responses_tools_none_arguments() -> None:
    response = _make_mock_response(None)

    result = reask_responses_tools({"messages": []}, response, ValueError("required"))

    msg = result["messages"][0]["content"]
    assert "MUST populate ALL required fields" in msg


def test_reask_responses_tools_no_tool_calls_adds_fallback_message() -> None:
    """Reask must add corrective feedback even when the output has no tool calls.

    Reasoning models can return only reasoning/message items instead of the
    forced function call. Without a fallback the retry resends the identical
    request with no feedback at all.
    """
    reasoning_item = MagicMock()
    reasoning_item.type = "reasoning"

    message_item = MagicMock()
    message_item.type = "message"

    response = MagicMock()
    response.output = [reasoning_item, message_item]

    error = ValueError(
        "1 validation error for ResponseToolModel\nname\n  Field required"
    )

    result = reask_responses_tools(
        {"messages": [{"role": "user", "content": "extract"}]}, response, error
    )

    assert len(result["messages"]) == 2
    fallback = result["messages"][-1]
    assert fallback["role"] == "user"
    assert "Validation Error found" in fallback["content"]
    assert "Field required" in fallback["content"]
    assert "Recall the function correctly" in fallback["content"]


def test_responses_tools_overrides_text_type_format() -> None:
    _, kwargs = OpenAIResponsesToolsHandler().prepare_request(
        ResponseToolModel,
        {"text": {"format": {"type": "text"}}},
    )

    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["name"] == "ResponseToolModel"


def test_parse_response_warns_on_empty_args(caplog) -> None:
    import logging

    response = _make_mock_response("{}")
    response.choices = []

    with caplog.at_level(logging.WARNING, logger="instructor"):
        try:
            OpenAIResponsesToolsHandler().parse_response(response, ResponseToolModel)
        except Exception:
            pass

    assert any("empty arguments" in record.message for record in caplog.records)


def test_parse_response_rejects_incomplete_max_output_tokens_response() -> None:
    """A valid tool payload does not make a token-truncated Responses turn complete."""
    import pytest

    from instructor.v2.core.errors import IncompleteOutputException

    response = _make_mock_response('{"name": "Ada"}')
    response.status = "incomplete"
    response.incomplete_details.reason = "max_output_tokens"

    with pytest.raises(IncompleteOutputException):
        OpenAIResponsesToolsHandler().parse_response(response, ResponseToolModel)
