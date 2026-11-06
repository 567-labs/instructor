"""Unit tests for xAI v2 handlers.

These tests verify handler behavior without requiring API keys by using mock responses.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry


class Answer(BaseModel):
    """Simple answer model for testing."""

    answer: float


class MockToolCall:
    """Mock tool call for testing."""

    def __init__(self, name: str, arguments: dict[str, Any] | str):
        self.function = MagicMock()
        self.function.name = name
        if isinstance(arguments, dict):
            self.function.arguments = json.dumps(arguments)
        else:
            self.function.arguments = arguments


class MockResponse:
    """Mock xAI response for testing."""

    def __init__(
        self,
        text: str | None = None,
        content: Any = None,
        tool_calls: list[MockToolCall] | None = None,
    ):
        self.text = text
        self.content = content
        self.tool_calls = tool_calls


# ============================================================================
# XAIToolsHandler Tests
# ============================================================================


class TestXAIToolsHandler:
    """Tests for XAIToolsHandler."""

    @pytest.fixture
    def handler(self):
        """Get the TOOLS handler from registry."""
        handlers = mode_registry.get_handlers(Provider.XAI, Mode.TOOLS)
        return handlers

    def test_prepare_request_with_none_model(self, handler):
        """Test prepare_request returns unchanged kwargs when response_model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_adds_tool_schema(self, handler):
        """Test prepare_request adds tool schema for response model."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        assert "_xai_tool" in result_kwargs
        assert result_kwargs["_xai_tool"]["name"] == "Answer"
        assert "parameters" in result_kwargs["_xai_tool"]

    def test_prepare_request_preserves_original_kwargs(self, handler):
        """Test prepare_request doesn't modify original kwargs."""
        original_kwargs = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100,
        }
        kwargs_copy = original_kwargs.copy()
        handler.request_handler(Answer, original_kwargs)

        # Original should be unchanged
        assert original_kwargs == kwargs_copy

    def test_parse_response_from_tool_calls(self, handler):
        """Test parsing response from tool_calls."""
        response = MockResponse(tool_calls=[MockToolCall("Answer", {"answer": 4.0})])

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 4.0

    def test_parse_response_from_tool_calls_dict_args(self, handler):
        """Test parsing when tool call arguments are already a dict."""
        mock_tool = MockToolCall("Answer", {"answer": 42.0})
        mock_tool.function.arguments = {"answer": 42.0}  # Dict instead of string
        response = MockResponse(tool_calls=[mock_tool])

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 42.0

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            pytest.param(MockResponse(text='{"answer": 5.0}'), 5.0, id="text"),
            pytest.param(MockResponse(content='{"answer": 6.0}'), 6.0, id="content"),
            pytest.param(
                MockResponse(content=['{"answer": 7.0}']),
                7.0,
                id="content-list",
            ),
            pytest.param(
                MockResponse(text='```json\n{"answer": 8.0}\n```'),
                8.0,
                id="markdown",
            ),
        ],
    )
    def test_parse_response_from_supported_content_shapes(
        self,
        handler,
        response: MockResponse,
        expected: float,
    ):
        """Test tools parsing from xAI's supported fallback content shapes."""

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == expected

    def test_parse_response_raises_on_no_content(self, handler):
        """Test parsing raises error when no content available."""
        response = MockResponse()

        with pytest.raises(ValueError, match="No tool calls returned"):
            handler.response_parser(response, Answer)

    def test_handle_reask_adds_messages(self, handler):
        """Test handle_reask adds assistant and user messages."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(text="Invalid response")
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        assert len(result["messages"]) == 3
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][2]["role"] == "user"
        assert "Validation Error" in result["messages"][2]["content"]

    def test_handle_reask_returns_new_dict(self, handler):
        """Test handle_reask returns a new dict (shallow copy)."""
        original_kwargs = {"messages": [{"role": "user", "content": "Test"}]}
        response = MockResponse(text="Error")
        exception = ValueError("Test error")

        result = handler.reask_handler(original_kwargs, response, exception)

        # Returns a new dict (shallow copy)
        assert result is not original_kwargs
        # But messages list is shared (shallow copy behavior)
        assert result["messages"] is original_kwargs["messages"]


# ============================================================================
# XAIJSONSchemaHandler Tests
# ============================================================================


class TestXAIJSONSchemaHandler:
    """Tests for XAIJSONSchemaHandler."""

    @pytest.fixture
    def handler(self):
        """Get the JSON_SCHEMA handler from registry."""
        handlers = mode_registry.get_handlers(Provider.XAI, Mode.JSON_SCHEMA)
        return handlers

    def test_prepare_request_with_none_model(self, handler):
        """Test prepare_request returns unchanged kwargs when response_model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert result_kwargs == kwargs

    def test_prepare_request_adds_json_schema(self, handler):
        """Test prepare_request adds JSON schema info."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is Answer
        assert "_xai_json_schema" in result_kwargs
        assert result_kwargs["_xai_json_schema"]["name"] == "Answer"
        assert "schema" in result_kwargs["_xai_json_schema"]

    def test_parse_response_from_tuple(self, handler):
        """Test parsing response when xAI returns (raw, parsed) tuple."""
        parsed_model = Answer(answer=10.0)
        raw_response = MockResponse(text="raw")
        response = (raw_response, parsed_model)

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 10.0
        assert hasattr(result, "_raw_response")

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            pytest.param(MockResponse(text='{"answer": 11.0}'), 11.0, id="text"),
            pytest.param(MockResponse(content='{"answer": 12.0}'), 12.0, id="content"),
        ],
    )
    def test_parse_response_from_supported_content_shapes(
        self,
        handler,
        response: MockResponse,
        expected: float,
    ):
        """Test schema parsing from xAI's supported content shapes."""

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == expected

    def test_parse_response_raises_on_no_content(self, handler):
        """Test parsing raises error when no content available."""
        response = MockResponse()

        with pytest.raises(ValueError, match="Could not parse"):
            handler.response_parser(response, Answer)

    def test_handle_reask_adds_message(self, handler):
        """Test handle_reask adds user message with error."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(text="Invalid")
        exception = ValueError("Schema validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        assert len(result["messages"]) == 2
        assert result["messages"][1]["role"] == "user"
        assert "Validation Errors" in result["messages"][1]["content"]


# ============================================================================
# XAIMDJSONHandler Tests
# ============================================================================


class TestXAIMDJSONHandler:
    """Tests for XAIMDJSONHandler."""

    @pytest.fixture
    def handler(self):
        """Get the MD_JSON handler from registry."""
        handlers = mode_registry.get_handlers(Provider.XAI, Mode.MD_JSON)
        return handlers

    def test_prepare_request_with_none_model(self, handler):
        """Test prepare_request returns unchanged kwargs when response_model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert result_kwargs == kwargs

    def test_prepare_request_adds_system_message(self, handler):
        """Test prepare_request adds system message with schema."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is Answer
        messages = result_kwargs["messages"]

        # Should have system message at start
        assert messages[0]["role"] == "system"
        assert "json_schema" in messages[0]["content"]

        # Should have additional user message requesting JSON
        assert any("json codeblock" in m.get("content", "").lower() for m in messages)

    def test_prepare_request_appends_to_existing_system(self, handler):
        """Test prepare_request appends to existing system message."""
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        }
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        messages = result_kwargs["messages"]
        system_msg = messages[0]

        assert system_msg["role"] == "system"
        assert "You are helpful." in system_msg["content"]
        assert "json_schema" in system_msg["content"]

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            pytest.param(
                MockResponse(text='```json\n{"answer": 13.0}\n```'),
                13.0,
                id="markdown",
            ),
            pytest.param(MockResponse(text='{"answer": 14.0}'), 14.0, id="text"),
            pytest.param(
                MockResponse(content=['{"answer": 15.0}']),
                15.0,
                id="content-list",
            ),
        ],
    )
    def test_parse_response_from_supported_content_shapes(
        self,
        handler,
        response: MockResponse,
        expected: float,
    ):
        """Test markdown parsing from xAI's supported content shapes."""

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == expected

    def test_parse_response_raises_on_no_content(self, handler):
        """Test parsing raises error when no content available."""
        response = MockResponse()

        with pytest.raises(ValueError, match="Could not extract JSON"):
            handler.response_parser(response, Answer)

    def test_handle_reask_adds_message(self, handler):
        """Test handle_reask adds user message with error."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(text="Invalid")
        exception = ValueError("JSON extraction failed")

        result = handler.reask_handler(kwargs, response, exception)

        assert len(result["messages"]) == 2
        assert result["messages"][1]["role"] == "user"
        assert "Validation Errors" in result["messages"][1]["content"]


# ============================================================================
# Handler Registration Tests
# ============================================================================
# Note: Common handler registration tests are unified in
# test_handler_registration_unified.py. Only provider-specific tests remain here.


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestXAIHandlerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tools_handler_with_complex_model(self):
        """Test TOOLS handler with nested model."""
        handlers = mode_registry.get_handlers(Provider.XAI, Mode.TOOLS)

        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        kwargs = {"messages": [{"role": "user", "content": "Get person info"}]}
        result_model, result_kwargs = handlers.request_handler(Person, kwargs)

        assert result_model is not None
        assert "_xai_tool" in result_kwargs
        schema = result_kwargs["_xai_tool"]["parameters"]
        assert "properties" in schema
        assert "address" in schema["properties"]

    def test_json_schema_handler_with_validation_context(self):
        """Test JSON_SCHEMA handler passes validation context."""
        handlers = mode_registry.get_handlers(Provider.XAI, Mode.JSON_SCHEMA)
        response = MockResponse(text='{"answer": 20.0}')

        result = handlers.response_parser(
            response,
            Answer,
            validation_context={"test": "context"},
        )

        assert isinstance(result, Answer)
        assert result.answer == 20.0

    def test_md_json_handler_with_strict_validation(self):
        """Test MD_JSON handler with strict validation."""
        handlers = mode_registry.get_handlers(Provider.XAI, Mode.MD_JSON)
        response = MockResponse(text='{"answer": 21.0}')

        result = handlers.response_parser(
            response,
            Answer,
            strict=True,
        )

        assert isinstance(result, Answer)
        assert result.answer == 21.0

    def test_tools_handler_preserves_extra_kwargs(self):
        """Test TOOLS handler preserves extra kwargs."""
        handlers = mode_registry.get_handlers(Provider.XAI, Mode.TOOLS)
        kwargs = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        result_model, result_kwargs = handlers.request_handler(Answer, kwargs)

        assert result_kwargs["max_tokens"] == 500
        assert result_kwargs["temperature"] == 0.7
