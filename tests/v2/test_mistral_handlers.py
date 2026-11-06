"""Unit tests for Mistral v2 handlers.

These tests verify handler behavior without requiring API keys by using mock responses.
Mistral has its own API format that differs from OpenAI.
"""

from __future__ import annotations

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
    """Mock tool call for testing Mistral responses."""

    _counter = 0

    def __init__(self, name: str, arguments: dict[str, Any] | str):
        MockToolCall._counter += 1
        self.id = f"call_{MockToolCall._counter}"
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        # Mistral can return arguments as dict or string
        if isinstance(arguments, dict):
            self.function.arguments = arguments
        else:
            self.function.arguments = arguments


class MockMessage:
    """Mock message for testing Mistral responses."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[MockToolCall] | None = None,
        role: str = "assistant",
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.role = role

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation."""
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return result


class MockChoice:
    """Mock choice for testing Mistral responses."""

    def __init__(
        self,
        message: MockMessage,
        finish_reason: str = "stop",
    ):
        self.message = message
        self.finish_reason = finish_reason


class MockResponse:
    """Mock Mistral response for testing."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[MockToolCall] | None = None,
        finish_reason: str = "stop",
    ):
        self.choices = [MockChoice(MockMessage(content, tool_calls), finish_reason)]


# ============================================================================
# MistralToolsHandler Tests
# ============================================================================


class TestMistralToolsHandler:
    """Tests for MistralToolsHandler."""

    @pytest.fixture
    def handler(self):
        """Get the TOOLS handler from registry."""
        handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.TOOLS)
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
        assert "tools" in result_kwargs
        assert len(result_kwargs["tools"]) == 1
        assert result_kwargs["tools"][0]["type"] == "function"

    def test_prepare_request_sets_tool_choice_any(self, handler):
        """Test prepare_request sets tool_choice to 'any' (Mistral-specific)."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["tool_choice"] == "any"

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

    def test_parse_response_from_tool_calls_dict_args(self, handler):
        """Test parsing response when arguments are a dict (Mistral format)."""
        response = MockResponse(tool_calls=[MockToolCall("Answer", {"answer": 4.0})])

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 4.0

    def test_parse_response_from_tool_calls_string_args(self, handler):
        """Test parsing response when arguments are a string."""
        response = MockResponse(tool_calls=[MockToolCall("Answer", '{"answer": 5.0}')])

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 5.0

    def test_parse_response_with_validation_context(self, handler):
        """Test parsing with validation context."""
        response = MockResponse(tool_calls=[MockToolCall("Answer", {"answer": 6.0})])

        result = handler.response_parser(
            response,
            Answer,
            validation_context={"test": "context"},
        )

        assert isinstance(result, Answer)
        assert result.answer == 6.0

    def test_handle_reask_adds_messages(self, handler):
        """Test handle_reask adds error message to conversation."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(tool_calls=[MockToolCall("Answer", {"answer": "bad"})])
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have added messages for reask
        assert len(result["messages"]) > 1
        # Should have tool result message with error
        tool_msg = result["messages"][-1]
        assert tool_msg["role"] == "tool"
        assert "Validation Error" in tool_msg["content"]

    def test_parse_response_without_tool_calls_raises_retryable_error(self, handler):
        """Test parsing a prose response with no tool calls raises ResponseParsingError.

        Some models ignore tool_choice="any" and reply with plain text, leaving
        tool_calls as None. Parsing must raise a retryable ResponseParsingError
        (not a bare TypeError) so the retry machinery can re-ask.
        """
        from instructor.v2.core.errors import ResponseParsingError

        response = MockResponse(content="I cannot answer that as a tool call.")

        with pytest.raises(ResponseParsingError, match="No tool calls found"):
            handler.response_parser(response, Answer)

    def test_handle_reask_without_tool_calls_adds_user_correction(self, handler):
        """Test handle_reask falls back to a user message when tool_calls is None.

        There is no tool_call_id to attach a tool-role message to, so the
        handler must append a plain user correction instead of iterating None.
        """
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(content="The answer is four.")
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        assert len(result["messages"]) == 3
        correction = result["messages"][-1]
        assert correction["role"] == "user"
        assert "Validation Error" in correction["content"]
        assert "Validation failed" in correction["content"]

    def test_parallel_parse_response_without_tool_calls_yields_nothing(self, handler):
        """Test the parallel generator tolerates tool_calls=None without crashing."""
        from collections.abc import Iterable
        from typing import Union

        class Other(BaseModel):
            value: int

        response = MockResponse(content="No tools were called.")

        result = handler.response_parser(response, Iterable[Union[Answer, Other]])

        assert list(result) == []

    def test_tools_handler_preserves_extra_kwargs(self, handler):
        """Test TOOLS handler preserves extra kwargs."""
        kwargs = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["max_tokens"] == 500
        assert result_kwargs["temperature"] == 0.7

    def test_tools_handler_with_complex_model(self, handler):
        """Test TOOLS handler with nested model."""

        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        kwargs = {"messages": [{"role": "user", "content": "Get person info"}]}
        result_model, result_kwargs = handler.request_handler(Person, kwargs)

        assert result_model is not None
        assert "tools" in result_kwargs
        # Schema should include nested properties
        schema = result_kwargs["tools"][0]["function"]
        assert "address" in str(schema)


# ============================================================================
# MistralJSONSchemaHandler Tests
# ============================================================================


class TestMistralJSONSchemaHandler:
    """Tests for MistralJSONSchemaHandler.

    Note: Most tests are skipped because they require the mistralai SDK.
    """

    @pytest.fixture
    def handler(self):
        """Get the JSON_SCHEMA handler from registry."""
        handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.JSON_SCHEMA)
        return handlers

    def test_parse_response_from_text_content(self, handler):
        """Test parsing JSON from text content."""
        response = MockResponse(content='{"answer": 7.0}')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 7.0

    def test_parse_response_with_validation_context(self, handler):
        """Test parsing with validation context."""
        response = MockResponse(content='{"answer": 8.0}')

        result = handler.response_parser(
            response,
            Answer,
            validation_context={"test": "context"},
        )

        assert isinstance(result, Answer)
        assert result.answer == 8.0

    def test_handle_reask_adds_messages(self, handler):
        """Test handle_reask adds user message with error."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(content='{"answer": "invalid"}')
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have added messages for reask
        assert len(result["messages"]) > 1
        # Should have assistant message with previous response
        assert result["messages"][-2]["role"] == "assistant"
        # Should have user message with error
        assert result["messages"][-1]["role"] == "user"
        assert "Validation Error" in result["messages"][-1]["content"]

    @pytest.mark.skipif(True, reason="Requires mistralai SDK")
    def test_prepare_request_uses_mistral_helper(self, handler):
        """Test prepare_request uses Mistral's response_format_from_pydantic_model."""
        # This test requires the mistralai SDK
        pass


# ============================================================================
# MistralMDJSONHandler Tests
# ============================================================================


class TestMistralMDJSONHandler:
    """Tests for MistralMDJSONHandler."""

    @pytest.fixture
    def handler(self):
        """Get the MD_JSON handler from registry."""
        handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.MD_JSON)
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

    def test_parse_response_from_markdown_codeblock(self, handler):
        """Test parsing JSON from markdown code block."""
        response = MockResponse(content='```json\n{"answer": 9.0}\n```')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 9.0

    def test_parse_response_from_plain_json(self, handler):
        """Test parsing plain JSON (no code block)."""
        response = MockResponse(content='{"answer": 10.0}')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 10.0

    def test_handle_reask_adds_message(self, handler):
        """Test handle_reask adds user message with error."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(content="Invalid")
        exception = ValueError("JSON extraction failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have added messages for reask
        assert len(result["messages"]) > 1
        # Should have assistant message with previous response
        assert result["messages"][-2]["role"] == "assistant"
        # Should have user message with error
        assert result["messages"][-1]["role"] == "user"
        assert "Validation Error" in result["messages"][-1]["content"]

    def test_md_json_handler_with_strict_validation(self, handler):
        """Test MD_JSON handler with strict validation."""
        response = MockResponse(content='{"answer": 11.0}')

        result = handler.response_parser(
            response,
            Answer,
            strict=True,
        )

        assert isinstance(result, Answer)
        assert result.answer == 11.0


# ============================================================================
# Handler Registration Tests
# ============================================================================
# Note: Common handler registration tests are unified in
# test_handler_registration_unified.py. Only provider-specific tests remain here.


# ============================================================================
# Mode Normalization Tests
# ============================================================================


class TestMistralModeNormalization:
    """Tests for Mistral mode handling in v2."""

    def test_mistral_tools_normalizes_to_tools(self):
        """Legacy MISTRAL_TOOLS remains accepted via normalization."""
        from instructor.v2.core.registry import mode_registry, normalize_mode

        result = normalize_mode(Provider.MISTRAL, Mode.MISTRAL_TOOLS)
        assert result == Mode.TOOLS
        assert mode_registry.is_registered(Provider.MISTRAL, Mode.MISTRAL_TOOLS)

    def test_mistral_structured_outputs_normalizes_to_json_schema(self):
        """Legacy structured outputs remains accepted via normalization."""
        from instructor.v2.core.registry import mode_registry, normalize_mode

        result = normalize_mode(Provider.MISTRAL, Mode.MISTRAL_STRUCTURED_OUTPUTS)
        assert result == Mode.JSON_SCHEMA
        assert mode_registry.is_registered(
            Provider.MISTRAL, Mode.MISTRAL_STRUCTURED_OUTPUTS
        )

    def test_generic_tools_passes_through(self):
        """Test generic TOOLS mode passes through unchanged."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.MISTRAL, Mode.TOOLS)
        assert result == Mode.TOOLS

    def test_generic_json_schema_passes_through(self):
        """Test generic JSON_SCHEMA mode passes through unchanged."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.MISTRAL, Mode.JSON_SCHEMA)
        assert result == Mode.JSON_SCHEMA


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestMistralHandlerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tools_handler_incomplete_output(self):
        """Test TOOLS handler raises on incomplete output."""
        from instructor.core.exceptions import IncompleteOutputException

        handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.TOOLS)
        response = MockResponse(
            tool_calls=[MockToolCall("Answer", {"answer": 1.0})],
            finish_reason="length",
        )

        with pytest.raises(IncompleteOutputException):
            handlers.response_parser(response, Answer)

    def test_json_schema_handler_incomplete_output(self):
        """Test JSON_SCHEMA handler raises on incomplete output."""
        from instructor.core.exceptions import IncompleteOutputException

        handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.JSON_SCHEMA)
        response = MockResponse(content='{"answer":', finish_reason="length")

        with pytest.raises(IncompleteOutputException):
            handlers.response_parser(response, Answer)

    def test_md_json_handler_incomplete_output(self):
        """Test MD_JSON handler raises on incomplete output."""
        from instructor.core.exceptions import IncompleteOutputException

        handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.MD_JSON)
        response = MockResponse(content='```json\n{"answer":', finish_reason="length")

        with pytest.raises(IncompleteOutputException):
            handlers.response_parser(response, Answer)

    def test_tools_handler_with_optional_fields(self):
        """Test TOOLS handler with optional fields."""

        class OptionalModel(BaseModel):
            required_field: str
            optional_field: str | None = None

        handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.TOOLS)
        kwargs = {"messages": [{"role": "user", "content": "Test"}]}

        result_model, result_kwargs = handlers.request_handler(OptionalModel, kwargs)

        assert result_model is not None
        assert "tools" in result_kwargs

    def test_md_json_handler_with_list_content_system_message(self):
        """Test MD_JSON handler with list-format system message content."""
        handlers = mode_registry.get_handlers(Provider.MISTRAL, Mode.MD_JSON)
        kwargs = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are helpful."}],
                },
                {"role": "user", "content": "What is 2+2?"},
            ]
        }

        result_model, result_kwargs = handlers.request_handler(Answer, kwargs)

        # Should have appended to the list content
        messages = result_kwargs["messages"]
        assert messages[0]["role"] == "system"


# ============================================================================
# Import Tests
# ============================================================================


class TestMistralImports:
    """Tests for Mistral v2 imports."""

    def test_from_mistral_importable_from_v2(self):
        """Test from_mistral can be imported from instructor.v2."""
        from instructor.v2 import from_mistral

        assert from_mistral is not None

    def test_handlers_importable(self):
        """Test handlers can be imported directly."""
        from instructor.v2.providers.mistral.handlers import (
            MistralToolsHandler,
            MistralJSONSchemaHandler,
            MistralMDJSONHandler,
        )

        assert MistralToolsHandler is not None
        assert MistralJSONSchemaHandler is not None
        assert MistralMDJSONHandler is not None
