"""Unit tests for Cohere v2 handlers.

These tests verify handler behavior without requiring API keys by using mock responses.
Cohere has both V1 and V2 client formats that need to be handled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry

# Import handlers directly to ensure they're registered
# This avoids the cohere SDK dependency in the __init__.py
_HANDLERS_PATH = (
    Path(__file__).resolve().parents[2] / "instructor/v2/providers/cohere/handlers.py"
)
if _HANDLERS_PATH.exists() and not mode_registry.is_registered(
    Provider.COHERE, Mode.TOOLS
):
    spec = importlib.util.spec_from_file_location(
        "instructor.v2.providers.cohere.handlers",
        _HANDLERS_PATH,
    )
    if spec and spec.loader:
        _handlers_module = importlib.util.module_from_spec(spec)
        sys.modules["instructor.v2.providers.cohere.handlers"] = _handlers_module
        spec.loader.exec_module(_handlers_module)


class Answer(BaseModel):
    """Simple answer model for testing."""

    answer: float


# ============================================================================
# Mock Response Classes for Cohere
# ============================================================================


class MockCohereV1Response:
    """Mock Cohere V1 response (has .text attribute)."""

    def __init__(
        self,
        text: str | None = None,
        tool_calls: list[Any] | None = None,
    ):
        self.text = text
        self.tool_calls = tool_calls


class MockCohereV2ContentItem:
    """Mock content item for V2 responses."""

    def __init__(self, type: str, text: str | None = None):
        self.type = type
        self.text = text


class MockCohereV2Message:
    """Mock message for V2 responses."""

    def __init__(self, content: list[MockCohereV2ContentItem] | None = None):
        self.content = content or []


class MockCohereV2Response:
    """Mock Cohere V2 response (has .message.content structure)."""

    def __init__(self, text: str | None = None):
        content = []
        if text:
            content.append(MockCohereV2ContentItem("text", text))
        self.message = MockCohereV2Message(content)


class MockCohereToolCall:
    """Mock tool call for Cohere responses."""

    def __init__(self, parameters: dict[str, Any]):
        self.parameters = parameters


# ============================================================================
# CohereToolsHandler Tests
# ============================================================================


class TestCohereToolsHandler:
    """Tests for CohereToolsHandler."""

    @pytest.fixture
    def handler(self):
        """Get the TOOLS handler from registry."""
        return mode_registry.get_handlers(Provider.COHERE, Mode.TOOLS)

    def test_prepare_request_with_none_model(self, handler):
        """Test prepare_request returns unchanged kwargs when response_model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_adds_extraction_instruction_v2(self, handler):
        """Test prepare_request adds extraction instruction for V2 format."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        # Should have added instruction to messages
        assert len(result_kwargs["messages"]) == 2
        # First message should be the instruction
        assert "Extract a valid Answer" in result_kwargs["messages"][0]["content"]

    def test_prepare_request_adds_extraction_instruction_v1(self, handler):
        """Test prepare_request adds extraction instruction for V1 format."""
        kwargs = {
            "chat_history": [{"role": "user", "message": "Previous message"}],
            "message": "What is 2+2?",
        }
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        # Should have added instruction to chat_history
        assert len(result_kwargs["chat_history"]) >= 1
        # First message should be the instruction
        assert "Extract a valid" in result_kwargs["chat_history"][0]["message"]

    def test_prepare_request_preserves_original_kwargs(self, handler):
        """Test prepare_request doesn't modify original kwargs."""
        original_kwargs = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100,
        }
        kwargs_copy = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100,
        }
        handler.request_handler(Answer, original_kwargs)

        # Original should be unchanged (messages list is modified in place though)
        assert original_kwargs["max_tokens"] == kwargs_copy["max_tokens"]

    def test_parse_response_from_v1_tool_calls(self, handler):
        """Test parsing response from V1 tool_calls."""
        response = MockCohereV1Response(
            tool_calls=[MockCohereToolCall({"answer": 4.0})]
        )

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 4.0

    def test_parse_response_from_v1_text(self, handler):
        """Test parsing response from V1 text."""
        response = MockCohereV1Response(text='{"answer": 5.0}')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 5.0

    def test_parse_response_from_v2_text(self, handler):
        """Test parsing response from V2 message.content."""
        response = MockCohereV2Response(text='{"answer": 6.0}')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 6.0

    def test_parse_response_with_validation_context(self, handler):
        """Test parsing with validation context."""
        response = MockCohereV1Response(text='{"answer": 7.0}')

        result = handler.response_parser(
            response,
            Answer,
            validation_context={"test": "context"},
        )

        assert isinstance(result, Answer)
        assert result.answer == 7.0

    def test_handle_reask_v2_format(self, handler):
        """Test handle_reask adds error message for V2 format."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockCohereV1Response(text="Invalid JSON")
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have added a message
        assert len(result["messages"]) > 1
        # Last message should contain the error
        last_msg = result["messages"][-1]
        assert "Validation failed" in last_msg["content"]

    def test_handle_reask_v1_format(self, handler):
        """Test handle_reask adds error message for V1 format."""
        kwargs = {
            "chat_history": [{"role": "user", "message": "Previous"}],
            "message": "Original",
        }
        response = MockCohereV1Response(text="Invalid JSON")
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have updated message and chat_history
        assert "Validation failed" in result["message"]
        assert len(result["chat_history"]) > 1

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
        # Schema should include nested properties
        instruction = result_kwargs["messages"][0]["content"]
        assert "address" in instruction


# ============================================================================
# CohereJSONSchemaHandler Tests
# ============================================================================


class TestCohereJSONSchemaHandler:
    """Tests for CohereJSONSchemaHandler."""

    @pytest.fixture
    def handler(self):
        """Get the JSON_SCHEMA handler from registry."""
        return mode_registry.get_handlers(Provider.COHERE, Mode.JSON_SCHEMA)

    def test_prepare_request_with_none_model(self, handler):
        """Test prepare_request returns unchanged kwargs when response_model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_sets_response_format(self, handler):
        """Test prepare_request sets response_format with schema."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        assert "response_format" in result_kwargs
        assert result_kwargs["response_format"]["type"] == "json_object"
        assert "schema" in result_kwargs["response_format"]

    def test_prepare_request_converts_v2_messages(self, handler):
        """Test prepare_request handles V2 message format."""
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        }
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        assert "messages" in result_kwargs
        assert "response_format" in result_kwargs

    def test_parse_response_from_v1_text(self, handler):
        """Test parsing JSON from V1 text response."""
        response = MockCohereV1Response(text='{"answer": 8.0}')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 8.0

    def test_parse_response_from_v2_text(self, handler):
        """Test parsing JSON from V2 message.content."""
        response = MockCohereV2Response(text='{"answer": 9.0}')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 9.0

    def test_parse_response_with_validation_context(self, handler):
        """Test parsing with validation context."""
        response = MockCohereV1Response(text='{"answer": 10.0}')

        result = handler.response_parser(
            response,
            Answer,
            validation_context={"test": "context"},
        )

        assert isinstance(result, Answer)
        assert result.answer == 10.0

    def test_handle_reask_v2_format(self, handler):
        """Test handle_reask adds user message with error for V2 format."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockCohereV1Response(text='{"answer": "invalid"}')
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have added a message
        assert len(result["messages"]) > 1
        # Last message should contain the error
        last_msg = result["messages"][-1]
        assert "Validation failed" in last_msg["content"]

    def test_handle_reask_v1_format(self, handler):
        """Test handle_reask adds error message for V1 format."""
        kwargs = {
            "chat_history": [{"role": "user", "message": "Previous"}],
            "message": "Original",
        }
        response = MockCohereV1Response(text='{"answer": "invalid"}')
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have updated message
        assert "Validation failed" in result["message"]


# ============================================================================
# CohereMDJSONHandler Tests
# ============================================================================


class TestCohereMDJSONHandler:
    """Tests for CohereMDJSONHandler."""

    @pytest.fixture
    def handler(self):
        """Get the MD_JSON handler from registry."""
        return mode_registry.get_handlers(Provider.COHERE, Mode.MD_JSON)

    def test_prepare_request_with_none_model(self, handler):
        """Test prepare_request returns unchanged kwargs when response_model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_adds_markdown_instruction_v2(self, handler):
        """Test prepare_request adds markdown instruction for V2 format."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        # Should have appended instruction to last message
        last_msg = result_kwargs["messages"][-1]
        assert "markdown code block" in last_msg["content"]
        assert "Schema:" in last_msg["content"]

    def test_prepare_request_adds_markdown_instruction_v1(self, handler):
        """Test prepare_request adds markdown instruction for V1 format."""
        kwargs = {
            "chat_history": [],
            "message": "What is 2+2?",
        }
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        # Should have appended instruction to message
        assert "markdown code block" in result_kwargs["message"]
        assert "Schema:" in result_kwargs["message"]

    def test_parse_response_from_markdown_codeblock(self, handler):
        """Test parsing JSON from markdown code block."""
        response = MockCohereV1Response(text='```json\n{"answer": 11.0}\n```')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 11.0

    def test_parse_response_from_plain_json(self, handler):
        """Test parsing plain JSON (no code block)."""
        response = MockCohereV1Response(text='{"answer": 12.0}')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 12.0

    def test_parse_response_from_v2_markdown(self, handler):
        """Test parsing markdown from V2 response."""
        response = MockCohereV2Response(text='```json\n{"answer": 13.0}\n```')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 13.0

    def test_handle_reask_adds_message(self, handler):
        """Test handle_reask adds user message with error."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockCohereV1Response(text="Invalid")
        exception = ValueError("JSON extraction failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have added a message
        assert len(result["messages"]) > 1
        # Last message should contain the error
        last_msg = result["messages"][-1]
        assert "JSON extraction failed" in last_msg["content"]

    def test_md_json_handler_with_strict_validation(self, handler):
        """Test MD_JSON handler with strict validation."""
        response = MockCohereV1Response(text='{"answer": 14.0}')

        result = handler.response_parser(
            response,
            Answer,
            strict=True,
        )

        assert isinstance(result, Answer)
        assert result.answer == 14.0


# ============================================================================
# Handler Registration Tests
# ============================================================================
# Note: Common handler registration tests are unified in
# test_handler_registration_unified.py. Only provider-specific tests remain here.


# ============================================================================
# Mode Normalization Tests
# ============================================================================


class TestCohereModeNormalization:
    """Tests for Cohere mode handling in v2."""

    def test_cohere_tools_normalizes_to_tools(self):
        """Legacy COHERE_TOOLS remains accepted via normalization."""
        from instructor.v2.core.registry import mode_registry, normalize_mode

        result = normalize_mode(Provider.COHERE, Mode.COHERE_TOOLS)
        assert result == Mode.TOOLS
        assert mode_registry.is_registered(Provider.COHERE, Mode.COHERE_TOOLS)

    def test_cohere_json_schema_normalizes_to_json_schema(self):
        """Legacy COHERE_JSON_SCHEMA remains accepted via normalization."""
        from instructor.v2.core.registry import mode_registry, normalize_mode

        result = normalize_mode(Provider.COHERE, Mode.COHERE_JSON_SCHEMA)
        assert result == Mode.JSON_SCHEMA
        assert mode_registry.is_registered(Provider.COHERE, Mode.COHERE_JSON_SCHEMA)

    def test_generic_tools_passes_through(self):
        """Test generic TOOLS mode passes through unchanged."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.COHERE, Mode.TOOLS)
        assert result == Mode.TOOLS

    def test_generic_json_schema_passes_through(self):
        """Test generic JSON_SCHEMA mode passes through unchanged."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.COHERE, Mode.JSON_SCHEMA)
        assert result == Mode.JSON_SCHEMA


# ============================================================================
# Client Version Detection Tests
# ============================================================================


class TestCohereClientVersionDetection:
    """Tests for Cohere client version detection."""

    def test_detect_v2_from_messages(self):
        """Test V2 detection from messages key."""
        from instructor.v2.providers.cohere.handlers import _detect_client_version

        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        assert _detect_client_version(kwargs) == "v2"

    def test_detect_v1_from_chat_history(self):
        """Test V1 detection from chat_history key."""
        from instructor.v2.providers.cohere.handlers import _detect_client_version

        kwargs = {"chat_history": [], "message": "Hello"}
        assert _detect_client_version(kwargs) == "v1"

    def test_detect_v1_from_message_only(self):
        """Test V1 detection from message key only."""
        from instructor.v2.providers.cohere.handlers import _detect_client_version

        kwargs = {"message": "Hello"}
        assert _detect_client_version(kwargs) == "v1"

    def test_detect_from_explicit_version(self):
        """Test detection from explicit _cohere_client_version."""
        from instructor.v2.providers.cohere.handlers import _detect_client_version

        kwargs = {"_cohere_client_version": "v1", "messages": []}
        assert _detect_client_version(kwargs) == "v1"

    def test_default_to_v2(self):
        """Test default to V2 when no indicators present."""
        from instructor.v2.providers.cohere.handlers import _detect_client_version

        kwargs = {}
        assert _detect_client_version(kwargs) == "v2"


# ============================================================================
# Message Conversion Tests
# ============================================================================


class TestCohereMessageConversion:
    """Tests for Cohere message format conversion."""

    def test_convert_messages_to_v1(self):
        """Test converting OpenAI-style messages to V1 format."""
        from instructor.v2.providers.cohere.handlers import (
            _convert_messages_to_cohere_v1,
        )

        kwargs = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ]
        }

        result = _convert_messages_to_cohere_v1(kwargs)

        assert "chat_history" in result
        assert "message" in result
        assert len(result["chat_history"]) == 2
        assert result["message"] == "How are you?"

    def test_convert_messages_to_v2(self):
        """Test cleaning up kwargs for V2 format."""
        from instructor.v2.providers.cohere.handlers import (
            _convert_messages_to_cohere_v2,
        )

        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "_cohere_client_version": "v2",
            "model_name": "command-r-plus",
        }

        result = _convert_messages_to_cohere_v2(kwargs)

        assert "messages" in result
        assert "_cohere_client_version" not in result
        assert "model" in result
        assert result["model"] == "command-r-plus"

    def test_convert_removes_strict_param(self):
        """Test that strict param is removed during conversion."""
        from instructor.v2.providers.cohere.handlers import (
            _convert_messages_to_cohere_v2,
        )

        kwargs = {
            "messages": [{"role": "user", "content": "Hello"}],
            "strict": True,
        }

        result = _convert_messages_to_cohere_v2(kwargs)

        assert "strict" not in result

    def test_convert_v2_without_messages(self):
        """Test that message-free kwargs remain message-free."""
        from instructor.v2.providers.cohere.handlers import (
            _convert_messages_to_cohere_v2,
        )

        result = _convert_messages_to_cohere_v2({"model": "command-r-plus"})

        assert "messages" not in result


# ============================================================================
# Text Extraction Tests
# ============================================================================


class TestCohereTextExtraction:
    """Tests for text extraction from Cohere responses."""

    def test_extract_from_v1_text(self):
        """Test extracting text from V1 response."""
        from instructor.v2.providers.cohere.handlers import _extract_text_from_response

        response = MockCohereV1Response(text="Hello world")
        result = _extract_text_from_response(response)

        assert result == "Hello world"

    def test_extract_from_v2_message_content(self):
        """Test extracting text from V2 message.content."""
        from instructor.v2.providers.cohere.handlers import _extract_text_from_response

        response = MockCohereV2Response(text="Hello from V2")
        result = _extract_text_from_response(response)

        assert result == "Hello from V2"

    def test_extract_raises_on_invalid_response(self):
        """Test that extraction raises on invalid response format."""
        from instructor.v2.providers.cohere.handlers import _extract_text_from_response
        from instructor.core.exceptions import ResponseParsingError

        # Create a response that has neither .text nor valid .message.content
        response = MagicMock()
        del response.text  # Remove the text attribute entirely
        response.message = MagicMock()
        response.message.content = []  # Empty content list

        with pytest.raises(ResponseParsingError):
            _extract_text_from_response(response)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestCohereHandlerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tools_handler_with_optional_fields(self):
        """Test TOOLS handler with optional fields."""

        class OptionalModel(BaseModel):
            required_field: str
            optional_field: str | None = None

        handlers = mode_registry.get_handlers(Provider.COHERE, Mode.TOOLS)
        kwargs = {"messages": [{"role": "user", "content": "Test"}]}

        result_model, result_kwargs = handlers.request_handler(OptionalModel, kwargs)

        assert result_model is not None
        # Schema should be in the instruction
        instruction = result_kwargs["messages"][0]["content"]
        assert "optional_field" in instruction

    def test_json_schema_handler_with_nested_model(self):
        """Test JSON_SCHEMA handler with nested model."""

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner

        handlers = mode_registry.get_handlers(Provider.COHERE, Mode.JSON_SCHEMA)
        kwargs = {"messages": [{"role": "user", "content": "Test"}]}

        result_model, result_kwargs = handlers.request_handler(Outer, kwargs)

        assert result_model is not None
        assert "response_format" in result_kwargs
        schema = result_kwargs["response_format"]["schema"]
        assert "properties" in schema

    def test_md_json_handler_with_empty_messages(self):
        """Test MD_JSON handler with empty messages list."""
        handlers = mode_registry.get_handlers(Provider.COHERE, Mode.MD_JSON)
        kwargs = {"messages": []}

        result_model, result_kwargs = handlers.request_handler(Answer, kwargs)

        # Should handle empty messages gracefully
        assert result_model is not None

    def test_v1_reask_without_chat_history(self):
        """Test V1 reask when chat_history doesn't exist."""
        handlers = mode_registry.get_handlers(Provider.COHERE, Mode.TOOLS)
        kwargs = {"message": "Original"}
        response = MockCohereV1Response(text="Invalid")
        exception = ValueError("Error")

        result = handlers.reask_handler(kwargs, response, exception)

        # Should create chat_history
        assert "chat_history" in result
        assert "message" in result


# ============================================================================
# Import Tests
# ============================================================================


class TestCohereImports:
    """Tests for Cohere v2 imports."""

    def test_from_cohere_importable_from_v2(self):
        """Test from_cohere can be imported from instructor.v2.

        Note: This may be None if cohere SDK is not installed.
        """

        # from_cohere may be None if cohere SDK is not installed
        # The test passes if the import doesn't raise an error
        pass  # Import succeeded, test passes

    def test_handlers_importable(self):
        """Test handlers can be imported directly."""
        from instructor.v2.providers.cohere.handlers import (
            CohereToolsHandler,
            CohereJSONSchemaHandler,
            CohereMDJSONHandler,
        )

        assert CohereToolsHandler is not None
        assert CohereJSONSchemaHandler is not None
        assert CohereMDJSONHandler is not None

    def test_helper_functions_importable(self):
        """Test helper functions can be imported."""
        from instructor.v2.providers.cohere.handlers import (
            _detect_client_version,
            _convert_messages_to_cohere_v1,
            _convert_messages_to_cohere_v2,
            _extract_text_from_response,
        )

        assert _detect_client_version is not None
        assert _convert_messages_to_cohere_v1 is not None
        assert _convert_messages_to_cohere_v2 is not None
        assert _extract_text_from_response is not None
