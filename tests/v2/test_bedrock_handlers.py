"""Unit tests for Bedrock v2 handlers."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field, create_model

from instructor import Mode, Provider
from instructor.v2.core.errors import ConfigurationError
from instructor.v2.core.registry import mode_registry


class Answer(BaseModel):
    """Simple answer model for tests."""

    answer: float


class User(BaseModel):
    """User model for tests."""

    name: str
    age: int


class Message(BaseModel):
    """Text payload used to verify JSON escape preservation."""

    text: str


class Address(BaseModel):
    """Nested address model for strict schema tests."""

    city: str


class Profile(BaseModel):
    """Nested profile model for strict schema tests."""

    user: User
    addresses: list[Address]


class FreeFormMetadata(BaseModel):
    """Model shape that Bedrock native structured outputs cannot represent."""

    metadata: dict[str, str]


class AnyOfConstrainedResponse(BaseModel):
    """Schema fixture with an unsupported constraint inside an anyOf branch."""

    @classmethod
    def model_json_schema(cls, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "string"},
                    ]
                }
            },
            "required": ["value"],
        }


def _bedrock_tool_response(
    args: dict[str, Any], tool_use_id: str = "tool-use-1", name: str = "Answer"
) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tool_use_id,
                            "name": name,
                            "input": args,
                        }
                    }
                ]
            }
        }
    }


def _bedrock_text_response(text: str) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "text": text,
                    }
                ]
            }
        }
    }


class TestBedrockToolsHandler:
    """Tests for BedrockToolsHandler."""

    @pytest.fixture
    def handler(self):
        """Get the TOOLS handler from registry."""
        return mode_registry.get_handlers(Provider.BEDROCK, Mode.TOOLS)

    def test_prepare_request_with_none_model(self, handler):
        """prepare_request returns unchanged model and converted kwargs."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_adds_tool_config(self, handler):
        """prepare_request adds Bedrock tool config."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        assert "toolConfig" in result_kwargs
        assert "tools" in result_kwargs["toolConfig"]
        assert "toolChoice" in result_kwargs["toolConfig"]

    def test_prepare_request_does_not_mutate_nested_config(self, handler):
        """Preparing Bedrock kwargs treats nested caller config as read-only."""
        kwargs = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0.2,
            "top_k": 10,
            "inferenceConfig": {"maxTokens": 100},
            "additionalModelRequestFields": {"thinking": {"type": "enabled"}},
        }
        original = deepcopy(kwargs)

        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert kwargs == original
        assert result_model is not None
        assert result_kwargs["inferenceConfig"] == {
            "maxTokens": 100,
            "temperature": 0.2,
        }
        assert result_kwargs["inferenceConfig"] is not kwargs["inferenceConfig"]
        assert result_kwargs["additionalModelRequestFields"] == {
            "thinking": {"type": "enabled"},
            "top_k": 10,
        }
        assert (
            result_kwargs["additionalModelRequestFields"]
            is not kwargs["additionalModelRequestFields"]
        )

    def test_parse_response_from_tool_use(self, handler):
        """parse_response extracts tool input."""
        response = _bedrock_tool_response({"answer": 4.0})
        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 4.0

    def test_handle_reask_adds_messages(self, handler):
        """handle_reask adds tool error messages."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        original = deepcopy(kwargs)
        response = _bedrock_tool_response({"answer": "bad"})
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        assert "messages" in result
        assert len(result["messages"]) > 1
        assert kwargs == original


class TestBedrockMDJSONHandler:
    """Tests for BedrockMDJSONHandler."""

    @pytest.fixture
    def handler(self):
        """Get the MD_JSON handler from registry."""
        return mode_registry.get_handlers(Provider.BEDROCK, Mode.MD_JSON)

    def test_prepare_request_with_none_model(self, handler):
        """prepare_request returns unchanged model and converted kwargs."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_with_none_model_converts_to_converse(self, handler):
        """response_model=None still converts OpenAI-style kwargs, like TOOLS."""
        kwargs = {"model": "m", "messages": [{"role": "user", "content": "Hello"}]}
        _, result_kwargs = handler.request_handler(None, kwargs)

        assert result_kwargs == {
            "modelId": "m",
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        }

    def test_prepare_request_adds_system_message(self, handler):
        """prepare_request adds system instructions for JSON output."""
        kwargs = {"messages": [{"role": "user", "content": "Extract user"}]}
        result_model, result_kwargs = handler.request_handler(User, kwargs)

        assert result_model is not None
        assert "system" in result_kwargs

    def test_parse_response_from_text(self, handler):
        """parse_response extracts JSON from Bedrock text."""
        response = _bedrock_text_response('{"answer": 4}')
        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 4.0

    def test_parse_response_from_codeblock(self, handler):
        """parse_response handles JSON in code blocks."""
        response = _bedrock_text_response('```json\n{"answer": 3}\n```')
        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 3.0

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                '<think>I first considered {"answer": 99}.</think>\n{"answer": 4}',
                4.0,
            ),
            (
                "<think>Reasoning with {not valid JSON} across\nmultiple lines.</think>\n"
                '```json\n{"answer": 5}\n```',
                5.0,
            ),
        ],
    )
    def test_parse_response_uses_final_json_after_reasoning(
        self, handler, text: str, expected: float
    ) -> None:
        """Reasoning content cannot replace or corrupt the final JSON value."""
        result = handler.response_parser(_bedrock_text_response(text), Answer)

        assert isinstance(result, Answer)
        assert result.answer == expected

    def test_parse_response_preserves_escaped_newlines(self, handler) -> None:
        """JSON escape sequences remain part of the parsed payload."""
        response = _bedrock_text_response('{"text": "first\\nsecond"}')

        result = handler.response_parser(response, Message)

        assert isinstance(result, Message)
        assert result.text == "first\nsecond"

    def test_prepare_request_does_not_mutate_native_system_messages(
        self, handler
    ) -> None:
        """Converting native Bedrock system content treats caller input as read-only."""
        kwargs = {
            "system": [{"text": "Existing instruction"}],
            "messages": [{"role": "user", "content": "Extract user"}],
        }
        original = deepcopy(kwargs)

        _, result_kwargs = handler.request_handler(User, kwargs)

        assert kwargs == original
        assert result_kwargs["system"][0] == {"text": "Existing instruction"}
        assert result_kwargs["messages"] == [
            {"role": "user", "content": [{"text": "Extract user"}]}
        ]

    def test_handle_reask_adds_messages(self, handler):
        """handle_reask adds user correction message."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        original = deepcopy(kwargs)
        response = _bedrock_text_response("Invalid response")
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        assert "messages" in result
        assert len(result["messages"]) > 1
        assert kwargs == original


class TestBedrockNativeStructuredOutputs:
    """Tests for opt-in Bedrock constrained decoding modes."""

    def test_json_schema_request_uses_recursive_strict_schema(self) -> None:
        handler = mode_registry.get_handlers(Provider.BEDROCK, Mode.JSON_SCHEMA)

        result_model, result_kwargs = handler.request_handler(
            Profile,
            {"messages": [{"role": "user", "content": "Extract profile"}]},
        )

        assert result_model is not None
        output_format = result_kwargs["outputConfig"]["textFormat"]
        assert output_format["type"] == "json_schema"
        json_schema = output_format["structure"]["jsonSchema"]
        assert json_schema["name"] == "Profile"
        schema = json.loads(json_schema["schema"])
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["User"]["additionalProperties"] is False
        assert schema["$defs"]["Address"]["additionalProperties"] is False

    def test_strict_tools_request_enforces_recursive_schema(self) -> None:
        handler = mode_registry.get_handlers(Provider.BEDROCK, Mode.TOOLS_STRICT)

        result_model, result_kwargs = handler.request_handler(
            Profile,
            {"messages": [{"role": "user", "content": "Extract profile"}]},
        )

        assert result_model is not None
        tool_spec = result_kwargs["toolConfig"]["tools"][0]["toolSpec"]
        assert tool_spec["strict"] is True
        schema = tool_spec["inputSchema"]["json"]
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["User"]["additionalProperties"] is False
        assert schema["$defs"]["Address"]["additionalProperties"] is False

    @pytest.mark.parametrize("mode", [Mode.JSON_SCHEMA, Mode.TOOLS_STRICT])
    def test_native_modes_reject_free_form_mappings(self, mode: Mode) -> None:
        handler = mode_registry.get_handlers(Provider.BEDROCK, mode)

        with pytest.raises(ConfigurationError, match="free-form mappings"):
            handler.request_handler(
                FreeFormMetadata,
                {"messages": [{"role": "user", "content": "Extract metadata"}]},
            )

    @pytest.mark.parametrize("mode", [Mode.JSON_SCHEMA, Mode.TOOLS_STRICT])
    @pytest.mark.parametrize(
        ("constraint", "field_type", "field"),
        [
            ("minimum", int, Field(ge=0)),
            ("maximum", int, Field(le=10)),
            ("multipleOf", int, Field(multiple_of=2)),
            ("minLength", str, Field(min_length=1)),
            ("maxLength", str, Field(max_length=10)),
            ("minItems", list[str], Field(min_length=2)),
        ],
    )
    def test_native_modes_reject_unsupported_schema_constraints(
        self,
        mode: Mode,
        constraint: str,
        field_type: Any,
        field: Any,
    ) -> None:
        constrained_model = create_model(
            f"{constraint}Value",
            value=(field_type, field),
        )
        response_model = create_model(
            f"{constraint}Wrapper",
            nested=(constrained_model, ...),
        )
        handler = mode_registry.get_handlers(Provider.BEDROCK, mode)

        with pytest.raises(ConfigurationError) as exc_info:
            handler.request_handler(
                response_model,
                {"messages": [{"role": "user", "content": "Extract value"}]},
            )

        message = str(exc_info.value)
        assert f"`{constraint}`" in message
        assert f"$.$defs.{constrained_model.__name__}.properties.value" in message

    @pytest.mark.parametrize("mode", [Mode.JSON_SCHEMA, Mode.TOOLS_STRICT])
    @pytest.mark.parametrize(
        "field_name",
        ["minimum", "maximum", "multipleOf", "minLength", "maxLength", "minItems"],
    )
    def test_native_modes_allow_properties_named_like_schema_constraints(
        self,
        mode: Mode,
        field_name: str,
    ) -> None:
        field_definitions: dict[str, Any] = {field_name: (str, ...)}
        response_model = create_model(
            "ConstraintNamedProperty",
            **field_definitions,
        )
        handler = mode_registry.get_handlers(Provider.BEDROCK, mode)

        _, result_kwargs = handler.request_handler(
            response_model,
            {"messages": [{"role": "user", "content": "Extract value"}]},
        )

        if mode is Mode.JSON_SCHEMA:
            schema = json.loads(
                result_kwargs["outputConfig"]["textFormat"]["structure"]["jsonSchema"][
                    "schema"
                ]
            )
        else:
            schema = result_kwargs["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"][
                "json"
            ]

        assert field_name in schema["properties"]

    @pytest.mark.parametrize("mode", [Mode.JSON_SCHEMA, Mode.TOOLS_STRICT])
    def test_native_modes_reject_constraints_inside_any_of(self, mode: Mode) -> None:
        handler = mode_registry.get_handlers(Provider.BEDROCK, mode)

        with pytest.raises(ConfigurationError) as exc_info:
            handler.request_handler(
                AnyOfConstrainedResponse,
                {"messages": [{"role": "user", "content": "Extract value"}]},
            )

        message = str(exc_info.value)
        assert "`minimum`" in message
        assert "$.properties.value.anyOf[0]" in message

    def test_json_schema_response_parses_text(self) -> None:
        handler = mode_registry.get_handlers(Provider.BEDROCK, Mode.JSON_SCHEMA)
        response = _bedrock_text_response(
            '{"user":{"name":"Ada","age":37},"addresses":[{"city":"London"}]}'
        )

        result = handler.response_parser(response, Profile)

        assert isinstance(result, Profile)
        assert result.user.name == "Ada"
        assert result.addresses[0].city == "London"

    def test_json_schema_response_rejects_streaming(self) -> None:
        handler = mode_registry.get_handlers(Provider.BEDROCK, Mode.JSON_SCHEMA)

        with pytest.raises(ConfigurationError, match="Streaming is not supported"):
            handler.response_parser(
                _bedrock_text_response('{"answer":4}'),
                Answer,
                stream=True,
            )

    def test_locked_sdk_exposes_native_converse_shapes(self) -> None:
        from botocore.session import Session

        operation = (
            Session().get_service_model("bedrock-runtime").operation_model("Converse")
        )
        input_shape = cast(Any, operation.input_shape)
        tool_spec = (
            input_shape.members["toolConfig"]
            .members["tools"]
            .member.members["toolSpec"]
        )

        assert "outputConfig" in input_shape.members
        assert "strict" in tool_spec.members
