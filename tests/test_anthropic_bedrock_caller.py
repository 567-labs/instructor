"""Test that reask_anthropic_tools excludes None fields when serializing content blocks.

AWS Bedrock does not populate the `caller` field on ToolUseBlock — it returns
caller=None.  When reask_anthropic_tools called content.model_dump() the
resulting dict contained "caller": null, which the Anthropic API rejected with
HTTP 400.

GitHub Issue: https://github.com/instructor-ai/instructor/issues/2277
"""

import pytest

from pydantic import ValidationError, BaseModel, field_validator

anthropic = pytest.importorskip("anthropic", reason="anthropic package required")


def _make_validation_error() -> ValidationError:
    class M(BaseModel):
        v: int = 0

        @field_validator("v")
        @classmethod
        def always_fail(cls, val: int) -> int:
            raise ValueError(f"forced failure: {val}")

    try:
        M(v=1)
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")  # pragma: no cover


def _build_bedrock_message() -> "anthropic.types.Message":
    from anthropic.types import Message, ToolUseBlock, Usage

    return Message(
        id="msg_bedrock_test",
        content=[
            ToolUseBlock(
                id="toolu_bedrock_abc",
                input={"summary": "hello"},
                name="MyExtraction",
                type="tool_use",
                caller=None,  # Bedrock leaves this field unpopulated
            )
        ],
        model="anthropic.claude-sonnet-4-6",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=10),
    )


class TestBedrockCallerNoneReask:
    """Regression tests for Bedrock caller=None serialisation bug."""

    def test_reask_does_not_include_null_caller(self):
        """Assistant content dicts must not contain 'caller': None after reask."""
        from instructor.v2.providers.anthropic.handlers import AnthropicToolsHandler

        kwargs = {
            "messages": [{"role": "user", "content": "extract something"}],
        }
        response = _build_bedrock_message()
        exception = _make_validation_error()

        result = AnthropicToolsHandler().handle_reask(kwargs, response, exception)

        # Find the assistant turn that was appended
        assistant_msgs = [m for m in result["messages"] if m.get("role") == "assistant"]
        assert assistant_msgs, "reask must append an assistant message"

        for block in assistant_msgs[0]["content"]:
            assert "caller" not in block, (
                f"'caller' field must be excluded when it is None, got: {block}"
            )

    def test_reask_tool_use_id_is_preserved(self):
        """The tool_use_id in the follow-up user message must match the block id."""
        from instructor.v2.providers.anthropic.handlers import AnthropicToolsHandler

        kwargs = {
            "messages": [{"role": "user", "content": "extract something"}],
        }
        response = _build_bedrock_message()
        exception = _make_validation_error()

        result = AnthropicToolsHandler().handle_reask(kwargs, response, exception)

        user_msgs = [m for m in result["messages"] if m.get("role") == "user"]
        # Last user message is the tool_result reask
        tool_result_msg = user_msgs[-1]
        assert isinstance(tool_result_msg["content"], list)
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_bedrock_abc"
