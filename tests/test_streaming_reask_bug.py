"""Test for streaming reask bug fix.

Bug: When using streaming mode with max_retries > 1, if validation fails,
the reask handlers crash with "'Stream' object has no attribute 'choices'"
because they expect a ChatCompletion but receive a Stream object.

GitHub Issue: https://github.com/jxnl/instructor/issues/1991
"""

from typing import Any, Optional

import pytest
from pydantic import ValidationError, BaseModel, field_validator

from instructor.mode import Mode
from instructor.processing.response import handle_reask_kwargs


class MockStream:
    """Mock Stream object that mimics openai.Stream behavior."""

    def __iter__(self):
        return iter([])

    def __next__(self):
        raise StopIteration


class MockResponsesToolCall:
    """Mock tool call item in a responses output list."""

    def __init__(
        self,
        arguments: str,
        name: Optional[str] = None,
        call_id: Optional[str] = None,
        item_type: str = "function_call",
    ) -> None:
        self.arguments = arguments
        self.name = name
        self.call_id = call_id
        self.type = item_type


class MockResponsesReasoningItem:
    """Mock reasoning item in a responses output list."""

    type = "reasoning"


class MockResponsesResponse:
    """Mock Responses API response with output items."""

    def __init__(self, output: list[Any]) -> None:
        self.output = output


def create_mock_validation_error():
    """Create a real Pydantic ValidationError for testing."""

    class TestModel(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def must_have_space(cls, v):
            if " " not in v:
                raise ValueError("must contain space")
            return v

    try:
        TestModel(name="John")
    except ValidationError as e:
        return e


class TestStreamingReaskBug:
    """Tests for the streaming reask bug fix."""

    def test_reask_tools_with_stream_object_does_not_crash(self):
        """Test that reask_tools handles Stream objects without crashing.

        Previously, this would crash with:
        "'Stream' object has no attribute 'choices'"
        """
        mock_stream = MockStream()
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"type": "function", "function": {"name": "test"}}],
        }
        exception = create_mock_validation_error()

        # This should not raise an AttributeError
        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.TOOLS,
            response=mock_stream,
            exception=exception,
        )

        # Should return modified kwargs with error message
        assert "messages" in result
        assert len(result["messages"]) > 1  # Original + error message

    def test_reask_anthropic_tools_with_stream_object(self):
        """Test that Anthropic reask handler handles Stream objects."""
        mock_stream = MockStream()
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
        }
        exception = create_mock_validation_error()

        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.ANTHROPIC_TOOLS,
            response=mock_stream,
            exception=exception,
        )

        assert "messages" in result

    def test_reask_with_none_response(self):
        """Test that reask handlers handle None response gracefully."""
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
        }
        exception = create_mock_validation_error()

        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.TOOLS,
            response=None,
            exception=exception,
        )

        assert "messages" in result

    def test_reask_responses_tools_skips_reasoning_items_and_includes_details(self):
        """Test responses reask ignores reasoning items and adds tool details."""
        mock_response = MockResponsesResponse(
            output=[
                MockResponsesReasoningItem(),
                MockResponsesToolCall(
                    arguments='{"name": "Jane"}',
                    name="extract_person",
                    call_id="call_123",
                ),
            ]
        )
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
        }
        exception = create_mock_validation_error()

        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.RESPONSES_TOOLS,
            response=mock_response,
            exception=exception,
        )

        assert "messages" in result
        assert len(result["messages"]) == 2
        reask_content = result["messages"][-1]["content"]
        assert "tool call name=extract_person, id=call_123" in reask_content
        assert '{"name": "Jane"}' in reask_content

    def test_reask_md_json_with_stream_object(self):
        """Test that MD_JSON reask handler handles Stream objects."""
        mock_stream = MockStream()
        kwargs = {
            "messages": [{"role": "user", "content": "test"}],
        }
        exception = create_mock_validation_error()

        result = handle_reask_kwargs(
            kwargs=kwargs,
            mode=Mode.MD_JSON,
            response=mock_stream,
            exception=exception,
        )

        assert "messages" in result


@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="openai not installed"),
    reason="openai not installed",
)
class TestStreamingReaskIntegration:
    """Integration tests that require OpenAI API key."""

    @pytest.fixture
    def client(self):
        """Create instructor client if API key available."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        import instructor
        from openai import OpenAI

        return instructor.from_openai(OpenAI())

    def test_streaming_with_retries_and_failing_validator(self, client):
        """Test that streaming with retries doesn't crash on validation failure.

        This test verifies that the reask handler doesn't crash with
        "'Stream' object has no attribute 'choices'" when validation fails
        during streaming. The actual validation outcome depends on LLM behavior.
        """

        class ImpossibleModel(BaseModel):
            """Model with a validator that always fails."""

            value: str

            @field_validator("value")
            @classmethod
            def always_fail(cls, v: str) -> str:  # noqa: ARG003
                raise ValueError("This validator always fails for testing")

        # This should not crash with AttributeError about Stream.choices
        # It should raise InstructorRetryException after retries are exhausted
        from instructor.core.exceptions import InstructorRetryException

        with pytest.raises(InstructorRetryException):
            list(
                client.chat.completions.create_partial(
                    model="gpt-4o-mini",
                    max_retries=2,
                    messages=[
                        {
                            "role": "user",
                            "content": "Return value='test'",
                        }
                    ],
                    response_model=ImpossibleModel,
                )
            )
