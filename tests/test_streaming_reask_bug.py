"""Test for streaming reask bug fix.

Bug: When using streaming mode with max_retries > 1, if validation fails,
the reask handlers crash with "'Stream' object has no attribute 'choices'"
because they expect a ChatCompletion but receive a Stream object.

GitHub Issue: https://github.com/jxnl/instructor/issues/1991
"""

from collections.abc import Generator
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


def test_streaming_retry_exhaustion_preserves_validation_error():
    import json

    import httpx
    import instructor
    from openai import OpenAI
    from instructor.core.exceptions import InstructorRetryException

    class ImpossibleModel(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def always_fail(cls, _value: str) -> str:
            raise ValueError("controlled invalid value")

    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        call = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": body["tools"][0]["function"]["name"],
                "arguments": '{"value":"test"}',
            },
        }
        chunk = {
            "id": "chatcmpl-local",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "local",
            "choices": [{"index": 0, "delta": {"tool_calls": [call]}}],
        }
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    transport = httpx.MockTransport(respond)
    with OpenAI(api_key="local", http_client=httpx.Client(transport=transport)) as sdk:
        client = instructor.from_openai(sdk)
        with pytest.raises(InstructorRetryException) as caught:
            list(
                client.chat.completions.create_partial(
                    model="local",
                    max_retries=2,
                    messages=[{"role": "user", "content": "test"}],
                    response_model=ImpossibleModel,
                )
            )
    assert len(requests) == caught.value.n_attempts == 3
    assert isinstance(caught.value.__cause__, ValidationError)
    assert "controlled invalid value" in str(caught.value.__cause__)


@pytest.mark.parametrize("outcome", ["empty", "close", "complete", "invalid-final"])
def test_sync_partial_retry_stream_lifecycle(outcome: str) -> None:
    from types import SimpleNamespace

    import instructor
    from instructor.v2.core.providers import Provider
    from instructor.v2.core.retry import retry_sync_v2
    from instructor.v2.dsl.partial import Partial

    class Result(BaseModel):
        value: int

    closed = False
    calls = 0

    def stream():
        nonlocal closed
        try:
            if outcome == "empty":
                return
            for content in (
                '{"value":',
                '"wrong"}' if outcome == "invalid-final" else "7}",
            ):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
                )
        finally:
            closed = True

    def create(**_kwargs: Any):
        nonlocal calls
        calls += 1
        return stream()

    result = retry_sync_v2(
        create,
        Partial[Result],
        Provider.OPENAI,
        instructor.Mode.JSON_SCHEMA,
        None,
        2,
        (),
        {"stream": True},
        True,
    )
    assert isinstance(result, Generator)
    if outcome == "close":
        result.close()
    elif outcome == "invalid-final":
        assert next(result).value is None
        with pytest.raises(ValidationError):
            next(result)
    else:
        assert list(result) == (
            []
            if outcome == "empty"
            else [Result.model_construct(value=None), Result(value=7)]
        )
    assert closed and calls == 1
