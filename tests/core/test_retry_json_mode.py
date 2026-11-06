"""
Test that retry mechanism works correctly with JSON mode.
Specifically tests that JSONDecodeError is properly caught by retry handler.

This is a regression test for issue #1856.
"""

import json
import pytest
from unittest.mock import Mock
from pydantic import BaseModel, ValidationError

import instructor
from instructor.core.exceptions import InstructorRetryException
from instructor.mode import Mode


class User(BaseModel):
    name: str
    age: int


def test_json_decode_error_caught_by_retry():
    """Test that JSON errors are caught by retry handler, not generic Exception handler.

    This is a regression test for issue #1856 where JSONDecodeError was wrapped
    in ValueError, causing it to be caught by the generic Exception handler instead
    of the specific validation error handler that calls handle_reask_kwargs.

    Note: In strict mode, Pydantic raises ValidationError with 'Invalid JSON' message.
    In non-strict mode, json.loads raises JSONDecodeError directly.
    Both are now properly caught by the retry handler.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "invalid json {"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = None

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=2,
        )

    exception = exc_info.value
    assert exception.n_attempts == 3
    assert exception.failed_attempts is not None
    assert len(exception.failed_attempts) == 3

    for attempt in exception.failed_attempts:
        assert isinstance(attempt.exception, (json.JSONDecodeError, ValidationError))
        if isinstance(attempt.exception, ValidationError):
            assert "Invalid JSON" in str(attempt.exception)


def test_validation_error_caught_by_retry():
    """Test that ValidationError is still caught by retry handler."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = '{"name": "John"}'
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = None

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    client = instructor.patch(mock_client, mode=Mode.JSON)

    with pytest.raises(InstructorRetryException) as exc_info:
        client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=2,
        )

    exception = exc_info.value
    assert exception.n_attempts == 3
    assert exception.failed_attempts is not None
    assert len(exception.failed_attempts) == 3

    for attempt in exception.failed_attempts:
        assert isinstance(attempt.exception, ValidationError)
