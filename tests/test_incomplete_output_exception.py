"""
Tests that IncompleteOutputException propagates directly from retry_sync/retry_async
without being wrapped in InstructorRetryException.

Regression tests for issue #2273.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

import instructor
from instructor.core.exceptions import IncompleteOutputException
from instructor.mode import Mode


class User(BaseModel):
    name: str
    age: int


def _make_truncated_response() -> Mock:
    """Return a mock response that simulates a MAX_TOKENS truncation."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = '{"name": "Alice"'  # truncated JSON
    mock_response.choices[0].finish_reason = "length"
    mock_response.usage = None
    return mock_response


def _raise_incomplete(*_args, **_kwargs):
    raise IncompleteOutputException(last_completion=None)


def test_incomplete_output_exception_is_catchable_sync():
    """IncompleteOutputException must be catchable directly, not wrapped."""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(side_effect=_raise_incomplete)

    client = instructor.patch(mock_client, mode=Mode.TOOLS)

    with pytest.raises(IncompleteOutputException):
        client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=0,
        )


def test_incomplete_output_not_wrapped_in_instructor_retry_exception_sync():
    """IncompleteOutputException must NOT be wrapped in InstructorRetryException."""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(side_effect=_raise_incomplete)

    client = instructor.patch(mock_client, mode=Mode.TOOLS)

    with pytest.raises(IncompleteOutputException):
        client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=3,
        )


def test_incomplete_output_exception_with_max_retries_zero():
    """IncompleteOutputException is raised immediately when max_retries=0."""
    call_count = {"n": 0}

    def _raise_incomplete_and_count(*_args, **_kwargs):
        call_count["n"] += 1
        raise IncompleteOutputException(last_completion=None)

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = _raise_incomplete_and_count

    client = instructor.patch(mock_client, mode=Mode.TOOLS)

    with pytest.raises(IncompleteOutputException):
        client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=0,
        )

    # Should only have been called once (no retries for IncompleteOutputException)
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_incomplete_output_exception_is_catchable_async():
    """Async path: IncompleteOutputException must be directly catchable."""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_raise_incomplete)

    client = instructor.patch(mock_client, mode=Mode.TOOLS)

    with pytest.raises(IncompleteOutputException):
        await client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=0,
        )
