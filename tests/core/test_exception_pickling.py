"""Offline regression tests for exception serialization."""

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import pytest

from instructor.v2.core import errors


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize(
    "exception",
    [
        errors.InstructorError("Request failed"),
        errors.IncompleteOutputException(
            message="Output truncated", last_completion={"content": "partial"}
        ),
        errors.InstructorRetryException(
            "Retries exhausted",
            n_attempts=3,
            total_usage={"total_tokens": 120},
            last_completion={"content": "invalid"},
            messages=[{"role": "user", "content": "Extract a user"}],
            create_kwargs={"model": "test-model", "max_tokens": 40},
        ),
        errors.TokenBudgetError(
            "Token budget reached", budget=100, n_attempts=2, total_usage=100
        ),
        errors.TokenBudgetExceeded(
            "Token budget exceeded", budget=100, n_attempts=2, total_usage=120
        ),
        errors.TokenUsageUnavailableError(
            "Usage metadata missing", budget=100, n_attempts=1, total_usage=None
        ),
        errors.ValidationError("Missing required field"),
        errors.ProviderError("openai", "Rate limit exceeded"),
        errors.ConfigurationError("Missing API key"),
        errors.ModeError("tools", "anthropic", ["anthropic_tools", "json"]),
        errors.ClientError("Client not initialized"),
        errors.AsyncValidationError("Async validation failed"),
        errors.ResponseParsingError(
            "Invalid JSON", mode="json", raw_response={"content": "{"}
        ),
        errors.MultimodalError(
            "Unsupported image", content_type="image", file_path="/tmp/image.bmp"
        ),
    ],
    ids=lambda exception: type(exception).__name__,
)
def test_exception_round_trip(exception: errors.InstructorError, protocol: int) -> None:
    restored = pickle.loads(pickle.dumps(exception, protocol=protocol))

    assert type(restored) is type(exception)
    assert str(restored) == str(exception)
    assert restored.args == exception.args
    assert restored.__dict__ == exception.__dict__


def test_failed_attempts_round_trip() -> None:
    exception = errors.InstructorRetryException(
        "Retries exhausted",
        n_attempts=2,
        total_usage=100,
        failed_attempts=[
            errors.FailedAttempt(
                1,
                errors.ResponseParsingError("Invalid JSON", mode="json"),
                {"content": "{"},
            ),
            errors.FailedAttempt(2, ValueError("Missing name"), None),
        ],
    )

    restored = pickle.loads(pickle.dumps(exception))

    assert str(restored) == str(exception)
    assert restored.args == exception.args
    assert len(restored.failed_attempts) == len(exception.failed_attempts)
    for actual, expected in zip(restored.failed_attempts, exception.failed_attempts):
        assert actual.attempt_number == expected.attempt_number
        assert actual.completion == expected.completion
        assert type(actual.exception) is type(expected.exception)
        assert actual.exception.args == expected.exception.args
        assert actual.exception.__dict__ == expected.exception.__dict__


def test_async_validation_errors_round_trip() -> None:
    exception = errors.AsyncValidationError(
        "Async validation failed", errors=[ValueError("Invalid name")]
    )

    restored = pickle.loads(pickle.dumps(exception))

    assert str(restored) == str(exception)
    assert restored.args == exception.args
    assert len(restored.errors) == 1
    assert type(restored.errors[0]) is ValueError
    assert restored.errors[0].args == exception.errors[0].args


def test_exception_chaining_is_not_pickled() -> None:
    # Tracebacks and chaining are omitted by standard exception pickling too.
    cause = ValueError(lambda: None)
    try:
        try:
            raise cause
        except ValueError:
            raise errors.InstructorError("Request failed") from cause
    except errors.InstructorError as exception:
        traceback = exception.__traceback__
        restored = pickle.loads(pickle.dumps(exception))

        assert exception.__cause__ is cause
        assert exception.__context__ is cause
        assert exception.__traceback__ is traceback
        assert exception.__suppress_context__ is True
        assert restored.__cause__ is None
        assert restored.__context__ is None
        assert restored.__traceback__ is None
        assert restored.__suppress_context__ is False


def _raise_retry_exception() -> None:
    raise errors.InstructorRetryException(
        "Retries exhausted", n_attempts=3, total_usage={"total_tokens": 120}
    )


def test_retry_exception_from_process_pool() -> None:
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        future = executor.submit(_raise_retry_exception)
        with pytest.raises(errors.InstructorRetryException) as exc_info:
            future.result(timeout=30)

    assert exc_info.value.n_attempts == 3
    assert exc_info.value.total_usage == {"total_tokens": 120}
    assert str(exc_info.value) == "Retries exhausted"
    assert exc_info.value.__cause__ is not None
