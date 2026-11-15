from __future__ import annotations

import pickle
from typing import Any

import pytest

from instructor.v2.core.errors import (
    AsyncValidationError,
    ClientError,
    ConfigurationError,
    FailedAttempt,
    IncompleteOutputException,
    InstructorError,
    InstructorRetryException,
    ModeError,
    MultimodalError,
    ProviderError,
    ResponseParsingError,
    TokenBudgetError,
    TokenBudgetExceeded,
    TokenUsageUnavailableError,
    ValidationError,
)

ATTEMPTS = [FailedAttempt(1, ValueError("bad"), "partial")]
RETRY_KWARGS: dict[str, Any] = {
    "last_completion": "done",
    "messages": [{"role": "user", "content": "hi"}],
    "n_attempts": 2,
    "total_usage": 30,
    "create_kwargs": {"model": "m"},
    "failed_attempts": ATTEMPTS,
}

CASES: list[tuple[BaseException, dict[str, Any]]] = [
    (InstructorError("base", failed_attempts=ATTEMPTS), {"failed_attempts": ATTEMPTS}),
    (IncompleteOutputException(last_completion="x"), {"last_completion": "x"}),
    (
        InstructorRetryException("retry", **RETRY_KWARGS),
        {"n_attempts": 2, "total_usage": 30, "create_kwargs": {"model": "m"}},
    ),
    (TokenBudgetError("tb", budget=10, **RETRY_KWARGS), {"budget": 10}),
    (TokenBudgetExceeded("tb", budget=10, **RETRY_KWARGS), {"budget": 10}),
    (TokenUsageUnavailableError("tb", budget=10, **RETRY_KWARGS), {"budget": 10}),
    (ValidationError("invalid"), {}),
    (ProviderError("openai", "down"), {"provider": "openai"}),
    (ConfigurationError("config"), {}),
    (
        ModeError("bad", "openai", ["tools", "json"]),
        {"mode": "bad", "provider": "openai", "valid_modes": ["tools", "json"]},
    ),
    (ClientError("client"), {}),
    (AsyncValidationError("async", errors=[ValueError("e")]), {}),
    (
        ResponseParsingError("parse", mode="tools", raw_response="raw"),
        {"mode": "tools", "raw_response": "raw"},
    ),
    (
        MultimodalError("mm", content_type="image", file_path="a.png"),
        {"content_type": "image", "file_path": "a.png"},
    ),
]


@pytest.mark.parametrize(
    ("error", "attrs"), CASES, ids=[type(e).__name__ for e, _ in CASES]
)
def test_instructor_errors_survive_pickle_round_trip(
    error: BaseException, attrs: dict[str, Any]
) -> None:
    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert restored.args == error.args
    for name, value in attrs.items():
        # Exceptions compare by identity, so compare nested values by repr.
        assert repr(getattr(restored, name)) == repr(value)
