"""Token-budget policy for the v2 runtime."""

from __future__ import annotations

from typing import Any

from instructor.v2.core.errors import (
    FailedAttempt,
    TokenBudgetError,
    TokenBudgetExceeded,
    TokenUsageUnavailableError,
)
from instructor.v2.core.messages import extract_messages
from instructor.v2.core.usage import _usage_snapshot, _usage_total_tokens


def _validate_token_budget(
    token_budget: int | None,
    *,
    response_model: object,
    kwargs: dict[str, Any],
) -> None:
    if token_budget is None:
        return
    if isinstance(token_budget, bool) or not isinstance(token_budget, int):
        raise TypeError("token_budget must be a positive integer or None")
    if token_budget <= 0:
        raise ValueError("token_budget must be greater than zero")
    if response_model is None:
        raise ValueError("token_budget requires a structured response_model")
    if kwargs.get("stream"):
        raise ValueError("token_budget is not supported for streaming responses")


def _budget_error(
    *,
    token_budget: int | None,
    usage_available: bool,
    total_usage: Any,
    attempt_number: int,
    response: Any,
    kwargs: dict[str, Any],
    failed_attempts: list[FailedAttempt],
) -> TokenBudgetError | None:
    if token_budget is None:
        return None

    error_kwargs = {
        "budget": token_budget,
        "last_completion": response,
        "messages": extract_messages(kwargs),
        "n_attempts": attempt_number,
        "total_usage": _usage_snapshot(total_usage),
        "create_kwargs": kwargs,
        "failed_attempts": failed_attempts,
    }
    if not usage_available:
        return TokenUsageUnavailableError(
            "Token budget cannot be enforced because the provider response did "
            "not include compatible usage metadata",
            **error_kwargs,
        )

    used_tokens = _usage_total_tokens(total_usage)
    if used_tokens is None:
        return TokenUsageUnavailableError(
            "Token budget cannot be enforced because total token usage is unavailable",
            **error_kwargs,
        )
    if used_tokens >= token_budget:
        return TokenBudgetExceeded(
            f"Token budget exhausted after {used_tokens} tokens across "
            f"{attempt_number} attempts (budget: {token_budget})",
            **error_kwargs,
        )
    return None
