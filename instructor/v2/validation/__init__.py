"""Validation components owned by the v2 runtime."""

from instructor.v2.core.errors import AsyncValidationError
from typing import TYPE_CHECKING, Any
from instructor.v2.validation.async_validators import (
    ASYNC_MODEL_VALIDATOR_KEY,
    ASYNC_VALIDATOR_KEY,
    AsyncValidationContext,
    async_field_validator,
    async_model_validator,
    model_declares_async_validators,
    run_async_validators,
)

if TYPE_CHECKING:
    from instructor.v2.core.validators import Validator
    from instructor.v2.validation.llm_validators import llm_validator, openai_moderation


def __getattr__(name: str) -> Any:
    if name == "Validator":
        from instructor.v2.core.validators import Validator

        return Validator
    if name in {"llm_validator", "openai_moderation"}:
        from instructor.v2.validation import llm_validators

        return getattr(llm_validators, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AsyncValidationContext",
    "AsyncValidationError",
    "async_field_validator",
    "async_model_validator",
    "model_declares_async_validators",
    "run_async_validators",
    "ASYNC_VALIDATOR_KEY",
    "ASYNC_MODEL_VALIDATOR_KEY",
    "Validator",
    "llm_validator",
    "openai_moderation",
]
