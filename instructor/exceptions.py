"""Backward compatibility module for instructor.exceptions imports.

.. deprecated:: 1.11.0
    This module is deprecated. Import exceptions from `instructor.core` instead.
    For example: `from instructor.core import InstructorRetryException`
"""

import warnings

# Show deprecation warning when this module is imported
warnings.warn(
    "Importing from 'instructor.exceptions' is deprecated and will be removed in a future version. "
    "Please import from 'instructor.core' instead. "
    "For example: 'from instructor.core import InstructorRetryException'",
    DeprecationWarning,
    stacklevel=2,
)

# Explicit re-exports for better IDE support and clarity
from .core.exceptions import (
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

__all__ = [
    "AsyncValidationError",
    "ClientError",
    "ConfigurationError",
    "FailedAttempt",
    "IncompleteOutputException",
    "InstructorError",
    "InstructorRetryException",
    "ModeError",
    "MultimodalError",
    "ProviderError",
    "ResponseParsingError",
    "TokenBudgetError",
    "TokenBudgetExceeded",
    "TokenUsageUnavailableError",
    "ValidationError",
]
