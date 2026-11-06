"""Exception handling utilities for v2 core infrastructure.

Provides centralized exception handling, validation, and error context
for the v2 registry-based architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from instructor.v2.core.errors import ConfigurationError

if TYPE_CHECKING:
    from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider


class RegistryError(ConfigurationError):
    """Exception raised for registry-related configuration errors.

    Raised when there are issues with handler registration, lookup,
    or mode/provider compatibility in the v2 registry.
    """


class ValidationContextError(ConfigurationError):
    """Exception raised for validation context configuration errors.

    Raised when there are conflicting or invalid validation context
    parameters passed to patched functions.
    """


class RegistryValidationMixin:
    """Mixin providing registry validation helper methods."""

    @staticmethod
    def validate_mode_registration(provider: Provider, mode: Mode) -> None:
        """Validate that a mode is registered for a provider.

        Args:
            provider: Provider enum value
            mode: Mode enum value

        Raises:
            RegistryError: If mode is not registered for provider
        """
        from instructor.v2.core.registry import mode_registry

        if not mode_registry.is_registered(provider, mode):
            available = mode_registry.list_modes()
            raise RegistryError(
                f"Mode {mode} is not registered for provider {provider}. "
                f"Available modes: {available}"
            )

    @staticmethod
    def validate_context_parameters(
        context: dict[str, Any] | None,
        validation_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Validate and merge context parameters.

        Args:
            context: New-style context parameter
            validation_context: Deprecated validation_context parameter

        Returns:
            Merged context dict or None

        Raises:
            ValidationContextError: If both parameters are provided
        """
        if context is not None and validation_context is not None:
            raise ValidationContextError(
                "Cannot provide both 'context' and 'validation_context'. "
                "Use 'context' instead."
            )

        if validation_context is not None and context is None:
            import warnings

            warnings.warn(
                "'validation_context' is deprecated. Use 'context' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return validation_context

        return context
