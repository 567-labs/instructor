"""Base handler class for v2 mode handlers.

Provides the common interface and default implementations for mode handlers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from instructor.v2.dsl.parallel import ParallelBase


class ModeHandler(ABC):
    """Base class for mode handlers.

    Subclasses must implement prepare_request, handle_reask, and parse_response.
    These methods define how requests are prepared, errors are handled, and
    responses are parsed for a specific mode.
    """

    @abstractmethod
    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request kwargs for this mode.

        Args:
            response_model: Pydantic model to extract (or None for unstructured)
            kwargs: Original request kwargs from user

        Returns:
            Tuple of (possibly modified response_model, modified kwargs)

        Example:
            For TOOLS mode, this adds "tools" and "tool_choice" to kwargs.
            For JSON mode, this adds JSON schema to system message.
        """
        ...

    @abstractmethod
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle validation failure and prepare retry request.

        Args:
            kwargs: Original request kwargs
            response: Failed API response
            exception: Validation exception that occurred

        Returns:
            Modified kwargs for retry attempt

        Example:
            For TOOLS mode, appends tool_result with error message.
            For JSON mode, appends user message with validation error.
        """
        ...

    @abstractmethod
    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel:
        """Parse API response into validated Pydantic model.

        Args:
            response: Raw API response
            response_model: Pydantic model to validate against
            validation_context: Optional context for Pydantic validation
            strict: Optional strict validation mode

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If response doesn't match model schema

        Example:
            For TOOLS mode, extracts tool_use blocks and validates.
            For JSON mode, extracts JSON from text blocks and validates.
        """
        ...

    def __repr__(self) -> str:
        """String representation of handler."""
        return f"<{self.__class__.__name__}>"

    def _finalize_parsed_result(
        self,
        response_model: type[BaseModel] | ParallelBase,
        response: Any,
        parsed: Any,
    ) -> Any:
        """Normalize common DSL results without interpreting provider payloads."""
        from instructor.v2.dsl.iterable import IterableBase
        from instructor.v2.dsl.parallel import ParallelBase
        from instructor.v2.dsl.simple_type import AdapterBase

        if isinstance(parsed, IterableBase):
            return [task for task in parsed.tasks]
        if isinstance(response_model, ParallelBase):
            return parsed
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed
