"""Protocol definitions for v2 mode handlers.

Defines the interfaces that all mode handlers must implement for type safety
and consistency across providers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator, Iterable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
RequestModel = TypeVar("RequestModel")


class RequestHandler(Protocol):
    """Prepares request kwargs for a specific mode.

    Takes the response model and existing kwargs, returns modified kwargs
    with mode-specific parameters (e.g., tools, response_format).
    """

    def __call__(
        self,
        response_model: type[RequestModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[RequestModel] | None, dict[str, Any]]:
        """Prepare request kwargs for this mode.

        Args:
            response_model: The Pydantic model to extract
            kwargs: Original request kwargs

        Returns:
            Tuple of (possibly modified response_model, modified kwargs)
        """
        ...


class ReaskHandler(Protocol):
    """Handles validation failures and prepares retry requests.

    Takes the original kwargs, failed response, and exception, returns
    modified kwargs for the retry attempt.
    """

    def __call__(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle validation failure and prepare retry.

        Args:
            kwargs: Original request kwargs
            response: Failed API response
            exception: Validation exception that occurred

        Returns:
            Modified kwargs for retry request
        """
        ...


class ResponseParser(Protocol):
    """Parses API response into validated Pydantic model.

    Extracts the structured data from the API response and validates
    it against the response model.
    """

    def __call__(
        self,
        response: Any,
        response_model: type[T],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> T:
        """Parse and validate response into model.

        Args:
            response: Raw API response
            response_model: Pydantic model to validate against
            validation_context: Optional context for validation
            strict: Optional strict validation mode
            stream: Whether the response is from a streaming request
            is_async: Whether the request is async

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If response doesn't match model
        """
        ...


class StreamExtractor(Protocol):
    """Extract JSON chunks from a streaming response."""

    def __call__(self, completion: Iterable[Any]) -> Generator[str, None, None]:
        """Yield JSON chunks from a streaming response."""
        ...


class AsyncStreamExtractor(Protocol):
    """Extract JSON chunks from an async streaming response."""

    async def __call__(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Yield JSON chunks from an async streaming response."""
        ...


class MessageConverter(Protocol):
    """Convert multimodal messages to provider-specific formats."""

    def __call__(
        self, messages: list[dict[str, Any]], autodetect_images: bool = False
    ) -> list[dict[str, Any]]:
        """Convert messages to provider-specific formats."""
        ...


class TemplateHandler(Protocol):
    """Apply template context to provider-specific message formats."""

    def __call__(
        self, kwargs: dict[str, Any], context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Return kwargs with templates applied."""
        ...
