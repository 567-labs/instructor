"""v2 MiniMax client factory."""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.providers.openai.client import _from_openai_compat

# Ensure handlers are registered.
from instructor.v2.providers.minimax import handlers  # noqa: F401


@overload
def from_minimax(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_minimax(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_minimax(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a MiniMax OpenAI-compatible client.

    Args:
        client: An ``openai.OpenAI`` or ``openai.AsyncOpenAI`` instance pointed at
            the MiniMax API (``base_url="https://api.minimax.io/v1"``).
        mode: The structured-output mode to use. Defaults to ``Mode.TOOLS``
            (tool calling). Use ``Mode.MD_JSON`` or the legacy ``Mode.MINIMAX_JSON``
            for system-prompt-based JSON output.
        model: Optional default model name.
        **kwargs: Additional keyword arguments forwarded to the Instructor constructor.

    Returns:
        An ``Instructor`` or ``AsyncInstructor`` instance.

    Examples:
        >>> import openai, instructor
        >>> client = openai.OpenAI(
        ...     api_key="your-key",
        ...     base_url="https://api.minimax.io/v1",
        ... )
        >>> ic = instructor.from_minimax(client)
    """
    return _from_openai_compat(
        client=client,
        provider=Provider.MINIMAX,
        mode=mode,
        model=model,
        **kwargs,
    )


__all__ = ["from_minimax"]
