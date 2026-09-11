"""v2 Requesty client factory."""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.providers.openai.client import _from_openai_compat

# Ensure Requesty handlers are registered (overrides JSON_SCHEMA).
from instructor.v2.providers.requesty import handlers  # noqa: F401


@overload
def from_requesty(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_requesty(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_requesty(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    return _from_openai_compat(
        client=client,
        provider=Provider.REQUESTY,
        mode=mode,
        model=model,
        **kwargs,
    )


__all__ = ["from_requesty"]
