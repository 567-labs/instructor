"""v2 Perplexity client factory."""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.providers.openai.client import _from_openai_compat

# Ensure handlers are registered.
from instructor.v2.providers.perplexity import handlers  # noqa: F401


@overload
def from_perplexity(
    client: openai.OpenAI,
    mode: Mode = Mode.MD_JSON,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_perplexity(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.MD_JSON,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_perplexity(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.MD_JSON,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    return _from_openai_compat(
        client=client,
        provider=Provider.PERPLEXITY,
        mode=mode,
        model=model,
        **kwargs,
    )


__all__ = ["from_perplexity"]
