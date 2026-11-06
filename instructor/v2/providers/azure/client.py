"""v2 Azure OpenAI / AI Foundry client factory.

Azure OpenAI is fully OpenAI-compatible, so this factory delegates to the
shared ``_from_openai_compat`` helper with ``Provider.AZURE_OPENAI``.
This enables ``from_azure()`` and ``from_provider("azure/...")`` ergonomics,
automatic ``AZURE_OPENAI_*`` env detection, and proper provider attribution.
"""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.providers.openai.client import _from_openai_compat

# Ensure OpenAI handlers are registered (decorators auto-register on import).
from instructor.v2.providers.openai import handlers  # noqa: F401


@overload
def from_azure(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_azure(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_azure(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Wrap a dated Azure client or a Foundry v1 OpenAI client.

    Use the deployment name as the model. Available modes depend on the
    deployment and API version.
    """
    return _from_openai_compat(
        client=client,
        provider=Provider.AZURE_OPENAI,
        mode=mode,
        model=model,
        **kwargs,
    )


__all__ = ["from_azure"]
