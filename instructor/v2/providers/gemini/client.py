"""v2 Gemini client factory."""

from __future__ import annotations

import importlib
from typing import Any, Literal, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered.
from instructor.v2.providers.gemini import handlers  # noqa: F401

try:
    genai: Any = importlib.import_module("google.generativeai")
except ImportError:
    genai = None


@overload
def from_gemini(
    client: Any,
    mode: Mode = Mode.MD_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_gemini(
    client: Any,
    mode: Mode = Mode.MD_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_gemini(
    client: Any,
    mode: Mode = Mode.MD_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    from instructor.v2.core.registry import mode_registry, normalize_mode

    normalized_mode = normalize_mode(Provider.GEMINI, mode)
    if not mode_registry.is_registered(Provider.GEMINI, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.GEMINI)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.GEMINI.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    if genai is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "google-generativeai is not installed. Install it with: "
            "pip install google-generativeai"
        )

    generative_model_type = getattr(genai, "GenerativeModel", None)
    if generative_model_type is None or not isinstance(client, generative_model_type):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "Client must be an instance of genai.GenerativeModel. "
            f"Got: {type(client).__name__}"
        )

    create = client.generate_content_async if use_async else client.generate_content
    patched_create = patch_v2(
        func=create,
        provider=Provider.GEMINI,
        mode=normalized_mode,
    )

    if use_async:
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.GEMINI,
            mode=normalized_mode,
            **kwargs,
        )
    return Instructor(
        client=client,
        create=patched_create,
        provider=Provider.GEMINI,
        mode=normalized_mode,
        **kwargs,
    )


__all__ = ["from_gemini"]
