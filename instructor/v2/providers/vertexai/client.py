"""v2 Vertex AI client factory."""

from __future__ import annotations

import warnings

from typing import Any, Literal, TYPE_CHECKING, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

if TYPE_CHECKING:
    import vertexai.generative_models as gm
else:
    try:
        import vertexai.generative_models as gm
    except ImportError:
        gm = None


@overload
def from_vertexai(
    client: gm.GenerativeModel,
    mode: Mode = Mode.TOOLS,
    use_async: None = None,
    *,
    _async: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_vertexai(
    client: gm.GenerativeModel,
    mode: Mode = Mode.TOOLS,
    use_async: Literal[False] = False,
    *,
    _async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_vertexai(
    client: gm.GenerativeModel,
    mode: Mode = Mode.TOOLS,
    use_async: Literal[True] = True,
    *,
    _async: Literal[False] = False,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_vertexai(
    client: gm.GenerativeModel,
    mode: Mode = Mode.TOOLS,
    use_async: bool | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    from instructor.v2.core.registry import mode_registry, normalize_mode

    if kwargs.pop("_async", False):
        if use_async is not None:
            from instructor.v2.core.errors import ConfigurationError

            raise ConfigurationError(
                "Cannot provide both '_async' and 'use_async'. Use 'use_async' instead."
            )
        warnings.warn(
            "'_async' is deprecated. Use 'use_async' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        use_async = True

    normalized_mode = normalize_mode(Provider.VERTEXAI, mode)
    if not mode_registry.is_registered(Provider.VERTEXAI, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.VERTEXAI)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.VERTEXAI.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    if gm is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "vertexai is not installed. Install it with: pip install google-cloud-aiplatform"
        )

    if not isinstance(client, gm.GenerativeModel):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "Client must be an instance of vertexai.generative_models.GenerativeModel. "
            f"Got: {type(client).__name__}"
        )

    create = client.generate_content_async if use_async else client.generate_content
    patched_create = patch_v2(
        func=create,
        provider=Provider.VERTEXAI,
        mode=normalized_mode,
    )

    if use_async:
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.VERTEXAI,
            mode=normalized_mode,
            **kwargs,
        )
    return Instructor(
        client=client,
        create=patched_create,
        provider=Provider.VERTEXAI,
        mode=normalized_mode,
        **kwargs,
    )


__all__ = ["from_vertexai"]
