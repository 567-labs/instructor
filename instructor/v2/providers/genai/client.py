from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import ClientError
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from ...core.patch import patch_v2
from ...core.registry import mode_registry, normalize_mode

# Ensure handlers are registered (decorators auto-register on import)
from . import handlers  # noqa: F401

if TYPE_CHECKING:
    from google.genai import Client
else:
    try:
        from google.genai import Client
    except ImportError:
        Client = None  # type: ignore[assignment]


@overload
def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


def from_genai(
    client: Client,
    mode: Mode = Mode.TOOLS,
    *,
    use_async: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """
    Create a v2 Instructor client from a google.genai.Client instance.

    Supports generic modes (TOOLS, JSON).

    Args:
        client: google.genai.Client instance
        mode: Mode to use (defaults to Mode.TOOLS)
        use_async: Whether to use async client
        model: Default model name to inject into requests if not provided
        **kwargs: Additional kwargs passed to Instructor constructor
    """
    if Client is None:
        raise ClientError(
            "google-genai is not installed. Install it with: pip install google-genai"
        )

    if not isinstance(client, Client):
        raise ClientError(
            f"Client must be an instance of google.genai.Client. Got: {type(client).__name__}"
        )

    # Normalize mode for handler lookup and client metadata.
    normalized_mode = normalize_mode(Provider.GENAI, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.GENAI, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.GENAI)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.GENAI.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    if use_async:

        async def async_wrapper(*_args: Any, **call_kwargs: Any) -> Any:
            # Extract model and stream from kwargs
            # default_model will be injected by patch_v2 if not present
            model_param: str = call_kwargs.pop("model", None) or model or ""
            stream = call_kwargs.pop("stream", False)

            # contents should be in call_kwargs from handler
            if stream:
                return await client.aio.models.generate_content_stream(
                    model=model_param, **call_kwargs
                )  # type: ignore[attr-defined]

            return await client.aio.models.generate_content(
                model=model_param, **call_kwargs
            )  # type: ignore[attr-defined]

        patched = patch_v2(
            func=async_wrapper,
            provider=Provider.GENAI,
            mode=normalized_mode,
            default_model=model,
        )
        return AsyncInstructor(
            client=client,
            create=patched,
            provider=Provider.GENAI,
            mode=normalized_mode,
            **kwargs,
        )

    def sync_wrapper(*_args: Any, **call_kwargs: Any) -> Any:
        # Extract model and stream from kwargs
        # default_model will be injected by patch_v2 if not present
        model_param: str = call_kwargs.pop("model", None) or model or ""
        stream = call_kwargs.pop("stream", False)

        # contents should be in call_kwargs from handler
        if stream:
            return client.models.generate_content_stream(
                model=model_param, **call_kwargs
            )

        return client.models.generate_content(model=model_param, **call_kwargs)

    patched = patch_v2(
        func=sync_wrapper,
        provider=Provider.GENAI,
        mode=normalized_mode,
        default_model=model,
    )
    return Instructor(
        client=client,
        create=patched,
        provider=Provider.GENAI,
        mode=normalized_mode,
        **kwargs,
    )
