"""v2 Cohere client factory.

Creates Instructor instances using v2 hierarchical registry system.
Supports both Cohere V1 and V2 client APIs.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, cast, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.cohere import handlers  # noqa: F401

if TYPE_CHECKING:
    import cohere
else:
    try:
        import cohere
    except ImportError:
        cohere = None  # type: ignore[assignment]


@overload
def from_cohere(
    client: cohere.Client,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cohere(
    client: cohere.ClientV2,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClient,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClientV2,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_cohere(
    client: cohere.Client | cohere.AsyncClient | cohere.ClientV2 | cohere.AsyncClientV2,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Cohere client using v2 registry.

    Args:
        client: A Cohere client instance (V1 or V2, sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Cohere
        ClientError: If client is not a valid Cohere client instance

    Examples:
        >>> import cohere
        >>> from instructor import Mode
        >>> from instructor.v2.providers.cohere import from_cohere
        >>>
        >>> # V2 client (recommended)
        >>> client = cohere.ClientV2()
        >>> instructor_client = from_cohere(client, mode=Mode.TOOLS)
        >>>
        >>> # V1 client
        >>> client = cohere.Client()
        >>> instructor_client = from_cohere(client, mode=Mode.JSON_SCHEMA)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    if cohere is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "cohere is not installed. Install it with: pip install cohere"
        )

    # Normalize provider-specific modes to generic modes
    # COHERE_TOOLS -> TOOLS, COHERE_JSON_SCHEMA -> JSON_SCHEMA
    normalized_mode = normalize_mode(Provider.COHERE, mode)

    # Validate mode is registered
    if not mode_registry.is_registered(Provider.COHERE, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.COHERE)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.COHERE.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    # Use normalized mode for patching
    mode = normalized_mode

    # Validate client type
    valid_client_types = (
        cohere.Client,
        cohere.AsyncClient,
        cohere.ClientV2,
        cohere.AsyncClientV2,
    )

    if not isinstance(client, valid_client_types):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    # Detect client version for request formatting
    if isinstance(client, (cohere.ClientV2, cohere.AsyncClientV2)):
        client_version = "v2"
    else:
        client_version = "v1"

    kwargs["_cohere_client_version"] = client_version

    # Determine if async client
    is_async = isinstance(client, (cohere.AsyncClient, cohere.AsyncClientV2))

    if is_async:

        async def async_wrapper(*args: Any, **call_kwargs: Any) -> Any:
            if call_kwargs.pop("stream", False):
                return client.chat_stream(*args, **call_kwargs)
            result = client.chat(*args, **call_kwargs)
            if inspect.isawaitable(result):
                return await cast(Awaitable[Any], result)
            return result

        patched_create = patch_v2(
            func=async_wrapper,
            provider=Provider.COHERE,
            mode=mode,
        )

        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.COHERE,
            mode=mode,
            **kwargs,
        )

    def sync_wrapper(*args: Any, **call_kwargs: Any) -> Any:
        if call_kwargs.pop("stream", False):
            return client.chat_stream(*args, **call_kwargs)
        return client.chat(*args, **call_kwargs)

    patched_create = patch_v2(
        func=sync_wrapper,
        provider=Provider.COHERE,
        mode=mode,
    )

    return Instructor(
        client=client,
        create=patched_create,
        provider=Provider.COHERE,
        mode=mode,
        **kwargs,
    )
