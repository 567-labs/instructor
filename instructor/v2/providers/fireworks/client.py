"""v2 Fireworks client factory.

Creates Instructor instances for Fireworks using v2 hierarchical registry system.
Fireworks uses an OpenAI-compatible API, so the client factory follows the same pattern.
"""

from __future__ import annotations

import inspect

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
# Fireworks uses OpenAI-compatible API, so handlers are registered via OpenAI handlers
from instructor.v2.providers.openai import handlers  # noqa: F401

_async_create_method = "create"
if TYPE_CHECKING:
    from fireworks.client import AsyncFireworks, Fireworks
else:
    try:
        from fireworks import AsyncFireworks, Fireworks
    except ImportError:
        _async_create_method = "acreate"
        try:
            from fireworks.client import AsyncFireworks, Fireworks
        except ImportError:
            AsyncFireworks = None
            Fireworks = None


@overload
def from_fireworks(
    client: Fireworks,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_fireworks(
    client: AsyncFireworks,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_fireworks(
    client: Fireworks | AsyncFireworks,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Fireworks client using v2 registry.

    Fireworks uses an OpenAI-compatible API, so this factory follows the same pattern
    as the OpenAI factory. Fireworks supports TOOLS and MD_JSON modes.

    Args:
        client: An instance of Fireworks client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Fireworks
        ClientError: If client is not a valid Fireworks client instance or fireworks not installed

    Examples:
        >>> from fireworks.client import Fireworks
        >>> from instructor import Mode
        >>> from instructor.v2.providers.fireworks import from_fireworks
        >>>
        >>> client = Fireworks()
        >>> instructor_client = from_fireworks(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use MD_JSON mode for text extraction
        >>> instructor_client = from_fireworks(client, mode=Mode.MD_JSON)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    # Check if fireworks is installed
    if Fireworks is None or AsyncFireworks is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "fireworks is not installed. Install it with: pip install fireworks-ai"
        )

    # Normalize provider-specific modes to generic modes
    # FIREWORKS_TOOLS -> TOOLS, FIREWORKS_JSON -> MD_JSON
    normalized_mode = normalize_mode(Provider.FIREWORKS, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.FIREWORKS, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.FIREWORKS)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.FIREWORKS.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    # Use normalized mode for patching
    mode = normalized_mode

    # Validate client type
    valid_client_types = (
        Fireworks,
        AsyncFireworks,
    )

    if not isinstance(client, valid_client_types):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    # Get create function - Fireworks uses chat.completions.create like OpenAI
    if isinstance(client, AsyncFireworks):
        completions = client.chat.completions
        sdk_create = getattr(completions, _async_create_method)

        async def async_create(*args: Any, **create_kwargs: Any) -> Any:
            # Fireworks 0.x defaults to streaming when this argument is omitted.
            create_kwargs.setdefault("stream", False)
            response = sdk_create(*args, **create_kwargs)
            return await response if inspect.isawaitable(response) else response

        create = async_create
    else:
        create = client.chat.completions.create

    # Patch using v2 registry, passing the model for injection
    patched_create = patch_v2(
        func=create,
        provider=Provider.FIREWORKS,
        mode=mode,
        default_model=model,
    )

    # Return sync or async instructor
    if isinstance(client, Fireworks):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.FIREWORKS,
            mode=mode,
            **kwargs,
        )
    return AsyncInstructor(
        client=client,
        create=patched_create,
        provider=Provider.FIREWORKS,
        mode=mode,
        **kwargs,
    )
