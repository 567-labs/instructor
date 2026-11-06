"""v2 Cerebras client factory.

Creates Instructor instances for Cerebras using v2 hierarchical registry system.
Cerebras uses an OpenAI-compatible API, so the client factory follows the same pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
# Cerebras uses OpenAI-compatible API, so handlers are registered via OpenAI handlers
from instructor.v2.providers.openai import handlers  # noqa: F401

if TYPE_CHECKING:
    from cerebras.cloud.sdk import AsyncCerebras, Cerebras
else:
    try:
        from cerebras.cloud.sdk import AsyncCerebras, Cerebras
    except ImportError:
        AsyncCerebras = None
        Cerebras = None


@overload
def from_cerebras(
    client: Cerebras,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cerebras(
    client: AsyncCerebras,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_cerebras(
    client: Cerebras | AsyncCerebras,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Cerebras client using v2 registry.

    Cerebras uses an OpenAI-compatible API, so this factory follows the same pattern
    as the OpenAI factory. Cerebras supports TOOLS and MD_JSON modes.

    Args:
        client: An instance of Cerebras client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Cerebras
        ClientError: If client is not a valid Cerebras client instance or SDK not installed

    Examples:
        >>> from cerebras.cloud.sdk import Cerebras
        >>> from instructor import Mode
        >>> from instructor.v2.providers.cerebras import from_cerebras
        >>>
        >>> client = Cerebras()
        >>> instructor_client = from_cerebras(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use MD_JSON mode for text extraction
        >>> instructor_client = from_cerebras(client, mode=Mode.MD_JSON)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    # Check if cerebras SDK is installed
    if Cerebras is None or AsyncCerebras is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "cerebras is not installed. Install it with: pip install cerebras-cloud-sdk"
        )

    # Normalize provider-specific modes to generic modes
    # CEREBRAS_TOOLS -> TOOLS, CEREBRAS_JSON -> MD_JSON
    normalized_mode = normalize_mode(Provider.CEREBRAS, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.CEREBRAS, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.CEREBRAS)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.CEREBRAS.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    # Use normalized mode for patching
    mode = normalized_mode

    # Validate client type
    valid_client_types = (
        Cerebras,
        AsyncCerebras,
    )

    if not isinstance(client, valid_client_types):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    # Get create function - Cerebras uses chat.completions.create like OpenAI
    create = client.chat.completions.create

    # Patch using v2 registry, passing the model for injection
    patched_create = patch_v2(
        func=create,
        provider=Provider.CEREBRAS,
        mode=mode,
        default_model=model,
    )

    # Return sync or async instructor
    if isinstance(client, Cerebras):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.CEREBRAS,
            mode=mode,
            **kwargs,
        )
    return AsyncInstructor(
        client=client,
        create=patched_create,
        provider=Provider.CEREBRAS,
        mode=mode,
        **kwargs,
    )
