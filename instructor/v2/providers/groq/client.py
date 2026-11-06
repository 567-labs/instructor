"""v2 Groq client factory.

Creates Instructor instances for Groq using v2 hierarchical registry system.
Groq uses an OpenAI-compatible API, so the client factory follows the same pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
# Groq uses OpenAI-compatible API, so handlers are registered via OpenAI handlers
from instructor.v2.providers.openai import handlers  # noqa: F401

if TYPE_CHECKING:
    import groq
else:
    try:
        import groq
    except ImportError:
        groq = None


@overload
def from_groq(
    client: groq.Groq,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_groq(
    client: groq.AsyncGroq,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_groq(
    client: groq.Groq | groq.AsyncGroq,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Groq client using v2 registry.

    Groq uses an OpenAI-compatible API, so this factory follows the same pattern
    as the OpenAI factory. Groq supports TOOLS and MD_JSON modes.

    Args:
        client: An instance of Groq client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Groq
        ClientError: If client is not a valid Groq client instance or groq not installed

    Examples:
        >>> import groq
        >>> from instructor import Mode
        >>> from instructor.v2.providers.groq import from_groq
        >>>
        >>> client = groq.Groq()
        >>> instructor_client = from_groq(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use MD_JSON mode for text extraction
        >>> instructor_client = from_groq(client, mode=Mode.MD_JSON)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    # Check if groq is installed
    if groq is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError("groq is not installed. Install it with: pip install groq")

    # Normalize provider-specific modes to generic modes
    normalized_mode = normalize_mode(Provider.GROQ, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.GROQ, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.GROQ)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.GROQ.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    # Use normalized mode for patching
    mode = normalized_mode

    # Validate client type
    valid_client_types = (
        groq.Groq,
        groq.AsyncGroq,
    )

    if not isinstance(client, valid_client_types):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    # Get create function
    create = client.chat.completions.create

    # Patch using v2 registry, passing the model for injection
    patched_create = patch_v2(
        func=create,
        provider=Provider.GROQ,
        mode=mode,
        default_model=model,
    )

    # Return sync or async instructor
    if isinstance(client, groq.Groq):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.GROQ,
            mode=mode,
            **kwargs,
        )
    return AsyncInstructor(
        client=client,
        create=patched_create,
        provider=Provider.GROQ,
        mode=mode,
        **kwargs,
    )
