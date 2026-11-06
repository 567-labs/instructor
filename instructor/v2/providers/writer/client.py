"""v2 Writer client factory.

Creates Instructor instances for Writer using v2 hierarchical registry system.
Writer uses the writerai SDK with Writer and AsyncWriter clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.writer import handlers  # noqa: F401

if TYPE_CHECKING:
    from writerai import AsyncWriter, Writer
else:
    try:
        from writerai import AsyncWriter, Writer
    except ImportError:
        AsyncWriter = None
        Writer = None


@overload
def from_writer(
    client: Writer,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_writer(
    client: AsyncWriter,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_writer(
    client: Writer | AsyncWriter,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Writer client using v2 registry.

    Writer uses the writerai SDK and supports TOOLS, JSON_SCHEMA, and MD_JSON modes.
    The API uses `client.chat.chat` for completions.

    Args:
        client: An instance of Writer client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for Writer
        ClientError: If client is not a valid Writer client instance or SDK not installed

    Examples:
        >>> from writerai import Writer
        >>> from instructor import Mode
        >>> from instructor.v2.providers.writer import from_writer
        >>>
        >>> client = Writer()
        >>> instructor_client = from_writer(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use MD_JSON mode for text extraction
        >>> instructor_client = from_writer(client, mode=Mode.MD_JSON)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    # Check if writerai SDK is installed
    if Writer is None or AsyncWriter is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "writerai is not installed. Install it with: pip install writer-sdk"
        )

    # Normalize provider-specific modes to generic modes
    # WRITER_TOOLS -> TOOLS, WRITER_JSON -> MD_JSON
    normalized_mode = normalize_mode(Provider.WRITER, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.WRITER, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.WRITER)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.WRITER.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    # Use normalized mode for patching
    mode = normalized_mode

    # Validate client type
    valid_client_types = (
        Writer,
        AsyncWriter,
    )

    if not isinstance(client, valid_client_types):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    # Get create function - Writer uses chat.chat instead of chat.completions.create
    create = client.chat.chat

    # Patch using v2 registry, passing the model for injection
    patched_create = patch_v2(
        func=create,
        provider=Provider.WRITER,
        mode=mode,
        default_model=model,
    )

    # Return sync or async instructor
    if isinstance(client, Writer):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.WRITER,
            mode=mode,
            **kwargs,
        )
    return AsyncInstructor(
        client=client,
        create=patched_create,
        provider=Provider.WRITER,
        mode=mode,
        **kwargs,
    )
