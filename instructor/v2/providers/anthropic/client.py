"""v2 Anthropic client factory.

Creates Instructor instances using v2 hierarchical registry system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.errors import ClientError
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.anthropic import handlers  # noqa: F401

if TYPE_CHECKING:
    import anthropic
else:
    try:
        import anthropic
    except ImportError:
        anthropic = None


@overload
def from_anthropic(
    client: (
        anthropic.Anthropic | anthropic.AnthropicBedrock | anthropic.AnthropicVertex
    ),
    mode: Mode = Mode.TOOLS,
    beta: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_anthropic(
    client: (
        anthropic.AsyncAnthropic
        | anthropic.AsyncAnthropicBedrock
        | anthropic.AsyncAnthropicVertex
    ),
    mode: Mode = Mode.TOOLS,
    beta: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_anthropic(
    client: (
        anthropic.Anthropic
        | anthropic.AsyncAnthropic
        | anthropic.AnthropicBedrock
        | anthropic.AsyncAnthropicBedrock
        | anthropic.AsyncAnthropicVertex
        | anthropic.AnthropicVertex
    ),
    mode: Mode = Mode.TOOLS,
    beta: bool = False,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from an Anthropic client using v2 registry.

    Args:
        client: An instance of Anthropic client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        beta: Whether to use beta API features (uses client.beta.messages.create)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ValueError: If mode is not registered
        TypeError: If client is not a valid Anthropic client instance

    Examples:
        >>> import anthropic
        >>> from instructor import Mode
        >>> from instructor.v2.providers.anthropic import from_anthropic
        >>>
        >>> client = anthropic.Anthropic()
        >>> instructor_client = from_anthropic(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use JSON mode
        >>> instructor_client = from_anthropic(client, mode=Mode.JSON)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    if anthropic is None:
        raise ClientError(
            "anthropic is not installed. Install it with: pip install anthropic"
        )

    # Normalize provider-specific modes to generic modes
    # ANTHROPIC_TOOLS -> TOOLS, ANTHROPIC_JSON -> JSON, ANTHROPIC_PARALLEL_TOOLS -> PARALLEL_TOOLS
    normalized_mode = normalize_mode(Provider.ANTHROPIC, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.ANTHROPIC, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.ANTHROPIC)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.ANTHROPIC.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    # Use normalized mode for patching
    mode = normalized_mode

    # Validate client type
    valid_client_types = (
        anthropic.Anthropic,
        anthropic.AsyncAnthropic,
        anthropic.AnthropicBedrock,
        anthropic.AnthropicVertex,
        anthropic.AsyncAnthropicBedrock,
        anthropic.AsyncAnthropicVertex,
    )

    if not isinstance(client, valid_client_types):
        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    # Get create function (beta or regular)
    if beta:
        create = client.beta.messages.create
    else:
        create = client.messages.create

    # Patch using v2 registry, passing the model for injection
    patched_create = patch_v2(
        func=create,
        provider=Provider.ANTHROPIC,
        mode=mode,
        default_model=model,
    )

    # Return sync or async instructor
    if isinstance(
        client,
        (anthropic.Anthropic, anthropic.AnthropicBedrock, anthropic.AnthropicVertex),
    ):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )
    return AsyncInstructor(
        client=client,
        create=patched_create,
        provider=Provider.ANTHROPIC,
        mode=mode,
        **kwargs,
    )
