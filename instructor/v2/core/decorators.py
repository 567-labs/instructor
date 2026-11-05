"""Decorator utilities for v2 mode registration."""

from __future__ import annotations

from collections.abc import Iterable
from inspect import signature

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.registry import mode_registry


def register_mode_handler(
    provider: Provider | Iterable[Provider],
    mode: Mode,
):
    """Decorator to register a mode handler class.

    The decorated class must implement RequestHandler, ReaskHandler,
    and ResponseParser protocols via prepare_request, handle_reask,
    and parse_response methods.

    Args:
        provider: Provider enum value (for tracking) or list of providers
        mode: Mode enum value

    Returns:
        Decorator function

    Raises:
        TypeError: If handler construction fails. Constructor errors propagate
            without retrying initialization with different arguments.

    Example:
        >>> from instructor import Mode
        >>> @register_mode_handler(Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS)
        ... class AnthropicToolsHandler:
        ...     def prepare_request(self, response_model, kwargs):
        ...         return response_model, kwargs
        ...     def handle_reask(self, kwargs, response, exception):
        ...         return kwargs
        ...     def parse_response(self, response, response_model, **kwargs):
        ...         return response_model.model_validate(response)
    """

    def decorator(handler_class: type) -> type:
        """Register the handler class."""
        providers = list(provider) if isinstance(provider, Iterable) else [provider]

        # Shared handler classes registered for multiple compatible providers should
        # remain the same object across those providers.
        try:
            handler_signature = signature(handler_class)
        except (TypeError, ValueError):
            handler_signature = None

        try:
            if handler_signature is not None:
                handler_signature.bind(mode=mode)
        except TypeError:
            handler = handler_class()
            if hasattr(handler, "mode"):
                handler.mode = mode
        else:
            handler = handler_class(mode=mode)

        for target_provider in providers:
            mode_registry.register(
                mode=mode,
                provider=target_provider,
                request_handler=handler.prepare_request,
                reask_handler=handler.handle_reask,
                response_parser=handler.parse_response,
                stream_extractor=getattr(handler, "extract_streaming_json", None),
                stream_extractor_async=getattr(
                    handler, "extract_streaming_json_async", None
                ),
                message_converter=getattr(handler, "convert_messages", None),
                template_handler=getattr(handler, "apply_templates", None),
            )

        return handler_class

    return decorator
