"""Mode handler registry for v2.

Central registry mapping Mode enum values to their handler implementations.
Supports lazy loading, dynamic registration, and queryable API.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.provider_specs import HANDLER_SPECS, PROVIDER_SPECS
from instructor.v2.core.protocols import (
    AsyncStreamExtractor,
    MessageConverter,
    ReaskHandler,
    RequestHandler,
    ResponseParser,
    StreamExtractor,
    TemplateHandler,
)

_HANDLER_ATTRS = {
    "request": "request_handler",
    "reask": "reask_handler",
    "response": "response_parser",
    "stream": "stream_extractor",
    "stream_async": "stream_extractor_async",
    "message": "message_converter",
    "template": "template_handler",
}


def normalize_mode(provider: Provider, mode: Mode) -> Mode:
    """Normalize backwards-compatible provider-specific aliases to core modes."""
    spec = PROVIDER_SPECS.get(provider)
    normalized = spec.legacy_modes.get(mode, mode) if spec is not None else mode
    if normalized is not mode:
        Mode.warn_deprecated_mode(mode)
    return normalized


@dataclass
class ModeHandlers:
    """Collection of handlers for a specific mode."""

    request_handler: RequestHandler
    reask_handler: ReaskHandler
    response_parser: ResponseParser
    stream_extractor: StreamExtractor | None = None
    stream_extractor_async: AsyncStreamExtractor | None = None
    message_converter: MessageConverter | None = None
    template_handler: TemplateHandler | None = None
    supports_schema_strict: bool = False


class ModeRegistry:
    """Central registry for mode handlers.

    Maps (Provider, Mode) tuples to their handler implementations.
    Supports lazy loading and dynamic registration.

    Example:
        >>> registry.register(
        ...     provider=Provider.ANTHROPIC,
        ...     mode=Mode.TOOLS,
        ...     request_handler=handle_request,
        ...     reask_handler=handle_reask,
        ...     response_parser=parse_response,
        ... )
        >>> # Preferred: get all handlers at once
        >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
        >>> handlers.request_handler(...)
        >>> handlers.reask_handler(...)
        >>> handlers.response_parser(...)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._handlers: dict[tuple[Provider, Mode], ModeHandlers] = {}
        self._lazy_loaders: dict[tuple[Provider, Mode], Callable[[], ModeHandlers]] = {}
        # Guards the lazy-load resolution (check -> pop -> import -> set) in
        # get_handlers(). Without this, concurrent first-callers for the same
        # mode_key race: one pops the loader, the others see neither dict
        # populated yet and raise KeyError. Held for the whole resolution
        # (not per-key) since lazy-loading only ever runs once per key.
        self._lazy_load_lock = threading.Lock()

    def register(
        self,
        provider: Provider,
        mode: Mode,
        request_handler: RequestHandler,
        reask_handler: ReaskHandler,
        response_parser: ResponseParser,
        stream_extractor: StreamExtractor | None = None,
        stream_extractor_async: AsyncStreamExtractor | None = None,
        message_converter: MessageConverter | None = None,
        template_handler: TemplateHandler | None = None,
        supports_schema_strict: bool = False,
    ) -> None:
        """Register handlers for a mode.

        Args:
            provider: Provider enum value
            mode: Mode enum value
            request_handler: Handler to prepare request kwargs
            reask_handler: Handler to handle validation failures
            response_parser: Handler to parse responses
            stream_extractor: Optional handler to extract streaming JSON chunks
            stream_extractor_async: Optional handler to extract async streaming JSON
            message_converter: Optional handler to convert multimodal messages
            template_handler: Optional handler to apply template context

        Raises:
            ConfigurationError: If mode is already registered with different handlers
        """
        mode_key = (provider, mode)
        if mode_key in self._lazy_loaders:
            self._lazy_loaders.pop(mode_key, None)

        self._handlers[mode_key] = ModeHandlers(
            request_handler=request_handler,
            reask_handler=reask_handler,
            response_parser=response_parser,
            stream_extractor=stream_extractor,
            stream_extractor_async=stream_extractor_async,
            message_converter=message_converter,
            template_handler=template_handler,
            supports_schema_strict=supports_schema_strict,
        )

    def register_lazy(
        self,
        provider: Provider,
        mode: Mode,
        loader: Callable[[], ModeHandlers],
    ) -> None:
        """Register a lazy loader for a mode.

        The loader will be called on first access to load handlers.

        Args:
            provider: Provider enum value
            mode: Mode enum value
            loader: Callable that returns ModeHandlers when invoked

        Raises:
            ConfigurationError: If mode is already registered
        """
        from instructor.v2.core.errors import ConfigurationError

        mode_key = (provider, mode)
        if mode_key in self._handlers or mode_key in self._lazy_loaders:
            raise ConfigurationError(f"Mode {mode_key} is already registered")

        self._lazy_loaders[mode_key] = loader

    def get_handlers(self, provider: Provider, mode: Mode) -> ModeHandlers:
        """Get all handlers for a mode.

        This is the preferred method for retrieving handlers. It performs
        a single registry lookup and returns all handlers at once, which is
        more efficient than calling get_handler() multiple times.

        Args:
            provider: Provider enum value
            mode: Mode enum value (provider-specific modes will be converted)

        Returns:
            ModeHandlers with all handler functions (request_handler,
            reask_handler, response_parser)

        Raises:
            KeyError: If mode is not registered

        Example:
            >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
            >>> handlers.request_handler(...)
            >>> handlers.reask_handler(...)
            >>> handlers.response_parser(...)
        """
        # Convert provider-specific modes to generic modes
        normalized_mode = normalize_mode(provider, mode)
        mode_key = (provider, normalized_mode)

        # Check if already loaded
        if mode_key in self._handlers:
            return self._handlers[mode_key]

        # Try lazy loading. Locked because resolving and publishing a loader is
        # not atomic; without the lock, concurrent first callers could run it
        # repeatedly.
        with self._lazy_load_lock:
            # Re-check: another thread may have finished loading this key
            # while we were waiting for the lock.
            if mode_key in self._handlers:
                return self._handlers[mode_key]

            if mode_key in self._lazy_loaders:
                loader = self._lazy_loaders[mode_key]
                handlers = loader()
                self._handlers[mode_key] = handlers
                self._lazy_loaders.pop(mode_key, None)
                return handlers

        raise KeyError(
            f"Mode {mode_key} is not registered. "
            f"Available modes: {list(self._handlers.keys())}"
        )

    def get_handler(
        self,
        provider: Provider,
        mode: Mode,
        handler_type: str = "request",
    ) -> (
        RequestHandler
        | ReaskHandler
        | ResponseParser
        | StreamExtractor
        | AsyncStreamExtractor
        | MessageConverter
        | TemplateHandler
    ):
        """Get a specific handler for a mode.

        This is a convenience method that internally calls get_handlers().
        For better performance when you need multiple handlers, use
        get_handlers() instead and access handlers via the returned object.

        Args:
            provider: Provider enum value
            mode: Mode enum value (provider-specific modes will be converted)
            handler_type: One of 'request', 'reask', 'response', 'stream',
                'stream_async', 'message', 'template'

        Returns:
            The requested handler function

        Raises:
            KeyError: If mode is not registered
            ValueError: If handler_type is invalid

        Example:
            >>> # Prefer this when you need multiple handlers:
            >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
            >>> handlers.request_handler(...)
            >>> handlers.reask_handler(...)

            >>> # Or use this convenience method for a single handler:
            >>> handler = registry.get_handler(Provider.ANTHROPIC, Mode.TOOLS, "request")
        """
        # get_handlers already handles normalization
        handlers = self.get_handlers(provider, mode)

        if handler_type not in _HANDLER_ATTRS:
            raise ValueError(
                f"Invalid handler_type: {handler_type}. "
                "Must be 'request', 'reask', 'response', 'stream', "
                "'stream_async', 'message', or 'template'"
            )

        attr_name = _HANDLER_ATTRS[handler_type]
        handler = getattr(handlers, attr_name)
        if handler is None:
            raise KeyError(f"No {attr_name} registered for {provider}, {mode}")
        return handler

    def is_registered(self, provider: Provider, mode: Mode) -> bool:
        """Check if a mode is registered.

        Args:
            provider: Provider enum value
            mode: Mode enum value (provider-specific modes will be converted)

        Returns:
            True if mode is registered (eagerly or lazily)
        """
        # Convert provider-specific modes to generic modes
        normalized_mode = normalize_mode(provider, mode)
        mode_key = (provider, normalized_mode)
        return mode_key in self._handlers or mode_key in self._lazy_loaders

    def get_modes_for_provider(self, provider: Provider) -> list[Mode]:
        """Get all registered modes for a provider.

        Args:
            provider: Provider enum value

        Returns:
            List of Mode values supported by this provider
        """
        modes = []
        for p, mt in self._handlers.keys():
            if p == provider:
                modes.append(mt)
        for p, mt in self._lazy_loaders.keys():
            if p == provider:
                modes.append(mt)
        return sorted(set(modes), key=lambda m: m.value)

    def get_providers_for_mode(self, mode: Mode) -> list[Provider]:
        """Get all providers that support a mode.

        Args:
            mode: Mode enum value

        Returns:
            List of Provider values that support this mode
        """
        providers = []
        for p, mt in self._handlers.keys():
            if mt == mode:
                providers.append(p)
        for p, mt in self._lazy_loaders.keys():
            if mt == mode:
                providers.append(p)
        return sorted(set(providers), key=lambda p: p.value)

    def list_modes(self) -> list[tuple[Provider, Mode]]:
        """List all registered modes.

        Returns:
            List of (Provider, Mode) tuples
        """
        all_modes = set(self._handlers.keys()) | set(self._lazy_loaders.keys())
        return sorted(all_modes, key=lambda m: (m[0].value, m[1].value))

    def get_handler_class(self, provider: Provider, mode: Mode) -> type | None:
        """Get the handler class for a mode.

        This method looks up the handler class that was registered for the given
        provider and mode. It's useful for testing and introspection.

        Args:
            provider: Provider enum value
            mode: Mode enum value (provider-specific modes will be converted)

        Returns:
            Handler class type if found, None otherwise
        """
        # Convert provider-specific modes to generic modes
        normalized_mode = normalize_mode(provider, mode)
        mode_key = (provider, normalized_mode)

        # Check if handlers are registered
        if mode_key not in self._handlers and mode_key not in self._lazy_loaders:
            return None

        # Try to get the handler class from the decorator registry
        # The decorator stores the handler class, but we need to find it
        # by looking at what was registered
        handlers = self.get_handlers(provider, mode)

        # The handlers are bound methods, so we need to get the class
        # We can inspect the handler's __self__ to get the instance, then __class__
        self_obj = getattr(handlers.request_handler, "__self__", None)
        if self_obj is not None:
            return self_obj.__class__

        # If it's a function, we can't easily get the class
        # Return None to indicate we couldn't determine it
        return None


# Global registry instance
mode_registry = ModeRegistry()

_DEFAULT_HANDLER_SPECS = HANDLER_SPECS


def _lazy_handler_loader(
    module_path: str, provider: Provider, mode: Mode
) -> ModeHandlers:
    __import__(module_path, fromlist=["__name__"])
    return mode_registry._handlers[(provider, mode)]


def _register_default_lazy_handlers() -> None:
    """Register built-in handlers without importing provider modules."""
    for provider, (module_path, modes) in _DEFAULT_HANDLER_SPECS.items():
        for mode in modes:
            if mode_registry.is_registered(provider, mode):
                continue
            mode_registry.register_lazy(
                provider,
                mode,
                lambda mp=module_path, p=provider, m=mode: _lazy_handler_loader(
                    mp, p, m
                ),
            )


_register_default_lazy_handlers()
