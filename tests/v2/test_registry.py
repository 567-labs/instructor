"""Tests for v2 mode registry."""

from __future__ import annotations

from typing import Any, cast

import pytest

from instructor import Mode
from instructor.v2 import Provider, mode_registry
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.protocols import ReaskHandler, RequestHandler
from instructor.v2.core.registry import ModeRegistry
from tests.v2.conftest import get_registered_provider_mode_pairs


def _get_registered_providers() -> list[Provider]:
    pairs = get_registered_provider_mode_pairs()
    return sorted({provider for provider, _ in pairs}, key=lambda p: p.value)


def _get_registered_modes() -> list[Mode]:
    pairs = get_registered_provider_mode_pairs()
    return sorted({mode for _, mode in pairs}, key=lambda m: m.value)


def _get_registered_provider_modes() -> list[tuple[Provider, Mode]]:
    return get_registered_provider_mode_pairs()


@pytest.fixture
def decorator_registry(monkeypatch: pytest.MonkeyPatch) -> ModeRegistry:
    registry = ModeRegistry()
    monkeypatch.setattr("instructor.v2.core.decorators.mode_registry", registry)
    return registry


def test_registry_registration(decorator_registry: ModeRegistry):
    """Test basic registration."""

    @register_mode_handler(Provider.DEEPSEEK, Mode.JSON)
    class TestHandler:
        mode = Mode.TOOLS

        def prepare_request(self, response_model, kwargs):
            return response_model, kwargs

        def handle_reask(self, kwargs, _response, _exception):
            return kwargs

        def parse_response(self, _response, response_model, **_kwargs):
            return response_model()

    # Check it's registered
    assert decorator_registry.is_registered(Provider.DEEPSEEK, Mode.JSON)

    # Get handlers
    handlers = decorator_registry.get_handlers(Provider.DEEPSEEK, Mode.JSON)
    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None
    assert cast(Any, handlers.request_handler).__self__.mode is Mode.JSON


@pytest.mark.parametrize("required_mode", [True, False])
def test_registration_preserves_constructor_type_errors(required_mode: bool) -> None:
    attempts: list[Mode | None] = []
    error = TypeError("invalid handler configuration")

    class RequiredModeHandler:
        def __init__(self, *, mode: Mode) -> None:
            attempts.append(mode)
            raise error

    class OptionalModeHandler:
        def __init__(self, *, mode: Mode | None = None) -> None:
            attempts.append(mode)
            raise error

    handler_class = RequiredModeHandler if required_mode else OptionalModeHandler

    with pytest.raises(TypeError, match="invalid handler configuration") as exc_info:
        register_mode_handler(Provider.DEEPSEEK, Mode.JSON)(handler_class)

    assert exc_info.value is error
    assert attempts == [Mode.JSON]


def test_registry_get_handler(decorator_registry: ModeRegistry):
    """Test getting specific handler types."""

    @register_mode_handler(Provider.OPENROUTER, Mode.TOOLS)
    class TestHandler:
        def prepare_request(self, response_model, _kwargs):
            return response_model, {"test": "request"}

        def handle_reask(self, _kwargs, _response, _exception):
            return {"test": "reask"}

        def parse_response(self, _response, response_model, **_kwargs):
            return response_model()

    # Get individual handlers
    request_handler = cast(
        RequestHandler,
        decorator_registry.get_handler(Provider.OPENROUTER, Mode.TOOLS, "request"),
    )
    result = request_handler(None, {})
    assert result[1]["test"] == "request"

    reask_handler = cast(
        ReaskHandler,
        decorator_registry.get_handler(Provider.OPENROUTER, Mode.TOOLS, "reask"),
    )
    retry_kwargs = reask_handler({}, None, ValueError("invalid response"))
    assert retry_kwargs["test"] == "reask"


@pytest.mark.parametrize("provider", _get_registered_providers())
def test_registry_query_by_provider(provider: Provider):
    """Test querying modes for a provider."""
    modes = mode_registry.get_modes_for_provider(provider)
    assert modes, f"{provider.value} should have at least one mode"

    expected_modes = {
        mode for prov, mode in get_registered_provider_mode_pairs() if prov == provider
    }
    assert expected_modes.issubset(set(modes))


@pytest.mark.parametrize("mode", _get_registered_modes())
def test_registry_query_by_mode_type(mode: Mode):
    """Test querying providers for a mode type."""
    providers = mode_registry.get_providers_for_mode(mode)
    assert providers, f"{mode.value} should have at least one provider"

    expected_providers = {
        provider
        for provider, registered_mode in get_registered_provider_mode_pairs()
        if registered_mode == mode
    }
    assert expected_providers.issubset(set(providers))


@pytest.mark.parametrize("provider,mode", _get_registered_provider_modes())
def test_registry_list_modes(provider: Provider, mode: Mode):
    """Test listing all registered modes."""
    all_modes = mode_registry.list_modes()
    assert (provider, mode) in all_modes


def test_registry_not_registered():
    """Test error when mode not registered."""
    with pytest.raises(KeyError, match="not registered"):
        mode_registry.get_handlers(Provider.GEMINI, Mode.JSON_SCHEMA)


@pytest.mark.parametrize("provider,mode", _get_registered_provider_modes())
def test_registry_invalid_handler_type(provider: Provider, mode: Mode):
    """Test error for invalid handler type."""
    with pytest.raises(ValueError, match="Invalid handler_type"):
        mode_registry.get_handler(provider, mode, "invalid_type")


def test_get_handlers_concurrent_first_access_does_not_race():
    """Regression test for #2422.

    Concurrent first callers for the same lazily-registered (provider, mode)
    key must all get the same handlers back, never a KeyError, even when the
    loader is slow (simulating a module import in flight).

    The loader sleeps briefly so every other thread has a chance to reach
    get_handlers() while the first thread's load is still in progress, this
    is the exact window in which the unlocked version raced: losing threads
    would see the lazy loader already popped and self._handlers not yet set,
    and raise KeyError for a mode that genuinely is registered.
    """
    import threading
    import time

    from instructor.v2.core.registry import ModeHandlers, ModeRegistry

    registry = ModeRegistry()
    n_threads = 8
    started = threading.Barrier(n_threads, timeout=5)
    load_count = 0
    load_count_lock = threading.Lock()

    def slow_loader() -> ModeHandlers:
        nonlocal load_count
        with load_count_lock:
            load_count += 1
        # Simulate a slow module import: gives every other thread time to
        # reach get_handlers() and start contending before this resolves.
        time.sleep(0.2)
        return ModeHandlers(
            request_handler=cast(Any, lambda *_a, **_k: None),
            reask_handler=cast(Any, lambda *_a, **_k: None),
            response_parser=cast(Any, lambda *_a, **_k: None),
        )

    registry.register_lazy(Provider.DEEPSEEK, Mode.TOOLS, slow_loader)

    results: list[object] = [None] * n_threads
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(idx: int) -> None:
        # Line up all threads so they call get_handlers() as close to
        # simultaneously as possible, maximizing contention.
        started.wait(timeout=5)
        try:
            results[idx] = registry.get_handlers(Provider.DEEPSEEK, Mode.TOOLS)
        except BaseException as exc:  # noqa: BLE001
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == [], f"get_handlers raised for concurrent callers: {errors}"
    assert all(r is not None for r in results)
    assert len({id(r) for r in results}) == 1, (
        "all callers should get the same ModeHandlers instance"
    )
    # The loader must run exactly once, not once per racing thread.
    assert load_count == 1


def test_registry_stays_registered_while_lazy_loader_runs():
    """The mode remains registered while its lazy loader is in progress."""
    import threading

    from instructor.v2.core.registry import ModeHandlers, ModeRegistry

    registry = ModeRegistry()
    load_started = threading.Event()
    finish_load = threading.Event()

    def slow_loader() -> ModeHandlers:
        load_started.set()
        finish_load.wait(timeout=5)
        return ModeHandlers(
            request_handler=cast(Any, lambda *_a, **_k: None),
            reask_handler=cast(Any, lambda *_a, **_k: None),
            response_parser=cast(Any, lambda *_a, **_k: None),
        )

    registry.register_lazy(Provider.DEEPSEEK, Mode.TOOLS, slow_loader)
    thread = threading.Thread(
        target=registry.get_handlers, args=(Provider.DEEPSEEK, Mode.TOOLS)
    )
    thread.start()
    assert load_started.wait(timeout=5)
    try:
        assert registry.is_registered(Provider.DEEPSEEK, Mode.TOOLS)
        assert (Provider.DEEPSEEK, Mode.TOOLS) in registry.list_modes()
    finally:
        finish_load.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_failed_lazy_loader_remains_retryable():
    """A transient loader failure must not permanently unregister the mode."""
    from instructor.v2.core.registry import ModeHandlers, ModeRegistry

    registry = ModeRegistry()
    attempts = 0

    def flaky_loader() -> ModeHandlers:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient import failure")
        return ModeHandlers(
            request_handler=cast(Any, lambda *_a, **_k: None),
            reask_handler=cast(Any, lambda *_a, **_k: None),
            response_parser=cast(Any, lambda *_a, **_k: None),
        )

    registry.register_lazy(Provider.DEEPSEEK, Mode.TOOLS, flaky_loader)

    with pytest.raises(RuntimeError, match="transient import failure"):
        registry.get_handlers(Provider.DEEPSEEK, Mode.TOOLS)

    assert registry.is_registered(Provider.DEEPSEEK, Mode.TOOLS)
    assert registry.get_handlers(Provider.DEEPSEEK, Mode.TOOLS)
    assert attempts == 2


def test_opaque_handler_signature_constructs_with_requested_mode(monkeypatch):
    from instructor.v2.providers.openai.handlers import OpenAIJSONHandler

    calls = []

    class OpaqueHandler(OpenAIJSONHandler):
        __signature__ = object()

        def __init__(self, *, mode):
            calls.append(mode)
            self.mode = mode

    registered = []
    monkeypatch.setattr(
        mode_registry, "register", lambda **kwargs: registered.append(kwargs)
    )
    assert (
        register_mode_handler(Provider.OPENAI, Mode.JSON)(OpaqueHandler)
        is OpaqueHandler
    )
    assert calls == [Mode.JSON]
    assert registered[0]["request_handler"].__self__.mode == Mode.JSON
