"""Behavior tests for v2 hook, registry, and provider edge cases."""

from __future__ import annotations

import builtins
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from instructor.v2.core import registry as registry_module
from instructor.v2.core.errors import ConfigurationError
from instructor.v2.core.hooks import HookName, Hooks
from instructor.v2.core.mode import Mode, reset_deprecated_mode_warnings
from instructor.v2.core.providers import (
    Provider,
    get_provider,
    normalize_mode_for_provider,
    provider_from_mode,
)
from instructor.v2.core.registry import ModeHandlers, ModeRegistry

T = TypeVar("T", bound=BaseModel)


def _request(response_model: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    return response_model, kwargs


def _reask(
    kwargs: dict[str, Any], response: Any, exception: Exception
) -> dict[str, Any]:
    del response, exception
    return kwargs


def _response(
    response: Any,
    response_model: type[T],
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    stream: bool = False,
    is_async: bool = False,
) -> T:
    del stream, is_async
    return response_model.model_validate(
        response, context=validation_context, strict=strict
    )


def _handlers() -> ModeHandlers:
    return ModeHandlers(_request, _reask, _response)


def test_hooks_reject_invalid_name_and_remove_or_clear_registered_handlers() -> None:
    hooks = Hooks()
    seen: list[str] = []

    def first(response: str) -> None:
        seen.append(f"first:{response}")

    def second(response: str) -> None:
        seen.append(f"second:{response}")

    def missing(response: str) -> None:
        seen.append(f"missing:{response}")

    hooks.on(HookName.COMPLETION_RESPONSE, first)
    hooks.on("completion:response", second)
    hooks.off("completion:kwargs", first)
    hooks.off("completion:response", missing)
    hooks.off("completion:response", first)
    hooks.emit_completion_response("one")
    assert seen == ["second:one"]

    hooks.clear("completion:response")
    hooks.emit_completion_response("two")
    assert seen == ["second:one"]

    hooks.on("completion:response", first)
    hooks.clear()
    hooks.emit_completion_response("three")
    assert seen == ["second:one"]

    with pytest.raises(ValueError, match="Invalid hook name: completion:missing"):
        hooks.get_hook_name(cast(Any, "completion:missing"))


def test_hooks_self_removal_does_not_skip_the_next_handler() -> None:
    hooks = Hooks()
    seen: list[str] = []

    def once(response: str) -> None:
        seen.append(f"once:{response}")
        hooks.off("completion:response", once)

    def always(response: str) -> None:
        seen.append(f"always:{response}")

    hooks.on("completion:response", once)
    hooks.on("completion:response", always)

    hooks.emit_completion_response("first")
    hooks.emit_completion_response("second")

    assert seen == ["once:first", "always:first", "always:second"]


def test_hooks_new_handlers_start_with_the_next_emission() -> None:
    hooks = Hooks()
    seen: list[str] = []

    def register(response: str) -> None:
        seen.append(f"register:{response}")
        if response == "first":
            hooks.on("completion:response", added)

    def added(response: str) -> None:
        seen.append(f"added:{response}")

    hooks.on("completion:response", register)

    hooks.emit_completion_response("first")
    hooks.emit_completion_response("second")

    assert seen == ["register:first", "register:second", "added:second"]


def test_hooks_emit_retry_events_with_metadata_and_remove_the_final_handler() -> None:
    hooks = Hooks()
    error = ValueError("invalid response")
    seen: list[tuple[str, Exception, dict[str, Any]]] = []

    def last_attempt(value: Exception, **kwargs: Any) -> None:
        seen.append(("last", value, kwargs))

    def parse_error(value: Exception, **kwargs: Any) -> None:
        seen.append(("parse", value, kwargs))

    hooks.on("completion:last_attempt", last_attempt)
    hooks.on("parse:error", parse_error)
    hooks.emit_completion_last_attempt(error, attempt_number=3, is_last_attempt=True)
    hooks.emit_parse_error(error, attempt_number=2)

    assert seen == [
        ("last", error, {"attempt_number": 3, "is_last_attempt": True}),
        ("parse", error, {"attempt_number": 2}),
    ]

    hooks.off("parse:error", parse_error)
    assert HookName.PARSE_ERROR not in hooks._handlers
    hooks.emit_parse_error(error, attempt_number=3)
    assert len(seen) == 2


def test_hooks_fall_back_for_legacy_handler_and_keep_emitting_after_error() -> None:
    hooks = Hooks()
    error = RuntimeError("request failed")
    seen: list[tuple[str, Exception]] = []

    def legacy_handler(value: Exception) -> None:
        seen.append(("legacy", value))

    def broken_handler(_value: Exception, **_kwargs: Any) -> None:
        raise LookupError("hook exploded")

    def final_handler(value: Exception, **_kwargs: Any) -> None:
        seen.append(("final", value))

    hooks.on("completion:error", legacy_handler)
    hooks.on("completion:error", broken_handler)
    hooks.on("completion:error", final_handler)

    with pytest.warns(UserWarning, match="Error in completion:error handler") as record:
        hooks.emit_completion_error(error, attempt_number=2, max_attempts=3)

    assert seen == [("legacy", error), ("final", error)]
    assert len(record) == 1
    assert "LookupError: hook exploded" in str(record[0].message)


def test_hooks_combine_in_place_and_copy_keep_handler_order_and_independence() -> None:
    first = Hooks()
    second = Hooks()
    third = Hooks()
    seen: list[str] = []

    def first_handler(*_args: Any, **_kwargs: Any) -> None:
        seen.append("first")

    def second_handler(*_args: Any, **_kwargs: Any) -> None:
        seen.append("second")

    def third_handler(*_args: Any, **_kwargs: Any) -> None:
        seen.append("third")

    first.on("completion:kwargs", first_handler)
    second.on("completion:kwargs", second_handler)
    third.on("completion:kwargs", third_handler)

    added = first + second
    added.emit_completion_arguments(model="example")
    assert seen == ["first", "second"]

    copied = added.copy()
    added.off("completion:kwargs", second_handler)
    seen.clear()
    copied.emit_completion_arguments(model="example")
    assert seen == ["first", "second"]

    seen.clear()
    first += second
    first.emit_completion_arguments(model="example")
    assert seen == ["first", "second"]

    seen.clear()
    combined = Hooks.combine(first, third)
    combined.emit_completion_arguments(model="example")
    assert seen == ["first", "second", "third"]

    assert first.__add__(cast(Any, object())) is NotImplemented
    assert first.__iadd__(cast(Any, object())) is NotImplemented
    with pytest.raises(TypeError, match="Expected Hooks instance"):
        Hooks.combine(first, cast(Any, object()))


def test_registry_rejects_duplicate_lazy_registration_for_eager_and_lazy_modes() -> (
    None
):
    registry = ModeRegistry()
    registry.register(Provider.OPENAI, Mode.JSON, _request, _reask, _response)

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register_lazy(Provider.OPENAI, Mode.JSON, _handlers)

    registry.register_lazy(Provider.ANTHROPIC, Mode.TOOLS, _handlers)
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register_lazy(Provider.ANTHROPIC, Mode.TOOLS, _handlers)


def test_registry_reports_lazy_providers_and_loads_each_handler_once() -> None:
    registry = ModeRegistry()
    loaded: list[str] = []

    def load_openai() -> ModeHandlers:
        loaded.append("openai")
        return _handlers()

    def load_anthropic() -> ModeHandlers:
        loaded.append("anthropic")
        return _handlers()

    registry.register_lazy(Provider.OPENAI, Mode.JSON, load_openai)
    registry.register_lazy(Provider.ANTHROPIC, Mode.JSON, load_anthropic)
    registry.register_lazy(Provider.COHERE, Mode.TOOLS, _handlers)

    assert registry.get_providers_for_mode(Mode.JSON) == [
        Provider.ANTHROPIC,
        Provider.OPENAI,
    ]
    assert loaded == []

    first = registry.get_handlers(Provider.OPENAI, Mode.JSON)
    second = registry.get_handlers(Provider.OPENAI, Mode.JSON)
    assert first is second
    assert loaded == ["openai"]

    with pytest.raises(KeyError, match="No stream_extractor registered"):
        registry.get_handler(Provider.OPENAI, Mode.JSON, "stream")


def test_registry_queries_eager_and_lazy_modes_and_validates_lookups() -> None:
    registry = ModeRegistry()
    registry.register_lazy(Provider.OPENAI, Mode.JSON, _handlers)
    registry.register(Provider.OPENAI, Mode.JSON, _request, _reask, _response)
    registry.register_lazy(Provider.OPENAI, Mode.TOOLS, _handlers)
    registry.register_lazy(Provider.COHERE, Mode.JSON, _handlers)
    registry.register(Provider.ANTHROPIC, Mode.TOOLS, _request, _reask, _response)

    assert registry.get_modes_for_provider(Provider.OPENAI) == [Mode.JSON, Mode.TOOLS]
    assert registry.get_providers_for_mode(Mode.TOOLS) == [
        Provider.ANTHROPIC,
        Provider.OPENAI,
    ]
    assert registry.get_handler(Provider.OPENAI, Mode.JSON, "request") is _request

    with pytest.raises(ValueError, match="Invalid handler_type: unknown"):
        registry.get_handler(Provider.OPENAI, Mode.JSON, "unknown")
    with pytest.raises(KeyError, match="is not registered"):
        registry.get_handlers(Provider.CEREBRAS, Mode.JSON)


def test_registry_normalizes_legacy_mode_before_handler_lookup() -> None:
    registry = ModeRegistry()
    registry.register(Provider.ANTHROPIC, Mode.TOOLS, _request, _reask, _response)

    reset_deprecated_mode_warnings()
    with pytest.warns(DeprecationWarning, match="Mode.ANTHROPIC_TOOLS is deprecated"):
        handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS)

    assert handlers.request_handler is _request
    assert registry.is_registered(Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS)
    reset_deprecated_mode_warnings()


def test_registry_finds_bound_handler_class_and_handles_unbound_or_missing_modes() -> (
    None
):
    registry = ModeRegistry()

    class Handler:
        def request(
            self, response_model: Any, kwargs: dict[str, Any]
        ) -> tuple[Any, dict[str, Any]]:
            return response_model, kwargs

    assert registry.get_handler_class(Provider.OPENAI, Mode.JSON) is None

    handler = Handler()
    registry.register(Provider.OPENAI, Mode.JSON, handler.request, _reask, _response)
    registry.register(Provider.ANTHROPIC, Mode.JSON, _request, _reask, _response)

    assert registry.get_handler_class(Provider.OPENAI, Mode.JSON) is Handler
    assert registry.get_handler_class(Provider.ANTHROPIC, Mode.JSON) is None


def test_register_default_lazy_handlers_preserves_existing_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModeRegistry()
    existing_loader = _handlers
    registry.register_lazy(Provider.OPENAI, Mode.JSON, existing_loader)
    monkeypatch.setattr(registry_module, "mode_registry", registry)
    monkeypatch.setattr(
        registry_module,
        "_DEFAULT_HANDLER_SPECS",
        {
            Provider.OPENAI: (
                "instructor.v2.providers.openai.handlers",
                (Mode.JSON, Mode.TOOLS),
            )
        },
    )

    registry_module._register_default_lazy_handlers()

    assert registry._lazy_loaders[(Provider.OPENAI, Mode.JSON)] is existing_loader
    assert (Provider.OPENAI, Mode.TOOLS) in registry._lazy_loaders
    assert registry.list_modes() == [
        (Provider.OPENAI, Mode.JSON),
        (Provider.OPENAI, Mode.TOOLS),
    ]


def test_lazy_handler_loader_imports_provider_module_and_returns_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModeRegistry()
    imports: list[str] = []
    original_import = builtins.__import__

    def import_and_register(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "instructor.v2.providers.example.handlers":
            imports.append(name)
            registry.register(Provider.OPENAI, Mode.JSON, _request, _reask, _response)
            return object()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(registry_module, "mode_registry", registry)
    monkeypatch.setattr(builtins, "__import__", import_and_register)

    handlers = registry_module._lazy_handler_loader(
        "instructor.v2.providers.example.handlers", Provider.OPENAI, Mode.JSON
    )

    assert imports == ["instructor.v2.providers.example.handlers"]
    assert handlers.request_handler is _request


def test_unknown_provider_url_does_not_guess_a_provider() -> None:
    assert get_provider("https://api.example.invalid/v1") is Provider.UNKNOWN


def test_provider_helpers_detect_known_urls_and_normalize_legacy_modes() -> None:
    assert get_provider("https://api.openai.com/v1") is Provider.OPENAI
    assert provider_from_mode(Mode.ANTHROPIC_TOOLS) is Provider.ANTHROPIC
    assert provider_from_mode(Mode.JSON, Provider.COHERE) is Provider.COHERE
    reset_deprecated_mode_warnings()
    with pytest.warns(DeprecationWarning, match="Mode.ANTHROPIC_TOOLS is deprecated"):
        normalized = normalize_mode_for_provider(
            Mode.ANTHROPIC_TOOLS, Provider.ANTHROPIC
        )
    assert normalized is Mode.TOOLS
    assert normalize_mode_for_provider(Mode.JSON, Provider.COHERE) is Mode.JSON
    reset_deprecated_mode_warnings()
