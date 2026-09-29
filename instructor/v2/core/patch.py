"""v2 patch mechanism using hierarchical registry.

Simplified patching logic that uses the v2 mode registry for handler dispatch.
"""

from __future__ import annotations

import logging
import warnings
from uuid import uuid4

from instructor.v2.validation.async_validators import reject_async_validators
from collections.abc import Awaitable
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, overload

from pydantic import BaseModel

from instructor.v2.core.budget import _validate_token_budget
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.hooks import Hooks
from instructor.v2.core.templating import handle_templating
from instructor.v2.core.utils import is_async
from instructor.v2.core.exceptions import RegistryValidationMixin
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.messages import isolate_retry_kwargs
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.core.retry import retry_async_v2, retry_sync_v2

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from openai import AsyncOpenAI, OpenAI
    from tenacity import AsyncRetrying, Retrying

logger = logging.getLogger("instructor.v2")


def _model_name(model: Any) -> str:
    """Name a response model the way the caller wrote it.

    `list[User]` is a generic alias, so `__name__` gives "list"; `str()` gives
    "<class 'str'>". Both are unhelpful in a warning about what the caller asked
    for, so unwrap the class form and keep everything else verbatim.
    """
    text = str(model)
    if text.startswith("<class '") and text.endswith("'>"):
        return text[len("<class '") : -len("'>")]
    return text


def _store_cached_result(
    cache: Any,
    key: str,
    response: Any,
    requested_model: Any,
    ttl: int | None,
) -> None:
    """Write a finished result to the cache, or say why it could not be written.

    Only a Pydantic result can be stored: `load_cached_response` rebuilds the
    value with `response_model.model_validate_json`, so what the store side
    writes has to be the model's own JSON. A list or scalar result has none --
    `list[User]` and `str` come back from the retry loop as a `ListResponse` or a
    plain `str` -- so the entry is skipped. That used to be silent, which reads
    as a cache that never hits rather than a response model it cannot hold.
    """
    if not isinstance(response, BaseModel):
        logger.warning(
            "Caching is not supported for this response model: %s returned a %s, "
            "which has no model JSON to store, so nothing was written to the cache "
            "and every call reaches the provider.",
            _model_name(requested_model),
            type(response).__name__,
        )
        return

    from instructor.v2.core.cache_response import store_cached_response

    try:
        store_cached_response(cache, key, response, ttl=ttl)
    except ModuleNotFoundError:
        pass


T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")


class InstructorChatCompletionCreate(Protocol):
    def __call__(
        self,
        response_model: type[T_Model] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | Retrying = 1,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model: ...


class AsyncInstructorChatCompletionCreate(Protocol):
    async def __call__(
        self,
        response_model: type[T_Model] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | AsyncRetrying = 1,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model: ...


@overload
def patch_v2(
    func: Callable[..., Awaitable[Any]],
    provider: Provider,
    mode: Mode,
    default_model: str | None = None,
) -> Callable[..., Awaitable[T_Model]]: ...


@overload
def patch_v2(
    func: Callable[..., Any],
    provider: Provider,
    mode: Mode,
    default_model: str | None = None,
) -> Callable[..., T_Model]: ...


def patch_v2(
    func: Callable[..., Any],
    provider: Provider,
    mode: Mode,
    default_model: str | None = None,
) -> Callable[..., Any]:
    """Patch a function to use v2 registry for structured outputs.

    Args:
        func: Function to patch (e.g., client.messages.create)
        provider: Provider enum value
        mode: Mode enum value
        default_model: Default model to inject if not provided in request

    Returns:
        Patched function that supports response_model parameter

    Raises:
        RegistryError: If mode is not registered for provider
    """
    logger.debug(f"Patching with v2 registry: {provider=}, {mode=}, {default_model=}")

    # Validate mode registration
    RegistryValidationMixin.validate_mode_registration(provider, mode)

    func_is_async = is_async(func)

    if func_is_async:
        return _create_async_wrapper(func, provider, mode, default_model)
    return _create_sync_wrapper(func, provider, mode, default_model)


@overload
def patch(
    client: OpenAI,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> OpenAI: ...


@overload
def patch(
    client: AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> AsyncOpenAI: ...


@overload
def patch(
    create: Callable[..., T_Retval],
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> InstructorChatCompletionCreate: ...


@overload
def patch(
    create: Awaitable[T_Retval],
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> InstructorChatCompletionCreate: ...


def patch(
    client: OpenAI | AsyncOpenAI | None = None,
    create: Callable[..., T_Retval] | None = None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> OpenAI | AsyncOpenAI | InstructorChatCompletionCreate:
    """Patch chat-completion create methods with v2 registry handlers."""
    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

    if create is not None:
        func = create
    elif client is not None:
        func = client.chat.completions.create
    else:
        raise ValueError("Either client or create must be provided")

    new_create = patch_v2(func=func, provider=provider, mode=mode)

    if client is not None:
        cast(Any, client.chat.completions).create = new_create
        return client
    return new_create


def apatch(
    client: AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> AsyncOpenAI:
    """Deprecated alias for :func:`patch`."""
    warnings.warn(
        "apatch is deprecated, use patch instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return patch(client, mode=mode, provider=provider)


def _create_sync_wrapper(
    func: Callable[..., Any],
    provider: Provider,
    mode: Mode,
    default_model: str | None = None,
) -> Callable[..., T_Model]:
    """Create synchronous wrapper for patched function."""
    cache_scope = uuid4().hex

    @wraps(func)
    def new_create_sync(
        response_model: type[T_Model] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | Retrying = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        token_budget: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model:
        """Patched synchronous create function."""
        reject_async_validators(response_model)
        _validate_token_budget(
            token_budget,
            response_model=response_model,
            kwargs=kwargs,
        )
        autodetect_images = bool(kwargs.get("autodetect_images", False))
        cache = kwargs.pop("cache", None)
        cache_namespace = kwargs.pop("cache_namespace", cache_scope)
        if not isinstance(cache_namespace, str) or not cache_namespace:
            raise ValueError("cache_namespace must be a non-empty string")
        cache_ttl_raw = kwargs.pop("cache_ttl", None)
        cache_ttl = cache_ttl_raw if isinstance(cache_ttl_raw, int) else None

        # Inject default model if not provided and available
        if default_model is not None and "model" not in kwargs:
            kwargs["model"] = default_model

        # Get handlers from registry
        handlers = mode_registry.get_handlers(provider, mode)

        # Kept for the cache warning: `response_model` is about to become the
        # prepared model (IterableUser for list[User]), which says nothing useful
        # to the caller who wrote list[User].
        requested_model = response_model

        if response_model is not None and mode not in Mode.parallel_modes():
            response_model = prepare_response_model(response_model)

        # Prepare request kwargs using registry handler
        prepared_model, new_kwargs = handlers.request_handler(
            response_model=response_model, kwargs=kwargs
        )
        if mode not in Mode.parallel_modes():
            response_model = prepared_model
        new_kwargs.pop("autodetect_images", None)
        if handlers.message_converter and "messages" in new_kwargs:
            new_kwargs["messages"] = handlers.message_converter(
                new_kwargs["messages"],
                autodetect_images=autodetect_images,
            )

        # Handle templating
        new_kwargs = handle_templating(
            new_kwargs,
            mode=mode,
            provider=provider,
            context=context,
        )

        # Compute identity once, before retries can mutate the prepared request.
        key = None
        if cache is not None and response_model is not None:
            from instructor.cache import (
                BaseCache,
                make_request_cache_key,
                client_cache_identity,
            )
            from instructor.v2.core.cache_response import load_cached_response

            if isinstance(cache, BaseCache):
                key = make_request_cache_key(
                    client_identity=client_cache_identity(func),
                    request=new_kwargs,
                    args=args,
                    response_model=response_model,
                    provider=provider.value,
                    mode=str(mode.value),
                    namespace=cache_namespace,
                    context=context,
                    strict=strict,
                )
                if key is not None:
                    cached = load_cached_response(
                        cache, key, response_model, context=context, strict=strict
                    )
                    if cached is not None:
                        return cached  # type: ignore[return-value]

        # Use v2 retry logic with registry handlers. Pass an isolated copy of the
        # messages list so reask-handler mutations cannot leak into caller state.
        response = retry_sync_v2(
            func=func,
            response_model=response_model,
            provider=provider,
            mode=mode,
            context=context,
            max_retries=max_retries,
            args=args,
            kwargs=isolate_retry_kwargs(new_kwargs),
            strict=strict,
            hooks=hooks,
            token_budget=token_budget,
        )

        if key is not None:
            _store_cached_result(cache, key, response, requested_model, cache_ttl)

        return response  # type: ignore[return-value]

    return new_create_sync  # type: ignore[return-value]


def _create_async_wrapper(
    func: Callable[..., Awaitable[Any]],
    provider: Provider,
    mode: Mode,
    default_model: str | None = None,
) -> Callable[..., Awaitable[T_Model]]:
    """Create asynchronous wrapper for patched function."""
    cache_scope = uuid4().hex

    @wraps(func)
    async def new_create_async(
        response_model: type[T_Model] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int | AsyncRetrying = 1,
        strict: bool = True,
        hooks: Hooks | None = None,
        token_budget: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model:
        """Patched asynchronous create function."""
        reject_async_validators(response_model)
        _validate_token_budget(
            token_budget,
            response_model=response_model,
            kwargs=kwargs,
        )
        autodetect_images = bool(kwargs.get("autodetect_images", False))
        cache = kwargs.pop("cache", None)
        cache_namespace = kwargs.pop("cache_namespace", cache_scope)
        if not isinstance(cache_namespace, str) or not cache_namespace:
            raise ValueError("cache_namespace must be a non-empty string")
        cache_ttl_raw = kwargs.pop("cache_ttl", None)
        cache_ttl = cache_ttl_raw if isinstance(cache_ttl_raw, int) else None

        # Inject default model if not provided and available
        if default_model is not None and "model" not in kwargs:
            kwargs["model"] = default_model

        # Get handlers from registry
        handlers = mode_registry.get_handlers(provider, mode)

        # Kept for the cache warning: `response_model` is about to become the
        # prepared model (IterableUser for list[User]), which says nothing useful
        # to the caller who wrote list[User].
        requested_model = response_model

        if response_model is not None and mode not in Mode.parallel_modes():
            response_model = prepare_response_model(response_model)

        # Prepare request kwargs using registry handler
        prepared_model, new_kwargs = handlers.request_handler(
            response_model=response_model, kwargs=kwargs
        )
        if mode not in Mode.parallel_modes():
            response_model = prepared_model
        new_kwargs.pop("autodetect_images", None)
        if handlers.message_converter and "messages" in new_kwargs:
            new_kwargs["messages"] = handlers.message_converter(
                new_kwargs["messages"],
                autodetect_images=autodetect_images,
            )

        # Handle templating
        new_kwargs = handle_templating(
            new_kwargs,
            mode=mode,
            provider=provider,
            context=context,
        )

        # Compute identity once, before retries can mutate the prepared request.
        key = None
        if cache is not None and response_model is not None:
            from instructor.cache import (
                BaseCache,
                make_request_cache_key,
                client_cache_identity,
            )
            from instructor.v2.core.cache_response import load_cached_response

            if isinstance(cache, BaseCache):
                key = make_request_cache_key(
                    client_identity=client_cache_identity(func),
                    request=new_kwargs,
                    args=args,
                    response_model=response_model,
                    provider=provider.value,
                    mode=str(mode.value),
                    namespace=cache_namespace,
                    context=context,
                    strict=strict,
                )
                if key is not None:
                    cached = load_cached_response(
                        cache, key, response_model, context=context, strict=strict
                    )
                    if cached is not None:
                        return cached  # type: ignore[return-value]

        # Use v2 retry logic with registry handlers. Pass an isolated copy of the
        # messages list so reask-handler mutations cannot leak into caller state.
        response = await retry_async_v2(
            func=func,
            response_model=response_model,
            provider=provider,
            mode=mode,
            context=context,
            max_retries=max_retries,
            args=args,
            kwargs=isolate_retry_kwargs(new_kwargs),
            strict=strict,
            hooks=hooks,
            token_budget=token_budget,
        )

        if key is not None:
            _store_cached_result(cache, key, response, requested_model, cache_ttl)

        return response  # type: ignore[return-value]

    return new_create_async  # type: ignore[return-value]
