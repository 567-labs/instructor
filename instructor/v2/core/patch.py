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
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, get_args, overload

from pydantic import BaseModel
from pydantic_core import PydanticUndefined, PydanticUndefinedType

from instructor.cache import (
    BaseCache,
    client_cache_identity,
    load_cached_response,
    make_request_cache_key,
    store_cached_response,
)
from instructor.v2.core.budget import _validate_token_budget
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.hooks import Hooks
from instructor.v2.core.templating import handle_templating
from instructor.v2.core.utils import is_async
from instructor.v2.core.exceptions import RegistryValidationMixin
from instructor.v2.core.registry import mode_registry
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.core.retry import retry_async_v2, retry_sync_v2

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from openai import AsyncOpenAI, OpenAI
    from tenacity import AsyncRetrying, Retrying

logger = logging.getLogger("instructor.v2")


def _store_cached_result(
    cache: Any, key: str, result: Any, requested_model: Any, ttl: int | None
) -> None:
    if isinstance(result, BaseModel):
        try:
            store_cached_response(cache, key, result, ttl=ttl)
        except ModuleNotFoundError:
            pass
        return

    name = getattr(requested_model, "__name__", type(requested_model).__name__)
    args = get_args(requested_model)
    if args and all(isinstance(arg, type) for arg in args):
        name += "[" + ", ".join(arg.__name__ for arg in args) + "]"
    logger.warning(
        "Caching is not supported for this response model: %s returned a %s. "
        "The next request will call the provider again.",
        name,
        type(result).__name__,
    )


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

    cache_scope = uuid4().hex

    def prepare(
        response_model: Any,
        context: dict[str, Any] | None,
        max_retries: Any,
        strict: bool | PydanticUndefinedType,
        hooks: Hooks | None,
        token_budget: int | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], Callable[[Any], Any]]:
        """Build retry arguments; return a cached result first when available."""
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

        if default_model is not None and "model" not in kwargs:
            kwargs["model"] = default_model

        handlers = mode_registry.get_handlers(provider, mode)
        if (
            isinstance(strict, bool)
            and response_model is not None
            and handlers.supports_schema_strict
        ):
            kwargs["strict"] = strict
        if not isinstance(strict, bool):
            strict = True

        requested_model = response_model
        if response_model is not None and mode not in Mode.parallel_modes():
            response_model = prepare_response_model(response_model)
        else:
            reject_async_validators(response_model)

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
        new_kwargs = handle_templating(
            new_kwargs, mode=mode, provider=provider, context=context
        )

        # Compute identity once, before retries can mutate the prepared request.
        key = cached = None
        if isinstance(cache, BaseCache) and response_model is not None:
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

        def finish(response: Any) -> Any:
            if key is not None:
                _store_cached_result(cache, key, response, requested_model, cache_ttl)
            return response

        retry_kwargs = {
            "func": func,
            "response_model": response_model,
            "provider": provider,
            "mode": mode,
            "context": context,
            "max_retries": max_retries,
            "args": args,
            "kwargs": new_kwargs,
            "strict": strict,
            "hooks": hooks,
            "token_budget": token_budget,
        }
        return cached, retry_kwargs, finish

    if is_async(func):

        @wraps(func)
        async def new_create_async(
            response_model: type[T_Model] | None = None,
            context: dict[str, Any] | None = None,
            max_retries: int | AsyncRetrying = 1,
            strict: bool | PydanticUndefinedType = PydanticUndefined,
            hooks: Hooks | None = None,
            token_budget: int | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> T_Model:
            cached, retry_kwargs, finish = prepare(
                response_model,
                context,
                max_retries,
                strict,
                hooks,
                token_budget,
                args,
                kwargs,
            )
            if cached is not None:
                return cached
            return finish(await retry_async_v2(**retry_kwargs))

        create: Callable[..., Any] = new_create_async
    else:

        @wraps(func)
        def new_create_sync(
            response_model: type[T_Model] | None = None,
            context: dict[str, Any] | None = None,
            max_retries: int | Retrying = 1,
            strict: bool | PydanticUndefinedType = PydanticUndefined,
            hooks: Hooks | None = None,
            token_budget: int | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> T_Model:
            cached, retry_kwargs, finish = prepare(
                response_model,
                context,
                max_retries,
                strict,
                hooks,
                token_budget,
                args,
                kwargs,
            )
            if cached is not None:
                return cached
            return finish(retry_sync_v2(**retry_kwargs))

        create = new_create_sync
    cast(Any, create)._instructor_patched_create = True
    return create


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
    *,
    create: Callable[..., Awaitable[Any]],
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> AsyncInstructorChatCompletionCreate: ...


@overload
def patch(
    *,
    create: Callable[..., T_Retval],
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> InstructorChatCompletionCreate: ...


def patch(
    client: OpenAI | AsyncOpenAI | None = None,
    create: Callable[..., Any] | None = None,
    mode: Mode = Mode.TOOLS,
    provider: Provider = Provider.OPENAI,
) -> (
    OpenAI
    | AsyncOpenAI
    | InstructorChatCompletionCreate
    | AsyncInstructorChatCompletionCreate
):
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
