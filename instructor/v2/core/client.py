from __future__ import annotations

from copy import copy
from types import MemberDescriptorType
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Callable,
    Literal,
    Any,
    cast,
    get_origin,
    get_args,
)

if TYPE_CHECKING:
    import openai
    from openai.types.chat import ChatCompletionMessageParam

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from tenacity import (
    AsyncRetrying,
    Retrying,
)
from collections.abc import Generator, Iterable, Awaitable, AsyncGenerator, Coroutine
from typing_extensions import Self, overload
from instructor.v2.dsl.partial import Partial
from instructor.v2.core.hooks import Hooks, HookName


T = TypeVar("T")


class _ResponseBase:
    @staticmethod
    def _normalize_messages(
        messages: str | list[ChatCompletionMessageParam] | None,
        kwargs: dict[str, Any],
    ) -> str | list[ChatCompletionMessageParam]:
        if messages is None:
            if "input" not in kwargs:
                raise TypeError("Either 'messages' or 'input' must be provided")
            messages = kwargs.pop("input")
        elif "input" in kwargs:
            raise TypeError("Pass only one of 'messages' or 'input'")

        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages


class Response(_ResponseBase):
    """Helper for responses API using a patched client."""

    def __init__(
        self,
        client: Instructor,
    ):
        self.client = client

    @overload
    def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] = ...,
        max_retries: int | Retrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        token_budget: int | None = None,
        **kwargs: Any,
    ) -> T: ...

    @overload
    def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: None = None,
        max_retries: int | Retrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        token_budget: None = None,
        **kwargs: Any,
    ) -> Any: ...

    def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] | None = None,
        max_retries: int | Retrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        token_budget: int | None = None,
        **kwargs,
    ) -> T | Any:
        messages = self._normalize_messages(messages, kwargs)
        if token_budget is not None:
            kwargs["token_budget"] = token_budget

        create = cast(Callable[..., Any], self.client.create)
        return create(
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            strict=strict,
            messages=messages,
            **kwargs,
        )

    @overload
    def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] = ...,
        max_retries: int | Retrying = 3,
        **kwargs: Any,
    ) -> tuple[T, Any]: ...

    @overload
    def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: None = None,
        max_retries: int | Retrying = 3,
        **kwargs: Any,
    ) -> tuple[Any, Any]: ...

    def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] | None = None,
        max_retries: int | Retrying = 3,
        **kwargs,
    ) -> tuple[T, Any]:
        messages = self._normalize_messages(messages, kwargs)

        create_with_completion = cast(
            Callable[..., tuple[T, Any]], self.client.create_with_completion
        )
        return create_with_completion(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )

    @overload
    def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] = ...,
        max_retries: int | Retrying = 3,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    @overload
    def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: None = None,
        max_retries: int | Retrying = 3,
        **kwargs: Any,
    ) -> Generator[Any, None, None]: ...

    def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] | None = None,
        max_retries: int | Retrying = 3,
        **kwargs,
    ) -> Generator[T, None, None]:
        messages = self._normalize_messages(messages, kwargs)

        create_iterable = cast(
            Callable[..., Generator[T, None, None]], self.client.create_iterable
        )
        return create_iterable(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )

    @overload
    def create_partial(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] = ...,
        max_retries: int | Retrying = 3,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    @overload
    def create_partial(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: None = None,
        max_retries: int | Retrying = 3,
        **kwargs: Any,
    ) -> Generator[Any, None, None]: ...

    def create_partial(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] | None = None,
        max_retries: int | Retrying = 3,
        **kwargs,
    ) -> Generator[T, None, None]:
        messages = self._normalize_messages(messages, kwargs)

        create_partial = cast(
            Callable[..., Generator[T, None, None]], self.client.create_partial
        )
        return create_partial(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )


class AsyncResponse(_ResponseBase):
    def __init__(self, client: AsyncInstructor):
        self.client = client

    @overload
    async def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] = ...,
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        token_budget: int | None = None,
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: None = None,
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        token_budget: None = None,
        **kwargs: Any,
    ) -> Any: ...

    async def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] | None = None,
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        token_budget: int | None = None,
        **kwargs,
    ) -> T | Any:
        messages = self._normalize_messages(messages, kwargs)
        if token_budget is not None:
            kwargs["token_budget"] = token_budget

        create = cast(Callable[..., Awaitable[Any]], self.client.create)
        return await create(
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            strict=strict,
            messages=messages,
            **kwargs,
        )

    @overload
    async def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] = ...,
        max_retries: int | AsyncRetrying = 3,
        **kwargs: Any,
    ) -> tuple[T, Any]: ...

    @overload
    async def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: None = None,
        max_retries: int | AsyncRetrying = 3,
        **kwargs: Any,
    ) -> tuple[Any, Any]: ...

    async def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] | None = None,
        max_retries: int | AsyncRetrying = 3,
        **kwargs,
    ) -> tuple[T, Any]:
        messages = self._normalize_messages(messages, kwargs)

        create_with_completion = cast(
            Callable[..., Awaitable[tuple[T, Any]]],
            self.client.create_with_completion,
        )
        return await create_with_completion(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )

    @overload
    async def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] = ...,
        max_retries: int | AsyncRetrying = 3,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    async def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: None = None,
        max_retries: int | AsyncRetrying = 3,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]: ...

    async def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] | None = None,
        max_retries: int | AsyncRetrying = 3,
        **kwargs,
    ) -> AsyncGenerator[T, None]:
        messages = self._normalize_messages(messages, kwargs)

        create_iterable = cast(
            Callable[..., AsyncGenerator[T, None]], self.client.create_iterable
        )
        return create_iterable(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )

    @overload
    def create_partial(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] = ...,
        max_retries: int | AsyncRetrying = 3,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_partial(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: None = None,
        max_retries: int | AsyncRetrying = 3,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]: ...

    def create_partial(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = None,
        response_model: type[T] | None = None,
        max_retries: int | AsyncRetrying = 3,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        messages = self._normalize_messages(messages, kwargs)

        create_partial = cast(
            Callable[..., AsyncGenerator[T, None]], self.client.create_partial
        )
        return create_partial(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )


class Instructor:
    """Sync client wrapper that adds structured output support."""

    client: Any | None
    create_fn: Callable[..., Any]
    mode: Mode
    default_model: str | None = None
    provider: Provider
    hooks: Hooks

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        mode: Mode = Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        if mode == Mode.FUNCTIONS:
            Mode.warn_mode_functions_deprecation()

        self.kwargs = kwargs
        self.provider = provider
        self.hooks = hooks or Hooks()

        if mode in {
            Mode.RESPONSES_TOOLS,
            Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }:
            import openai as _openai

            assert isinstance(client, (_openai.OpenAI, _openai.AsyncOpenAI))
            self.responses = Response(client=self)

    def on(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "completion:usage",
                "parse:error",
            ]
        ),
        handler: Callable[[Any], None],
    ) -> None:
        self.hooks.on(hook_name, handler)

    def off(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "completion:usage",
                "parse:error",
            ]
        ),
        handler: Callable[[Any], None],
    ) -> None:
        self.hooks.off(hook_name, handler)

    def clear(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "completion:usage",
                "parse:error",
            ]
        )
        | None = None,
    ) -> None:
        self.hooks.clear(hook_name)

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    @property
    def messages(self) -> Self:
        return self

    @overload
    def create(
        self: AsyncInstructor,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        token_budget: int | None = None,
        **kwargs: Any,
    ) -> Awaitable[T]: ...

    @overload
    def create(
        self: Self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        token_budget: int | None = None,
        **kwargs: Any,
    ) -> T: ...

    @overload
    def create(
        self: AsyncInstructor,
        response_model: None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        token_budget: None = None,
        **kwargs: Any,
    ) -> Awaitable[Any]: ...

    @overload
    def create(
        self: Self,
        response_model: None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        token_budget: None = None,
        **kwargs: Any,
    ) -> Any: ...

    def create(
        self,
        response_model: type[T] | None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        token_budget: int | None = None,
        **kwargs: Any,
    ) -> T | Any | Awaitable[T] | Awaitable[Any]:
        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(response_model)
        kwargs = self.handle_kwargs(kwargs)
        if token_budget is not None:
            kwargs["token_budget"] = token_budget

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        return self.create_fn(
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            context=context,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )

    @overload
    def create_partial(
        self: AsyncInstructor,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_partial(
        self: Self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    def create_partial(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        kwargs["stream"] = True

        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(response_model)
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        response_model = Partial[response_model]  # type: ignore
        return self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            context=context,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )

    @overload
    def create_iterable(
        self: AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_iterable(
        self: Self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    def create_iterable(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        kwargs["stream"] = True
        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(response_model)
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        response_model = Iterable[response_model]  # type: ignore
        return self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            context=context,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )

    @overload
    def create_with_completion(
        self: AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Awaitable[tuple[T, Any]]: ...

    @overload
    def create_with_completion(
        self: Self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> tuple[T, Any]: ...

    def create_with_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> tuple[T, Any] | Awaitable[tuple[T, Any]]:
        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(response_model)
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        model = self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            context=context,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )
        return model, model._raw_response

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Handle and process keyword arguments for the API call.

        This method merges the provided kwargs with the default kwargs stored in the instance.
        It ensures that any kwargs passed to the method call take precedence over the default ones.
        """
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs

    def __copy__(self) -> Self:
        copied = object.__new__(type(self))
        copied.__dict__.update(self.__dict__)
        for cls in type(self).__mro__:
            for descriptor in vars(cls).values():
                if isinstance(descriptor, MemberDescriptorType):
                    try:
                        value = descriptor.__get__(self, type(self))
                    except AttributeError:
                        continue
                    descriptor.__set__(copied, value)
        try:
            responses = object.__getattribute__(self, "responses")
        except AttributeError:
            responses = None
        if (
            isinstance(responses, (Response, AsyncResponse))
            and responses.client is self
        ):
            copied.responses = copy(responses)
            copied.responses.client = copied
        return copied

    def __getattr__(self, attr: str) -> Any:
        if attr not in {"create", "chat", "messages"}:
            return getattr(object.__getattribute__(self, "client"), attr)

        return getattr(self, attr)


class AsyncInstructor(Instructor):
    """Async client wrapper that adds structured output support."""

    client: Any | None
    create_fn: Callable[..., Any]
    mode: Mode
    default_model: str | None = None
    provider: Provider
    hooks: Hooks

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        mode: Mode = Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        self.kwargs = kwargs
        self.provider = provider
        self.hooks = hooks or Hooks()

        if mode in {
            Mode.RESPONSES_TOOLS,
            Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }:
            import openai as _openai

            assert isinstance(client, (_openai.OpenAI, _openai.AsyncOpenAI))
            self.responses = AsyncResponse(client=self)

    async def create(  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self,
        response_model: type[T] | None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        token_budget: int | None = None,
        **kwargs: Any,
    ) -> T | Any:
        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(response_model)
        kwargs = self.handle_kwargs(kwargs)
        if token_budget is not None:
            kwargs["token_budget"] = token_budget

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        # Check if the response model is an iterable type
        if (
            get_origin(response_model) in {Iterable}
            and get_args(response_model)
            and get_args(response_model)[0] is not None
            and self.mode not in Mode.parallel_modes()
        ):
            return self.create_iterable(
                messages=messages,
                response_model=get_args(response_model)[0],
                max_retries=max_retries,
                context=context,
                strict=strict,
                hooks=hooks,  # Pass the per-call hooks to create_iterable
                **kwargs,
            )

        return await self.create_fn(
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )

    async def create_partial(  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(response_model)
        kwargs = self.handle_kwargs(kwargs)
        kwargs["stream"] = True

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        async for item in await self.create_fn(
            response_model=Partial[response_model],  # type: ignore
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        ):
            yield item

    async def create_iterable(  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(response_model)
        kwargs = self.handle_kwargs(kwargs)
        kwargs["stream"] = True

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        iterable_model: Any = Iterable
        async for item in await self.create_fn(
            response_model=iterable_model[response_model],
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        ):
            yield item

    async def create_with_completion(  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        from instructor.v2.validation.async_validators import reject_async_validators

        reject_async_validators(response_model)
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        response = await self.create_fn(
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )
        return response, response._raw_response


@overload
def from_openai(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Compatibility wrapper for the v2 OpenAI factory."""
    from instructor.v2.providers.openai.client import from_openai as from_openai_v2

    return from_openai_v2(client=client, mode=mode, **kwargs)


@overload
def from_litellm(
    completion: Callable[..., Coroutine[Any, Any, T]],
    mode: Mode = Mode.TOOLS,
    *,
    async_client: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_litellm(
    completion: Callable[..., Awaitable[T]],
    mode: Mode = Mode.TOOLS,
    *,
    async_client: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_litellm(
    completion: Callable[..., object],
    mode: Mode = Mode.TOOLS,
    *,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


def from_litellm(
    completion: Callable[..., object] | Callable[..., Awaitable[Any]],
    mode: Mode = Mode.TOOLS,
    *,
    async_client: bool | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Compatibility wrapper for the v2 LiteLLM factory."""
    from instructor.v2.providers.litellm.client import from_litellm as from_litellm_v2

    if async_client is True:
        return from_litellm_v2(
            completion=cast(Callable[..., Awaitable[Any]], completion),
            mode=mode,
            async_client=True,
            **kwargs,
        )
    if async_client is False:
        return from_litellm_v2(
            completion=completion,
            mode=mode,
            async_client=False,
            **kwargs,
        )
    return from_litellm_v2(
        completion=completion,
        mode=mode,
        **kwargs,
    )
