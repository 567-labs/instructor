from __future__ import annotations
from functools import wraps
from typing import (
    Any,
    Callable,
    Protocol,
    TypeVar,
    Union,
    overload,
)
from collections.abc import Awaitable
from typing_extensions import ParamSpec

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from instructor.process_response import handle_response_model
from instructor.retry import retry_async, retry_sync
from instructor.utils import is_async

from instructor.mode import Mode
import logging

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")


class InstructorChatCompletionCreate(Protocol):
    def __call__(
        self,
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        max_retries: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model: ...


class AsyncInstructorChatCompletionCreate(Protocol):
    async def __call__(
        self,
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        max_retries: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> T_Model: ...


def handle_context(
    context: dict[str, Any] | None = None,
    validation_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Handle the context and validation_context parameters.
    If both are provided, raise an error.
    If validation_context is provided, issue a deprecation warning and use it as context.
    If neither is provided, return None.
    """
    if context is not None and validation_context is not None:
        raise ValueError(
            "Cannot provide both 'context' and 'validation_context'. Use 'context' instead."
        )
    if validation_context is not None and context is None:
        import warnings

        warnings.warn(
            "'validation_context' is deprecated. Use 'context' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        context = validation_context
    return context


def handle_templating(
    messages: list[dict[str, Any]], context: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """
    Handle templating for messages using the provided context.

    This function processes a list of messages, applying Jinja2 templating to their content
    using the provided context. It supports both standard OpenAI message format and
    Anthropic's format with parts.

    Args:
        messages (list[dict[str, Any]]): A list of message dictionaries to process.
        context (dict[str, Any] | None, optional): A dictionary of variables to use in templating.
            Defaults to None.

    Returns:
        list[dict[str, Any]]: The processed list of messages with templated content.

    Note:
        - If no context is provided, the original messages are returned unchanged.
        - For OpenAI format, the 'content' field is processed if it's a string.
        - For Anthropic format, each 'text' part within the 'content' list is processed.
        - The function uses Jinja2 for templating and applies textwrap.dedent for formatting.

    TODO: Gemini, Cohere, formats are missing here.
    """
    if context is None:
        return messages

    from jinja2 import Template
    from textwrap import dedent

    for message in messages:
        # Handle OpenAI format
        if isinstance(message.get("content"), str):
            message["content"] = dedent(Template(message["content"]).render(**context))
            continue

        # Handle Anthropic format
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    part["text"] = dedent(Template(part["text"]).render(**context))

        # TODO: Add handling for Gemini and Cohere formats

    return messages


@overload
def patch(
    client: OpenAI,
    mode: Mode = Mode.TOOLS,
) -> OpenAI: ...


@overload
def patch(
    client: AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
) -> AsyncOpenAI: ...


@overload
def patch(
    create: Callable[T_ParamSpec, T_Retval],
    mode: Mode = Mode.TOOLS,
) -> InstructorChatCompletionCreate: ...


@overload
def patch(
    create: Awaitable[T_Retval],
    mode: Mode = Mode.TOOLS,
) -> InstructorChatCompletionCreate: ...


def patch(  # type: ignore
    client: Union[OpenAI, AsyncOpenAI] | None = None,
    create: Callable[T_ParamSpec, T_Retval] | None = None,
    mode: Mode = Mode.TOOLS,
) -> Union[OpenAI, AsyncOpenAI]:
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """

    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

    if create is not None:
        func = create
    elif client is not None:
        func = client.chat.completions.create
    else:
        raise ValueError("Either client or create must be provided")

    func_is_async = is_async(func)

    @wraps(func)  # type: ignore
    async def new_create_async(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int = 1,
        strict: bool = True,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:

        context = handle_context(context, validation_context)

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )
        new_kwargs["messages"] = handle_templating(new_kwargs["messages"], context)

        response = await retry_async(
            func=func,  # type: ignore
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            strict=strict,
            mode=mode,
        )
        return response

    @wraps(func)  # type: ignore
    def new_create_sync(
        response_model: type[T_Model] | None = None,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_retries: int = 1,
        strict: bool = True,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:

        context = handle_context(context, validation_context)

        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )
        new_kwargs["messages"] = handle_templating(new_kwargs["messages"], context)
        response = retry_sync(
            func=func,  # type: ignore
            response_model=response_model,
            context=context,
            max_retries=max_retries,
            args=args,
            strict=strict,
            kwargs=new_kwargs,
            mode=mode,
        )
        return response

    new_create = new_create_async if func_is_async else new_create_sync

    if client is not None:
        client.chat.completions.create = new_create  # type: ignore
        return client
    else:
        return new_create  # type: ignore


def apatch(client: AsyncOpenAI, mode: Mode = Mode.TOOLS) -> AsyncOpenAI:
    """
    No longer necessary, use `patch` instead.

    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """
    import warnings

    warnings.warn(
        "apatch is deprecated, use patch instead", DeprecationWarning, stacklevel=2
    )
    return patch(client, mode=mode)
