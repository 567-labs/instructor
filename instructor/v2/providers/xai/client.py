"""v2 xAI client factory.

Creates Instructor instances for xAI's Grok models using the v2 registry system.

The xAI SDK has a unique API that differs from OpenAI. This client handles
the translation between Instructor's interface and xAI's native SDK.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast, overload

from pydantic import BaseModel

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import get_types_array, model_for_tool_name
from instructor.v2.dsl.partial import PartialBase
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.core.response_model import prepare_response_model

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.xai import handlers

if TYPE_CHECKING:
    from xai_sdk.sync.client import Client as SyncClient
    from xai_sdk.aio.client import Client as AsyncClient
    from xai_sdk import chat as xchat
else:
    try:
        from xai_sdk.sync.client import Client as SyncClient
        from xai_sdk.aio.client import Client as AsyncClient
        from xai_sdk import chat as xchat
    except ImportError:
        SyncClient = None
        AsyncClient = None
        xchat = None


def _get_model_schema(response_model: Any) -> dict[str, Any]:
    """Get JSON schema from a response model."""
    if hasattr(response_model, "model_json_schema") and callable(
        response_model.model_json_schema
    ):
        return response_model.model_json_schema()
    return {}


def _get_model_name(response_model: Any) -> str:
    """Get the name of a response model."""
    return getattr(response_model, "__name__", "Model")


def _finalize_parsed_response(parsed: Any, raw_response: Any) -> Any:
    """Finalize parsed response, attaching raw response."""
    if isinstance(parsed, BaseModel):
        parsed._raw_response = raw_response  # type: ignore[attr-defined]
    if isinstance(parsed, IterableBase):
        return [task for task in parsed.tasks]
    if isinstance(parsed, AdapterBase):
        return parsed.content
    return parsed


def _convert_messages(messages: list[dict[str, Any]]) -> list[Any]:
    return handlers._convert_messages_with_chat(messages, xchat)


def _add_md_json_instructions(
    messages: list[dict[str, Any]], response_model: Any
) -> list[dict[str, Any]]:
    """Ensure MD_JSON requests include a schema instruction for xAI."""
    schema = _get_model_schema(response_model)
    if not schema:
        return list(messages)

    instruction = (
        "Return your answer as JSON that matches this schema. "
        "Respond with JSON only (preferably inside a ```json code block). "
        f"Schema: {json.dumps(schema, indent=2)}"
    )

    new_messages = list(messages)
    if new_messages and new_messages[0].get("role") == "system":
        content = new_messages[0].get("content", "")
        new_messages[0] = {
            **new_messages[0],
            "content": f"{content}\n\n{instruction}" if content else instruction,
        }
        return new_messages

    return [{"role": "system", "content": instruction}, *new_messages]


def _iter_tool_call_arg_deltas(stream_iter: Any) -> Any:
    """Yield tool call argument deltas from sync xAI streams."""
    last_tool_args: dict[str, str] = {}
    last_args_value = ""
    for resp, _ in stream_iter:
        tool_calls = getattr(resp, "tool_calls", None) or []
        for index, tool_call in enumerate(tool_calls):
            function = getattr(tool_call, "function", None)
            args = getattr(function, "arguments", None)
            if args is None:
                continue
            if isinstance(args, dict):
                args = json.dumps(args)
            tool_id = getattr(tool_call, "id", None) or str(index)
            delta = args
            previous = last_tool_args.get(tool_id, "")
            if previous and delta.startswith(previous):
                delta = delta[len(previous) :]
            elif last_args_value and delta.startswith(last_args_value):
                delta = delta[len(last_args_value) :]
            last_tool_args[tool_id] = args
            last_args_value = args
            if delta:
                yield delta


async def _aiter_tool_call_arg_deltas(stream_iter: Any) -> Any:
    """Yield tool call argument deltas from async xAI streams."""
    last_tool_args: dict[str, str] = {}
    last_args_value = ""
    async for resp, _ in stream_iter:
        tool_calls = getattr(resp, "tool_calls", None) or []
        for index, tool_call in enumerate(tool_calls):
            function = getattr(tool_call, "function", None)
            args = getattr(function, "arguments", None)
            if args is None:
                continue
            if isinstance(args, dict):
                args = json.dumps(args)
            tool_id = getattr(tool_call, "id", None) or str(index)
            delta = args
            previous = last_tool_args.get(tool_id, "")
            if previous and delta.startswith(previous):
                delta = delta[len(previous) :]
            elif last_args_value and delta.startswith(last_args_value):
                delta = delta[len(last_args_value) :]
            last_tool_args[tool_id] = args
            last_args_value = args
            if delta:
                yield delta


@overload
def from_xai(
    client: SyncClient,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_xai(
    client: AsyncClient,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_xai(
    client: SyncClient | AsyncClient,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from an xAI client using v2 registry.

    Args:
        client: An instance of xAI client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for xAI
        ClientError: If client is not a valid xAI client instance

    Examples:
        >>> from xai_sdk.sync.client import Client
        >>> from instructor import Mode
        >>> from instructor.v2.providers.xai import from_xai
        >>>
        >>> client = Client()
        >>> instructor_client = from_xai(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use JSON_SCHEMA mode
        >>> instructor_client = from_xai(client, mode=Mode.JSON_SCHEMA)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    # Normalize provider-specific modes to generic modes
    # XAI_TOOLS -> TOOLS, XAI_JSON -> MD_JSON
    normalized_mode = normalize_mode(Provider.XAI, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.XAI, normalized_mode):
        from instructor.v2.core.errors import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.XAI)
        raise ModeError(
            mode=str(mode.value),
            provider=Provider.XAI.value,
            valid_modes=[str(m.value) for m in available_modes],
        )

    # Use normalized mode
    mode = normalized_mode

    # Validate client type
    if SyncClient is None or AsyncClient is None:
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            "The xai-sdk package is required to use the xAI provider. "
            'Install it with `uv pip install "instructor[xai]"` '
            '(or `pip install "instructor[xai]"`).'
        )

    if not isinstance(client, (SyncClient, AsyncClient)):
        from instructor.v2.core.errors import ClientError

        raise ClientError(
            f"Client must be an instance of xai_sdk.sync.client.Client or "
            f"xai_sdk.aio.client.Client. Got: {type(client).__name__}"
        )

    # Validate that handlers are registered for this mode
    mode_registry.get_handlers(Provider.XAI, mode)

    # Create async wrapper for xAI's unique API
    async def acreate(
        response_model: type[BaseModel] | None,
        messages: list[dict[str, Any]],
        strict: bool = True,
        **call_kwargs: Any,
    ) -> Any:
        model = call_kwargs.pop("model")
        # Remove instructor-specific kwargs that xAI doesn't support
        call_kwargs.pop("max_retries", None)
        call_kwargs.pop("validation_context", None)
        call_kwargs.pop("context", None)
        call_kwargs.pop("hooks", None)
        if call_kwargs.pop("token_budget", None) is not None:
            raise ValueError("token_budget is not supported for xAI requests")
        is_stream = call_kwargs.pop("stream", False)

        prepared_model = response_model
        if response_model is not None and (
            mode in {Mode.TOOLS, Mode.PARALLEL_TOOLS, Mode.MD_JSON} or is_stream
        ):
            prepared_model = prepare_response_model(response_model)
            if mode == Mode.MD_JSON:
                messages = _add_md_json_instructions(messages, prepared_model)

        x_messages = _convert_messages(messages)
        chat = cast(
            Any, client.chat.create(model=model, messages=x_messages, **call_kwargs)
        )

        if response_model is None:
            return await chat.sample()  # type: ignore[misc]

        if mode == Mode.JSON_SCHEMA:
            if is_stream:
                chat.proto.response_format.CopyFrom(
                    xchat.chat_pb2.ResponseFormat(
                        format_type=xchat.chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                        schema=json.dumps(_get_model_schema(prepared_model)),
                    )
                )
                json_chunks = (chunk.content async for _, chunk in chat.stream())  # type: ignore[misc]
                rm = cast(type[BaseModel], prepared_model)
                if issubclass(rm, IterableBase):
                    return rm.tasks_from_chunks_async(json_chunks)
                if issubclass(rm, PartialBase):
                    return rm.model_from_chunks_async(json_chunks)  # type: ignore
                raise ValueError(
                    f"Unsupported response model type for streaming: {_get_model_name(response_model)}"
                )
            raw, parsed = await chat.parse(response_model)  # type: ignore[misc]
            parsed._raw_response = raw
            return parsed
        if mode == Mode.TOOLS:
            tool_obj = xchat.tool(
                name=_get_model_name(prepared_model),
                description=prepared_model.__doc__ or "",
                parameters=_get_model_schema(prepared_model),
            )
            chat.proto.tools.append(cast(Any, tool_obj))
            tool_name = tool_obj.function.name  # type: ignore[attr-defined]
            chat.proto.tool_choice.CopyFrom(xchat.required_tool(tool_name))
            if is_stream:
                stream_iter = chat.stream()  # type: ignore[misc]
                args = _aiter_tool_call_arg_deltas(stream_iter)
                rm = cast(type[BaseModel], prepared_model)
                if issubclass(rm, IterableBase):
                    return rm.tasks_from_chunks_async(args)
                if issubclass(rm, PartialBase):
                    return rm.model_from_chunks_async(args)  # type: ignore
                raise ValueError(
                    f"Unsupported response model type for streaming: {_get_model_name(response_model)}"
                )
            resp = await chat.sample()  # type: ignore[misc]
            if not resp.tool_calls:  # type: ignore[attr-defined]
                # Try to extract from text content
                from instructor.v2.core.function_calls import (
                    _validate_model_from_json,
                )
                from instructor.v2.core.json import extract_json_from_codeblock

                text_content: str = ""
                if hasattr(resp, "text") and resp.text:  # type: ignore[attr-defined]
                    text_content = str(resp.text)  # type: ignore[attr-defined]
                elif hasattr(resp, "content") and resp.content:  # type: ignore[attr-defined]
                    content = resp.content  # type: ignore[attr-defined]
                    if isinstance(content, str):
                        text_content = content
                    elif isinstance(content, list) and content:
                        text_content = str(content[0])

                if text_content:
                    json_str = extract_json_from_codeblock(text_content)
                    model_for_validation = cast(type[Any], prepared_model)
                    parsed = _validate_model_from_json(
                        model_for_validation, json_str, None, strict
                    )
                    return _finalize_parsed_response(parsed, resp)

                raise ValueError(
                    f"No tool calls returned from xAI and no text content available. "
                    f"Response: {resp}"
                )

            args = resp.tool_calls[0].function.arguments  # type: ignore[index,attr-defined]
            from instructor.v2.core.function_calls import (
                _validate_model_from_json,
            )

            model_for_validation = cast(type[Any], prepared_model)
            parsed = _validate_model_from_json(model_for_validation, args, None, strict)
            return _finalize_parsed_response(parsed, resp)
        if mode == Mode.PARALLEL_TOOLS:
            for model_type in get_types_array(response_model):  # type: ignore[arg-type]
                tool_obj = xchat.tool(
                    name=_get_model_name(model_type),
                    description=model_type.__doc__ or "",
                    parameters=_get_model_schema(model_type),
                )
                chat.proto.tools.append(cast(Any, tool_obj))
            resp = await chat.sample()  # type: ignore[misc]
            type_registry = {
                model_type.__name__: model_type
                for model_type in get_types_array(response_model)  # type: ignore[arg-type]
            }
            from instructor.v2.core.function_calls import _validate_model_from_json

            return iter(
                _validate_model_from_json(
                    model_for_tool_name(
                        type_registry,
                        tool_call.function.name,
                        mode=mode,
                        raw_response=resp,
                    ),
                    tool_call.function.arguments,
                    None,
                    strict,
                )
                for tool_call in resp.tool_calls
            )
        # MD_JSON mode - use sample() and extract from text
        resp = await chat.sample()  # type: ignore[misc]
        from instructor.v2.core.function_calls import _validate_model_from_json
        from instructor.v2.core.json import extract_json_from_codeblock

        text_content = ""
        if hasattr(resp, "text") and resp.text:
            text_content = str(resp.text)
        elif hasattr(resp, "content") and resp.content:
            content = resp.content
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list) and content:
                text_content = str(content[0])

        if text_content:
            json_str = extract_json_from_codeblock(text_content)
            model_for_validation = cast(type[Any], prepared_model)
            parsed = _validate_model_from_json(
                model_for_validation, json_str, None, strict
            )
            return _finalize_parsed_response(parsed, resp)

        raise ValueError(f"Could not extract JSON from xAI response: {resp}")

    # Create sync wrapper for xAI's unique API
    def create(
        response_model: type[BaseModel] | None,
        messages: list[dict[str, Any]],
        strict: bool = True,
        **call_kwargs: Any,
    ) -> Any:
        model = call_kwargs.pop("model")
        # Remove instructor-specific kwargs that xAI doesn't support
        call_kwargs.pop("max_retries", None)
        call_kwargs.pop("validation_context", None)
        call_kwargs.pop("context", None)
        call_kwargs.pop("hooks", None)
        if call_kwargs.pop("token_budget", None) is not None:
            raise ValueError("token_budget is not supported for xAI requests")
        is_stream = call_kwargs.pop("stream", False)

        prepared_model = response_model
        if response_model is not None and (
            mode in {Mode.TOOLS, Mode.PARALLEL_TOOLS, Mode.MD_JSON} or is_stream
        ):
            prepared_model = prepare_response_model(response_model)
            if mode == Mode.MD_JSON:
                messages = _add_md_json_instructions(messages, prepared_model)

        x_messages = _convert_messages(messages)
        chat = cast(
            Any, client.chat.create(model=model, messages=x_messages, **call_kwargs)
        )

        if response_model is None:
            return chat.sample()  # type: ignore[misc]

        if mode == Mode.JSON_SCHEMA:
            if is_stream:
                chat.proto.response_format.CopyFrom(
                    xchat.chat_pb2.ResponseFormat(
                        format_type=xchat.chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                        schema=json.dumps(_get_model_schema(prepared_model)),
                    )
                )
                json_chunks = (chunk.content for _, chunk in chat.stream())  # type: ignore[misc]
                rm = cast(type[BaseModel], prepared_model)
                if issubclass(rm, IterableBase):
                    return rm.tasks_from_chunks(json_chunks)
                if issubclass(rm, PartialBase):
                    return cast(Any, rm).model_from_chunks(json_chunks)
                raise ValueError(
                    f"Unsupported response model type for streaming: {_get_model_name(response_model)}"
                )
            raw, parsed = chat.parse(response_model)  # type: ignore[misc]
            parsed._raw_response = raw
            return parsed
        if mode == Mode.TOOLS:
            tool_obj = xchat.tool(
                name=_get_model_name(prepared_model),
                description=prepared_model.__doc__ or "",
                parameters=_get_model_schema(prepared_model),
            )
            chat.proto.tools.append(cast(Any, tool_obj))
            tool_name = tool_obj.function.name  # type: ignore[attr-defined]
            chat.proto.tool_choice.CopyFrom(xchat.required_tool(tool_name))
            if is_stream:
                stream_iter = chat.stream()  # type: ignore[misc]
                args = _iter_tool_call_arg_deltas(stream_iter)
                rm = cast(type[BaseModel], prepared_model)
                if issubclass(rm, IterableBase):
                    return rm.tasks_from_chunks(args)
                if issubclass(rm, PartialBase):
                    return cast(Any, rm).model_from_chunks(args)
                raise ValueError(
                    f"Unsupported response model type for streaming: {_get_model_name(response_model)}"
                )
            resp = chat.sample()  # type: ignore[misc]
            if not resp.tool_calls:  # type: ignore[attr-defined]
                # Try to extract from text content
                from instructor.v2.core.function_calls import (
                    _validate_model_from_json,
                )
                from instructor.v2.core.json import extract_json_from_codeblock

                text_content: str = ""
                if hasattr(resp, "text") and resp.text:  # type: ignore[attr-defined]
                    text_content = str(resp.text)  # type: ignore[attr-defined]
                elif hasattr(resp, "content") and resp.content:  # type: ignore[attr-defined]
                    content = resp.content  # type: ignore[attr-defined]
                    if isinstance(content, str):
                        text_content = content
                    elif isinstance(content, list) and content:
                        text_content = str(content[0])

                if text_content:
                    json_str = extract_json_from_codeblock(text_content)
                    model_for_validation = cast(type[Any], prepared_model)
                    parsed = _validate_model_from_json(
                        model_for_validation, json_str, None, strict
                    )
                    return _finalize_parsed_response(parsed, resp)

                raise ValueError(
                    f"No tool calls returned from xAI and no text content available. "
                    f"Response: {resp}"
                )

            args = resp.tool_calls[0].function.arguments  # type: ignore[index,attr-defined]
            from instructor.v2.core.function_calls import (
                _validate_model_from_json,
            )

            model_for_validation = cast(type[Any], prepared_model)
            parsed = _validate_model_from_json(model_for_validation, args, None, strict)
            return _finalize_parsed_response(parsed, resp)
        if mode == Mode.PARALLEL_TOOLS:
            for model_type in get_types_array(response_model):  # type: ignore[arg-type]
                tool_obj = xchat.tool(
                    name=_get_model_name(model_type),
                    description=model_type.__doc__ or "",
                    parameters=_get_model_schema(model_type),
                )
                chat.proto.tools.append(cast(Any, tool_obj))
            resp = chat.sample()  # type: ignore[misc]
            type_registry = {
                model_type.__name__: model_type
                for model_type in get_types_array(response_model)  # type: ignore[arg-type]
            }
            from instructor.v2.core.function_calls import _validate_model_from_json

            return iter(
                _validate_model_from_json(
                    model_for_tool_name(
                        type_registry,
                        tool_call.function.name,
                        mode=mode,
                        raw_response=resp,
                    ),
                    tool_call.function.arguments,
                    None,
                    strict,
                )
                for tool_call in resp.tool_calls
            )
        # MD_JSON mode - use sample() and extract from text
        resp = chat.sample()  # type: ignore[misc]
        from instructor.v2.core.function_calls import _validate_model_from_json
        from instructor.v2.core.json import extract_json_from_codeblock

        text_content = ""
        if hasattr(resp, "text") and resp.text:
            text_content = str(resp.text)
        elif hasattr(resp, "content") and resp.content:
            content = resp.content
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list) and content:
                text_content = str(content[0])

        if text_content:
            json_str = extract_json_from_codeblock(text_content)
            model_for_validation = cast(type[Any], prepared_model)
            parsed = _validate_model_from_json(
                model_for_validation, json_str, None, strict
            )
            return _finalize_parsed_response(parsed, resp)

        raise ValueError(f"Could not extract JSON from xAI response: {resp}")

    # Return sync or async instructor
    if isinstance(client, AsyncClient):
        return AsyncInstructor(
            client=client,
            create=acreate,
            provider=Provider.XAI,
            mode=mode,
            **kwargs,
        )
    return Instructor(
        client=client,
        create=create,
        provider=Provider.XAI,
        mode=mode,
        **kwargs,
    )
