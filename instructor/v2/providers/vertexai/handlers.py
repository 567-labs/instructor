"""VertexAI v2 mode handlers."""

from __future__ import annotations

import inspect
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from typing import Any, cast, get_origin

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import ConfigurationError
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase, get_types_array
from instructor.v2.dsl.partial import PartialBase
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.providers.gemini.utils import (
    handle_vertexai_json,
    handle_vertexai_parallel_tools,
    handle_vertexai_tools,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


def _gm() -> Any:
    try:
        import vertexai.generative_models as gm  # type: ignore[import-not-found]
    except ImportError as err:
        raise ImportError(
            "vertexai is required for VertexAI handler operations. "
            "Install it with: pip install google-cloud-aiplatform"
        ) from err
    return gm


def _tool_config_cls() -> Any:
    try:
        from vertexai.preview.generative_models import (  # type: ignore[import-not-found]
            ToolConfig,
        )
    except ImportError as err:
        raise ImportError(
            "vertexai is required for VertexAI handler operations. "
            "Install it with: pip install google-cloud-aiplatform"
        ) from err
    return ToolConfig


def vertexai_message_parser(
    message: dict[str, Any],
) -> Any:
    gm = _gm()
    if isinstance(message["content"], str):
        return gm.Content(
            role=message["role"],
            parts=[gm.Part.from_text(message["content"])],
        )
    if isinstance(message["content"], list):
        parts: list[gm.Part] = []
        for item in message["content"]:
            if isinstance(item, str):
                parts.append(gm.Part.from_text(item))
            elif isinstance(item, gm.Part):
                parts.append(item)
            else:
                raise ValueError(f"Unsupported content type in list: {type(item)}")
        return gm.Content(
            role=message["role"],
            parts=parts,
        )
    raise ValueError("Unsupported message content type")


def vertexai_message_list_parser(
    messages: list[dict[str, Any]],
) -> list[Any]:
    return [
        vertexai_message_parser(message) if isinstance(message, dict) else message
        for message in messages
    ]


def vertexai_function_response_parser(response: Any, exception: Exception) -> Any:
    gm = _gm()
    return gm.Content(
        parts=[
            gm.Part.from_function_response(
                name=response.candidates[0].content.parts[0].function_call.name,
                response={
                    "content": (
                        "Validation Error found:\n"
                        f"{exception}\nRecall the function correctly, fix the errors"
                    )
                },
            )
        ]
    )


def reask_vertexai_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Build a VertexAI tool reask payload after validation failure."""
    kwargs = kwargs.copy()
    reask_msgs = [
        response.candidates[0].content,
        vertexai_function_response_parser(response, exception),
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_vertexai_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Build a VertexAI JSON reask payload after validation failure."""
    kwargs = kwargs.copy()
    reask_msgs = [
        response.candidates[0].content,
        vertexai_message_parser(
            {
                "role": "user",
                "content": (
                    f"Validation Errors found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors found in the following attempt:\n{response.text}"
                ),
            }
        ),
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def parse_vertexai_tools(
    response_model: type[BaseModel],
    completion: Any,
    validation_context: dict[str, Any] | None = None,
) -> BaseModel:
    """Parse VertexAI function-call responses."""
    tool_call = completion.candidates[0].content.parts[0].function_call.args
    model = {field: tool_call[field] for field in tool_call}
    return response_model.model_validate(
        model,
        context=validation_context,
        strict=False,
    )


def parse_vertexai_json(
    response_model: type[BaseModel],
    completion: Any,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> BaseModel:
    """Parse VertexAI text JSON responses."""
    return response_model.model_validate_json(
        completion.text,
        context=validation_context,
        strict=strict,
    )


def _create_gemini_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    if get_origin(model) is not None:
        raise TypeError(f"Expected concrete model class, got type hint {model}")

    try:
        import jsonref
    except ImportError as err:
        raise ConfigurationError(
            "The 'jsonref' package is required for VertexAI structured output. "
            "Install it with: pip install 'instructor[vertexai]'"
        ) from err

    schema = model.model_json_schema()
    schema_without_refs: dict[str, Any] = jsonref.replace_refs(schema)  # type: ignore[assignment]
    gemini_schema: dict[Any, Any] = {
        "type": schema_without_refs["type"],
        "properties": schema_without_refs["properties"],
        "required": (
            schema_without_refs["required"] if "required" in schema_without_refs else []
        ),
    }
    return gemini_schema


def _create_vertexai_tool(
    models: type[BaseModel] | list[type[BaseModel]] | Any,
) -> Any:
    """Create a tool with function declarations for model(s)."""
    gm = _gm()
    if get_origin(models) is not None:
        model_list = list(get_types_array(cast(Any, models)))
    else:
        model_list = models if isinstance(models, list) else [models]

    declarations = []
    for model in model_list:
        parameters = _create_gemini_json_schema(model)
        declaration = gm.FunctionDeclaration(
            name=model.__name__,
            description=model.__doc__,
            parameters=parameters,
        )
        declarations.append(declaration)

    return gm.Tool(function_declarations=declarations)


def vertexai_process_response(
    call_kwargs: dict[str, Any],
    model: type[BaseModel] | list[type[BaseModel]] | Any,
):
    ToolConfig = _tool_config_cls()
    messages: list[dict[str, str]] = call_kwargs.pop("messages")
    contents = vertexai_message_list_parser(messages)  # type: ignore[arg-type]

    tool = _create_vertexai_tool(models=model)

    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        )
    )
    return contents, [tool], tool_config


def vertexai_process_json_response(call_kwargs: dict[str, Any], model: type[BaseModel]):
    gm = _gm()
    messages: list[dict[str, str]] = call_kwargs.pop("messages")
    contents = vertexai_message_list_parser(messages)  # type: ignore[arg-type]

    config: dict[str, Any] | None = call_kwargs.pop("generation_config", None)
    response_schema = _create_gemini_json_schema(model)

    generation_config = gm.GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        **(config if config else {}),
    )

    return contents, generation_config


class VertexAIHandlerBase(ModeHandler):
    """Base handler for VertexAI modes."""

    mode: Mode

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from VertexAI streaming responses."""
        for chunk in completion:
            try:
                # `candidates` can be empty (e.g. a safety-blocked chunk),
                # which raises IndexError rather than AttributeError — skip
                # the chunk either way.
                if self.mode == Mode.TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                else:
                    yield chunk.candidates[0].content.parts[0].text
            except (AttributeError, IndexError):
                continue

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from VertexAI async streams."""
        async for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                else:
                    yield chunk.candidates[0].content.parts[0].text
            except (AttributeError, IndexError):
                continue

    def _parse_streaming(
        self,
        response_model: type[BaseModel],
        response: Any,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        parse_kwargs: dict[str, Any] = {}
        if validation_context is not None:
            parse_kwargs["context"] = validation_context
        if strict is not None:
            parse_kwargs["strict"] = strict

        streaming_model = cast(Any, response_model)
        task_parser = None
        if (
            self.mode == Mode.TOOLS
            and inspect.isclass(response_model)
            and issubclass(response_model, IterableBase)
        ):
            task_parser = streaming_model.tasks_from_task_list_chunks

        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            if task_parser is not None:
                parse_kwargs["task_parser"] = (
                    streaming_model.tasks_from_task_list_chunks_async
                )
            return streaming_model.from_streaming_response_async(
                response,
                stream_extractor=self.extract_streaming_json_async,
                **parse_kwargs,
            )

        if task_parser is not None:
            parse_kwargs["task_parser"] = task_parser
        generator = streaming_model.from_streaming_response(
            response,
            stream_extractor=self.extract_streaming_json,
            **parse_kwargs,
        )
        if inspect.isclass(response_model) and issubclass(response_model, IterableBase):
            return generator
        return list(generator)

    def _finalize(
        self,
        response_model: type[BaseModel] | ParallelBase,  # noqa: ARG002
        response: Any,
        parsed: BaseModel,
    ) -> Any:
        if isinstance(parsed, AdapterBase):
            return cast(Any, parsed).content
        cast(Any, parsed)._raw_response = response
        return parsed


@register_mode_handler(Provider.VERTEXAI, Mode.TOOLS)
class VertexAIToolsHandler(VertexAIHandlerBase):
    """Handler for VertexAI TOOLS mode."""

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        return handle_vertexai_tools(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_vertexai_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | ParallelBase,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming(
                cast(type[BaseModel], response_model),
                response,
                validation_context,
                strict,
            )
        if isinstance(response_model, ParallelBase):
            parallel_model = cast(ParallelBase[Any], response_model)
            return parallel_model.from_response(
                response,
                mode=Mode.VERTEXAI_PARALLEL_TOOLS,
                validation_context=validation_context,
                strict=strict,
            )
        parsed = parse_vertexai_tools(response_model, response, validation_context)
        return self._finalize(response_model, response, parsed)


@register_mode_handler(Provider.VERTEXAI, Mode.MD_JSON)
class VertexAIJSONHandler(VertexAIHandlerBase):
    """Handler for VertexAI JSON mode."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        return handle_vertexai_json(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_vertexai_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming(
                response_model, response, validation_context, strict
            )
        parsed = parse_vertexai_json(
            response_model, response, validation_context, strict
        )
        return self._finalize(response_model, response, parsed)


@register_mode_handler(Provider.VERTEXAI, Mode.PARALLEL_TOOLS)
class VertexAIParallelToolsHandler(VertexAIHandlerBase):
    """Handler for VertexAI parallel tools mode."""

    mode = Mode.PARALLEL_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        if response_model is None:
            return None, new_kwargs
        return handle_vertexai_parallel_tools(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_vertexai_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | ParallelBase,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if isinstance(response_model, ParallelBase):
            parallel_model = cast(ParallelBase[Any], response_model)
            return parallel_model.from_response(
                response,
                mode=Mode.VERTEXAI_PARALLEL_TOOLS,
                validation_context=validation_context,
                strict=strict,
            )
        parsed = parse_vertexai_tools(response_model, response, validation_context)
        return self._finalize(response_model, response, parsed)


__all__ = [
    "vertexai_function_response_parser",
    "vertexai_message_list_parser",
    "vertexai_message_parser",
    "vertexai_process_json_response",
    "vertexai_process_response",
    "VertexAIToolsHandler",
    "VertexAIJSONHandler",
    "VertexAIParallelToolsHandler",
]
