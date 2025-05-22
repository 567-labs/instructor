from __future__ import annotations

from typing import Any, Union, get_origin

from vertexai.preview.generative_models import ToolConfig  # type: ignore
import vertexai.generative_models as gm  # type: ignore
from pydantic import BaseModel
import instructor
from instructor.dsl.parallel import get_types_array
import jsonref


def _create_gemini_json_schema(model: BaseModel):
    # Add type check to ensure we have a concrete model class
    if get_origin(model) is not None:
        raise TypeError(f"Expected concrete model class, got type hint {model}")

    schema = model.model_json_schema()
    schema_without_refs: dict[str, Any] = jsonref.replace_refs(schema)  # type: ignore
    gemini_schema: dict[Any, Any] = {
        "type": schema_without_refs["type"],
        "properties": schema_without_refs["properties"],
        "required": (
            schema_without_refs["required"] if "required" in schema_without_refs else []
        ),  # TODO: Temporary Fix for Iterables which throw an error when their tasks field is specified in the required field
    }
    return gemini_schema


def _create_vertexai_tool(
    models: BaseModel | list[BaseModel] | type,
) -> gm.Tool:  # noqa: UP007
    """Creates a tool with function declarations for single model or list of models"""
    # Handle Iterable case first
    if get_origin(models) is not None:
        model_list = list(get_types_array(models))  # type: ignore
    else:
        # Handle both single model and list of models
        model_list = models if isinstance(models, list) else [models]

    declarations = []
    for model in model_list:  # type: ignore
        parameters = _create_gemini_json_schema(model)  # type: ignore
        declaration = gm.FunctionDeclaration(
            name=model.__name__,  # type: ignore
            description=model.__doc__,  # type: ignore
            parameters=parameters,
        )
        declarations.append(declaration)  # type: ignore

    return gm.Tool(function_declarations=declarations)  # type: ignore


def vertexai_message_parser(
    message: dict[str, str | gm.Part | list[str | gm.Part]],
) -> gm.Content:
    if isinstance(message["content"], str):
        return gm.Content(
            role=message["role"],  # type:ignore
            parts=[gm.Part.from_text(message["content"])],
        )
    elif isinstance(message["content"], list):
        parts: list[gm.Part] = []
        for item in message["content"]:
            if isinstance(item, str):
                parts.append(gm.Part.from_text(item))
            elif isinstance(item, gm.Part):
                parts.append(item)
            else:
                raise ValueError(f"Unsupported content type in list: {type(item)}")
        return gm.Content(
            role=message["role"],  # type:ignore
            parts=parts,
        )
    else:
        raise ValueError("Unsupported message content type")


def _vertexai_message_list_parser(
    messages: list[dict[str, str | gm.Part | list[str | gm.Part]]],
) -> list[gm.Content]:
    contents = [
        vertexai_message_parser(message) if isinstance(message, dict) else message
        for message in messages
    ]
    return contents


def vertexai_function_response_parser(
    response: gm.GenerationResponse, exception: Exception
) -> gm.Content:
    return gm.Content(
        parts=[
            gm.Part.from_function_response(
                name=response.candidates[0].content.parts[0].function_call.name,
                response={
                    "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                },
            )
        ]
    )


def vertexai_process_response(
    _kwargs: dict[str, Any],
    model: Union[BaseModel, list[BaseModel], type],  # noqa: UP007
):
    messages: list[dict[str, str]] = _kwargs.pop("messages")
    contents = _vertexai_message_list_parser(messages)  # type: ignore

    tool = _create_vertexai_tool(models=model)

    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        )
    )
    return contents, [tool], tool_config


def vertexai_process_json_response(_kwargs: dict[str, Any], model: BaseModel):
    messages: list[dict[str, str]] = _kwargs.pop("messages")
    contents = _vertexai_message_list_parser(messages)  # type: ignore

    config: dict[str, Any] | None = _kwargs.pop("generation_config", None)

    response_schema = _create_gemini_json_schema(model)

    generation_config = gm.GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        **(config if config else {}),
    )

    return contents, generation_config


def from_vertexai(
    client: gm.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.VERTEXAI_TOOLS,
    _async: bool = False,
    use_async: bool | None = None,
    **kwargs: Any,
) -> instructor.Instructor:
    valid_modes = {
        instructor.Mode.VERTEXAI_PARALLEL_TOOLS,
        instructor.Mode.VERTEXAI_TOOLS,
        instructor.Mode.VERTEXAI_JSON,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="VertexAI",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, gm.GenerativeModel):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of vertexai.generative_models.GenerativeModel. "
            f"Got: {type(client).__name__}"
        )

    if use_async is not None and _async != False:
        from instructor.exceptions import ConfigurationError
        
        raise ConfigurationError(
            "Cannot provide both '_async' and 'use_async'. Use 'use_async' instead."
        )
        
    if _async and use_async is None:
        import warnings
        
        warnings.warn(
            "'_async' is deprecated. Use 'use_async' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        use_async = _async
    
    is_async = use_async if use_async is not None else _async

    create = client.generate_content_async if is_async else client.generate_content

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.VERTEXAI,
        mode=mode,
        **kwargs,
    )
