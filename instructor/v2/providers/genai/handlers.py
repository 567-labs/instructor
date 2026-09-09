from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

from pydantic import BaseModel

from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.errors import IncompleteOutputException
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.mode import Mode
from instructor.v2.core.multimodal import extract_genai_multimodal_content
from instructor.v2.core.providers import Provider
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.parallel import ParallelBase
from instructor.v2.dsl.partial import Partial, PartialBase
from instructor.v2.dsl.simple_type import AdapterBase
from instructor.v2.providers.gemini import utils as gemini_utils
from instructor.v2.providers.genai.request import update_genai_kwargs


def reask_genai_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Build a GenAI tool reask payload after validation failure."""
    from google.genai import types

    kwargs = kwargs.copy()
    existing_contents = kwargs.get("contents")
    if isinstance(existing_contents, list):
        kwargs["contents"] = existing_contents.copy()
    elif existing_contents is None:
        kwargs["contents"] = []
    else:
        kwargs["contents"] = list(existing_contents)

    model_content = None
    function_call_content = None
    function_call = None

    candidates = getattr(response, "candidates", None) if response is not None else None
    if isinstance(candidates, list):
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            if model_content is None:
                model_content = content
            parts = getattr(content, "parts", None) or []
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    function_call_content = content
                    break
            if function_call is not None:
                break

    error_msg = (
        f"Validation Error found:\n{exception}\n"
        "Recall the function correctly, fix the errors"
    )
    if function_call is None:
        if model_content is not None:
            kwargs["contents"].append(model_content)
        kwargs["contents"].append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=error_msg)],
            )
        )
        return kwargs

    function_response_part = types.Part.from_function_response(
        name=function_call.name,
        response={"error": error_msg},
    )
    kwargs["contents"].append(function_call_content)
    kwargs["contents"].append(
        types.Content(role="user", parts=[function_response_part])
    )
    return kwargs


def reask_genai_structured_outputs(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Build a GenAI structured-output reask payload after validation failure."""
    from google.genai import types

    kwargs = kwargs.copy()
    genai_response = (
        response.text
        if response and hasattr(response, "text")
        else "You must generate a response to the user's request that is consistent with the response model"
    )
    kwargs["contents"].append(
        types.ModelContent(
            parts=[
                types.Part.from_text(
                    text=f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors in the following attempt:\n{genai_response}"
                ),
            ]
        ),
    )
    return kwargs


def parse_genai_structured_outputs(
    response_model: type[BaseModel],
    completion: Any,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> BaseModel:
    """Parse GenAI native structured-output responses."""
    return response_model.model_validate_json(
        completion.text,
        context=validation_context,
        strict=strict,
    )


def parse_genai_tools(
    response_model: type[BaseModel],
    completion: Any,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> BaseModel:
    """Parse GenAI function-call responses."""
    from google.genai import types

    assert isinstance(completion, types.GenerateContentResponse)
    assert len(completion.candidates) == 1
    parts = completion.candidates[0].content.parts
    non_thought_parts = [
        part for part in parts if not (hasattr(part, "thought") and part.thought)
    ]
    assert len(non_thought_parts) == 1, (
        "Instructor does not support multiple function calls, use List[Model] instead"
    )
    function_call = non_thought_parts[0].function_call
    assert function_call is not None, "Please return your response as a function call"
    assert function_call.name == gemini_utils._get_model_name(response_model)
    return response_model.model_validate(
        obj=function_call.args,
        context=validation_context,
        strict=strict,
    )


class GenAIHandlerBase(ModeHandler):
    """Common utilities shared across GenAI mode handlers."""

    def __init__(self, mode: Mode | None = None) -> None:
        """Initialize handler with optional mode."""
        self.mode = mode

    def _clone_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        new_kwargs = kwargs.copy()
        generation_config = new_kwargs.get("generation_config")
        if isinstance(generation_config, dict):
            new_kwargs["generation_config"] = generation_config.copy()
        return new_kwargs

    def _pop_autodetect_images(self, kwargs: dict[str, Any]) -> bool:
        return bool(kwargs.pop("autodetect_images", False))

    def _extract_system_instruction(self, kwargs: dict[str, Any]) -> str | None:
        if "system" in kwargs and kwargs["system"] is not None:
            return cast(str, kwargs.pop("system"))
        if "messages" in kwargs:
            return gemini_utils.extract_genai_system_message(
                cast(list[dict[str, Any]], kwargs["messages"])
            )
        return None

    def extract_streaming_json(self, completion: Any) -> Generator[str, None, None]:
        """Extract JSON chunks from GenAI streaming responses."""
        for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                else:
                    yield gemini_utils.extract_gemini_chunk_text(chunk)
            except AttributeError:
                continue

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from GenAI async streams."""
        async for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                else:
                    yield gemini_utils.extract_gemini_chunk_text(chunk)
            except AttributeError:
                continue

    def _wrap_streaming_model(
        self,
        response_model: type[BaseModel] | None,
        stream: bool,
    ) -> type[BaseModel] | None:
        if response_model is None:
            return None
        if (
            stream
            and isinstance(response_model, type)
            and not issubclass(response_model, PartialBase)
        ):
            return Partial[response_model]  # type: ignore[return-value]
        return response_model

    def _convert_messages_to_contents(
        self,
        kwargs: dict[str, Any],
        autodetect_images: bool,
    ) -> dict[str, Any]:
        contents = gemini_utils.convert_to_genai_messages(kwargs.get("messages", []))
        kwargs["contents"] = extract_genai_multimodal_content(
            contents, autodetect_images
        )
        kwargs.pop("messages", None)
        return kwargs

    def _cleanup_provider_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        # Keep 'model' as it's required by GenAI API
        # Remove other OpenAI-specific params that should be in config
        for key in (
            "response_model",
            "generation_config",
            "cached_content",
            "safety_settings",
            "thinking_config",
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "kwargs",  # Remove any nested kwargs key
        ):
            kwargs.pop(key, None)
        return kwargs

    def _prepare_without_response_model(
        self,
        kwargs: dict[str, Any],
        autodetect_images: bool,
    ) -> dict[str, Any]:
        from google.genai import types

        system_instruction = self._extract_system_instruction(kwargs)
        kwargs = self._convert_messages_to_contents(kwargs, autodetect_images)
        user_config = kwargs.get("config")
        if isinstance(user_config, dict):
            config = user_config.copy()
        elif user_config is not None:
            config = user_config.model_dump(exclude_none=True)
        else:
            config = {}
        cached_content = kwargs.pop("cached_content", None)
        if config.get("cached_content") is None and cached_content is not None:
            config["cached_content"] = cached_content
        if config.get("cached_content") is not None:
            for field in ("system_instruction", "tools", "tool_config"):
                config.pop(field, None)
        elif system_instruction:
            config["system_instruction"] = system_instruction
        if config:
            kwargs["config"] = types.GenerateContentConfig(**config)
        return self._cleanup_provider_kwargs(kwargs)

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        raise NotImplementedError

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | None,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel | Any:
        if response_model is None:
            return response

        if (
            stream
            and isinstance(response_model, type)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            if is_async:
                return response_model.from_streaming_response_async(  # type: ignore
                    response,
                    stream_extractor=self.extract_streaming_json_async,
                )
            generator = response_model.from_streaming_response(  # type: ignore
                response,
                stream_extractor=self.extract_streaming_json,
            )
            if issubclass(response_model, IterableBase):
                return generator
            return list(generator)

        # A response truncated at max_tokens is not evaluable: the model never
        # emitted the remaining fields, so parsing it yields schema defaults that
        # are indistinguishable from values the model actually chose.
        # Restores the check shipped in v1.15.0 (#2232), dropped by 60cc815.
        from google.genai import types as genai_types

        candidates = getattr(response, "candidates", None)
        if (
            candidates
            and candidates[0].finish_reason == genai_types.FinishReason.MAX_TOKENS
        ):
            raise IncompleteOutputException(last_completion=response)

        if self.mode == Mode.TOOLS:
            model = parse_genai_tools(
                response_model,
                response,
                validation_context,
                strict,
            )
        else:
            model = parse_genai_structured_outputs(
                response_model,
                response,
                validation_context,
                strict,
            )

        if isinstance(model, IterableBase):
            return list(cast(Any, model).tasks)

        if isinstance(response_model, ParallelBase):
            return model

        if isinstance(model, AdapterBase):
            return cast(Any, model).content

        cast(Any, model)._raw_response = response
        return model

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,  # noqa: ARG002
        exception: Exception,  # noqa: ARG002
    ) -> dict[str, Any]:
        return kwargs.copy()


@register_mode_handler(Provider.GENAI, Mode.TOOLS)
class GenAIToolsHandler(GenAIHandlerBase):
    """Mode handler for GenAI tools/function calling."""

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        from google.genai import types

        new_kwargs = self._clone_kwargs(kwargs)
        autodetect_images = self._pop_autodetect_images(new_kwargs)
        stream = bool(new_kwargs.get("stream", False))

        prepared_model = prepare_response_model(response_model)
        if prepared_model is None:
            return None, self._prepare_without_response_model(
                new_kwargs, autodetect_images
            )

        prepared_model = self._wrap_streaming_model(prepared_model, stream)
        schema = gemini_utils.map_to_genai_schema(
            gemini_utils._get_model_schema(prepared_model)
        )
        function_decl = types.FunctionDeclaration(
            name=gemini_utils._get_model_name(prepared_model),
            description=getattr(prepared_model, "__doc__", None),
            parameters=schema,
        )

        system_instruction = self._extract_system_instruction(new_kwargs)

        # Move OpenAI-style params to generation_config for conversion
        generation_config_dict = new_kwargs.pop("generation_config", {})
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
        ):
            if key in new_kwargs:
                generation_config_dict[key] = new_kwargs.pop(key)

        base_config = {
            "system_instruction": system_instruction,
            "tools": [types.Tool(function_declarations=[function_decl])],
            "tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[
                        gemini_utils._get_model_name(prepared_model)
                    ],
                ),
            ),
        }
        # Temporarily put generation_config back for update_genai_kwargs to process
        new_kwargs["generation_config"] = generation_config_dict
        generation_config = update_genai_kwargs(new_kwargs, base_config)
        new_kwargs.pop("generation_config", None)  # Remove it after processing
        new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
        new_kwargs = self._convert_messages_to_contents(new_kwargs, autodetect_images)
        new_kwargs = self._cleanup_provider_kwargs(new_kwargs)
        return prepared_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_genai_tools(
            kwargs.copy(),
            response,
            exception,
        )


@register_mode_handler(Provider.GENAI, Mode.JSON)
class GenAIStructuredOutputsHandler(GenAIHandlerBase):
    """Mode handler for GenAI structured outputs / JSON schema."""

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        from google.genai import types

        new_kwargs = self._clone_kwargs(kwargs)
        autodetect_images = self._pop_autodetect_images(new_kwargs)
        stream = bool(new_kwargs.get("stream", False))

        prepared_model = prepare_response_model(response_model)
        if prepared_model is None:
            return None, self._prepare_without_response_model(
                new_kwargs, autodetect_images
            )

        prepared_model = self._wrap_streaming_model(prepared_model, stream)
        # Validate schema for unsupported union types
        gemini_utils.map_to_gemini_function_schema(
            gemini_utils._get_model_schema(prepared_model)
        )

        system_instruction = self._extract_system_instruction(new_kwargs)

        # Move OpenAI-style params to generation_config for conversion
        generation_config_dict = new_kwargs.pop("generation_config", {})
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
        ):
            if key in new_kwargs:
                generation_config_dict[key] = new_kwargs.pop(key)

        base_config = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": prepared_model,
        }
        # Temporarily put generation_config back for update_genai_kwargs to process
        new_kwargs["generation_config"] = generation_config_dict
        generation_config = update_genai_kwargs(new_kwargs, base_config)
        new_kwargs.pop("generation_config", None)  # Remove it after processing
        new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
        new_kwargs = self._convert_messages_to_contents(new_kwargs, autodetect_images)
        new_kwargs = self._cleanup_provider_kwargs(new_kwargs)
        return prepared_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_genai_structured_outputs(
            kwargs.copy(),
            response,
            exception,
        )
