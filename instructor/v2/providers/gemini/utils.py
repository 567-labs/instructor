"""Google-specific utilities (Gemini, GenAI, VertexAI)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from instructor.v2.dsl.partial import Partial, PartialBase
from instructor.v2.core.errors import ConfigurationError
from instructor.v2.core.multimodal import Audio, Image, PDF
from instructor.v2.core.messages import get_message_content

if TYPE_CHECKING:
    from google.genai.types import Content as GenAIContent
    from google.genai.types import File as GenAIFile
else:
    GenAIContent = Any
    GenAIFile = Any

_OPENAI_TO_GEMINI_MAP = {
    "max_tokens": "max_output_tokens",
    "temperature": "temperature",
    "n": "candidate_count",
    "top_p": "top_p",
    "stop": "stop_sequences",
}


@lru_cache(maxsize=1)
def _default_safety_thresholds() -> dict[Any, Any] | None:
    try:
        from google.genai.types import HarmBlockThreshold, HarmCategory
    except ImportError:
        try:
            from google.generativeai.types import (  # type: ignore
                HarmBlockThreshold,
                HarmCategory,
            )
        except ImportError:
            return None

    return {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }


def _get_model_schema(response_model: Any) -> dict[str, Any]:
    if hasattr(response_model, "model_json_schema") and callable(
        response_model.model_json_schema
    ):
        return response_model.model_json_schema()
    return getattr(response_model, "model_json_schema", {})  # type: ignore[return-value]


def _get_model_name(response_model: Any) -> str:
    return getattr(response_model, "__name__", "Model")


def extract_gemini_chunk_text(chunk: Any) -> str:
    try:
        return chunk.text
    except (AttributeError, ValueError):
        part_text = chunk.candidates[0].content.parts[0].text
        if part_text:
            return part_text
        raise


def transform_to_gemini_prompt(
    messages_chatgpt: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    if not messages_chatgpt:
        return []

    system_prompts: list[str] = []
    for message in messages_chatgpt:
        if message.get("role") == "system":
            content = message.get("content", "")
            if isinstance(content, str) and content:
                system_prompts.append(content)
            elif isinstance(content, list):
                for item in content:
                    item_dict = cast(dict[str, Any], item)
                    text = item_dict.get("text")
                    if item_dict.get("type") == "text" and isinstance(text, str):
                        system_prompts.append(text)

    system_prompt = ""
    if system_prompts:
        system_prompt = "\n\n".join(filter(None, system_prompts))

    messages_gemini: list[dict[str, Any]] = []
    role_map = {
        "user": "user",
        "assistant": "model",
    }

    for message in messages_chatgpt:
        role = message.get("role", "")
        if role in role_map:
            gemini_role = role_map[role]
            messages_gemini.append(
                {"role": gemini_role, "parts": get_message_content(message)}
            )

    if system_prompt:
        if messages_gemini:
            first_message = messages_gemini[0]
            first_message["parts"].insert(0, f"*{system_prompt}*")
        else:
            messages_gemini.append({"role": "user", "parts": [f"*{system_prompt}*"]})

    return messages_gemini


def verify_no_unions(obj: dict[str, Any]) -> bool:  # noqa: ARG001
    return True


def map_to_gemini_function_schema(obj: dict[str, Any]) -> dict[str, Any]:
    try:
        import jsonref
    except ImportError as err:
        raise ConfigurationError(
            "The 'jsonref' package is required for Gemini structured output. "
            "Install it with: pip install 'instructor[google-genai]'"
        ) from err

    class FunctionSchema(BaseModel):
        description: str | None = None
        enum: list[str] | None = None
        example: Any | None = None
        format: str | None = None
        nullable: bool | None = None
        items: FunctionSchema | None = None
        required: list[str] | None = None
        type: str | None = None
        anyOf: list[dict[str, Any]] | None = None
        properties: dict[str, FunctionSchema] | None = None

    schema: dict[str, Any] = jsonref.replace_refs(obj, lazy_load=False)
    schema.pop("$defs", None)

    def transform_schema_node(node: Any) -> Any:
        if isinstance(node, list):
            return [transform_schema_node(item) for item in node]

        if not isinstance(node, dict):
            return node

        transformed = {}

        for key, value in node.items():
            if key == "enum":
                transformed[key] = value
                transformed["format"] = "enum"
            elif key == "anyOf" and isinstance(value, list) and len(value) == 2:
                non_null_items = [
                    item
                    for item in value
                    if not (isinstance(item, dict) and item.get("type") == "null")
                ]

                if len(non_null_items) == 1:
                    actual_type = transform_schema_node(non_null_items[0])
                    transformed.update(actual_type)
                    transformed["nullable"] = True
                else:
                    transformed[key] = transform_schema_node(value)
            else:
                transformed[key] = transform_schema_node(value)

        return transformed

    schema = transform_schema_node(schema)

    if not verify_no_unions(schema):
        raise ValueError(
            "Gemini does not support Union types (except Optional). Please change your function schema"
        )

    return FunctionSchema(**schema).model_dump(exclude_none=True, exclude_unset=True)


if TYPE_CHECKING:
    from google.genai import types as genai_types


def map_to_genai_schema(obj: dict[str, Any]) -> genai_types.Schema:
    from google.genai import types

    schema = map_to_gemini_function_schema(obj)

    def normalize(node: Any) -> Any:
        if isinstance(node, list):
            return [normalize(item) for item in node]

        if not isinstance(node, dict):
            return node

        key_map = {
            "anyOf": "any_of",
            "$ref": "ref",
            "$defs": "defs",
            "maxItems": "max_items",
            "minItems": "min_items",
            "maxLength": "max_length",
            "minLength": "min_length",
            "maxProperties": "max_properties",
            "minProperties": "min_properties",
        }

        normalized: dict[str, Any] = {}
        for key, value in node.items():
            normalized[key_map.get(key, key)] = normalize(value)
        return normalized

    return types.Schema.model_validate(normalize(schema))


def update_genai_kwargs(
    kwargs: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    """Compatibility wrapper for the GenAI-owned request policy."""
    from instructor.v2.providers.genai.request import (
        update_genai_kwargs as update_genai_kwargs_impl,
    )

    return update_genai_kwargs_impl(kwargs, base_config)


def update_gemini_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    result = kwargs.copy()

    if "generation_config" in result:
        gen_config = result["generation_config"].copy()
        result["generation_config"] = gen_config

        for openai_key, gemini_key in _OPENAI_TO_GEMINI_MAP.items():
            if openai_key in gen_config:
                val = gen_config.pop(openai_key)
                if val is not None:
                    gen_config[gemini_key] = val

    if "messages" in result:
        result["contents"] = transform_to_gemini_prompt(result.pop("messages"))

    default_safety_thresholds = _default_safety_thresholds()
    if default_safety_thresholds is None:
        result.setdefault("safety_settings", {})
        return result

    safety_settings = result.get("safety_settings", {}).copy()
    result["safety_settings"] = safety_settings

    for category, threshold in default_safety_thresholds.items():
        current = safety_settings.get(category)
        if current is None or current > threshold:
            safety_settings[category] = threshold

    return result


def extract_genai_system_message(
    messages: list[dict[str, Any]],
) -> str:
    system_messages = ""

    for message in messages:
        if isinstance(message, str):
            continue
        if isinstance(message, dict):
            if message.get("role") == "system":
                if isinstance(message.get("content"), str):
                    system_messages += message.get("content", "") + "\n\n"
                elif isinstance(message.get("content"), list):
                    for item in message.get("content", []):
                        if isinstance(item, str):
                            system_messages += item + "\n\n"

    if system_messages and len(messages) == 1:
        raise ValueError(
            "At least one user message must be included. A system message alone is not sufficient."
        )

    if re.search(r"{{.*?}}|{%.*?%}", system_messages):
        raise ValueError(
            "Jinja templating is not supported in system messages with Google GenAI, only user messages."
        )

    return system_messages


def convert_to_genai_messages(
    messages: list[
        str | dict[str, Any] | list[dict[str, Any]] | GenAIContent | GenAIFile
    ],
) -> list[Any]:
    from google.genai import types

    result: list[types.Content | types.File] = []

    for message in messages:
        if isinstance(message, str):
            result.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
            )
        elif isinstance(message, types.Content):
            result.append(message)
        elif isinstance(message, types.File):
            result.append(message)
        elif isinstance(message, dict):
            assert "role" in message
            assert "content" in message

            if message["role"] == "system":
                continue

            if message["role"] not in {"user", "model"}:
                raise ValueError(f"Unsupported role: {message['role']}")

            if isinstance(message["content"], str):
                result.append(
                    types.Content(
                        role=message["role"],
                        parts=[types.Part.from_text(text=message["content"])],
                    )
                )

            elif isinstance(message["content"], list):
                content_parts = []

                for content_item in message["content"]:
                    if isinstance(content_item, str):
                        content_parts.append(types.Part.from_text(text=content_item))
                    elif isinstance(content_item, (Image, Audio, PDF)):
                        content_parts.append(content_item.to_genai())
                    else:
                        raise ValueError(
                            f"Unsupported content item type: {type(content_item)}"
                        )

                result.append(
                    types.Content(
                        role=message["role"],
                        parts=content_parts,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}"
                )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    return result


def handle_genai_message_conversion(
    new_kwargs: dict[str, Any], autodetect_images: bool = False
) -> dict[str, Any]:
    from google.genai import types

    messages = new_kwargs.get("messages", [])

    new_kwargs["contents"] = convert_to_genai_messages(messages)

    from instructor.v2.core.multimodal import extract_genai_multimodal_content

    new_kwargs["contents"] = extract_genai_multimodal_content(
        new_kwargs["contents"], autodetect_images
    )

    base_config: dict[str, Any] = {}
    system_message = new_kwargs.pop("system", None)
    if system_message is None:
        system_message = extract_genai_system_message(messages)
    if system_message:
        base_config["system_instruction"] = system_message

    # Fold `generation_config`, `safety_settings` and `thinking_config` into `config` the same
    # way the structured-output path does. Otherwise these would be passed as raw keyword
    # arguments to `generate_content`, which raises e.g. "got an unexpected keyword argument
    # 'generation_config'" when `response_model=None` (see #2366).
    generation_config = update_genai_kwargs(new_kwargs, base_config)
    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)

    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)
    new_kwargs.pop("thinking_config", None)

    return new_kwargs


def handle_gemini_json(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    if "model" in new_kwargs:
        raise ConfigurationError(
            "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
        )

    if response_model is None:
        new_kwargs = update_gemini_kwargs(new_kwargs)
        return None, new_kwargs

    message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(_get_model_schema(response_model), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    messages = new_kwargs.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        new_kwargs.setdefault("messages", []).insert(
            0, {"role": "system", "content": message}
        )
    else:
        messages[0]["content"] += f"\n\n{message}"

    new_kwargs["generation_config"] = new_kwargs.get("generation_config", {}) | {
        "response_mime_type": "application/json"
    }

    new_kwargs = update_gemini_kwargs(new_kwargs)
    return response_model, new_kwargs


def handle_gemini_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    if "model" in new_kwargs:
        raise ConfigurationError(
            "Gemini `model` must be set while patching the client, not passed as a parameter to the create method"
        )

    if response_model is None:
        new_kwargs = update_gemini_kwargs(new_kwargs)
        return None, new_kwargs

    new_kwargs["tools"] = [response_model.gemini_schema]
    new_kwargs["tool_config"] = {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [_get_model_name(response_model)],
        },
    }

    new_kwargs = update_gemini_kwargs(new_kwargs)
    return response_model, new_kwargs


def handle_genai_structured_outputs(
    response_model: type[Any] | None,
    new_kwargs: dict[str, Any],
    autodetect_images: bool = False,
) -> tuple[type[Any] | None, dict[str, Any]]:
    from google.genai import types

    if response_model is None:
        new_kwargs = handle_genai_message_conversion(new_kwargs, autodetect_images)
        return None, new_kwargs

    if new_kwargs.get("stream", False) and not issubclass(response_model, PartialBase):
        response_model = Partial[response_model]

    user_config = new_kwargs.get("config")
    user_thinking_config = None
    user_cached_content = None
    if isinstance(user_config, dict):
        user_thinking_config = user_config.get("thinking_config")
        user_cached_content = user_config.get("cached_content")
    elif user_config is not None:
        if hasattr(user_config, "thinking_config"):
            user_thinking_config = user_config.thinking_config
        if hasattr(user_config, "cached_content"):
            user_cached_content = user_config.cached_content

    if "thinking_config" not in new_kwargs and user_thinking_config is not None:
        new_kwargs["thinking_config"] = user_thinking_config

    if new_kwargs.get("system"):
        system_message = new_kwargs.pop("system")
    elif new_kwargs.get("messages"):
        system_message = extract_genai_system_message(new_kwargs["messages"])
    else:
        system_message = None

    new_kwargs["contents"] = convert_to_genai_messages(new_kwargs["messages"])

    from instructor.v2.core.multimodal import extract_genai_multimodal_content

    new_kwargs["contents"] = extract_genai_multimodal_content(
        new_kwargs["contents"], autodetect_images
    )

    map_to_gemini_function_schema(_get_model_schema(response_model))

    base_config: dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": response_model,
    }

    if user_cached_content is None:
        base_config["system_instruction"] = system_message

    generation_config = update_genai_kwargs(new_kwargs, base_config)

    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)
    new_kwargs.pop("response_model", None)
    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)
    new_kwargs.pop("thinking_config", None)

    return response_model, new_kwargs


def handle_genai_tools(
    response_model: type[Any] | None,
    new_kwargs: dict[str, Any],
    autodetect_images: bool = False,
) -> tuple[type[Any] | None, dict[str, Any]]:
    from google.genai import types

    if response_model is None:
        new_kwargs = handle_genai_message_conversion(new_kwargs, autodetect_images)
        return None, new_kwargs

    if new_kwargs.get("stream", False) and not issubclass(response_model, PartialBase):
        response_model = Partial[response_model]

    user_config = new_kwargs.get("config")
    user_thinking_config = None
    user_cached_content = None
    if isinstance(user_config, dict):
        user_thinking_config = user_config.get("thinking_config")
        user_cached_content = user_config.get("cached_content")
    elif user_config is not None:
        if hasattr(user_config, "thinking_config"):
            user_thinking_config = user_config.thinking_config
        if hasattr(user_config, "cached_content"):
            user_cached_content = user_config.cached_content

    if "thinking_config" not in new_kwargs and user_thinking_config is not None:
        new_kwargs["thinking_config"] = user_thinking_config

    schema = map_to_genai_schema(_get_model_schema(response_model))
    function_definition = types.FunctionDeclaration(
        name=_get_model_name(response_model),
        description=getattr(response_model, "__doc__", None),
        parameters=schema,
    )

    if new_kwargs.get("system"):
        system_message = new_kwargs.pop("system")
    elif new_kwargs.get("messages"):
        system_message = extract_genai_system_message(new_kwargs["messages"])
    else:
        system_message = None

    base_config: dict[str, Any] = {}

    if user_cached_content is None:
        base_config["system_instruction"] = system_message
        base_config["tools"] = [types.Tool(function_declarations=[function_definition])]
        base_config["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
                allowed_function_names=[_get_model_name(response_model)],
            )
        )

    new_kwargs["contents"] = convert_to_genai_messages(new_kwargs["messages"])

    from instructor.v2.core.multimodal import extract_genai_multimodal_content

    new_kwargs["contents"] = extract_genai_multimodal_content(
        new_kwargs["contents"], autodetect_images
    )

    generation_config = update_genai_kwargs(new_kwargs, base_config)

    new_kwargs["config"] = types.GenerateContentConfig(**generation_config)

    new_kwargs.pop("response_model", None)
    new_kwargs.pop("messages", None)
    new_kwargs.pop("generation_config", None)
    new_kwargs.pop("safety_settings", None)
    new_kwargs.pop("thinking_config", None)

    return response_model, new_kwargs


def handle_vertexai_parallel_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    from typing import get_args

    from instructor.v2.providers.vertexai.parallel import VertexAIParallelModel
    from instructor.v2.providers.vertexai.handlers import vertexai_process_response

    if new_kwargs.get("stream", False):
        raise ConfigurationError(
            "stream=True is not supported when using VERTEXAI_PARALLEL_TOOLS mode"
        )

    model_types = list(get_args(response_model))
    contents, tools, tool_config = vertexai_process_response(new_kwargs, model_types)
    new_kwargs["contents"] = contents
    new_kwargs["tools"] = tools
    new_kwargs["tool_config"] = tool_config

    return VertexAIParallelModel(typehint=response_model), new_kwargs


def handle_vertexai_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    from instructor.v2.providers.vertexai.handlers import vertexai_process_response

    if response_model is None:
        return None, new_kwargs

    contents, tools, tool_config = vertexai_process_response(new_kwargs, response_model)

    new_kwargs["contents"] = contents
    new_kwargs["tools"] = tools
    new_kwargs["tool_config"] = tool_config
    return response_model, new_kwargs


def handle_vertexai_json(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    from instructor.v2.providers.vertexai.handlers import (
        vertexai_process_json_response,
    )

    if response_model is None:
        return None, new_kwargs

    contents, generation_config = vertexai_process_json_response(
        new_kwargs, response_model
    )

    new_kwargs["contents"] = contents
    new_kwargs["generation_config"] = generation_config
    return response_model, new_kwargs


def reask_genai_tools(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility shim for the GenAI-owned reask helper."""
    from instructor.v2.providers.genai.handlers import reask_genai_tools as impl

    return impl(*args, **kwargs)
