"""Bedrock v2 mode handlers."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
import mimetypes
from textwrap import dedent
from typing import Any, cast

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.errors import ConfigurationError, ResponseParsingError
from instructor.v2.core.remote import MAX_IMAGE_BYTES, fetch_remote_content
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.json import extract_json_from_codeblock


def _prepare_bedrock_strict_schema(response_model: type[Any]) -> dict[str, Any]:
    """Return a Bedrock-compatible schema without mutating model-owned data."""
    schema = deepcopy(response_model.model_json_schema())
    unsupported_constraints = {
        "minimum",
        "maximum",
        "multipleOf",
        "minLength",
        "maxLength",
    }
    schema_map_keywords = {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
    schema_keywords = {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
    schema_list_keywords = {"allOf", "anyOf", "oneOf", "prefixItems"}

    def normalize(value: Any, path: str) -> None:
        if not isinstance(value, dict):
            return

        if unsupported := unsupported_constraints.intersection(value):
            constraint = sorted(unsupported)[0]
            raise ConfigurationError(
                "Bedrock native structured outputs do not support "
                f"the `{constraint}` JSON Schema constraint at {path}."
            )

        min_items = value.get("minItems")
        if min_items not in (None, 0, 1):
            raise ConfigurationError(
                "Bedrock native structured outputs only support `minItems` "
                f"values of 0 or 1 at {path}; got {min_items}."
            )

        additional_properties = value.get("additionalProperties")
        if additional_properties is not None and additional_properties is not False:
            raise ConfigurationError(
                "Bedrock native structured outputs do not support free-form "
                f"mappings at {path}; `additionalProperties` must be false."
            )
        if value.get("type") == "object" or "properties" in value:
            value["additionalProperties"] = False

        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in schema_map_keywords and isinstance(item, dict):
                for name, child_schema in item.items():
                    normalize(child_schema, f"{child_path}.{name}")
            elif key in schema_keywords or key in schema_list_keywords:
                child_schemas = item if isinstance(item, list) else [item]
                for index, child_schema in enumerate(child_schemas):
                    suffix = f"[{index}]" if isinstance(item, list) else ""
                    normalize(child_schema, f"{child_path}{suffix}")

    normalize(schema, "$")
    return schema


def generate_bedrock_schema(
    response_model: type[Any], *, strict: bool = False
) -> dict[str, Any]:
    """Generate Bedrock tool schema from a Pydantic model."""
    schema = (
        _prepare_bedrock_strict_schema(response_model)
        if strict
        else response_model.model_json_schema()
    )
    tool_spec: dict[str, Any] = {
        "name": response_model.__name__,
        "description": response_model.__doc__
        or f"Correctly extracted `{response_model.__name__}` with all the required parameters with correct types",
        "inputSchema": {"json": schema},
    }
    if strict:
        tool_spec["strict"] = True

    return {"toolSpec": tool_spec}


def reask_bedrock_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
) -> dict[str, Any]:
    """Handle reask for Bedrock JSON mode when validation fails."""
    new_kwargs = kwargs.copy()
    new_kwargs["messages"] = list(kwargs.get("messages", []))
    reask_msgs = [response["output"]["message"]]
    reask_msgs.append(
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "Correct your JSON ONLY RESPONSE, based on the following errors:\n"
                        f"{exception}"
                    )
                },
            ],
        }
    )
    new_kwargs["messages"].extend(reask_msgs)
    return new_kwargs


def reask_bedrock_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
) -> dict[str, Any]:
    """Handle reask for Bedrock tools mode when validation fails."""
    new_kwargs = kwargs.copy()
    new_kwargs["messages"] = list(kwargs.get("messages", []))

    assistant_message = response["output"]["message"]
    reask_msgs = [assistant_message]

    tool_use_id = None
    if "content" in assistant_message:
        for content_block in assistant_message["content"]:
            if "toolUse" in content_block:
                tool_use_id = content_block["toolUse"]["toolUseId"]
                break

    if tool_use_id:
        reask_msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [
                                {
                                    "text": (
                                        "Validation Error found:\n"
                                        f"{exception}\n"
                                        "Recall the function correctly, fix the errors"
                                    )
                                }
                            ],
                            "status": "error",
                        }
                    }
                ],
            }
        )
    else:
        reask_msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Validation Error due to no tool invocation:\n"
                            f"{exception}\nRecall the function correctly, fix the errors"
                        )
                    }
                ],
            }
        )

    new_kwargs["messages"].extend(reask_msgs)
    return new_kwargs


def _normalize_bedrock_image_format(mime_or_ext: str) -> str:
    """Map common image types to Bedrock format enum."""
    if not mime_or_ext:
        return "jpeg"
    val = mime_or_ext.strip().lower()
    if "/" in val:
        val = val.split("/", 1)[1]
    if val in ("jpg", "pjpeg", "x-jpeg", "x-jpg"):
        return "jpeg"
    if val in ("png", "x-png"):
        return "png"
    if val in ("gif", "x-gif"):
        return "gif"
    if val in ("webp", "image/webp"):
        return "webp"
    return "jpeg"


def _openai_image_part_to_bedrock(part: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI-style image parts to Bedrock content."""
    image_url = (part.get("image_url") or {}).get("url")
    if not image_url:
        raise ValueError("image_url.url is required for OpenAI-style image parts")

    if image_url.startswith("data:"):
        try:
            header, b64 = image_url.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid data URL in image_url.url") from exc
        if ";base64" not in header:
            raise ValueError("Only base64 data URLs are supported for Bedrock")
        meta = header[5:]
        mime = meta.split(";", 1)[0]
        if not mime or "/" not in mime:
            guessed = None
            for token in meta.split(";")[1:]:
                if token.startswith("name="):
                    name = token[len("name=") :].strip().strip('"')
                    guessed = mimetypes.guess_type(name)[0]
                    if guessed:
                        break
            mime = guessed or "image/jpeg"
        fmt = _normalize_bedrock_image_format(mime)
        return {"image": {"format": fmt, "source": {"bytes": base64.b64decode(b64)}}}

    if image_url.startswith("https://"):
        response = fetch_remote_content(
            image_url,
            max_bytes=MAX_IMAGE_BYTES,
            timeout=30,
        )
        mime = response.content_type or mimetypes.guess_type(image_url)[0]
        fmt = _normalize_bedrock_image_format(mime or "")
        return {"image": {"format": fmt, "source": {"bytes": response.content}}}

    raise ValueError(
        "Unsupported image_url scheme for Bedrock. "
        "Use data:image/...;base64,... or pass Bedrock-native image bytes."
    )


def _to_bedrock_content_items(content: Any) -> list[dict[str, Any]]:
    """Normalize content into Bedrock Converse content list."""
    if isinstance(content, str):
        return [{"text": content}]

    if isinstance(content, list):
        items: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and "type" in part:
                part_type = part.get("type")
                if part_type in ("text", "input_text"):
                    txt = part.get("text") or part.get("input_text") or ""
                    items.append({"text": txt})
                    continue
                if part_type == "image_url":
                    items.append(_openai_image_part_to_bedrock(part))
                    continue
                raise ValueError(
                    f"Unsupported OpenAI-style part type for Bedrock: {part_type}"
                )

            if isinstance(part, dict):
                if (
                    "text" in part
                    and isinstance(part["text"], str)
                    and set(part.keys()) == {"text"}
                ):
                    items.append(part)
                    continue
                if "image" in part and isinstance(part["image"], dict):
                    items.append(part)
                    continue
                if "document" in part and isinstance(part["document"], dict):
                    items.append(part)
                    continue
                if "cachePoint" in part:
                    items.append(part)
                    continue
                raise ValueError(f"Unsupported dict content for Bedrock: {part}")

            if isinstance(part, str):
                items.append({"text": part})
                continue

            raise ValueError(f"Unsupported content part for Bedrock: {type(part)}")
        return items

    raise ValueError(f"Unsupported message content type for Bedrock: {type(content)}")


def _prepare_bedrock_converse_kwargs_internal(
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Prepare kwargs for the Bedrock Converse API."""
    if "system" in call_kwargs and isinstance(call_kwargs["system"], list):
        system_content = call_kwargs.pop("system")
        if (
            system_content
            and isinstance(system_content[0], dict)
            and "text" in system_content[0]
        ):
            system_text = system_content[0]["text"]
            messages = list(call_kwargs.get("messages", []))
            messages.insert(0, {"role": "system", "content": system_text})
            call_kwargs["messages"] = messages

    if "model" in call_kwargs and "modelId" not in call_kwargs:
        call_kwargs["modelId"] = call_kwargs.pop("model")

    inference_config_params = {}
    if "temperature" in call_kwargs:
        inference_config_params["temperature"] = call_kwargs.pop("temperature")

    if "max_tokens" in call_kwargs:
        inference_config_params["maxTokens"] = call_kwargs.pop("max_tokens")
    elif "maxTokens" in call_kwargs:
        inference_config_params["maxTokens"] = call_kwargs.pop("maxTokens")

    if "top_p" in call_kwargs:
        inference_config_params["topP"] = call_kwargs.pop("top_p")
    elif "topP" in call_kwargs:
        inference_config_params["topP"] = call_kwargs.pop("topP")

    if "stop" in call_kwargs:
        stop_val = call_kwargs.pop("stop")
        if isinstance(stop_val, str):
            inference_config_params["stopSequences"] = [stop_val]
        elif isinstance(stop_val, list):
            inference_config_params["stopSequences"] = stop_val
    elif "stop_sequences" in call_kwargs:
        inference_config_params["stopSequences"] = call_kwargs.pop("stop_sequences")
    elif "stopSequences" in call_kwargs:
        inference_config_params["stopSequences"] = call_kwargs.pop("stopSequences")

    # top_k is not part of the Bedrock InferenceConfiguration base set
    # (maxTokens/stopSequences/temperature/topP). Model-specific parameters
    # like top_k must be routed through additionalModelRequestFields, otherwise
    # the leftover kwarg reaches client.converse() and boto3 raises
    # ParamValidationError: Unknown parameter "top_k".
    additional_model_request_fields = {}
    if "top_k" in call_kwargs:
        additional_model_request_fields["top_k"] = call_kwargs.pop("top_k")
    elif "topK" in call_kwargs:
        additional_model_request_fields["top_k"] = call_kwargs.pop("topK")

    if additional_model_request_fields:
        if "additionalModelRequestFields" in call_kwargs:
            existing_additional_fields = call_kwargs[
                "additionalModelRequestFields"
            ].copy()
            for key, value in additional_model_request_fields.items():
                if key not in existing_additional_fields:
                    existing_additional_fields[key] = value
            call_kwargs["additionalModelRequestFields"] = existing_additional_fields
        else:
            call_kwargs["additionalModelRequestFields"] = (
                additional_model_request_fields
            )

    if inference_config_params:
        if "inferenceConfig" in call_kwargs:
            existing_inference_config = call_kwargs["inferenceConfig"].copy()
            for key, value in inference_config_params.items():
                if key not in existing_inference_config:
                    existing_inference_config[key] = value
            call_kwargs["inferenceConfig"] = existing_inference_config
        else:
            call_kwargs["inferenceConfig"] = inference_config_params

    if "messages" in call_kwargs and isinstance(call_kwargs["messages"], list):
        original_input_messages = call_kwargs.pop("messages")
        bedrock_system_list: list[dict[str, Any]] = []
        bedrock_user_assistant_messages_list: list[dict[str, Any]] = []

        for msg_dict in original_input_messages:
            if not isinstance(msg_dict, dict):
                bedrock_user_assistant_messages_list.append(msg_dict)
                continue

            current_message_for_api = msg_dict.copy()
            role = current_message_for_api.get("role")
            content = current_message_for_api.get("content")

            if role == "system":
                if isinstance(content, str):
                    bedrock_system_list.append({"text": content})
                else:
                    raise ValueError(
                        "System message content must be a string for Bedrock processing by this handler. "
                        f"Found type: {type(content)}."
                    )
            else:
                if "content" in current_message_for_api:
                    if isinstance(content, list):
                        for part in content:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "image_url"
                            ):
                                image_url = (part.get("image_url") or {}).get("url", "")
                                if image_url.startswith(("http://", "https://")):
                                    raise ValueError(
                                        "Unsupported image_url scheme for Bedrock. "
                                        "Use data:image/...;base64,... or pass Bedrock-native image bytes."
                                    )
                    current_message_for_api["content"] = _to_bedrock_content_items(
                        content
                    )
                bedrock_user_assistant_messages_list.append(current_message_for_api)

        if bedrock_system_list:
            call_kwargs["system"] = bedrock_system_list

        call_kwargs["messages"] = bedrock_user_assistant_messages_list
    return call_kwargs


def handle_bedrock_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle Bedrock JSON mode."""
    new_kwargs = _prepare_bedrock_converse_kwargs_internal(new_kwargs)
    json_message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        and don't include any other text in the response apart from the json
        """
    )
    system_message = new_kwargs.pop("system", None)
    if not system_message:
        new_kwargs["system"] = [{"text": json_message}]
    else:
        if not isinstance(system_message, list):
            raise ValueError(
                """system must be a list of SystemMessage, refer to:
                https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
                """
            )
        system_message.append({"text": json_message})
        new_kwargs["system"] = system_message

    return response_model, new_kwargs


def handle_bedrock_tools(
    response_model: type[Any] | None,
    new_kwargs: dict[str, Any],
    *,
    strict: bool = False,
) -> tuple[type[Any] | None, dict[str, Any]]:
    """Handle Bedrock tools mode."""
    new_kwargs = _prepare_bedrock_converse_kwargs_internal(new_kwargs)

    if response_model is None:
        return None, new_kwargs

    tool_schema = generate_bedrock_schema(response_model, strict=strict)
    new_kwargs["toolConfig"] = {
        "tools": [tool_schema],
        "toolChoice": {"tool": {"name": response_model.__name__}},
    }

    return response_model, new_kwargs


def handle_bedrock_json_schema(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """Handle native Bedrock JSON schema constrained decoding."""
    new_kwargs = _prepare_bedrock_converse_kwargs_internal(new_kwargs)

    if response_model is None:
        return None, new_kwargs

    schema = _prepare_bedrock_strict_schema(response_model)
    schema_name = response_model.__name__
    new_kwargs["outputConfig"] = {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "schema": json.dumps(schema),
                    "name": schema_name,
                    "description": response_model.__doc__
                    or f"Correctly extracted `{schema_name}` with all the required parameters with correct types",
                }
            },
        }
    }
    return response_model, new_kwargs


def _extract_bedrock_text(response: Any) -> str:
    """Extract text from Bedrock response formats."""
    if isinstance(response, dict):
        content = response.get("output", {}).get("message", {}).get("content", [])
        text_block = next((block for block in content if "text" in block), None)
        if not text_block:
            raise ResponseParsingError(
                "Unexpected Bedrock response format: No text content found.",
                mode="BEDROCK_JSON",
                raw_response=response,
            )
        return text_block["text"]
    if hasattr(response, "text"):
        return response.text
    raise ResponseParsingError(
        "Unexpected Bedrock response format: no text attribute found.",
        mode="BEDROCK_JSON",
        raw_response=response,
    )


def _extract_bedrock_tool_input(
    response: Any, response_model: type[BaseModel]
) -> dict[str, Any]:
    """Extract tool input from Bedrock tool-use responses."""
    if not isinstance(response, dict):
        raise ResponseParsingError(
            "Unexpected Bedrock response format: expected dict response.",
            mode="BEDROCK_TOOLS",
            raw_response=response,
        )

    message = response.get("output", {}).get("message", {})
    content = message.get("content", [])
    for content_block in content:
        if "toolUse" in content_block:
            tool_use = content_block["toolUse"]
            if tool_use.get("name") != response_model.__name__:
                raise ResponseParsingError(
                    f"Tool name mismatch: expected {response_model.__name__}, "
                    f"got {tool_use.get('name')}",
                    mode="BEDROCK_TOOLS",
                    raw_response=response,
                )
            return tool_use.get("input", {})

    raise ResponseParsingError(
        "No tool use found in Bedrock response.",
        mode="BEDROCK_TOOLS",
        raw_response=response,
    )


@register_mode_handler(Provider.BEDROCK, Mode.TOOLS)
class BedrockToolsHandler(ModeHandler):
    """Handler for Bedrock TOOLS mode."""

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        if response_model is None:
            return handle_bedrock_tools(None, new_kwargs)

        prepared_model = cast(type[BaseModel], prepare_response_model(response_model))
        return handle_bedrock_tools(prepared_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_bedrock_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        if stream:
            raise ConfigurationError(
                "Streaming is not supported for Bedrock in TOOLS mode."
            )
        tool_input = _extract_bedrock_tool_input(response, response_model)
        return response_model.model_validate(
            tool_input,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.BEDROCK, Mode.TOOLS_STRICT)
class BedrockToolsStrictHandler(BedrockToolsHandler):
    """Handler for Bedrock tool use with native schema enforcement."""

    mode = Mode.TOOLS_STRICT

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        if response_model is None:
            return handle_bedrock_tools(None, new_kwargs, strict=True)

        prepared_model = cast(type[BaseModel], prepare_response_model(response_model))
        return handle_bedrock_tools(prepared_model, new_kwargs, strict=True)


@register_mode_handler(Provider.BEDROCK, Mode.MD_JSON)
class BedrockMDJSONHandler(ModeHandler):
    """Handler for Bedrock MD_JSON mode."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        if response_model is None:
            return None, _prepare_bedrock_converse_kwargs_internal(new_kwargs)

        prepared_model = cast(type[BaseModel], prepare_response_model(response_model))
        return handle_bedrock_json(prepared_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_bedrock_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        if stream:
            raise ConfigurationError(
                "Streaming is not supported for Bedrock in MD_JSON mode."
            )
        text = _extract_bedrock_text(response)
        text = extract_json_from_codeblock(text)
        return response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.BEDROCK, Mode.JSON_SCHEMA)
class BedrockJSONSchemaHandler(ModeHandler):
    """Handler for Bedrock native JSON schema constrained decoding."""

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        if response_model is None:
            return handle_bedrock_json_schema(None, new_kwargs)

        prepared_model = cast(type[BaseModel], prepare_response_model(response_model))
        return handle_bedrock_json_schema(prepared_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_bedrock_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        if stream:
            raise ConfigurationError(
                "Streaming is not supported for Bedrock in JSON_SCHEMA mode."
            )
        return response_model.model_validate_json(
            _extract_bedrock_text(response),
            context=validation_context,
            strict=strict,
        )


__all__ = [
    "BedrockToolsHandler",
    "BedrockToolsStrictHandler",
    "BedrockMDJSONHandler",
    "BedrockJSONSchemaHandler",
]
