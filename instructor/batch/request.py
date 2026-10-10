"""
Batch request models and schema utilities.

This module contains the BatchRequest class and related models for creating
provider-specific batch requests with JSON schema generation.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Generic, cast
from pydantic import BaseModel, Field, ConfigDict
import json
import io
from .models import T


class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Tool(BaseModel):
    type: str
    function: Function


class RequestBody(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = Field(default=1000)
    temperature: float | None = Field(default=1.0)
    tools: list[Tool] | None
    tool_choice: dict[str, Any] | None


class BatchModel(BaseModel):
    custom_id: str
    body: RequestBody
    url: str
    method: str


class BatchRequest(BaseModel, Generic[T]):
    """Unified batch request that works across all providers using JSON schema"""

    custom_id: str
    messages: list[dict[str, Any]]
    response_model: type[T]
    model: str
    max_tokens: int | None = Field(default=1000)
    temperature: float | None = Field(default=0.1)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_json_schema(self) -> dict[str, Any]:
        """Generate JSON schema from response_model"""
        return self.response_model.model_json_schema()

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI batch format with JSON schema"""
        from openai.lib._pydantic import resolve_ref

        strict_schema = deepcopy(self.get_json_schema())

        def make_strict_schema(schema: dict[str, Any] | bool) -> None:
            # The SDK strict helper rejects boolean schemas, which batch callers
            # already use. Traverse schema keywords only, leaving metadata intact.
            if isinstance(schema, bool):
                return
            properties = schema.get("properties")
            additional = schema.get("additionalProperties")
            if (
                isinstance(additional, dict)
                or (additional is True and not properties)
                or (
                    schema.get("type") == "object"
                    and "properties" not in schema
                    and additional is None
                )
                or schema.get("patternProperties")
            ):
                raise ValueError(
                    "Arbitrary mapping schemas are not supported by OpenAI batch "
                    "strict mode; use a model with named fields instead"
                )
            schema_type = schema.get("type")
            if (
                schema_type == "object"
                or (isinstance(schema_type, list) and "object" in schema_type)
                or isinstance(properties, dict)
            ):
                schema["additionalProperties"] = False
            if isinstance(properties, dict):
                schema["required"] = list(properties)
                for child in properties.values():
                    make_strict_schema(child)
            for keyword in ("$defs", "definitions"):
                for child in schema.get(keyword, {}).values():
                    make_strict_schema(child)
            if "items" in schema:
                make_strict_schema(schema["items"])
            for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
                for child in schema.get(keyword, []):
                    make_strict_schema(child)
            all_of = schema.get("allOf")
            if (
                isinstance(all_of, list)
                and len(all_of) == 1
                and isinstance(all_of[0], dict)
            ):
                schema.update(all_of[0])
                del schema["allOf"]
            # Match the SDK's nullable-default handling; retain non-null defaults.
            if "default" in schema and schema["default"] is None:
                del schema["default"]
            ref = schema.get("$ref")
            if isinstance(ref, str) and len(schema) > 1:
                resolved = cast(
                    dict[str, Any], resolve_ref(root=strict_schema, ref=ref)
                )
                schema.update({**resolved, **schema})
                del schema["$ref"]
                make_strict_schema(schema)

        make_strict_schema(strict_schema)

        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_model.__name__,
                        "strict": True,
                        "schema": strict_schema,
                    },
                },
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic batch format with JSON schema"""
        from instructor.v2.providers.anthropic.handlers import (
            combine_system_messages,
        )

        schema = self.get_json_schema()

        # Ensure schema has proper format for Anthropic
        if "type" not in schema:
            schema["type"] = "object"
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        # Extract system message and convert to system parameter
        system_message = None
        filtered_messages = []

        for message in self.messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if content:
                    system_message = combine_system_messages(system_message, content)
            else:
                filtered_messages.append(message)

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": filtered_messages,
            "tools": [
                {
                    "name": "extract_data",
                    "description": f"Extract data matching the {self.response_model.__name__} schema",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "extract_data"},
        }

        # Add system parameter if system message exists
        if system_message:
            params["system"] = system_message

        return {
            "custom_id": self.custom_id,
            "params": params,
        }

    def save_to_file(
        self, file_path_or_buffer: str | io.BytesIO, provider: str
    ) -> None:
        """Save batch request to file or BytesIO buffer in provider-specific format"""
        if provider == "openai":
            data = self.to_openai_format()
        elif provider == "anthropic":
            data = self.to_anthropic_format()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        json_line = json.dumps(data) + "\n"

        if isinstance(file_path_or_buffer, str):
            with open(file_path_or_buffer, "a", encoding="utf-8") as f:
                f.write(json_line)
        elif isinstance(file_path_or_buffer, io.BytesIO):
            file_path_or_buffer.write(json_line.encode("utf-8"))
        else:
            raise ValueError(
                f"Unsupported file_path_or_buffer type: {type(file_path_or_buffer)}"
            )
