"""
Batch request models and schema utilities.

This module contains the BatchRequest class and related models for creating
provider-specific batch requests with JSON schema generation.
"""

from __future__ import annotations
from typing import Any, Generic
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

    strict: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_json_schema(self) -> dict[str, Any]:
        """Generate JSON schema from response_model"""
        return self.response_model.model_json_schema()

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI batch format with JSON schema"""
        from instructor.v2.providers.openai.schema import make_strict_schema

        schema = self.get_json_schema()
        if self.strict:
            schema = make_strict_schema(schema)

        if self.model == "gpt-6-luna":
            options: dict[str, Any] = {
                "max_completion_tokens": self.max_tokens,
                "reasoning_effort": "none",
            }
        else:
            options = {"max_tokens": self.max_tokens}
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": self.messages,
                **options,
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_model.__name__,
                        "strict": self.strict,
                        "schema": schema,
                    },
                },
            },
        }

    def to_mistral_format(self) -> dict[str, Any]:
        """Serialize one Mistral request. The job sets the model for all requests."""
        from mistralai.extra import response_format_from_pydantic_model

        response_format = response_format_from_pydantic_model(self.response_model)
        # SDK 1 returns a Pydantic model; SDK 2 returns the wire dictionary.
        if isinstance(response_format, BaseModel):
            response_format = response_format.model_dump(
                by_alias=True, exclude_none=True
            )
        return {
            "custom_id": self.custom_id,
            "body": {
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_format": response_format,
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

        if self.temperature is not None and self.model != "claude-haiku-5-5":
            params["temperature"] = self.temperature

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
        elif provider == "mistral":
            data = self.to_mistral_format()
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
