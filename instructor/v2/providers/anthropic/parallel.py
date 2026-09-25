"""Anthropic-specific parallel response helpers."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable
from typing import Any, TypeVar

from pydantic import BaseModel

from instructor.v2.core.function_calls import openai_schema
from instructor.v2.core.mode import Mode
from instructor.v2.dsl.parallel import (
    ParallelBase,
    get_types_array,
    model_for_tool_name,
)

T = TypeVar("T", bound=BaseModel)


def handle_parallel_model(typehint: type[Iterable[T]]) -> list[dict[str, Any]]:
    """Build Anthropic tool schemas for a parallel model."""
    return [
        openai_schema(model).anthropic_schema for model in get_types_array(typehint)
    ]


class AnthropicParallelBase(ParallelBase[T]):
    def from_response(
        self,
        response: Any,
        mode: Mode,
        validation_context: Any | None = None,
        strict: bool | None = None,
    ) -> Generator[T, None, None]:
        if not response or not hasattr(response, "content"):
            return

        for content in response.content:
            if getattr(content, "type", None) == "tool_use":
                name = content.name
                arguments = content.input
                model = model_for_tool_name(
                    self.registry, name, mode=mode, raw_response=response
                )
                json_str = json.dumps(arguments)
                yield model.model_validate_json(
                    json_str, context=validation_context, strict=strict
                )


def AnthropicParallelModel(typehint: type[Iterable[T]]) -> AnthropicParallelBase[T]:
    return AnthropicParallelBase(*get_types_array(typehint))
