"""VertexAI-specific parallel response helpers."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable
from typing import Any, TypeVar

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.dsl.parallel import (
    ParallelBase,
    get_types_array,
    model_for_tool_name,
)

T = TypeVar("T", bound=BaseModel)


class VertexAIParallelBase(ParallelBase[T]):
    def from_response(
        self,
        response: Any,
        mode: Mode,
        validation_context: Any | None = None,
        strict: bool | None = None,
    ) -> Generator[T, None, None]:
        if not response or not response.candidates:
            return

        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call is not None:
                    name = part.function_call.name
                    arguments = part.function_call.args
                    model = model_for_tool_name(
                        self.registry, name, mode=mode, raw_response=response
                    )
                    json_str = json.dumps(arguments)
                    yield model.model_validate_json(
                        json_str, context=validation_context, strict=strict
                    )


def VertexAIParallelModel(typehint: type[Iterable[T]]) -> VertexAIParallelBase[T]:
    return VertexAIParallelBase(*get_types_array(typehint))
