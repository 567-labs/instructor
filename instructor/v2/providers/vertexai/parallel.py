"""VertexAI-specific parallel response helpers."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable
from typing import Any, TypeVar

from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.dsl.parallel import ParallelBase, get_types_array

T = TypeVar("T", bound=BaseModel)


class VertexAIParallelBase(ParallelBase[T]):
    def _get_model_name(self, model: type[T]) -> str:
        # VertexAI function declarations use the Python class name.
        return model.__name__

    def from_response(
        self,
        response: Any,
        mode: Mode,  # noqa: ARG002
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
                    if name in self.registry:
                        json_str = json.dumps(arguments)
                        yield self.registry[name].model_validate_json(
                            json_str, context=validation_context, strict=strict
                        )


def VertexAIParallelModel(typehint: type[Iterable[T]]) -> VertexAIParallelBase[T]:
    return VertexAIParallelBase(*get_types_array(typehint))
