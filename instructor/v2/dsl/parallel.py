import sys
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from collections.abc import Generator
from pydantic import BaseModel
from collections.abc import Iterable

from instructor.v2.core.mode import Mode

T = TypeVar("T", bound=BaseModel)


class ParallelBase(Generic[T]):
    def __init__(self, *models: type[T]):
        # Note that for everything else we've created a class, but for parallel base it is an instance
        assert len(models) > 0, "At least one model is required"
        self.models = models
        self.registry: dict[str, type[T]] = {
            self._get_model_name(model): model for model in models
        }

    def _get_model_name(self, model: type[T]) -> str:
        # OpenAI and Anthropic tool declarations use the JSON schema title.
        return model.model_json_schema()["title"]

    def from_response(
        self,
        response: Any,
        mode: Mode,
        validation_context: Optional[Any] = None,
        strict: Optional[bool] = None,
    ) -> Generator[T, None, None]:
        # Validate the complete non-streaming response before it leaves retry.
        from instructor.v2.core.errors import ResponseParsingError

        choices = getattr(response, "choices", None)
        if not choices:
            raise ResponseParsingError(
                "No choices in OpenAI response",
                mode=str(mode.value),
                raw_response=response,
            )

        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            raise ResponseParsingError(
                "No tool calls in response",
                mode=str(mode.value),
                raw_response=response,
            )

        results = []
        for tool_call in tool_calls:
            name = tool_call.function.name
            arguments = tool_call.function.arguments
            model = model_for_tool_name(
                self.registry, name, mode=mode, raw_response=response
            )
            results.append(
                model.model_validate_json(
                    arguments, context=validation_context, strict=strict
                )
            )
        return (result for result in results)


if sys.version_info >= (3, 10):
    from types import UnionType

    def is_union_type(typehint: type[Iterable[T]]) -> bool:
        return get_origin(get_args(typehint)[0]) in (Union, UnionType)

else:

    def is_union_type(typehint: type[Iterable[T]]) -> bool:
        return get_origin(get_args(typehint)[0]) is Union


def model_for_tool_name(
    registry: dict[str, type[T]],
    name: str,
    *,
    mode: Any = None,
    raw_response: Any = None,
) -> type[T]:
    """Return the model for a tool name, or raise if the name was not registered."""
    model = registry.get(name)
    if model is not None:
        return model
    from instructor.v2.core.errors import ResponseParsingError

    expected = ", ".join(sorted(registry))
    mode_value = getattr(mode, "value", mode)
    raise ResponseParsingError(
        f"Unknown tool call {name!r}. Expected one of: {expected}.",
        mode=None if mode_value is None else str(mode_value),
        raw_response=raw_response,
    )


def get_types_array(typehint: type[Iterable[T]]) -> tuple[type[T], ...]:
    should_be_iterable = get_origin(typehint)

    if should_be_iterable is not Iterable:
        raise TypeError(f"Model should be with Iterable instead of {typehint}")

    if is_union_type(typehint):
        # works for Iterable[Union[int, str]], Iterable[int | str]
        return get_args(get_args(typehint)[0])

    # works for Iterable[int]
    return get_args(typehint)


def handle_parallel_model(typehint: type[Iterable[T]]) -> list[dict[str, Any]]:
    # Import at runtime to avoid circular import
    from instructor.v2.core.function_calls import openai_schema

    the_types = get_types_array(typehint)
    return [
        {"type": "function", "function": openai_schema(model).openai_schema}
        for model in the_types
    ]


def handle_anthropic_parallel_model(
    typehint: type[Iterable[T]],
) -> list[dict[str, Any]]:
    """Compatibility shim for Anthropic-owned parallel schema generation."""
    from instructor.v2.providers.anthropic.parallel import handle_parallel_model

    return handle_parallel_model(typehint)


def ParallelModel(typehint: type[Iterable[T]]) -> ParallelBase[T]:
    the_types = get_types_array(typehint)
    return ParallelBase(*[model for model in the_types])


def VertexAIParallelModel(typehint: type[Iterable[T]]) -> ParallelBase[T]:
    """Compatibility shim for the VertexAI-owned parallel model."""
    from instructor.v2.providers.vertexai.parallel import (
        VertexAIParallelModel as factory,
    )

    return factory(typehint)


def AnthropicParallelModel(typehint: type[Iterable[T]]) -> ParallelBase[T]:
    """Compatibility shim for the Anthropic-owned parallel model."""
    from instructor.v2.providers.anthropic.parallel import (
        AnthropicParallelModel as factory,
    )

    return factory(typehint)
