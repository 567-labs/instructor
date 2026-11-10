"""Non-streaming dispatch keeps the requested provider and legacy parser hooks."""

from collections.abc import Iterable
from typing import Any, cast

import pytest
from anthropic.types import Message
from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

from instructor import Mode, Provider
from instructor.v2.core.function_calls import ResponseSchema
from instructor.processing.response import process_response, process_response_async
from instructor.v2.core.response_model import prepare_response_model
from instructor.v2.dsl.iterable import IterableBase
from instructor.v2.dsl.response_list import ListResponse


class Item(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def apply_context(cls, value: int, info: ValidationInfo) -> int:
        return value + (info.context or {}).get("offset", 0)


def completion(model: type[BaseModel], mode: Mode, tasks: list[Any]) -> Message:
    import json

    content: list[Any] = (
        [{"type": "text", "text": json.dumps({"tasks": tasks})}]
        if mode in (Mode.JSON, Mode.ANTHROPIC_JSON)
        else [
            {
                "type": "tool_use",
                "id": "tool_local",
                "name": model.__name__,
                "input": {"tasks": tasks},
            }
        ]
    )
    return Message(
        id="msg_local",
        type="message",
        role="assistant",
        model="test-model",
        content=content,
        stop_reason="end_turn",
        stop_sequence=None,
        usage={"input_tokens": 1, "output_tokens": 1},
    )


@pytest.mark.parametrize(
    ("mode", "provider"),
    [
        (Mode.JSON, Provider.ANTHROPIC),
        (Mode.TOOLS, Provider.ANTHROPIC),
        (Mode.ANTHROPIC_JSON, Provider.OPENAI),
        (Mode.ANTHROPIC_TOOLS, Provider.OPENAI),
    ],
)
@pytest.mark.parametrize("scalar", [False, True], ids=["models", "scalars"])
@pytest.mark.asyncio
async def test_nonstreaming_iterable_uses_requested_provider(
    mode: Mode, provider: Provider, scalar: bool
) -> None:
    annotation = Iterable[int] if scalar else list[Item]
    model = cast(Any, prepare_response_model(annotation))
    tasks = [{"content": 0}, {"content": 3}] if scalar else [{"value": 0}, {"value": 3}]
    expected = [0, 3] if scalar else [Item(value=10), Item(value=13)]
    raw = cast(Any, completion(model, mode, tasks))
    kwargs: dict[str, Any] = dict(
        response_model=model,
        stream=False,
        mode=mode,
        provider=provider,
        validation_context={"offset": 10},
        strict=True,
    )

    sync_result = process_response(raw, **kwargs)
    async_result = await process_response_async(raw, **kwargs)
    for result in (sync_result, async_result):
        assert isinstance(result, ListResponse)
        assert result == expected
        assert result.get_raw_response() is raw


@pytest.mark.parametrize("mode", [Mode.JSON, Mode.TOOLS])
@pytest.mark.asyncio
async def test_nonstreaming_iterable_preserves_strict_validation(mode: Mode) -> None:
    model = cast(Any, prepare_response_model(list[Item]))
    raw = cast(Any, completion(model, mode, [{"value": "3"}]))
    kwargs: dict[str, Any] = dict(
        response_model=model,
        stream=False,
        mode=mode,
        provider=Provider.ANTHROPIC,
        strict=True,
    )
    with pytest.raises(ValidationError):
        process_response(raw, **kwargs)
    with pytest.raises(ValidationError):
        await process_response_async(raw, **kwargs)


@pytest.mark.asyncio
async def test_inherited_custom_from_response_still_runs_without_provider_keyword() -> (
    None
):
    class CustomEnvelope(ResponseSchema, IterableBase):
        tasks: list[Item]

        @classmethod
        # Legacy custom hooks do not have to accept a provider keyword.
        def from_response(
            cls, response, *, validation_context, strict, mode
        ):  # ty: ignore[invalid-method-override]
            assert mode is Mode.JSON
            return cls.model_validate(
                response, context=validation_context, strict=strict
            )

    class ChildEnvelope(CustomEnvelope):
        pass

    raw: Any = {"tasks": [{"value": 1}]}
    kwargs: dict[str, Any] = dict(
        response_model=ChildEnvelope,
        stream=False,
        mode=Mode.JSON,
        validation_context={"offset": 2},
        strict=True,
    )
    for result in (
        process_response(raw, **kwargs),
        await process_response_async(raw, **kwargs),
    ):
        assert result == [Item(value=3)]
        assert result.get_raw_response() is raw
