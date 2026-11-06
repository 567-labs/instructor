from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import pytest
from pydantic import BaseModel

from instructor.dsl.iterable import IterableBase
from instructor.dsl.response_list import ListResponse
from instructor.mode import Mode
from instructor.processing.response import process_response, process_response_async
from instructor.utils.core import prepare_response_model


class DummyIterableModel(BaseModel, IterableBase):
    tasks: list[int]

    @classmethod
    def from_response(cls, completion, **kwargs):  # noqa: ANN001,ARG003
        return cls(tasks=[1, 2])

    @classmethod
    def from_streaming_response(  # ty: ignore[invalid-method-override]  # noqa: ANN001
        cls, _completion, mode: Mode, **_kwargs
    ) -> Generator[int, None, None]:
        del mode
        yield 1
        yield 2

    @classmethod
    def from_streaming_response_async(  # ty: ignore[invalid-method-override]  # noqa: ANN001
        cls, _completion: AsyncGenerator[object, None], mode: Mode, **_kwargs
    ) -> AsyncGenerator[int, None]:
        del mode

        async def gen() -> AsyncGenerator[int, None]:
            yield 1
            yield 2

        return gen()


class DummyCompletion(BaseModel):
    """Minimal stand-in for a provider completion object."""


def test_process_response_returns_list_response_for_iterable_model():
    raw = DummyCompletion()

    result = process_response(
        raw,
        response_model=DummyIterableModel,
        stream=False,
        mode=Mode.TOOLS,
    )

    assert isinstance(result, ListResponse)
    assert list(result) == [1, 2]
    assert result._raw_response == raw


def test_process_response_streaming_returns_list_response_for_iterable_model():
    raw = DummyCompletion()

    result = process_response(
        raw,
        response_model=DummyIterableModel,
        stream=True,
        mode=Mode.TOOLS,
    )

    # Streaming IterableBase should preserve generator behavior (used by create_iterable()).
    assert list(result) == [1, 2]


@pytest.mark.asyncio
async def test_process_response_async_streaming_returns_list_response_for_iterable_model():
    async def completion_stream() -> AsyncGenerator[object, None]:
        yield object()

    raw = completion_stream()

    result = await process_response_async(
        raw,  # ty: ignore[invalid-argument-type]
        response_model=DummyIterableModel,
        stream=True,
        mode=Mode.TOOLS,
    )

    # Streaming IterableBase should preserve async generator behavior (used by create_iterable()).
    collected: list[int] = []
    async for item in result:
        collected.append(item)
    assert collected == [1, 2]


def test_prepare_response_model_treats_list_as_iterable_model():
    class User(BaseModel):
        name: str

    prepared = prepare_response_model(list[User])
    assert prepared is not None
    assert issubclass(prepared, IterableBase)
