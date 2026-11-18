"""Container completeness at Iterable EOF without provider cutoff metadata."""

from collections.abc import AsyncGenerator
from typing import Any, cast

from pydantic import BaseModel
import pytest

from instructor.dsl.iterable import IterableModel
from instructor.v2.core.errors import IncompleteOutputException


class Row(BaseModel):
    value: int


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('[{"value":1}', [1]),
        ('[{"value":1},', [1]),
        ("[", []),
        ('{"tasks":[{"value":1}', [1]),
        ('{"tasks":[{"value":1}]', [1]),
        ('{"tasks":[', []),
        ('{"tasks":[{"value":1}],"meta":{"note":"o', [1]),
    ],
)
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_unclosed_container_preserves_items_then_raises(
    payload: str,
    expected: list[int],
    split: bool,
    is_async: bool,
) -> None:
    model = cast(Any, IterableModel(Row))
    chunks = list(payload) if split else [payload]
    received: list[int] = []

    async def source() -> AsyncGenerator[str, None]:
        for chunk in chunks:
            yield chunk

    with pytest.raises(IncompleteOutputException):
        if is_async:
            async for row in model.tasks_from_chunks_async(source()):
                received.append(row.value)
        else:
            for row in model.tasks_from_chunks(chunks):
                received.append(row.value)
    assert received == expected
