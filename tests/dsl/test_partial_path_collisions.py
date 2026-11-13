from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
import json
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field, ValidationError

from instructor import Partial
from instructor.v2.dsl.json_tracker import JsonCompleteness


class Name(BaseModel):
    text: str = Field(min_length=5)


class User(BaseModel):
    name: Name


class DottedEnvelope(BaseModel):
    marker: int = Field(alias="user.name")
    user: User


class IndexedEnvelope(BaseModel):
    marker: int = Field(alias="items[0]")
    items: list[Name]


class EmptyAliasEnvelope(BaseModel):
    marker: int = Field(alias="")
    user: User


class PlainEnvelope(BaseModel):
    marker: int
    user: User


CASES = [
    (
        DottedEnvelope,
        {"user.name": 1, "user": {"name": {"text": "Alice"}}},
        ("user", "name", "text"),
    ),
    (
        IndexedEnvelope,
        {"items[0]": 1, "items": [{"text": "Alice"}]},
        ("items", 0, "text"),
    ),
    (
        EmptyAliasEnvelope,
        {"": 1, "user": {"name": {"text": "Alice"}}},
        ("user", "name", "text"),
    ),
    (
        PlainEnvelope,
        {"marker": 1, "user": {"name": {"text": "Alice"}}},
        ("user", "name", "text"),
    ),
]


async def _collect(
    model: type[BaseModel], chunks: list[str], asynchronous: bool
) -> list[BaseModel]:
    partial = cast(Any, Partial[model])
    if asynchronous:

        async def stream() -> AsyncIterator[str]:
            for chunk in chunks:
                yield chunk

        return [item async for item in partial.model_from_chunks_async(stream())]
    return list(partial.model_from_chunks(chunks))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("model", "payload", "text_path"),
    CASES,
    ids=["dotted-alias", "indexed-alias", "empty-alias", "plain-control"],
)
async def test_partial_alias_does_not_complete_another_path(
    model: type[BaseModel],
    payload: dict[str, Any],
    text_path: tuple[str | int, ...],
    asynchronous: bool,
) -> None:
    text = json.dumps(payload)
    split = text.index("Alice") + 2
    expected = model.model_validate_json(text)

    snapshots = await _collect(model, [text[:split], text[split:]], asynchronous)

    assert len(snapshots) == 2
    partial_text: Any = snapshots[0].model_dump()
    for part in text_path:
        partial_text = partial_text[part]
    assert partial_text == "Al"
    assert snapshots[-1] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("model", "payload", "text_path"),
    CASES,
    ids=["dotted-alias", "indexed-alias", "empty-alias", "plain-control"],
)
async def test_completed_alias_payload_still_rejects_invalid_data(
    model: type[BaseModel],
    payload: dict[str, Any],
    text_path: tuple[str | int, ...],
    asynchronous: bool,
) -> None:
    invalid = deepcopy(payload)
    parent: Any = invalid
    for part in text_path[:-1]:
        parent = parent[part]
    parent[text_path[-1]] = "Al"

    with pytest.raises(ValidationError) as failure:
        await _collect(model, [json.dumps(invalid)], asynchronous)
    assert any(error["type"] == "string_too_short" for error in failure.value.errors())


def test_empty_key_is_not_a_complete_root() -> None:
    tracker = JsonCompleteness()
    tracker.analyze('{"": 1, "pending": "hel')
    assert not tracker.is_root_complete()
    assert not tracker.is_path_complete("")
    assert "" not in tracker.get_complete_paths()

    tracker.analyze('{"": {"nested": 1}, "pending": "hel')
    assert tracker.get_complete_paths() == {"nested"}

    tracker.analyze('{"": 1}')
    assert tracker.is_root_complete()
    assert tracker.is_path_complete("")
    assert "" in tracker.get_complete_paths()
