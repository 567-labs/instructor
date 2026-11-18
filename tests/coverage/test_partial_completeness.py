from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, Field, model_validator

from instructor.v2.dsl.json_tracker import JsonCompleteness
from instructor.v2.dsl.partial import Partial


class Reading(BaseModel):
    number: int = Field(alias="n")


class Readings(BaseModel):
    featured: Reading = Field(validation_alias=AliasChoices("lead", "lead.value"))
    selected: Reading | None = Field(
        default=None, validation_alias=AliasPath("payload", "rows", -1)
    )
    items: list[Reading] = Field(alias="entries[0]")
    mapping: dict[str, Reading] = Field(alias="map.values")


class CollidingPaths(BaseModel):
    dotted: Reading = Field(validation_alias=AliasPath("node", "sample"))
    bracketed: Reading = Field(validation_alias=AliasPath("rows", 0))


class MutatingPayload(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def add_reading(cls, data: dict[str, Any]) -> dict[str, Any]:
        data["new"] = {"n": "9"}
        return data


class ValidatorInput(BaseModel):
    payload: MutatingPayload
    selected: Reading = Field(validation_alias=AliasPath("payload", "new"))


def readings_snapshot(
    featured: int | str,
    selected: int | str | None = None,
    items: list[int | str] | None = None,
    mapping: dict[str, int | str] | None = None,
) -> dict[str, Any]:
    return {
        "featured": {"number": featured},
        "selected": {"number": selected} if selected is not None else None,
        "items": [{"number": item} for item in items] if items is not None else None,
        "mapping": (
            {key: {"number": item} for key, item in mapping.items()}
            if mapping is not None
            else None
        ),
    }


@pytest.mark.parametrize(
    ("model", "chunks", "expected"),
    [
        pytest.param(
            Readings,
            [
                '{"lead.value":{"n":"1',
                '"},"payload":{"rows":[{"n":"2"}',
                ',{"n":"3"}]},"entries[0]":[{"n":"4"}',
                ',{"n":"5"}],"map.values":{"some.key":{"n":"6"}',
                ',"other[0]":{"n":"7"}',
                '},"lead":{"n":"8"}}',
            ],
            [
                readings_snapshot("1"),
                readings_snapshot(1, "2"),
                readings_snapshot(1, 3, ["4"]),
                readings_snapshot(1, 3, [4, 5], {"some.key": "6"}),
                readings_snapshot(1, 3, [4, 5], {"some.key": 6, "other[0]": "7"}),
                readings_snapshot(8, 3, [4, 5], {"some.key": 6, "other[0]": 7}),
            ],
            id="nested-aliases",
        ),
        pytest.param(
            CollidingPaths,
            [
                '{"node.sample":0,"rows[0]":0,"node":{"sample":{"n":"8',
                '"}},"rows":[{"n":"9"}',
                "]}",
            ],
            [
                {"dotted": {"number": 8}, "bracketed": {"number": None}},
                {"dotted": {"number": 8}, "bracketed": {"number": 9}},
                {"dotted": {"number": 8}, "bracketed": {"number": 9}},
            ],
            id="flattened-path-collisions",
        ),
        pytest.param(
            ValidatorInput,
            ['{"payload":{"existing":0},"still_open":"x'],
            [{"payload": {}, "selected": {"number": "9"}}],
            id="validator-mutates-completed-input",
        ),
    ],
)
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_partial_stream_preserves_each_snapshot(
    model: type[BaseModel],
    chunks: list[str],
    expected: list[dict[str, Any]],
    use_async: bool,
) -> None:
    # Golden snapshots captured before the completeness tracking optimization.
    # Strings remain raw while a nested model is open and coerce when complete.
    partial_model = cast(Any, Partial[model])
    if use_async:

        async def source() -> AsyncGenerator[str, None]:
            for chunk in chunks:
                yield chunk

        snapshots = [
            item.model_dump(warnings=False)
            async for item in partial_model.model_from_chunks_async(source())
        ]
    else:
        snapshots = [
            item.model_dump(warnings=False)
            for item in partial_model.model_from_chunks(chunks)
        ]

    assert snapshots == expected


@pytest.mark.parametrize(
    ("document", "expected", "absent"),
    [
        (
            '{"done":{"values":[{"x":1},false]},"open":{"next":[0,',
            {
                "done",
                "done.values",
                "done.values[0]",
                "done.values[0].x",
                "done.values[1]",
            },
            {"", "done.missing", "done.values[0].missing", "done.values[2]", "open"},
        ),
        (
            '{"node.sample":{"n":1},"rows[0]":{"n":2},"node":{"sample":',
            {"node.sample", "node.sample.n", "rows[0]", "rows[0].n"},
            {"", "node", "node.sample.missing", "rows"},
        ),
        (
            '{"node":{"sample":{"n":1}},"rows":[{"n":2}],"node.sample":{"n":3',
            {"node", "node.sample", "node.sample.n", "rows", "rows[0]", "rows[0].n"},
            {"", "node.sample.missing", "rows[1]"},
        ),
        (
            '{"":{"n":1},"next":0,"open":"abc',
            {"", "n", "next"},
            {"open", ".n", "missing"},
        ),
        (
            '{"node":{"":{"value":1},"last":0},"open":"abc',
            {"node", "node.", "node..value", "node.last"},
            {"", "node.value", "node.missing", "open"},
        ),
        (
            '[{"a.b":{"[0]":true}},{',
            {"[0]", "[0].a.b", "[0].a.b.[0]"},
            {"", "[0].a", "[0].a.b[0]", "[1]"},
        ),
        (
            '{"node":{"empty":{},"zero":0},"list":[]}',
            {"", "list", "node", "node.empty", "node.zero"},
            {"node.empty.missing", "node.zero.missing", "list[0]"},
        ),
    ],
)
def test_tracker_keeps_exact_flattened_paths(
    document: str, expected: set[str], absent: set[str]
) -> None:
    tracker = JsonCompleteness()
    tracker.analyze(document)
    assert tracker.is_root_complete() == ("" in expected)
    assert all(tracker.is_path_complete(path) for path in expected)
    assert not any(tracker.is_path_complete(path) for path in absent)
    returned = tracker.get_complete_paths()
    assert returned == expected
    returned.clear()
    assert tracker.get_complete_paths() == expected

    tracker.analyze(" ")
    assert not tracker.is_root_complete()
    assert tracker.get_complete_paths() == set()
