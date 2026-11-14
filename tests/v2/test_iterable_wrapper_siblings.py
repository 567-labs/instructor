from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from instructor.v2.dsl.iterable import IterableModel


class U(BaseModel):
    name: str


PAYLOAD = '{"tasks":[{"name":"a"}],"meta":{"name":"zz"}}'


def test_tasks_wrapper_sibling_object_is_ignored_whole_string() -> None:
    model = cast(Any, IterableModel(U))

    assert list(model.tasks_from_chunks([PAYLOAD])) == [U(name="a")]


def test_tasks_wrapper_sibling_object_is_ignored_char_by_char() -> None:
    model = cast(Any, IterableModel(U))

    assert list(model.tasks_from_chunks(list(PAYLOAD))) == [U(name="a")]


def test_tasks_wrapper_sibling_with_brackets_in_strings_is_ignored() -> None:
    model = cast(Any, IterableModel(U))
    payload = '{"tasks":[{"name":"a"}],"meta":[{"name":"]}"},{"name":"zz"}]}'

    assert list(model.tasks_from_chunks(list(payload))) == [U(name="a")]


def test_objects_after_tasks_wrapper_closes_are_still_parsed() -> None:
    model = cast(Any, IterableModel(U))
    payload = '{"tasks":[{"name":"a"}],"meta":{"name":"zz"}}{"name":"b"}'

    assert list(model.tasks_from_chunks([payload])) == [U(name="a"), U(name="b")]
