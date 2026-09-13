from __future__ import annotations

import sys
import types
import typing
from enum import Enum

import pytest
from pydantic import BaseModel, ValidationError

import instructor.v2.dsl.simple_type as simple_type
from instructor.v2.dsl.citation import CitationMixin
from instructor.v2.dsl.json_tracker import JsonCompleteness, is_json_complete
from instructor.v2.dsl.maybe import Maybe
from instructor.v2.dsl.response_list import ListResponse, ResponseList
from instructor.v2.dsl.simple_type import (
    ModelAdapter,
    is_simple_type,
    validateIsSubClass,
)


class Claim(CitationMixin):
    fact: str


class User(BaseModel):
    name: str


def test_citation_keeps_quotes_without_context_and_recovers_fuzzy_quotes() -> None:
    original = Claim.model_validate(
        {"fact": "age", "substring_quotes": ["Jaxon is 20 years old"]}
    )
    assert original.substring_quotes == ["Jaxon is 20 years old"]

    context = "Betty was a student. Jason is 20 years old."
    cited = Claim.model_validate(
        {
            "fact": "age",
            "substring_quotes": ["Jaxon is 20 years old", "not in the text"],
        },
        context={"context": context},
    )
    assert cited.substring_quotes == ["Jason is 20 years old"]
    assert list(cited.get_spans(context)) == [(21, 42)]


def test_citation_keeps_quotes_when_validation_context_lacks_context_key() -> None:
    quotes = ["Jaxon is 20 years old"]

    cited = Claim.model_validate(
        {"fact": "age", "substring_quotes": quotes},
        context={"request_id": "req_123"},
    )

    assert cited.substring_quotes == quotes


@pytest.mark.parametrize(
    ("value", "complete"),
    [("", False), ("  \n\t", False), ('{"ok": true}', True), ('{"ok":', False)],
)
def test_is_json_complete_handles_empty_complete_and_partial_values(
    value: str, complete: bool
) -> None:
    assert is_json_complete(value) is complete


def test_json_tracker_marks_every_nested_path_and_returns_a_copy() -> None:
    tracker = JsonCompleteness()
    tracker.analyze('{"user": {"name": "Ada"}, "items": [{"id": 1}, 2]}')

    expected = {
        "",
        "user",
        "user.name",
        "items",
        "items[0]",
        "items[0].id",
        "items[1]",
    }
    assert tracker.is_root_complete()
    assert tracker.get_complete_paths() == expected
    assert all(tracker.is_path_complete(path) for path in expected)

    returned = tracker.get_complete_paths()
    returned.clear()
    assert tracker.get_complete_paths() == expected

    tracker.analyze(" \n ")
    assert not tracker.is_root_complete()
    assert tracker.get_complete_paths() == set()


def test_json_tracker_uses_siblings_for_partial_lists_and_objects() -> None:
    tracker = JsonCompleteness()
    tracker.analyze(
        '{"done": true, "items": [{"name": "A"}, {"name": "B", "unfinished": "hel'
    )

    assert tracker.get_complete_paths() == {
        "done",
        "items[0]",
        "items[0].name",
        "items[1].name",
    }
    assert not tracker.is_root_complete()
    assert not tracker.is_path_complete("items")
    assert not tracker.is_path_complete("items[1]")
    assert not tracker.is_path_complete("items[1].unfinished")

    tracker.analyze("}")
    assert tracker.get_complete_paths() == set()


def test_maybe_model_validates_results_defaults_and_truthiness() -> None:
    maybe_user = Maybe(User)

    present = maybe_user.model_validate({"result": {"name": "Ada"}})
    absent = maybe_user.model_validate({"error": True, "message": "No user found"})

    assert maybe_user.__name__ == "MaybeUser"
    assert present.result == User(name="Ada")
    assert present.error is False
    assert present.message is None
    assert bool(present)
    assert absent.result is None
    assert absent.error is True
    assert absent.message == "No user found"
    assert not absent
    assert maybe_user.model_fields["result"].description.startswith(
        "Correctly extracted result"
    )

    with pytest.raises(ValidationError):
        maybe_user.model_validate({"result": {"name": 123}})


def test_response_list_preserves_raw_response_on_slices_and_alias() -> None:
    raw_response = {"provider": "test", "request_id": "req-1"}
    values = ListResponse.from_list([1, 2, 3], raw_response=raw_response)

    assert ResponseList is ListResponse
    assert values.get_raw_response() is raw_response
    assert values[1] == 2
    assert isinstance(values[1:], ListResponse)
    assert values[1:] == [2, 3]
    assert values[1:].get_raw_response() is raw_response
    assert ListResponse().get_raw_response() is None


def test_model_adapter_accepts_simple_values_and_rejects_models() -> None:
    adapted = typing.cast(type[BaseModel], ModelAdapter[int])
    assert adapted.model_validate({"content": 7}).model_dump() == {"content": 7}
    assert adapted.model_json_schema()["properties"]["content"]["type"] == "integer"

    with pytest.raises(AssertionError, match="Only simple types are supported"):
        ModelAdapter[User]


def test_validate_is_subclass_handles_generic_alias_and_legacy_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11))
    assert validateIsSubClass(User)
    assert not validateIsSubClass(list[User])

    monkeypatch.setattr(sys, "version_info", (3, 9))
    assert not validateIsSubClass(User)
    assert validateIsSubClass(list[User])


def test_validate_is_subclass_tolerates_a_broken_generic_alias_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11))
    monkeypatch.setattr(types, "GenericAlias", object())
    assert validateIsSubClass(User)


def test_is_simple_type_handles_unions_and_type_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert is_simple_type(list[typing.Union[int, str]])

    monkeypatch.setattr(sys, "version_info", (3, 11))

    def invalid_subclass_check(_: type) -> bool:
        raise TypeError("unsupported generic")

    monkeypatch.setattr(simple_type, "validateIsSubClass", invalid_subclass_check)
    assert not is_simple_type(User)


def test_is_simple_type_covers_list_shapes_and_old_issubclass_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert is_simple_type(list[typing.Union[int, str]])
    assert not is_simple_type(list[User])
    assert not is_simple_type(list[object])
    assert is_simple_type(typing.List)  # noqa: UP006

    monkeypatch.setattr(simple_type, "hasattr", lambda *_: False, raising=False)
    assert is_simple_type(list[int])
    assert not is_simple_type(list[typing.Literal["one"]])

    def legacy_issubclass(value: object, base: type) -> bool:
        if value is int:
            raise TypeError("legacy generic alias")
        if not isinstance(value, type):
            raise TypeError("issubclass() arg 1 must be a class")
        return issubclass(value, base)

    monkeypatch.setattr(sys, "version_info", (3, 11))
    monkeypatch.setattr(simple_type, "issubclass", legacy_issubclass, raising=False)
    assert is_simple_type(list[int])


def test_is_simple_type_handles_legacy_iterable_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_get_origin = typing.get_origin
    union_iterable = typing.Iterable[typing.Union[int, str]]
    simple_iterable = typing.Iterable[int]
    literal_iterable = typing.Iterable[typing.Literal["one"]]
    bare_iterable = typing.Iterable

    def legacy_get_origin(value: object) -> object:
        if value in {
            union_iterable,
            simple_iterable,
            literal_iterable,
            bare_iterable,
        }:
            return typing.Iterable
        return original_get_origin(value)

    monkeypatch.setattr(typing, "get_origin", legacy_get_origin)
    assert is_simple_type(union_iterable)
    assert is_simple_type(simple_iterable)

    monkeypatch.setattr(simple_type, "hasattr", lambda *_: False, raising=False)
    assert is_simple_type(simple_iterable)
    assert not is_simple_type(literal_iterable)
    assert not is_simple_type(bare_iterable)


def test_is_simple_type_recognizes_core_scalars_annotations_and_enums() -> None:
    class Color(Enum):
        RED = "red"

    assert is_simple_type(str)
    assert is_simple_type(typing.Annotated[int, "count"])
    assert is_simple_type(typing.Literal["ok"])
    assert is_simple_type(Color)
    assert not is_simple_type(dict[str, int])
