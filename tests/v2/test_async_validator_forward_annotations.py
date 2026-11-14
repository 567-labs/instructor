"""Reject unsupported validators behind postponed container annotations."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Any, ClassVar, NamedTuple

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from instructor.v2.core.response import handle_response_model
from instructor.v2.validation import async_field_validator
from instructor.v2.validation.async_validators import reject_async_validators


class ValidatedValue(BaseModel):
    value: str

    @async_field_validator("value")
    async def validate_value(cls, value: str) -> str:
        raise ValueError(f"Unsupported validation must not be skipped for {value!r}")


class TypedValues(TypedDict):
    item: ValidatedValue


class TupleValues(NamedTuple):
    item: ValidatedValue


@dataclass
class DataclassValues:
    item: ValidatedValue


class TypedEnvelope(BaseModel):
    values: TypedValues


class TupleEnvelope(BaseModel):
    values: TupleValues


class DataclassEnvelope(BaseModel):
    values: DataclassValues


@pytest.mark.parametrize(
    "model",
    [
        TypedValues,
        TupleValues,
        DataclassValues,
        TypedEnvelope,
        TupleEnvelope,
        DataclassEnvelope,
    ],
)
def test_postponed_container_annotations_reject_async_validators(model: Any) -> None:
    with pytest.raises(ValueError, match="async validators are not supported"):
        reject_async_validators(model)


@pytest.mark.parametrize("model", [TypedEnvelope, TupleEnvelope, DataclassEnvelope])
def test_request_preparation_rejects_nested_postponed_validators(model: Any) -> None:
    with pytest.raises(ValueError, match="async validators are not supported"):
        handle_response_model(model, messages=[])


class PlainValues(TypedDict):
    item: str


class PlainTuple(NamedTuple):
    item: str


@dataclass
class PlainDataclass:
    item: str


class RecursiveValues(TypedDict):
    children: list[RecursiveValues]


@pytest.mark.parametrize(
    "container", [PlainValues, PlainTuple, PlainDataclass, RecursiveValues]
)
def test_supported_postponed_containers_are_not_rejected(container: Any) -> None:
    assert reject_async_validators(container) is None


class UnresolvedAnnotations:
    __annotations__ = {"item": "NotYetDefined"}


def test_unresolved_annotations_are_left_to_schema_preparation() -> None:
    assert reject_async_validators(UnresolvedAnnotations) is None


@dataclass
class InheritedValues(DataclassValues):
    label: str


def test_inherited_postponed_dataclass_fields_are_checked() -> None:
    with pytest.raises(ValueError, match="async validators are not supported"):
        reject_async_validators(InheritedValues)


def test_unresolved_annotation_preserves_eager_validator_detection() -> None:
    class MixedAnnotations:
        __annotations__ = {"ready": ValidatedValue, "pending": "NotYetDefined"}

    with pytest.raises(ValueError, match="async validators are not supported"):
        reject_async_validators(MixedAnnotations)


@dataclass
class DataclassWithFactory:
    value: str
    factory: ClassVar[type[ValidatedValue]] = ValidatedValue


class FactoryEnvelope(BaseModel):
    values: DataclassWithFactory


def test_class_variables_do_not_reject_valid_requests() -> None:
    schema = FactoryEnvelope.model_json_schema()
    assert "factory" not in schema["$defs"]["DataclassWithFactory"]["properties"]
    model, kwargs = handle_response_model(FactoryEnvelope, messages=[])
    assert model is not None
    assert kwargs["tools"][0]["function"]["name"] == "FactoryEnvelope"


def test_classvar_annotations_are_not_validated() -> None:
    assert reject_async_validators(ClassVar[type[ValidatedValue]]) is None


@dataclass
class DataclassInitValues:
    value: InitVar[ValidatedValue]


class InitEnvelope(BaseModel):
    values: DataclassInitValues


def test_initvars_are_validated_inputs() -> None:
    schema = InitEnvelope.model_json_schema()
    assert "value" in schema["$defs"]["DataclassInitValues"]["properties"]
    with pytest.raises(ValueError, match="async validators are not supported"):
        handle_response_model(InitEnvelope, messages=[])


@dataclass
class PlainInitValues:
    value: InitVar[str]


def test_plain_initvars_are_not_rejected() -> None:
    assert reject_async_validators(PlainInitValues) is None
