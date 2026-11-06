"""Async-validator response models must fail closed before provider execution."""

from __future__ import annotations

from typing import Annotated, Any, Generic, Optional, TypeVar

import pytest
from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import TypeAliasType

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import retry_async_v2, retry_sync_v2
from instructor.v2.validation import async_field_validator, async_model_validator
from instructor.v2.validation.async_validators import (
    model_declares_async_validators,
    reject_async_validators,
)


class Email(BaseModel):
    address: str

    @async_field_validator("address")
    async def must_contain_at(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Invalid email address")
        return value.lower()


class Account(BaseModel):
    email: Email

    @async_model_validator()
    async def normalize(self) -> Account:
        return self


class Plain(BaseModel):
    value: int


class NestedEmails(BaseModel):
    values: list[int]
    emails: dict[str, tuple[list[Email], ...]]


T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    items: list[T]


EmailGroups = TypeAliasType("EmailGroups", list[Email], type_params=(T,))


@pytest.mark.parametrize(
    "annotation,payload,expected",
    [
        (list[int], [1], [1]),
        (list[Plain], [{"value": 1}], [Plain(value=1)]),
        (dict[str, list[Plain]], {"a": [{"value": 1}]}, {"a": [Plain(value=1)]}),
        (tuple[list[int], Plain], [[1], {"value": 1}], ([1], Plain(value=1))),
        (Optional[list[Plain]], [{"value": 1}], [Plain(value=1)]),
        (Annotated[list[int], Field(min_length=1)], [1], [1]),
        (Envelope[int], {"items": [1]}, Envelope[int](items=[1])),
    ],
)
def test_supported_generic_models_validate_without_async_markers(
    annotation: Any, payload: Any, expected: Any
) -> None:
    assert not model_declares_async_validators(annotation)
    assert reject_async_validators(annotation) is None
    assert TypeAdapter(annotation).validate_python(payload) == expected


@pytest.mark.parametrize(
    "response_model",
    [
        Email,
        Account,
        list[Email],
        dict[str, list[Email]],
        NestedEmails,
        list[NestedEmails],
        Envelope[Email],
        EmailGroups[int],
    ],
)
def test_sync_retry_rejects_async_validators_before_provider(
    response_model: Any,
) -> None:
    def unexpected_request(**_kwargs: Any) -> None:
        raise AssertionError("Provider must not be called for unsupported validators")

    with pytest.raises(ValueError, match="async validators are not supported"):
        retry_sync_v2(
            func=unexpected_request,
            response_model=response_model,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs={},
            strict=True,
            hooks=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_model",
    [
        Email,
        Account,
        list[Email],
        dict[str, list[Email]],
        NestedEmails,
        list[NestedEmails],
        Envelope[Email],
        EmailGroups[int],
    ],
)
async def test_async_retry_rejects_async_validators_before_provider(
    response_model: Any,
) -> None:
    async def unexpected_request(**_kwargs: Any) -> None:
        raise AssertionError("Provider must not be called for unsupported validators")

    with pytest.raises(ValueError, match="async validators are not supported"):
        await retry_async_v2(
            func=unexpected_request,
            response_model=response_model,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs={},
            strict=True,
            hooks=None,
        )
