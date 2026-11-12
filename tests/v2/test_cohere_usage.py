"""Native tokens-only projection; optional billing never substitutes tokens."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("cohere")
from instructor.v2.providers.cohere.usage import (
    _usage_types,
    accumulate_usage,
    initialize_usage,
    token_counts,
)
from instructor.v2.core.usage import _usage_total_tokens


def usage(**changes: Any) -> Any:
    usage_type, tokens_type = _usage_types()
    return usage_type.model_validate(
        {"tokens": {"input_tokens": 100, "output_tokens": 20}, **changes}
    )


@pytest.mark.parametrize(
    "value", [None, -1, 1.5, True, float("nan"), float("inf"), "100"]
)
@pytest.mark.parametrize("name", ["input_tokens", "output_tokens"])
def test_invalid_native_counts(value: Any, name: str) -> None:
    usage_type, tokens_type = _usage_types()
    fields: dict[str, Any] = {"input_tokens": 100, "output_tokens": 20, name: value}
    raw = usage_type.model_construct(tokens=tokens_type.model_construct(**fields))
    assert token_counts(raw) is None
    assert accumulate_usage(SimpleNamespace(usage=raw), initialize_usage()) is None


@pytest.mark.parametrize("fields", [{}, {"input_tokens": 0}, {"output_tokens": 0}])
def test_missing_is_not_zero(fields: dict[str, int]) -> None:
    usage_type, tokens_type = _usage_types()
    assert token_counts(usage_type.model_validate({"tokens": fields})) is None


def test_projection_and_raw_are_independent() -> None:
    raw = usage(
        billed_units={"input_tokens": 10, "output_tokens": 5},
        cached_tokens=70,
        total_tokens=99999,
        opaque={"ids": [1]},
    )
    before = raw.model_dump()
    total = initialize_usage()
    first = accumulate_usage(SimpleNamespace(usage=raw), total)
    assert first is not None
    second = accumulate_usage(SimpleNamespace(usage=raw), first)
    assert second is not None
    assert token_counts(total) == (0, 0)
    assert token_counts(first) == (100, 20)
    assert token_counts(second) == (200, 40)
    assert _usage_total_tokens(second) == 240
    assert second.model_fields_set == {"tokens"}
    assert second.model_extra in ({}, None)
    assert raw.model_dump() == before
    assert (
        first is not second and cast(Any, first).tokens is not cast(Any, second).tokens
    )
    assert cast(Any, first).tokens is not raw.tokens


def test_native_identity_required() -> None:
    assert (
        token_counts(
            SimpleNamespace(tokens=SimpleNamespace(input_tokens=1, output_tokens=2))
        )
        is None
    )


def test_invalid_cumulative_float_representation_is_unavailable() -> None:
    usage_type, tokens_type = _usage_types()
    large = usage_type.model_validate(
        {"tokens": {"input_tokens": 2**53, "output_tokens": 0}}
    )
    one = usage_type.model_validate({"tokens": {"input_tokens": 1, "output_tokens": 0}})
    assert token_counts(large) == (2**53, 0)
    assert accumulate_usage(SimpleNamespace(usage=one), large) is None
    extreme = usage_type.model_validate(
        {"tokens": {"input_tokens": 1e308, "output_tokens": 0}}
    )
    assert token_counts(extreme) is not None
    assert accumulate_usage(SimpleNamespace(usage=extreme), extreme) is None
    zero = usage_type.model_validate(
        {"tokens": {"input_tokens": 0, "output_tokens": 0}}
    )
    assert token_counts(zero) == (0, 0)
    assert accumulate_usage(SimpleNamespace(usage=zero), large) is not None
