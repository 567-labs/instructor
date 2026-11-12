"""Native usage contract, independent of a live Groq account."""

from types import SimpleNamespace

import pytest

pytest.importorskip("groq")
from instructor.v2.core.retry import _initialize_usage
from instructor.v2.core.usage import (
    has_compatible_usage,
    update_total_usage,
    _usage_snapshot,
)
from instructor.v2.core.providers import Provider
from instructor.v2.providers.groq.usage import _usage_type


def usage(**changes):
    data = dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    data.update(changes)
    return _usage_type().model_construct(**data)


@pytest.mark.parametrize(
    "field", ["prompt_tokens", "completion_tokens", "total_tokens"]
)
@pytest.mark.parametrize("value", [None, -1, 1.5, True, "10"])
def test_invalid_counts_are_unavailable_and_untouched(field, value):
    raw = usage()
    setattr(raw, field, value)
    response = SimpleNamespace(usage=raw)
    total = _initialize_usage(Provider.GROQ)
    assert not has_compatible_usage(response, total)
    assert update_total_usage(response, total) is response
    assert response.usage is raw
    assert total.total_tokens == 0


def test_inconsistent_counts_are_unavailable():
    assert not has_compatible_usage(
        SimpleNamespace(usage=usage(total_tokens=16)), _initialize_usage(Provider.GROQ)
    )


@pytest.mark.parametrize(
    "breakdown",
    [{}, [], {"models": []}, {"models": [{"model": "child"}]}, {"unknown": 1}],
)
def test_compound_usage_is_not_guessed(breakdown):
    response = SimpleNamespace(usage=usage(), usage_breakdown=breakdown)
    total = _initialize_usage(Provider.GROQ)
    assert not has_compatible_usage(response, total)
    update_total_usage(response, total)
    assert total.total_tokens == 0


def test_latest_metadata_and_snapshot_do_not_alias_or_sum():
    total = _initialize_usage(Provider.GROQ)
    first = usage(
        queue_time=1.5,
        cost=99,
        _trace={"ids": [1]},
        schema={"x": [2]},
        prompt_tokens_details={"cached_tokens": 4},
    )
    response = SimpleNamespace(usage=first, x_groq={"id": "raw"})
    assert update_total_usage(response, total) is response
    assert response.usage is first
    snapshot = _usage_snapshot(total)
    first.model_extra["_trace"]["ids"].append(5)
    assert total.model_extra["_trace"] == {"ids": [1]}
    second = usage(
        queue_time=0.25,
        cost=7,
        _trace={"ids": [3]},
        schema={"x": [4]},
        prompt_tokens_details={"cached_tokens": 2},
    )
    response.usage = second
    update_total_usage(response, total)
    assert response.usage is second
    assert total.total_tokens == second.total_tokens == 30
    assert total.prompt_tokens == 20
    assert total.completion_tokens == 10
    assert total.queue_time == 0.25
    assert total.model_extra["cost"] == 7
    assert total.model_dump()["schema"] == {"x": [4]}
    assert total.model_dump()["_trace"] == {"ids": [3]}
    assert snapshot.total_tokens == 15
    assert snapshot.model_dump()["schema"] == {"x": [2]}
    assert response.x_groq == {"id": "raw"}
    update_total_usage(SimpleNamespace(usage=usage()), total)
    assert "schema" not in total.model_dump()
    assert total.total_tokens == 45


def test_zero_is_valid_and_duck_types_are_not():
    total = _initialize_usage(Provider.GROQ)
    assert has_compatible_usage(
        SimpleNamespace(
            usage=usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        ),
        total,
    )
    assert not has_compatible_usage(
        SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        ),
        total,
    )
