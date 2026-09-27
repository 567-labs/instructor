"""Regression tests for PARALLEL_TOOLS through the v2 patch wrapper.

The existing handler-level tests call ``handlers.request_handler(...)`` directly
and bypass ``patch_v2``; these exercise the wrapper path ``from_openai`` uses.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ConfigDict

from instructor.v2.core.mode import Mode
from instructor.v2.core.patch import patch_v2
from instructor.v2.core.providers import Provider
from tests.coverage._openai import chat_completion, tool_call


class A(BaseModel):
    a: str


class B(BaseModel):
    b: str


def _member_types(response_model: Any) -> tuple[type[BaseModel], ...]:
    inner = get_args(response_model)[0]
    origin = get_origin(inner)
    if origin is Union:
        return get_args(inner)
    return (inner,)


def _parallel_completion(response_model: Any) -> Any:
    members = _member_types(response_model)
    payload = {"A": {"a": "alpha"}, "B": {"b": "beta"}}
    return chat_completion(
        tool_calls=[
            tool_call(
                m.__name__, payload[m.__name__], call_id=f"call_{m.__name__.lower()}"
            )
            for m in members
        ],
        finish_reason="tool_calls",
    )


def _assert_parallel_result(result: Any, response_model: Any) -> None:
    members = _member_types(response_model)
    expected_names = [m.__name__ for m in members]
    items = list(result)
    assert [type(x).__name__ for x in items] == expected_names
    assert items[0].a == "alpha"
    if B in members:
        assert items[1].b == "beta"


@pytest.mark.parametrize(
    "response_model",
    [
        pytest.param(Iterable[A], id="Iterable[A]"),
        pytest.param(Iterable[Union[A, B]], id="Iterable[Union[A,B]]"),
    ],
)
def test_parallel_tools_sync_wrapper(response_model: Any) -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _parallel_completion(response_model)

    patched = patch_v2(create, Provider.OPENAI, Mode.PARALLEL_TOOLS)
    result = patched(
        response_model=response_model,
        messages=[{"role": "user", "content": "run both"}],
    )

    assert calls, "create was never called"
    _assert_parallel_result(result, response_model)
    tool_names = {t["function"]["name"] for t in calls[0]["tools"]}
    assert tool_names == {m.__name__ for m in _member_types(response_model)}


@pytest.mark.asyncio
async def test_parallel_tools_async_wrapper() -> None:
    response_model = Iterable[Union[A, B]]
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _parallel_completion(response_model)

    patched = patch_v2(create, Provider.OPENAI, Mode.PARALLEL_TOOLS)
    result = await patched(
        response_model=response_model,
        messages=[{"role": "user", "content": "run both"}],
    )

    assert calls, "create was never called"
    _assert_parallel_result(result, response_model)


class CustomTitledModel(BaseModel):
    model_config = ConfigDict(title="custom_tool_title")
    x: int


def test_parallel_tools_custom_schema_title_wrapper() -> None:
    response_model = Iterable[Union[CustomTitledModel, B]]
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return chat_completion(
            tool_calls=[
                tool_call("custom_tool_title", '{"x": 123}', call_id="call_custom"),
                tool_call("B", '{"b": "beta"}', call_id="call_b"),
            ],
            finish_reason="tool_calls",
        )

    patched = patch_v2(create, Provider.OPENAI, Mode.PARALLEL_TOOLS)
    result = patched(
        response_model=response_model,
        messages=[{"role": "user", "content": "run both"}],
    )

    assert calls, "create was never called"
    tool_names = {t["function"]["name"] for t in calls[0]["tools"]}
    assert tool_names == {"custom_tool_title", "B"}
    items = list(result)
    assert len(items) == 2
    assert isinstance(items[0], CustomTitledModel)
    assert items[0].x == 123
    assert isinstance(items[1], B)
    assert items[1].b == "beta"
