"""The legacy class-keyed streaming marker is consumed exactly once."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from weakref import WeakKeyDictionary

from pydantic import BaseModel

from instructor.v2.core.streaming import StreamingModelState
from instructor.v2.dsl.partial import Partial


class Item(BaseModel):
    value: int


def test_streaming_flag_is_consumed_once() -> None:
    state = StreamingModelState()
    model = Partial[Item]

    state.mark_streaming_model(model, True)

    assert state._consume_streaming_flag(model) is True
    assert state._consume_streaming_flag(model) is False


def test_unmarked_or_non_class_models_are_not_consumed() -> None:
    state = StreamingModelState()

    assert state._consume_streaming_flag(Partial[Item]) is False
    assert state._consume_streaming_flag(None) is False


class _InterleavingDict(WeakKeyDictionary):  # type: ignore[type-arg]
    """Runs a competing consumer right after a membership check.

    This reproduces the interleaving where two consumers both see the marker
    and both try to delete it.
    """

    def __init__(self, on_check: Callable[[Any], None]) -> None:
        super().__init__()
        self._on_check: Callable[[Any], None] | None = on_check

    def __contains__(self, key: object) -> bool:
        found = super().__contains__(key)
        hook, self._on_check = self._on_check, None
        if found and hook is not None:
            hook(key)
        return found


def test_concurrent_consumers_do_not_raise() -> None:
    state = StreamingModelState()
    model = Partial[Item]
    results: list[bool] = []
    state._streaming_models = _InterleavingDict(
        lambda key: results.append(state._consume_streaming_flag(key))
    )
    state.mark_streaming_model(model, True)

    results.append(state._consume_streaming_flag(model))

    assert results.count(True) == 1
    assert state._consume_streaming_flag(model) is False
