"""Real threads retire legacy markers without racing explicit stream routing."""

from concurrent.futures import ThreadPoolExecutor
import gc
import sys
from threading import Barrier
from weakref import ref

from pydantic import BaseModel, create_model
import pytest

from instructor import Partial
from instructor.v2.core.streaming import StreamingModelState


class Row(BaseModel):
    value: int


@pytest.mark.parametrize("explicit", [False, True])
def test_concurrent_explicit_stream_marker_consumption(explicit: bool) -> None:
    state = StreamingModelState()
    model = Partial[Row]
    barrier = Barrier(8)

    def consume() -> None:
        barrier.wait(timeout=10)
        for _ in range(10_000):
            state.mark_streaming_model(model, True)
            assert state._should_parse_streaming(model, explicit) is explicit

    original_interval = sys.getswitchinterval()
    try:
        # Exercise actual Python thread interleavings, not a replacement dictionary.
        sys.setswitchinterval(0.000001)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(consume) for _ in range(8)]
            for future in futures:
                future.result(timeout=30)
    finally:
        sys.setswitchinterval(original_interval)
    assert not state._consume_streaming_flag(model)


def test_marker_presence_is_not_marker_truthiness() -> None:
    state = StreamingModelState()
    model = Partial[Row]
    assert state._consume_streaming_flag(model) is False
    state.mark_streaming_model(model, True)
    # Stored values are None; a present marker must nevertheless return True.
    assert state._consume_streaming_flag(model) is True
    assert state._consume_streaming_flag(model) is False
    state.mark_streaming_model(model, False)
    assert state._consume_streaming_flag(model) is False
    assert state._consume_streaming_flag(None) is False
    assert state._consume_streaming_flag(Row) is False
    state.mark_streaming_model(model, True)
    assert state._should_parse_streaming(model, None) is True
    assert state._should_parse_streaming(model, None) is False


def test_markers_do_not_keep_models_alive() -> None:
    state = StreamingModelState()

    # Create an uncached subclass; Partial[...] itself intentionally caches classes.
    ephemeral = create_model("Ephemeral", __base__=Partial[Row])

    reference = ref(ephemeral)
    state.mark_streaming_model(ephemeral, True)
    del ephemeral
    gc.collect()
    assert reference() is None
    assert len(state._streaming_models) == 0
