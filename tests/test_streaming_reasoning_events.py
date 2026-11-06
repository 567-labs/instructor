"""Tests for reasoning summary event surfacing in RESPONSES_TOOLS streaming.

These tests verify the fix for issue #2291: ResponseReasoningSummaryTextDeltaEvent
and ResponseReasoningSummaryTextDoneEvent are forwarded to an on_event callback
instead of being silently dropped during create_partial streaming.

All tests use mock event objects — no API key required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from instructor.dsl.partial import PartialBase
from instructor.mode import Mode


# ---------------------------------------------------------------------------
# Lightweight mock event classes matching openai.types.responses shapes
# ---------------------------------------------------------------------------


class _MockFunctionCallArgumentsDeltaEvent:
    """Mimics ResponseFunctionCallArgumentsDeltaEvent."""

    type: str = "response.function_call_arguments.delta"

    def __init__(self, delta: str) -> None:
        self.delta = delta


class _MockReasoningSummaryTextDeltaEvent:
    """Mimics ResponseReasoningSummaryTextDeltaEvent."""

    type: str = "response.reasoning_summary_text.delta"

    def __init__(self, delta: str) -> None:
        self.delta = delta


class _MockReasoningSummaryTextDoneEvent:
    """Mimics ResponseReasoningSummaryTextDoneEvent."""

    type: str = "response.reasoning_summary_text.done"

    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------------------
# Monkeypatch helper: replace the openai import inside extract_json
# so our mocks are recognised by isinstance checks.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_openai_response_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace openai.types.responses event classes with our mocks."""
    import openai.types.responses as responses_mod

    monkeypatch.setattr(
        responses_mod,
        "ResponseFunctionCallArgumentsDeltaEvent",
        _MockFunctionCallArgumentsDeltaEvent,
    )
    monkeypatch.setattr(
        responses_mod,
        "ResponseReasoningSummaryTextDeltaEvent",
        _MockReasoningSummaryTextDeltaEvent,
    )
    monkeypatch.setattr(
        responses_mod,
        "ResponseReasoningSummaryTextDoneEvent",
        _MockReasoningSummaryTextDoneEvent,
    )


# ---------------------------------------------------------------------------
# Sync tests — extract_json
# ---------------------------------------------------------------------------


class TestExtractJsonReasoningEvents:
    """Tests for PartialBase.extract_json with RESPONSES_TOOLS mode."""

    def test_yields_function_call_deltas(self) -> None:
        """Existing behaviour: FunctionCallArgumentsDelta events yield JSON text."""
        chunks = [
            _MockFunctionCallArgumentsDeltaEvent('{"name":'),
            _MockFunctionCallArgumentsDeltaEvent('"Alice"}'),
        ]
        result = list(PartialBase.extract_json(iter(chunks), Mode.RESPONSES_TOOLS))
        assert result == ['{"name":', '"Alice"}']

    def test_forwards_reasoning_delta_to_on_event(self) -> None:
        """Reasoning delta events are forwarded to on_event callback."""
        reasoning_chunk = _MockReasoningSummaryTextDeltaEvent("thinking...")
        fn_chunk = _MockFunctionCallArgumentsDeltaEvent('{"a":1}')
        chunks = [reasoning_chunk, fn_chunk]

        callback = MagicMock()
        json_parts = list(
            PartialBase.extract_json(
                iter(chunks), Mode.RESPONSES_TOOLS, on_event=callback
            )
        )

        # JSON yield is preserved
        assert json_parts == ['{"a":1}']
        # Callback was invoked with the reasoning event
        callback.assert_called_once_with(reasoning_chunk)

    def test_forwards_reasoning_done_to_on_event(self) -> None:
        """Reasoning done events are forwarded to on_event callback."""
        done_chunk = _MockReasoningSummaryTextDoneEvent("full reasoning text")
        chunks = [done_chunk]

        callback = MagicMock()
        list(
            PartialBase.extract_json(
                iter(chunks), Mode.RESPONSES_TOOLS, on_event=callback
            )
        )

        callback.assert_called_once_with(done_chunk)

    def test_no_on_event_silently_ignores_reasoning_events(self) -> None:
        """Without on_event, reasoning events are silently dropped (backward compat)."""
        chunks: list[Any] = [
            _MockReasoningSummaryTextDeltaEvent("ignored"),
            _MockFunctionCallArgumentsDeltaEvent('{"ok":true}'),
        ]

        # Should not raise — reasoning events are simply skipped
        result = list(PartialBase.extract_json(iter(chunks), Mode.RESPONSES_TOOLS))
        assert result == ['{"ok":true}']

    def test_works_with_inbuilt_tools_mode(self) -> None:
        """on_event also works with RESPONSES_TOOLS_WITH_INBUILT_TOOLS mode."""
        reasoning_chunk = _MockReasoningSummaryTextDeltaEvent("reasoning")
        fn_chunk = _MockFunctionCallArgumentsDeltaEvent('{"b":2}')
        chunks = [reasoning_chunk, fn_chunk]

        callback = MagicMock()
        json_parts = list(
            PartialBase.extract_json(
                iter(chunks),
                Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                on_event=callback,
            )
        )

        assert json_parts == ['{"b":2}']
        callback.assert_called_once_with(reasoning_chunk)


# ---------------------------------------------------------------------------
# Async tests — extract_json_async
# ---------------------------------------------------------------------------


async def _async_iter(items: list[Any]):
    """Convert a list into an async generator."""
    for item in items:
        yield item


class TestExtractJsonAsyncReasoningEvents:
    """Tests for PartialBase.extract_json_async with RESPONSES_TOOLS mode."""

    @pytest.mark.asyncio
    async def test_forwards_reasoning_events_async(self) -> None:
        """Async path forwards reasoning events to sync callback."""
        reasoning_chunk = _MockReasoningSummaryTextDeltaEvent("async thinking")
        fn_chunk = _MockFunctionCallArgumentsDeltaEvent('{"c":3}')

        callback = MagicMock()
        json_parts: list[str] = []
        async for part in PartialBase.extract_json_async(
            _async_iter([reasoning_chunk, fn_chunk]),
            Mode.RESPONSES_TOOLS,
            on_event=callback,
        ):
            json_parts.append(part)

        assert json_parts == ['{"c":3}']
        callback.assert_called_once_with(reasoning_chunk)

    @pytest.mark.asyncio
    async def test_async_on_event_callback_awaited(self) -> None:
        """Async callbacks are properly awaited in extract_json_async."""
        received: list[Any] = []

        async def async_callback(event: Any) -> None:
            received.append(event)

        reasoning_chunk = _MockReasoningSummaryTextDeltaEvent("awaited")
        fn_chunk = _MockFunctionCallArgumentsDeltaEvent('{"d":4}')

        json_parts: list[str] = []
        async for part in PartialBase.extract_json_async(
            _async_iter([reasoning_chunk, fn_chunk]),
            Mode.RESPONSES_TOOLS,
            on_event=async_callback,
        ):
            json_parts.append(part)

        assert json_parts == ['{"d":4}']
        assert len(received) == 1
        assert received[0] is reasoning_chunk
