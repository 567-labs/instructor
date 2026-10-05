"""A streaming chunk may carry deltas for several parallel tool calls at once.

extract_streaming_json previously read only ``tool_calls[0]`` of every chunk, so
the fragments of every call after the first were silently discarded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from instructor.v2.providers.openai.handlers import OpenAIToolsHandler


def _delta_chunk(*tool_calls: SimpleNamespace) -> Any:
    delta = SimpleNamespace(content=None, tool_calls=list(tool_calls))
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def _call(index: int, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=f"call-{index}",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_streaming_extractor_keeps_every_tool_call_in_a_chunk() -> None:
    handler = OpenAIToolsHandler()
    chunks = [
        _delta_chunk(_call(0, "_Foo", '{"name": "A'), _call(1, "_Foo", '{"name": "B')),
        _delta_chunk(_call(0, "_Foo", 'lice"}'), _call(1, "_Foo", 'ob"}')),
    ]

    parts = list(handler.extract_streaming_json(iter(chunks)))

    assert parts == ['{"name": "A', '{"name": "B', 'lice"}', 'ob"}']
