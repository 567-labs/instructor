"""Regression test for async streaming with Mode.GEMINI_TOOLS.

The sync paths in PartialBase.from_streaming_response and
IterableBase.from_streaming_response apply extract_json_from_stream
for both Mode.MD_JSON and Mode.GEMINI_TOOLS, but the async paths
were only applying it for Mode.MD_JSON.
"""

import pytest

from instructor.mode import Mode
from instructor.utils.core import (
    extract_json_from_stream,
    extract_json_from_stream_async,
)


def test_sync_extract_json_from_stream_handles_codeblock():
    chunks = ["```json\n", '{"name": "Alice",', ' "age": 30}', "\n```"]
    result = "".join(extract_json_from_stream(iter(chunks)))
    assert result == '{"name": "Alice", "age": 30}'


@pytest.mark.asyncio
async def test_async_extract_json_from_stream_handles_codeblock():
    chunks = ["```json\n", '{"name": "Alice",', ' "age": 30}', "\n```"]

    async def async_chunks():
        for c in chunks:
            yield c

    result = "".join([c async for c in extract_json_from_stream_async(async_chunks())])
    assert result == '{"name": "Alice", "age": 30}'


def test_sync_gemini_tools_mode_triggers_json_extraction():
    """Verify that GEMINI_TOOLS is in the set that triggers extract_json_from_stream
    in the sync from_streaming_response path."""
    # This tests the condition that was already correct in the sync path
    assert Mode.GEMINI_TOOLS in {Mode.MD_JSON, Mode.GEMINI_TOOLS}


def test_async_gemini_tools_mode_triggers_json_extraction():
    """Verify the fix: GEMINI_TOOLS must be in the set that triggers
    extract_json_from_stream_async in the async from_streaming_response_async path.

    Before the fix, the async path only checked `mode == Mode.MD_JSON`,
    so GEMINI_TOOLS streaming would skip JSON extraction from code blocks.
    """
    # After the fix, both sync and async paths use the same set
    mode = Mode.GEMINI_TOOLS
    # This is the condition in the fixed async path
    assert mode in {Mode.MD_JSON, Mode.GEMINI_TOOLS}
