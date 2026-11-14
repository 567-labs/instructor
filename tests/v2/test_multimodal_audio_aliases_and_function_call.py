from __future__ import annotations

import base64
from pathlib import Path

import pytest

from instructor.v2.core.multimodal import Audio, convert_messages
from instructor.v2.core.mode import Mode


@pytest.mark.parametrize(
    ("suffix", "media_type"),
    [(".flac", "audio/flac"), (".m4a", "audio/m4a"), (".aac", "audio/aac")],
)
def test_audio_from_path_accepts_platform_aliases(
    tmp_path: Path, suffix: str, media_type: str
) -> None:
    path = tmp_path / f"clip{suffix}"
    path.write_bytes(b"audio-bytes")

    audio = Audio.from_path(path)

    assert audio.media_type == media_type
    assert audio.data == base64.b64encode(b"audio-bytes").decode()


@pytest.mark.parametrize(
    ("alias", "media_type"),
    [
        ("audio/x-flac", "audio/flac"),
        ("audio/x-aac", "audio/aac"),
        ("audio/x-m4a", "audio/m4a"),
        ("audio/mp4a-latm", "audio/m4a"),
    ],
)
def test_audio_from_base64_accepts_aliases(alias: str, media_type: str) -> None:
    audio = Audio.from_base64(f"data:{alias};base64,YWJj")

    assert audio.media_type == media_type


def test_convert_messages_keeps_legacy_function_call_without_content() -> None:
    message = {
        "role": "assistant",
        "function_call": {"name": "lookup", "arguments": "{}"},
    }

    assert convert_messages([message], Mode.TOOLS) == [message]


def test_convert_messages_keeps_legacy_function_call_with_none_content() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "function_call": {"name": "lookup", "arguments": "{}"},
    }

    assert convert_messages([message], Mode.TOOLS) == [message]
