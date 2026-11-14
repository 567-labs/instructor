from __future__ import annotations

import base64
import io
import mimetypes
import wave
from pathlib import Path

import pytest

from instructor import Mode
from instructor.v2.core.multimodal import Audio, autodetect_media, convert_messages


@pytest.fixture
def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\x00\x00" * 80)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "media_type", ["audio/wav", "audio/x-wav", "AUDIO/X-WAV;charset=utf-8"]
)
def test_wav_data_uri_matches_file_audio(
    media_type: str, wav_bytes: bytes, tmp_path: Path
) -> None:
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes)
    encoded = base64.b64encode(wav_bytes).decode()
    source = f"data:{media_type};base64,{encoded}"

    from_file = Audio.from_path(path)
    from_data = Audio.from_base64(source)
    assert from_data.media_type == from_file.media_type == "audio/wav"
    assert from_data.data == from_file.data == encoded
    assert from_data.source == source
    assert Audio.autodetect(source) == from_data
    assert autodetect_media(source) == from_data
    assert from_data.to_openai(Mode.TOOLS) == from_file.to_openai(Mode.TOOLS)


def test_wav_alias_is_converted_to_audio_message(wav_bytes: bytes) -> None:
    encoded = base64.b64encode(wav_bytes).decode()
    source = f"data:audio/x-wav;base64,{encoded}"

    converted = convert_messages(
        [{"role": "user", "content": [source]}],
        Mode.TOOLS,
        autodetect_images=True,
    )

    assert converted[0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}
    ]


def test_wav_alias_preserves_bytes_in_genai_content(wav_bytes: bytes) -> None:
    pytest.importorskip("google.genai")
    encoded = base64.b64encode(wav_bytes).decode()
    audio = Audio.from_base64(f"data:audio/x-wav;base64,{encoded}")

    part = audio.to_genai()

    assert part.inline_data.mime_type == "audio/wav"
    assert part.inline_data.data == wav_bytes


def test_unsupported_audio_data_uri_still_raises(wav_bytes: bytes) -> None:
    encoded = base64.b64encode(wav_bytes).decode()
    with pytest.raises(ValueError, match="Unsupported audio format"):
        Audio.from_base64(f"data:video/mp4;base64,{encoded}")


def test_aac_alias_data_uri_is_detected_and_normalized() -> None:
    source = "data:audio/vnd.dlna.adts;base64,AA=="

    assert Audio.is_base64(source)
    audio = Audio.autodetect(source)
    assert audio.media_type == "audio/aac"
    assert audio.data == "AA=="
    assert audio.source == source
    assert autodetect_media(source) == audio
    with pytest.raises(ValueError, match="Expected WAV or MP3"):
        audio.to_openai(Mode.TOOLS)


def test_audio_path_with_unknown_media_type(tmp_path: Path, wav_bytes: bytes) -> None:
    path = tmp_path / "clip.instructor-no-mime"
    path.write_bytes(wav_bytes)
    with pytest.raises(ValueError, match="Unsupported audio format"):
        Audio.from_path(path)


@pytest.mark.parametrize(
    "filename, media_type",
    [
        ("song.flac", "audio/flac"),
        ("clip.m4a", "audio/m4a"),
        ("clip.aac", "audio/aac"),
    ],
)
def test_audio_path_uses_canonical_media_type(
    filename: str, media_type: str, tmp_path: Path
) -> None:
    path = tmp_path / filename
    path.write_bytes(b"\x00")

    assert mimetypes.guess_type(str(path))[0] == media_type
    audio = Audio.from_path(path)
    assert audio.media_type == media_type
    assert audio.data == "AA=="
    assert Audio.autodetect(path) == audio
    assert autodetect_media(str(path)) == audio


@pytest.mark.parametrize(
    "alias, media_type",
    [
        ("audio/x-flac", "audio/flac"),
        ("audio/mp4a-latm", "audio/m4a"),
        ("audio/x-aac", "audio/aac"),
        ("audio/x-m4a", "audio/m4a"),
    ],
)
def test_audio_alias_data_uri_is_detected_and_normalized(
    alias: str, media_type: str
) -> None:
    source = f"data:{alias};base64,AA=="

    assert Audio.is_base64(source)
    audio = Audio.from_base64(source)
    assert audio.media_type == media_type
    assert audio.data == "AA=="
    assert audio.source == source
    assert Audio.autodetect(source) == audio
    assert autodetect_media(source) == audio


@pytest.mark.parametrize(
    "filename, alias, media_type",
    [
        ("song.flac", "audio/x-flac", "audio/flac"),
        ("clip.m4a", "audio/mp4a-latm", "audio/m4a"),
        ("clip.aac", "audio/x-aac", "audio/aac"),
    ],
)
def test_audio_path_normalizes_platform_aliases(
    filename: str,
    alias: str,
    media_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / filename
    path.write_bytes(b"\x00")
    monkeypatch.setattr(
        mimetypes, "guess_type", lambda *_args, **_kwargs: (alias, None)
    )

    audio = Audio.from_path(path)
    assert audio.media_type == media_type
    assert audio.data == "AA=="
