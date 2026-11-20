from __future__ import annotations

import base64
import io
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


@pytest.mark.parametrize("platform_db", [True, False], ids=["platform", "fallback"])
@pytest.mark.parametrize(
    ("suffix", "media_types"),
    # Linux maps .m4a to audio/mp4; other platforms report an m4a alias.
    [
        (".flac", {"audio/flac"}),
        (".m4a", {"audio/m4a", "audio/mp4"}),
        (".aac", {"audio/aac"}),
    ],
)
def test_audio_from_path_accepts_platform_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    media_types: set[str],
    platform_db: bool,
) -> None:
    if not platform_db:
        # Simulate a mimetypes database with no entry for the suffix.
        monkeypatch.setattr("mimetypes.guess_type", lambda _path: (None, None))
    path = tmp_path / f"clip{suffix}"
    path.write_bytes(b"audio-bytes")

    audio = Audio.from_path(path)

    assert audio.media_type in media_types
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
def test_audio_data_uri_aliases_are_normalized(alias: str, media_type: str) -> None:
    assert Audio.from_base64(f"data:{alias};base64,YWJj").media_type == media_type


def test_audio_direct_construction_with_data_uri_encodes_for_openai(
    wav_bytes: bytes,
) -> None:
    encoded = base64.b64encode(wav_bytes).decode()
    source = f"data:audio/wav;base64,{encoded}"
    audio = Audio(source=source, media_type="audio/wav")

    result = audio.to_openai(Mode.TOOLS)
    assert result == {
        "type": "input_audio",
        "input_audio": {
            "data": encoded,
            "format": "wav",
        },
    }


def test_audio_missing_data_raises_value_error() -> None:
    audio = Audio(source="http://example.com/audio.wav", media_type="audio/wav")
    with pytest.raises(ValueError, match="Audio data is missing for base64 encoding"):
        audio.to_openai(Mode.TOOLS)


def test_audio_direct_construction_with_data_uri_encodes_for_genai(
    wav_bytes: bytes,
) -> None:
    pytest.importorskip("google.genai")
    encoded = base64.b64encode(wav_bytes).decode()
    source = f"data:audio/wav;base64,{encoded}"
    audio = Audio(source=source, media_type="audio/wav")

    part = audio.to_genai()
    assert part.inline_data.mime_type == "audio/wav"
    assert part.inline_data.data == wav_bytes
