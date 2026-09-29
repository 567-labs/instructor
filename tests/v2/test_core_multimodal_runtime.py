from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from instructor import Mode
from instructor.v2.core import multimodal as multimodal_module
from instructor.v2.core.multimodal import (
    Audio,
    Image,
    ImageWithCacheControl,
    PDF,
    autodetect_media,
    convert_contents,
    convert_messages,
)
from instructor.v2.core.remote import RemoteContent


def test_convert_contents_uses_responses_text_shape() -> None:
    assert convert_contents("hello", Mode.RESPONSES_TOOLS) == "hello"
    assert convert_contents(["hello"], Mode.RESPONSES_TOOLS) == [
        {"type": "input_text", "text": "hello"}
    ]


def test_convert_contents_routes_pdf_to_mistral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = PDF(
        source="https://example.com/doc.pdf",
        media_type="application/pdf",
        data=None,
    )
    monkeypatch.setattr(PDF, "to_mistral", lambda _self: {"target": "mistral"})
    monkeypatch.setattr(PDF, "to_openai", lambda _self, _mode: {"target": "openai"})

    assert convert_contents(pdf, Mode.MISTRAL_TOOLS) == [{"target": "mistral"}]


def test_convert_contents_rejects_unknown_object() -> None:
    with pytest.raises(ValueError, match="Unsupported content type"):
        convert_contents(
            [object()],  # ty: ignore[invalid-argument-type]
            Mode.TOOLS,
        )


@pytest.mark.parametrize(
    ("media_type", "autodetect"),
    [
        ("audio", Audio.autodetect),
        ("PDF", PDF.autodetect),
    ],
)
def test_autodetect_rejects_unsupported_source_types(
    media_type: str, autodetect: Any
) -> None:
    with pytest.raises(
        ValueError, match=rf"Unsupported {media_type} source type: bytes"
    ):
        autodetect(b"not a string or path")


def test_image_autodetect_rejects_unsupported_source_type() -> None:
    invalid_source: Any = {"not": "a string or path"}

    with pytest.raises(ValueError, match="Unsupported image source type: dict"):
        Image.autodetect(invalid_source)


@pytest.mark.parametrize(
    ("source", "media_type"),
    [
        (b"\xff\xd8\xff\xe0" + b"\x00" * 100, "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png"),
        (b"GIF89a" + b"\x00" * 100, "image/gif"),
        (b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 100, "image/webp"),
    ],
)
def test_image_autodetect_accepts_raw_bytes(source: bytes, media_type: str) -> None:
    image = Image.autodetect(source)

    assert image.media_type == media_type
    assert image.data is not None


def test_image_autodetect_rejects_unknown_byte_content() -> None:
    with pytest.raises(ValueError, match="Invalid or unsupported base64 image data"):
        Image.autodetect(b"not an image")


def test_webp_mime_type_is_registered() -> None:
    import mimetypes

    assert mimetypes.guess_type("image.webp")[0] == "image/webp"


def test_autodetect_media_uses_mime_type_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image(source="file.png", media_type="image/png", data="AA==")
    monkeypatch.setattr("mimetypes.guess_type", lambda _: ("image/png", None))
    monkeypatch.setattr(Image, "autodetect_safely", lambda _source: image)

    def fail_audio_autodetect(_source: str) -> None:
        raise AssertionError("audio autodetect should not be called")

    monkeypatch.setattr(Audio, "autodetect_safely", fail_audio_autodetect)

    assert autodetect_media("picture.png") is image


def test_convert_messages_autodetects_image_params_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = ImageWithCacheControl(
        source="data:image/png;base64,AA==",
        media_type="image/png",
        data="AA==",
        cache_control={"type": "ephemeral"},
    )
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        ImageWithCacheControl,
        "from_image_params",
        classmethod(lambda _cls, _params: image),
    )
    monkeypatch.setattr(
        "instructor.v2.core.multimodal.autodetect_media",
        lambda value: f"detected:{value}",
    )

    def fake_convert_contents(contents: Any, mode: Mode) -> list[dict[str, str]]:
        seen["contents"] = contents
        seen["mode"] = mode
        return [{"type": "text", "text": "converted"}]

    monkeypatch.setattr(
        "instructor.v2.core.multimodal.convert_contents", fake_convert_contents
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                "photo.png",
                {"type": "image", "source": "data:image/png;base64,AA=="},
            ],
            "name": "tester",
        }
    ]

    converted = convert_messages(messages, Mode.TOOLS, autodetect_images=True)

    assert seen["contents"] == ["detected:photo.png", image]
    assert seen["mode"] == Mode.TOOLS
    assert converted == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "converted"}],
            "name": "tester",
        }
    ]


def test_convert_messages_rejects_unknown_typed_message() -> None:
    with pytest.raises(ValueError, match="Unsupported message type"):
        convert_messages(
            [{"type": "video", "role": "user", "content": "x"}], Mode.TOOLS
        )


def test_pdf_to_bedrock_uses_s3_source_and_sanitizes_name() -> None:
    pdf = PDF(source="s3://bucket/folder/My Report (Final)!.pdf", data=None)

    result = pdf.to_bedrock(name="My Report (Final)!.pdf")

    assert result["document"]["name"] == "My Report (Final)pdf"
    assert result["document"]["source"]["s3Location"]["uri"] == pdf.source


def test_pdf_to_bedrock_decodes_base64_bytes() -> None:
    encoded = base64.b64encode(b"%PDF-1.7 demo").decode("utf-8")
    pdf = PDF(source="data:application/pdf;base64," + encoded, data=encoded)

    result = pdf.to_bedrock(name="Doc")

    assert result["document"]["name"] == "Doc"
    assert result["document"]["source"]["bytes"] == b"%PDF-1.7 demo"


def test_pdf_to_bedrock_requires_data_for_non_s3_source() -> None:
    pdf = PDF(source="https://example.com/report.pdf", data=None)

    with pytest.raises(ValueError, match="PDF data is missing"):
        pdf.to_bedrock()


def test_audio_from_path_normalizes_windows_wav_and_aac_mime_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"RIFFdemo")
    aac_path = tmp_path / "clip.aac"
    aac_path.write_bytes(b"demo")

    mime_types = {
        str(wav_path): ("audio/x-wav", None),
        str(aac_path): ("audio/vnd.dlna.adts", None),
    }
    monkeypatch.setattr("mimetypes.guess_type", lambda path: mime_types[str(path)])

    wav_audio = Audio.from_path(wav_path)
    aac_audio = Audio.from_path(aac_path)

    assert wav_audio.media_type == "audio/wav"
    assert aac_audio.media_type == "audio/aac"


def test_audio_from_url_raises_value_error_for_unsupported_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        multimodal_module,
        "fetch_remote_content",
        lambda url, **_kwargs: RemoteContent(
            content=b"\x00\x01\x02",
            content_type="video/mp4",
            url=url,
        ),
    )

    with pytest.raises(ValueError, match="Unsupported audio format"):
        Audio.from_url("https://example.com/clip.mp4")


def test_audio_from_path_raises_value_error_for_unsupported_format(
    tmp_path: Path,
) -> None:
    txt_path = tmp_path / "clip.txt"
    txt_path.write_bytes(b"not audio")

    with pytest.raises(ValueError, match="Unsupported audio format"):
        Audio.from_path(txt_path)


def test_audio_from_path_raises_value_error_for_empty_file(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.wav"
    empty_path.write_bytes(b"")

    with pytest.raises(ValueError, match="Audio file is empty"):
        Audio.from_path(empty_path)


def test_audio_from_path_raises_file_not_found_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        Audio.from_path(missing_path)


def test_audio_to_openai_format_follows_media_type() -> None:
    from instructor.v2.providers.openai.multimodal import audio_to_openai

    mp3 = Audio(source="clip.mp3", media_type="audio/mpeg", data="ZmFrZQ==")
    assert audio_to_openai(mp3, Mode.TOOLS)["input_audio"]["format"] == "mp3"

    wav = Audio(source="clip.wav", media_type="audio/wav", data="ZmFrZQ==")
    assert audio_to_openai(wav, Mode.TOOLS)["input_audio"]["format"] == "wav"

    aac = Audio(source="clip.aac", media_type="audio/aac", data="ZmFrZQ==")
    with pytest.raises(ValueError, match="Expected WAV or MP3"):
        audio_to_openai(aac, Mode.TOOLS)


@pytest.mark.parametrize("padding", [0, 1024, 8192])
def test_pdf_autodetect_accepts_raw_base64(padding: int) -> None:
    body = b"%PDF-1.7\n" + b" " * padding + b"\n%%EOF"
    encoded = base64.b64encode(body).decode("ascii")

    pdf = PDF.autodetect(encoded)

    assert pdf.media_type == "application/pdf"
    assert pdf.source == encoded
    assert pdf.data == encoded


def test_autodetect_media_recognizes_long_raw_base64_pdf() -> None:
    encoded = base64.b64encode(b"%PDF-1.7\n" + b" " * 1024).decode("ascii")

    result = autodetect_media(encoded)

    assert isinstance(result, PDF)
    assert result.data == encoded


def test_convert_messages_sends_long_raw_base64_as_pdf() -> None:
    encoded = base64.b64encode(b"%PDF-1.7\n" + b" " * 1024).decode("ascii")
    messages = [{"role": "user", "content": [encoded]}]

    converted = convert_messages(messages, Mode.ANTHROPIC_TOOLS, autodetect_images=True)

    assert converted[0]["content"] == [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": encoded,
            },
        }
    ]
    assert messages == [{"role": "user", "content": [encoded]}]
