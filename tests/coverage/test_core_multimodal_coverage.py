from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import requests

from instructor import Mode
import instructor.v2.core.multimodal as multimodal_module
from instructor.v2.core.errors import MultimodalError
from instructor.v2.core.multimodal import (
    Audio,
    Image,
    ImageWithCacheControl,
    PDF,
    PDFWithGenaiFile,
    _normalize_media_type,
    autodetect_media,
    convert_messages,
)
from instructor.v2.core.remote import RemoteContent


@pytest.fixture(autouse=True)
def isolate_multimodal_http(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def fetch(url: str, *, max_bytes: int, timeout: int) -> RemoteContent:  # noqa: ARG001
        result = requests.get(url, timeout=timeout)
        result.raise_for_status()
        return RemoteContent(
            content=result.content,
            content_type=_normalize_media_type(result.headers.get("Content-Type")),
            url=url,
        )

    def probe(url: str, *, timeout: int) -> str | None:
        result = requests.head(url, allow_redirects=True, timeout=timeout)
        result.raise_for_status()
        return _normalize_media_type(result.headers.get("Content-Type"))

    monkeypatch.setattr(multimodal_module, "fetch_remote_content", fetch)
    monkeypatch.setattr(multimodal_module, "probe_remote_content_type", probe)
    yield


def response(body: bytes, media_type: str, status: int = 200) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result._content = body
    result.headers["Content-Type"] = media_type
    return result


def test_image_gcs_loading_and_autodetection(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"\x89PNG\r\n\x1a\nimage"
    calls: list[tuple[str, int]] = []

    def get(url: str, timeout: int = 30) -> requests.Response:
        calls.append((url, timeout))
        return response(body, "image/png")

    monkeypatch.setattr(requests, "get", get)
    direct = Image.from_gs_url("gs://pictures/photo.png", timeout=7)
    detected = Image.autodetect("gs://pictures/photo.png")
    via_url = Image.from_url("gs://pictures/photo.png")

    assert direct.data == base64.b64encode(body).decode()
    assert detected.media_type == via_url.media_type == "image/png"
    assert calls == [
        ("https://storage.googleapis.com/pictures/photo.png", 7),
        ("https://storage.googleapis.com/pictures/photo.png", 30),
        ("https://storage.googleapis.com/pictures/photo.png", 30),
    ]


def test_image_gcs_rejects_invalid_sources_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="URL must start with gs://"):
        Image.from_gs_url("https://example.test/photo.png")

    monkeypatch.setattr(
        requests, "get", lambda *_args, **_kwargs: response(b"text", "text/plain")
    )
    with pytest.raises(ValueError, match="Unsupported image format: text/plain"):
        Image.from_gs_url("gs://pictures/readme.txt")

    monkeypatch.setattr(
        requests,
        "get",
        lambda *_args, **_kwargs: response(b"missing", "image/png", status=404),
    )
    with pytest.raises(ValueError, match="Failed to access GCS image"):
        Image.from_gs_url("gs://pictures/missing.png")


def test_image_url_errors_missing_path_and_unsupported_data_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        requests,
        "head",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.ConnectionError()),
    )
    with pytest.raises(ValueError, match="Failed to fetch image from URL"):
        Image.from_url("https://example.test/no-extension")

    with pytest.raises(ValueError, match="Unsupported image format: text/plain"):
        Image.from_url("https://example.test/readme.txt")

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        Image.from_path(tmp_path / "missing.png")

    with pytest.raises(MultimodalError, match="Unsupported image format: image/tiff"):
        Image.from_base64("data:image/tiff;base64,aW1hZ2U=")


def test_remote_media_types_ignore_http_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _normalize_media_type(None) is None

    def head(url: str, **_kwargs: Any) -> requests.Response:
        media_type = "image/png" if "photo" in url else "application/pdf"
        return response(b"", f"{media_type}; charset=binary")

    def get(_url: str, **_kwargs: Any) -> requests.Response:
        return response(b"audio", "Audio/WAV; codecs=1")

    monkeypatch.setattr(requests, "head", head)
    monkeypatch.setattr(requests, "get", get)

    image = Image.from_url("https://example.test/photo")
    audio = Audio.from_url("https://example.test/audio")
    pdf = PDF.from_url("https://example.test/report")

    assert image.media_type == "image/png"
    assert audio.media_type == "audio/wav"
    assert pdf.media_type == "application/pdf"


def test_data_uri_media_types_are_case_insensitive() -> None:
    # RFC 2396: media types are case-insensitive, and optional parameters are
    # allowed after them, so these are all equivalent to `data:image/png`.
    for media_type in ("image/PNG", "IMAGE/PNG", "Image/Png"):
        image = Image.from_base64(f"data:{media_type};base64,aW1hZ2U=")

        assert image.media_type == "image/png"
        assert image.data == "aW1hZ2U="

    for media_type in ("application/PDF", "APPLICATION/Pdf"):
        pdf = PDF.from_base64(f"data:{media_type};base64,JVBERi0=")

        assert pdf.media_type == "application/pdf"
        assert pdf.data == "JVBERi0="


def test_data_uri_media_types_still_reject_unsupported_types() -> None:
    with pytest.raises(MultimodalError, match="Unsupported image format: image/tiff"):
        Image.from_base64("data:image/TIFF;base64,aW1hZ2U=")

    with pytest.raises(ValueError, match="Unsupported PDF format: image/png"):
        PDF.from_base64("data:image/PNG;base64,JVBERi0=")


def test_data_uri_with_parameters_keeps_payload_intact() -> None:
    image = Image.from_base64("data:image/PNG;charset=utf-8;base64,aW1hZ2U=")

    assert image.media_type == "image/png"
    assert image.data == "aW1hZ2U="
    assert image.source == "data:image/PNG;charset=utf-8;base64,aW1hZ2U="


def test_image_path_probe_falls_back_to_raw_base64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = base64.b64encode(b"\xff\xd8\xffimage").decode()
    monkeypatch.setattr(Path, "is_file", lambda _self: (_ for _ in ()).throw(OSError()))

    image = Image.autodetect(encoded)

    assert image.media_type == "image/jpeg"
    assert image.data == encoded


@pytest.mark.parametrize(
    ("body", "media_type"),
    [
        (b"GIF87aimage", "image/gif"),
        (b"GIF89aimage", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPimage", "image/webp"),
    ],
)
def test_image_raw_base64_detects_supported_signatures(
    body: bytes, media_type: str
) -> None:
    encoded = base64.b64encode(body).decode()

    image = Image.from_raw_base64(encoded)

    assert image.media_type == media_type
    assert image.source == image.data == encoded


def test_image_raw_base64_rejects_non_webp_riff_data() -> None:
    encoded = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEaudio").decode()

    with pytest.raises(ValueError, match="Invalid or unsupported base64 image data"):
        Image.from_raw_base64(encoded)


def test_image_url_to_base64_does_not_cache_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def get(url: str, *, timeout: int) -> requests.Response:
        calls.append((url, timeout))
        return response(b"image-bytes", "image/png")

    monkeypatch.setattr(requests, "get", get)
    first = Image.url_to_base64("https://example.test/photo.png")
    second = Image.url_to_base64("https://example.test/photo.png")

    assert first == second == base64.b64encode(b"image-bytes").decode()
    assert calls == [
        ("https://example.test/photo.png", 30),
        ("https://example.test/photo.png", 30),
    ]


def test_audio_autodetects_url_gcs_string_path_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFFclip")

    def get(url: str, **_kwargs: Any) -> requests.Response:
        return response(url.encode(), "audio/wav")

    monkeypatch.setattr(requests, "get", get)

    http = Audio.autodetect("https://example.test/clip.wav")
    gcs = Audio.autodetect("gs://sounds/clip.wav")
    from_string_path = Audio.autodetect(str(audio_path))
    from_path = Audio.autodetect(audio_path)
    gcs_from_url = Audio.from_url("gs://sounds/clip.wav")

    assert http.source == "https://example.test/clip.wav"
    assert gcs.source == gcs_from_url.source == "gs://sounds/clip.wav"
    assert (
        from_string_path.data
        == from_path.data
        == base64.b64encode(b"RIFFclip").decode()
    )
    assert {http.media_type, gcs.media_type, from_path.media_type} == {"audio/wav"}


def test_audio_invalid_inputs_and_conversions(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="Unsupported audio format: audio/ogg"):
        Audio.from_base64("data:audio/ogg;base64,YXVkaW8=")

    monkeypatch.setattr(Path, "is_file", lambda _self: (_ for _ in ()).throw(OSError()))
    with pytest.raises(ValueError, match="Unable to determine audio source"):
        Audio.autodetect("not-an-audio-file")

    audio = Audio(source="clip.wav", data="UklGRg==", media_type="audio/wav")
    with pytest.raises(NotImplementedError, match="Anthropic is not supported yet"):
        audio.to_anthropic()


def test_audio_gcs_rejects_invalid_sources_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="URL must start with gs://"):
        Audio.from_gs_url("https://example.test/clip.wav")

    monkeypatch.setattr(
        requests, "get", lambda *_args, **_kwargs: response(b"text", "text/plain")
    )
    with pytest.raises(ValueError, match="Unsupported audio format: text/plain"):
        Audio.from_gs_url("gs://sounds/readme.txt")

    monkeypatch.setattr(
        requests,
        "get",
        lambda *_args, **_kwargs: response(b"missing", "audio/wav", status=404),
    )
    with pytest.raises(ValueError, match="Failed to access GCS audio"):
        Audio.from_gs_url("gs://sounds/missing.wav")


def test_image_params_keep_cache_control_and_encode_anthropic_image() -> None:
    image = ImageWithCacheControl.from_image_params(
        {
            "type": "image",
            "source": "data:image/png;base64,iVBORw0KGgo=",
            "cache_control": {"type": "ephemeral"},
        }
    )

    assert image.to_anthropic() == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgo=",
        },
        "cache_control": {"type": "ephemeral"},
    }


def test_pdf_autodetects_gcs_path_and_raw_base64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"%PDF-1.7\nexample"
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(body)
    monkeypatch.setattr(
        requests, "get", lambda *_args, **_kwargs: response(body, "application/pdf")
    )
    gcs = PDF.autodetect("gs://reports/report.pdf")
    gcs_from_url = PDF.from_url("gs://reports/report.pdf")
    local = PDF.autodetect(pdf_path)
    raw = PDF.autodetect(base64.b64encode(body).decode())

    expected = base64.b64encode(body).decode()
    assert gcs.data == gcs_from_url.data == local.data == raw.data == expected
    assert {gcs.media_type, local.media_type, raw.media_type} == {"application/pdf"}


def test_pdf_autodetects_url_from_response_media_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool, int]] = []

    def head(
        url: str, allow_redirects: bool = False, *, timeout: int
    ) -> requests.Response:
        calls.append((url, allow_redirects, timeout))
        return response(b"", "application/pdf")

    monkeypatch.setattr(requests, "head", head)

    pdf = PDF.autodetect("https://example.test/reports/latest")

    assert pdf.source == "https://example.test/reports/latest"
    assert pdf.media_type == "application/pdf"
    assert pdf.data is None
    assert calls == [("https://example.test/reports/latest", True, 30)]


def test_pdf_rejects_raw_base64_with_non_pdf_content() -> None:
    encoded = base64.b64encode(b"plain text, not a PDF").decode()

    with pytest.raises(ValueError, match="Invalid or unsupported base64 PDF data"):
        PDF.from_raw_base64(encoded)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError("gone"), "PDF file not found"),
        (OSError(63, "too long"), "PDF file name too long"),
        (OSError(5, "read error"), "Unable to read PDF file"),
    ],
)
def test_pdf_autodetect_reports_path_probe_errors(
    error: OSError, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "is_file", lambda _self: (_ for _ in ()).throw(error))

    with pytest.raises(MultimodalError, match=message) as exc_info:
        PDF.autodetect("report.pdf")

    assert exc_info.value.content_type == "pdf"
    assert exc_info.value.file_path == "report.pdf"


def test_pdf_rejects_invalid_data_and_local_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.touch()
    text = tmp_path / "notes.txt"
    text.write_bytes(b"not pdf")

    with pytest.raises(ValueError, match="Unsupported PDF format: application/json"):
        PDF.from_base64("data:application/json;base64,e30=")
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        PDF.from_path(tmp_path / "missing.pdf")
    with pytest.raises(ValueError, match="PDF file is empty"):
        PDF.from_path(empty)
    with pytest.raises(ValueError, match="Unsupported PDF format: text/plain"):
        PDF.from_path(text)


def test_pdf_gcs_rejects_invalid_sources_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="URL must start with gs://"):
        PDF.from_gs_url("https://example.test/report.pdf")

    monkeypatch.setattr(
        requests, "get", lambda *_args, **_kwargs: response(b"text", "text/plain")
    )
    with pytest.raises(ValueError, match="Unsupported PDF format: text/plain"):
        PDF.from_gs_url("gs://reports/readme.txt")

    monkeypatch.setattr(
        requests,
        "get",
        lambda *_args, **_kwargs: response(b"missing", "application/pdf", status=404),
    )
    with pytest.raises(ValueError, match="Failed to access GCS PDF"):
        PDF.from_gs_url("gs://reports/missing.pdf")


def test_pdf_url_errors_and_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "head",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.ConnectionError()),
    )
    with pytest.raises(ValueError, match="Failed to fetch PDF from URL"):
        PDF.from_url("https://example.test/no-extension")

    with pytest.raises(ValueError, match="Unsupported PDF format: text/plain"):
        PDF.from_url("https://example.test/readme.txt")

    body = b"%PDF-1.7\nexample"
    encoded = base64.b64encode(body).decode()
    pdf = PDF(source="report.pdf", data=encoded)

    assert pdf.to_anthropic() == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": encoded,
        },
    }


def test_pdf_genai_and_file_helpers_forward_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import instructor.v2.providers.genai.multimodal as genai_multimodal

    calls: list[tuple[Any, ...]] = []

    def encode(pdf: PDF) -> dict[str, Any]:
        calls.append(("encode", pdf.source))
        return {"mime_type": pdf.media_type, "data": pdf.data}

    def upload(
        cls: type[PDFWithGenaiFile], file_path: str, retry_delay: int, max_retries: int
    ) -> PDFWithGenaiFile:
        calls.append(("upload", cls, file_path, retry_delay, max_retries))
        return cls(source="uploaded://new", data=None)

    def load(cls: type[PDFWithGenaiFile], file_name: str) -> PDFWithGenaiFile:
        calls.append(("load", cls, file_name))
        return cls(source="uploaded://existing", data=None)

    monkeypatch.setattr(genai_multimodal, "pdf_to_genai", encode)
    monkeypatch.setattr(genai_multimodal, "upload_new_pdf_file", upload)
    monkeypatch.setattr(genai_multimodal, "load_existing_pdf_file", load)

    pdf = PDF(source="report.pdf", data="JVBERi0xLjc=")
    new_file = PDFWithGenaiFile.from_new_genai_file("report.pdf", 2, 4)
    existing_file = PDFWithGenaiFile.from_existing_genai_file("files/report")

    assert pdf.to_genai() == {"mime_type": "application/pdf", "data": "JVBERi0xLjc="}
    assert new_file.source == "uploaded://new"
    assert existing_file.source == "uploaded://existing"
    assert calls == [
        ("upload", PDFWithGenaiFile, "report.pdf", 2, 4),
        ("load", PDFWithGenaiFile, "files/report"),
        ("encode", "report.pdf"),
    ]


def test_pdf_bedrock_falls_back_to_document_name_for_unknown_source() -> None:
    pdf = PDF.model_construct(
        source=object(), media_type="application/pdf", data="JVBERi0xLjc="
    )

    assert pdf.to_bedrock()["document"] == {
        "format": "pdf",
        "name": "document",
        "source": {"bytes": b"%PDF-1.7"},
    }


def test_autodetect_media_uses_existing_audio_pdf_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image(source="photo.png", media_type="image/png", data="aW1hZ2U=")
    audio = Audio(source="clip.wav", media_type="audio/wav", data="YXVkaW8=")
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"ID3clip")
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nexample")

    assert autodetect_media(image) is image
    assert isinstance(autodetect_media(audio_path), Audio)
    assert isinstance(autodetect_media(pdf_path), PDF)

    monkeypatch.setattr(Image, "autodetect_safely", lambda _source: "unknown")
    monkeypatch.setattr(Audio, "autodetect_safely", lambda _source: audio)

    assert autodetect_media("opaque-source") is audio


def test_convert_messages_keeps_typed_and_mixed_content_and_detects_single_image() -> (
    None
):
    typed = {"type": "audio", "role": "user", "content": "already converted"}
    image_params = {
        "type": "image",
        "source": "data:image/png;base64,iVBORw0KGgo=",
        "cache_control": {"type": "ephemeral"},
    }
    messages: list[dict[str, Any]] = [
        typed,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "keep me"},
                "data:audio/wav;base64,UklGRg==",
            ],
            "name": "speaker",
        },
        {"role": "user", "content": image_params},
    ]

    converted = convert_messages(messages, Mode.TOOLS, autodetect_images=True)

    assert converted[0] is typed
    assert converted[1] == {
        "role": "user",
        "name": "speaker",
        "content": [
            {"type": "text", "text": "keep me"},
            {
                "type": "input_audio",
                "input_audio": {"data": "UklGRg==", "format": "wav"},
            },
        ],
    }
    assert converted[2] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            }
        ],
    }


def test_convert_messages_keeps_a_single_detected_image() -> None:
    image = Image(
        source="data:image/png;base64,iVBORw0KGgo=",
        media_type="image/png",
        data="iVBORw0KGgo=",
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": image}]

    assert convert_messages(messages, Mode.TOOLS, autodetect_images=True) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                }
            ],
        }
    ]
