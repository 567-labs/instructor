import base64

from instructor.v2.core.multimodal import Audio, PDF, Image


PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 32
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _data_uri(media_type: str, payload: bytes) -> str:
    return f"data:{media_type};base64,{base64.b64encode(payload).decode()}"


def test_pdf_from_url_accepts_data_uri():
    pdf = PDF.from_url(_data_uri("application/pdf", PDF_BYTES))

    assert pdf.source.startswith("data:application/pdf;base64,")
    assert pdf.media_type == "application/pdf"


def test_audio_from_url_accepts_data_uri():
    audio = Audio.from_url(_data_uri("audio/mpeg", MP3_BYTES))

    assert audio.source.startswith("data:audio/mpeg;base64,")
    assert audio.media_type == "audio/mpeg"


def test_image_from_url_accepts_data_uri():
    image = Image.from_url(_data_uri("image/png", PNG_BYTES))

    assert image.source.startswith("data:image/png;base64,")
    assert image.media_type == "image/png"
