"""Google GenAI-specific multimodal encoders."""

from __future__ import annotations

import base64
from typing import Any

from instructor.v2.core.multimodal import Audio, Image, PDF, autodetect_media
from instructor.v2.core.remote import (
    MAX_IMAGE_BYTES,
    MAX_PDF_BYTES,
    fetch_remote_content,
)


def _types() -> Any:
    try:
        from google.genai import types
    except ImportError as err:
        raise ImportError(
            "google-genai package is required for GenAI integration. Install with: pip install google-genai"
        ) from err
    return types


def image_to_genai(image: Any) -> Any:
    types = _types()
    if isinstance(image.source, str) and image.source.startswith("gs://"):
        return types.Part.from_bytes(data=image.data, mime_type=image.media_type)
    if isinstance(image.source, str) and image.source.startswith(
        ("http://", "https://")
    ):
        response = fetch_remote_content(
            image.source,
            max_bytes=MAX_IMAGE_BYTES,
            timeout=30,
        )
        return types.Part.from_bytes(data=response.content, mime_type=image.media_type)
    if image.data or image.is_base64(str(image.source)):
        data = image.data or str(image.source).split(",", 1)[1]
        return types.Part.from_bytes(
            data=base64.b64decode(data),
            mime_type=image.media_type,
        )
    raise ValueError("Image data is missing for base64 encoding.")


def audio_to_genai(audio: Any) -> Any:
    types = _types()
    return types.Part.from_bytes(
        data=base64.b64decode(audio.data),
        mime_type=audio.media_type,
    )


def pdf_to_genai(pdf: Any) -> Any:
    types = _types()
    if (
        isinstance(pdf.source, str)
        and pdf.source.startswith(("http://", "https://"))
        and not pdf.data
    ):
        data = fetch_remote_content(
            pdf.source,
            max_bytes=MAX_PDF_BYTES,
            timeout=30,
        ).content
        encoded = base64.b64encode(data).decode("utf-8")
        return types.Part.from_bytes(
            data=base64.b64decode(encoded),
            mime_type=pdf.media_type,
        )
    if pdf.data:
        return types.Part.from_bytes(
            data=base64.b64decode(pdf.data),
            mime_type=pdf.media_type,
        )
    raise ValueError("Unsupported PDF format")


def upload_new_pdf_file(
    cls: type[Any], file_path: str, retry_delay: int = 10, max_retries: int = 20
) -> Any:
    from google.genai import Client
    from google.genai.types import FileState
    import time

    client = Client()
    file = client.files.upload(file=file_path)
    while file.state != FileState.ACTIVE:
        file_name = file.name
        if file_name is None:
            raise ValueError("Cannot poll a pending GenAI file without a file name")
        if max_retries <= 0:
            raise Exception(
                "Max retries reached. File upload has been started but is still pending"
            )
        time.sleep(retry_delay)
        file = client.files.get(name=file_name)
        max_retries -= 1
    return cls(source=file.uri, media_type=file.mime_type, data=None)


def load_existing_pdf_file(cls: type[Any], file_name: str) -> Any:
    from google.genai import Client, types
    from google.genai.types import FileState

    client = Client()
    file = client.files.get(name=file_name)
    if file.source == types.FileSource.UPLOADED and file.state == FileState.ACTIVE:
        return cls(source=file.uri, media_type=file.mime_type, data=None)
    raise ValueError("We only support uploaded PDFs for now")


def uploaded_pdf_to_genai(pdf: Any) -> Any:
    types = _types()
    if (
        pdf.source
        and isinstance(pdf.source, str)
        and "https://generativelanguage.googleapis.com/v1beta/files/" in pdf.source
    ):
        return types.Part.from_uri(file_uri=pdf.source, mime_type=pdf.media_type)
    return pdf_to_genai(pdf)


def extract_multimodal_content(
    contents: list[Any],
    autodetect_images: bool = True,
) -> list[Any]:
    """Convert typed Google GenAI contents, auto-detecting media when needed."""
    types = _types()
    result: list[Any] = []
    for content in contents:
        if isinstance(content, types.File):
            result.append(content)
            continue
        if not isinstance(content, types.Content):
            raise ValueError(
                f"Unsupported content type: {type(content)}. This should only be used for the Google types"
            )
        converted_contents: list[Any] = []
        if not content.parts:
            raise ValueError("Content parts are empty")
        for content_part in content.parts:
            if content_part.text and autodetect_images:
                converted_item = autodetect_media(content_part.text)
                if isinstance(converted_item, (Image, Audio, PDF)):
                    converted_contents.append(converted_item.to_genai())
                    continue
            converted_contents.append(content_part)
        result.append(types.Content(parts=converted_contents, role=content.role))
    return result
