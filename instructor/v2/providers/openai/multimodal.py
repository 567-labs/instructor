"""OpenAI-specific multimodal encoders."""

from __future__ import annotations

import base64
from typing import Any

from instructor.v2.core.mode import Mode
from instructor.v2.core.remote import MAX_PDF_BYTES, fetch_remote_content

RESPONSES_MODES = {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}


def image_to_openai(image: Any, mode: Mode) -> dict[str, Any]:
    image_type = "input_image" if mode in RESPONSES_MODES else "image_url"
    if (
        isinstance(image.source, str)
        and image.source.startswith(("http://", "https://"))
        and not image.is_base64(image.source)
    ):
        if mode in RESPONSES_MODES:
            return {"type": "input_image", "image_url": image.source}
        return {"type": image_type, "image_url": {"url": image.source}}
    if image.data or image.is_base64(str(image.source)):
        data = image.data or str(image.source).split(",", 1)[1]
        if mode in RESPONSES_MODES:
            return {
                "type": "input_image",
                "image_url": f"data:{image.media_type};base64,{data}",
            }
        return {
            "type": image_type,
            "image_url": {"url": f"data:{image.media_type};base64,{data}"},
        }
    raise ValueError("Image data is missing for base64 encoding.")


def audio_to_openai(audio: Any, mode: Mode) -> dict[str, Any]:
    if mode in RESPONSES_MODES:
        raise ValueError("OpenAI Responses doesn't support audio")

    # Chat Completions accepts only WAV or MP3. Reject other formats instead of
    # sending mislabeled bytes that the API cannot decode.
    media_type = (getattr(audio, "media_type", "") or "").lower()
    format_by_media_type = {
        "audio/mp3": "mp3",
        "audio/mpeg": "mp3",
        "audio/mpga": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
    }
    if media_type not in format_by_media_type:
        raise ValueError(
            f"Unsupported OpenAI audio format: {media_type or 'unknown'}. "
            "Expected WAV or MP3."
        )

    return {
        "type": "input_audio",
        "input_audio": {
            "data": audio.data,
            "format": format_by_media_type[media_type],
        },
    }


def pdf_to_openai(pdf: Any, mode: Mode) -> dict[str, Any]:
    input_file_type = "input_file" if mode in RESPONSES_MODES else "file"
    if (
        isinstance(pdf.source, str)
        and pdf.source.startswith(("http://", "https://"))
        and not pdf.data
    ):
        response = fetch_remote_content(
            pdf.source,
            max_bytes=MAX_PDF_BYTES,
            timeout=30,
        )
        data = base64.b64encode(response.content).decode("utf-8")
        if mode in RESPONSES_MODES:
            return {
                "type": input_file_type,
                "filename": pdf.source,
                "file_data": f"data:{pdf.media_type};base64,{data}",
            }
        return {
            "type": input_file_type,
            "file": {
                "filename": pdf.source,
                "file_data": f"data:{pdf.media_type};base64,{data}",
            },
        }
    if pdf.data or pdf.is_base64(str(pdf.source)):
        data = pdf.data or str(pdf.source).split(",", 1)[1]
        filename = pdf.source if isinstance(pdf.source, str) else str(pdf.source)
        if mode in RESPONSES_MODES:
            return {
                "type": input_file_type,
                "filename": filename,
                "file_data": f"data:{pdf.media_type};base64,{data}",
            }
        return {
            "type": input_file_type,
            "file": {
                "filename": filename,
                "file_data": f"data:{pdf.media_type};base64,{data}",
            },
        }
    raise ValueError("PDF data is missing for base64 encoding.")
