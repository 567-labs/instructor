"""Anthropic-specific multimodal encoders."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from instructor.v2.core.remote import MAX_PDF_BYTES, fetch_remote_content


def image_to_anthropic(image: Any) -> dict[str, Any]:
    if (
        isinstance(image.source, str)
        and image.source.startswith(("http://", "https://"))
        and not image.data
    ):
        image.data = image.url_to_base64(image.source)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": image.media_type,
            "data": image.data,
        },
    }


def _read_local_pdf(source: Any) -> bytes | None:
    """Return the bytes of ``source`` when it points at a readable local file.

    Returns ``None`` when the source is not a usable filesystem path — for
    example an inline base64 payload or a non-HTTP remote URI — so callers can
    raise a clear error instead of failing deeper in the stack.
    """
    try:
        path = Path(source)
    except TypeError:
        return None
    try:
        if not path.is_file():
            return None
        return path.read_bytes()
    except OSError:
        # Inline base64 payloads and remote URIs are not valid path names.
        return None


def pdf_to_anthropic(pdf: Any) -> dict[str, Any]:
    if (
        isinstance(pdf.source, str)
        and pdf.source.startswith(("http://", "https://"))
        and not pdf.data
    ):
        return {"type": "document", "source": {"type": "url", "url": pdf.source}}
    if not pdf.data:
        local_bytes = _read_local_pdf(pdf.source)
        if local_bytes is None:
            if not isinstance(pdf.source, str) or not pdf.source.lower().startswith(
                ("http://", "https://")
            ):
                raise ValueError("PDF data is missing for base64 encoding.")
            response = fetch_remote_content(str(pdf.source), max_bytes=MAX_PDF_BYTES)
            local_bytes = response.content
        pdf.data = base64.b64encode(local_bytes).decode("utf-8")
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": pdf.media_type,
            "data": pdf.data,
        },
    }


def image_with_cache_control_to_anthropic(image: Any) -> dict[str, Any]:
    result = image_to_anthropic(image)
    if image.cache_control:
        result["cache_control"] = image.cache_control
    return result


def pdf_with_cache_control_to_anthropic(pdf: Any) -> dict[str, Any]:
    result = pdf_to_anthropic(pdf)
    result["cache_control"] = {"type": "ephemeral"}
    return result


def audio_to_anthropic(_audio: Any) -> dict[str, Any]:
    raise NotImplementedError("Anthropic is not supported yet")
