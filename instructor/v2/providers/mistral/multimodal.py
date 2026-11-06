"""Mistral-specific multimodal encoders."""

from __future__ import annotations

from typing import Any


def pdf_to_mistral(pdf: Any) -> dict[str, Any]:
    if (
        isinstance(pdf.source, str)
        and pdf.source.startswith(("http://", "https://"))
        and not pdf.data
    ):
        return {"type": "document_url", "document_url": pdf.source}
    raise ValueError("Mistral only supports document URLs for now")
