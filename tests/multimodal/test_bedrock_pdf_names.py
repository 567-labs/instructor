"""Bedrock document names conform to the public DocumentBlock constraints."""

import base64
from pathlib import Path

import pytest

from instructor.processing.multimodal import PDF


PDF_BYTES = b"%PDF-1.4\ntest content"
PDF_DATA = base64.b64encode(PDF_BYTES).decode("ascii")


@pytest.mark.parametrize("s3_source", [False, True])
@pytest.mark.parametrize(
    "name,expected",
    [
        ("monthly_report", "monthlyreport"),
        ("___", "document"),
        ("!!!", "document"),
        ("", "document"),
        (" \t\n ", "document"),
        ("x" * 201, "x" * 200),
        ("x" * 200, "x" * 200),
        ("x" * 199 + "   tail", "x" * 199),
        ("valid - (2026) [final]", "valid - (2026) [final]"),
        ("report \t  final", "report final"),
        ("R\u00e9sum\u00e9_2026", "R\u00e9sum\u00e92026"),
    ],
)
def test_bedrock_pdf_name_constraints(
    name: str, expected: str, s3_source: bool
) -> None:
    """Sanitize names while preserving byte and S3 document payloads."""
    pdf = PDF(
        source="s3://bucket/report.pdf" if s3_source else "inline-pdf",
        data=None if s3_source else PDF_DATA,
    )
    original = pdf.model_dump()

    document = pdf.to_bedrock(name=name)["document"]

    assert document["name"] == expected
    assert document["format"] == "pdf"
    assert document["source"] == (
        {"s3Location": {"uri": pdf.source}} if s3_source else {"bytes": PDF_BYTES}
    )
    assert pdf.model_dump() == original


@pytest.mark.parametrize("string_path", [False, True])
def test_bedrock_pdf_name_from_local_path(tmp_path: Path, string_path: bool) -> None:
    """A file name with underscores must be sanitized after extraction."""
    path = tmp_path / "monthly_report.pdf"
    path.write_bytes(PDF_BYTES)
    pdf = PDF.from_path(str(path) if string_path else path)

    document = pdf.to_bedrock()["document"]

    assert document["name"] == "monthlyreportpdf"
    assert document["source"] == {"bytes": PDF_BYTES}


@pytest.mark.parametrize("scheme", ["https", "gs"])
def test_bedrock_pdf_name_from_url_with_data(scheme: str) -> None:
    """URL names use the same sanitizer without changing supplied PDF bytes."""
    pdf = PDF(source=f"{scheme}://example.com/monthly_report.pdf", data=PDF_DATA)

    document = pdf.to_bedrock()["document"]

    assert document["name"] == "monthlyreportpdf"
    assert document["source"] == {"bytes": PDF_BYTES}


def test_bedrock_pdf_s3_default_name() -> None:
    """The existing neutral default and S3 source remain unchanged."""
    pdf = PDF(source="s3://bucket/monthly_report.pdf")

    assert pdf.to_bedrock()["document"] == {
        "format": "pdf",
        "name": "document",
        "source": {"s3Location": {"uri": pdf.source}},
    }
