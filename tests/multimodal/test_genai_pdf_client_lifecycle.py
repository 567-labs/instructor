from typing import Any, Literal

import pytest

genai = pytest.importorskip("google.genai")
from google.genai import types

from instructor.v2.core.multimodal import PDFWithGenaiFile


@pytest.mark.parametrize("operation", ["upload", "load"])
@pytest.mark.parametrize("outcome", ["active", "failed", "request_error"])
def test_pdf_helpers_manage_owned_client(
    monkeypatch: pytest.MonkeyPatch,
    operation: Literal["upload", "load"],
    outcome: Literal["active", "failed", "request_error"],
) -> None:
    client = genai.Client(api_key="local-test-key", vertexai=False)
    close = getattr(client, "close", None)
    http_client = getattr(client._api_client, "_httpx_client", None)
    monkeypatch.setattr(genai, "Client", lambda: client)
    file = types.File(
        name="files/test",
        uri="https://example.com/test.pdf",
        mime_type="application/pdf",
        source=types.FileSource.UPLOADED,
        state=types.FileState.FAILED if outcome == "failed" else types.FileState.ACTIVE,
    )

    def request(**_kwargs: Any) -> Any:
        if outcome == "request_error":
            raise RuntimeError("request failed")
        return file

    monkeypatch.setattr(client.files, "upload", request)
    monkeypatch.setattr(client.files, "get", request)
    try:
        if outcome == "active":
            result = (
                PDFWithGenaiFile.from_new_genai_file("test.pdf")
                if operation == "upload"
                else PDFWithGenaiFile.from_existing_genai_file("files/test")
            )
            assert result.source == file.uri
        else:
            with pytest.raises(
                RuntimeError if outcome == "request_error" else ValueError
            ):
                if operation == "upload":
                    PDFWithGenaiFile.from_new_genai_file("test.pdf")
                else:
                    PDFWithGenaiFile.from_existing_genai_file("files/test")
        if callable(close):
            assert http_client is not None
            assert http_client.is_closed
    finally:
        if callable(close):
            close()
