from __future__ import annotations
import base64
import pytest


@pytest.fixture(scope="session")
def tiny_png_bytes() -> bytes:
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        b"ASsJTYQAAAAASUVORK5CYII="
    )


@pytest.fixture(scope="session")
def tiny_png_data_url(tiny_png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(tiny_png_bytes).decode("utf-8")


@pytest.fixture(scope="session")
def tiny_pdf_bytes() -> bytes:
    return base64.b64decode(
        b"JVBERi0xLjQKJSVPRgoAAAAQAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    )
