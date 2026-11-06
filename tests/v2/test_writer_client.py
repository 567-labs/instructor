"""Import tests for Writer v2 handlers."""

from __future__ import annotations


def test_writer_handlers_are_importable() -> None:
    from instructor.v2.providers.writer.handlers import (
        WriterJSONSchemaHandler,
        WriterMDJSONHandler,
        WriterToolsHandler,
    )

    assert WriterToolsHandler is not None
    assert WriterJSONSchemaHandler is not None
    assert WriterMDJSONHandler is not None
