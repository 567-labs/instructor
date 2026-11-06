"""Test backwards compatibility of exception handling."""

import pytest
from instructor.core.exceptions import (
    InstructorError,
    ResponseParsingError,
    MultimodalError,
    AsyncValidationError,
)


def test_response_parsing_error_is_value_error():
    """Test that ResponseParsingError can be caught as ValueError."""
    with pytest.raises(ValueError):
        raise ResponseParsingError("Test error", mode="TOOLS")

    # Should also be catchable as InstructorError
    with pytest.raises(InstructorError):
        raise ResponseParsingError("Test error", mode="TOOLS")

    # And as the specific type
    with pytest.raises(ResponseParsingError):
        raise ResponseParsingError("Test error", mode="TOOLS")


def test_multimodal_error_is_value_error():
    """Test that MultimodalError can be caught as ValueError."""
    with pytest.raises(ValueError):
        raise MultimodalError("Test error", content_type="image")

    # Should also be catchable as InstructorError
    with pytest.raises(InstructorError):
        raise MultimodalError("Test error", content_type="image")

    # And as the specific type
    with pytest.raises(MultimodalError):
        raise MultimodalError("Test error", content_type="image")


def test_async_validation_error_is_value_error():
    """Test that AsyncValidationError can be caught as ValueError."""
    with pytest.raises(ValueError):
        raise AsyncValidationError("Test error")

    # Should also be catchable as InstructorError
    with pytest.raises(InstructorError):
        raise AsyncValidationError("Test error")


def test_exception_inheritance_chain():
    """Test that new exceptions have correct inheritance."""
    # ResponseParsingError
    assert issubclass(ResponseParsingError, ValueError)
    assert issubclass(ResponseParsingError, InstructorError)
    assert issubclass(ResponseParsingError, Exception)

    # MultimodalError
    assert issubclass(MultimodalError, ValueError)
    assert issubclass(MultimodalError, InstructorError)
    assert issubclass(MultimodalError, Exception)

    # AsyncValidationError
    assert issubclass(AsyncValidationError, ValueError)
    assert issubclass(AsyncValidationError, InstructorError)
    assert issubclass(AsyncValidationError, Exception)


def test_mixed_exception_catching():
    """Test catching multiple exception types including ValueError."""

    def raise_parsing_error():
        raise ResponseParsingError("Parsing failed", mode="JSON")

    def raise_multimodal_error():
        raise MultimodalError(
            "File not found", content_type="image", file_path="/test.jpg"
        )

    # Catch as ValueError
    with pytest.raises(ValueError):
        raise_parsing_error()

    with pytest.raises(ValueError):
        raise_multimodal_error()

    # Catch as InstructorError
    with pytest.raises(InstructorError):
        raise_parsing_error()

    with pytest.raises(InstructorError):
        raise_multimodal_error()


def test_exception_attributes_preserved():
    """Test that exception attributes are preserved when caught as ValueError."""
    try:
        raise ResponseParsingError(
            "Parse failed", mode="TOOLS", raw_response={"test": "data"}
        )
    except ValueError as e:
        # Should still be able to access ResponseParsingError attributes
        assert isinstance(e, ResponseParsingError)
        assert e.mode == "TOOLS"
        assert e.raw_response == {"test": "data"}

    try:
        raise MultimodalError("File error", content_type="pdf", file_path="/test.pdf")
    except ValueError as e:
        # Should still be able to access MultimodalError attributes
        assert isinstance(e, MultimodalError)
        assert e.content_type == "pdf"
        assert e.file_path == "/test.pdf"
