"""Tests for in-memory batch processing functionality."""

import io
import json
import pytest
from pydantic import BaseModel
from instructor.batch.request import BatchRequest
from instructor.batch.providers.openai import OpenAIProvider
from instructor.batch.providers.anthropic import AnthropicProvider

# Mark all tests in this module as unit tests (not integration)
pytestmark = pytest.mark.unit


class User(BaseModel):
    name: str
    age: int
    email: str


class TestBatchRequestInMemory:
    """Test BatchRequest with BytesIO support."""

    def test_save_to_bytesio_openai(self):
        """Test saving BatchRequest to BytesIO for OpenAI format."""
        buffer = io.BytesIO()

        batch_request = BatchRequest[User](
            custom_id="test-1",
            messages=[{"role": "user", "content": "Extract user info"}],
            response_model=User,
            model="gpt-4",
            max_tokens=100,
            temperature=0.1,
        )

        # Save to BytesIO
        batch_request.save_to_file(buffer, "openai")

        # Read back and verify
        buffer.seek(0)
        content = buffer.read().decode("utf-8")
        data = json.loads(content.strip())

        assert data["custom_id"] == "test-1"
        assert data["method"] == "POST"
        assert data["url"] == "/v1/chat/completions"
        assert "body" in data
        assert data["body"]["model"] == "gpt-4"
        assert "response_format" in data["body"]

    def test_save_to_bytesio_anthropic(self):
        """Test saving BatchRequest to BytesIO for Anthropic format."""
        buffer = io.BytesIO()

        batch_request = BatchRequest[User](
            custom_id="test-1",
            messages=[{"role": "user", "content": "Extract user info"}],
            response_model=User,
            model="claude-3-sonnet",
            max_tokens=100,
            temperature=0.1,
        )

        # Save to BytesIO
        batch_request.save_to_file(buffer, "anthropic")

        # Read back and verify
        buffer.seek(0)
        content = buffer.read().decode("utf-8")
        data = json.loads(content.strip())

        assert data["custom_id"] == "test-1"
        assert "params" in data
        assert data["params"]["model"] == "claude-3-sonnet"
        assert "tools" in data["params"]

    def test_save_to_file_still_works(self):
        """Test that original file-based saving still works."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            temp_path = f.name

        try:
            batch_request = BatchRequest[User](
                custom_id="test-1",
                messages=[{"role": "user", "content": "Extract user info"}],
                response_model=User,
                model="gpt-4",
                max_tokens=100,
                temperature=0.1,
            )

            # Save to file
            batch_request.save_to_file(temp_path, "openai")

            # Read back and verify
            with open(temp_path) as f:
                content = f.read()

            data = json.loads(content.strip())
            assert data["custom_id"] == "test-1"
            assert "body" in data

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_multiple_requests_in_buffer(self):
        """Test writing multiple requests to the same BytesIO buffer."""
        buffer = io.BytesIO()

        for i in range(3):
            batch_request = BatchRequest[User](
                custom_id=f"request-{i}",
                messages=[{"role": "user", "content": f"Extract user {i}"}],
                response_model=User,
                model="gpt-4",
                max_tokens=100,
                temperature=0.1,
            )
            batch_request.save_to_file(buffer, "openai")

        # Read back and verify
        buffer.seek(0)
        content = buffer.read().decode("utf-8")
        lines = [line for line in content.split("\n") if line.strip()]

        assert len(lines) == 3

        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["custom_id"] == f"request-{i}"

    def test_invalid_buffer_type_raises_error(self):
        """Test that invalid buffer types raise appropriate errors."""
        batch_request = BatchRequest[User](
            custom_id="test-1",
            messages=[{"role": "user", "content": "Extract user info"}],
            response_model=User,
            model="gpt-4",
            max_tokens=100,
            temperature=0.1,
        )

        with pytest.raises(ValueError, match="Unsupported file_path_or_buffer type"):
            batch_request.save_to_file(
                123,  # ty: ignore[invalid-argument-type] - deliberately invalid input
                "openai",
            )


class TestProviderInMemorySupport:
    """Test that providers support BytesIO buffers."""

    def test_provider_invalid_type_raises_error(self):
        """Test that providers raise errors for invalid types."""
        openai_provider = OpenAIProvider()
        anthropic_provider = AnthropicProvider()

        with pytest.raises(ValueError, match="Unsupported file_path_or_buffer type"):
            openai_provider.submit_batch(
                123  # ty: ignore[invalid-argument-type] - deliberately invalid input
            )

        with pytest.raises(ValueError, match="Unsupported file_path_or_buffer type"):
            anthropic_provider.submit_batch(
                123  # ty: ignore[invalid-argument-type] - deliberately invalid input
            )
