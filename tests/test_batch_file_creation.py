"""Batch file creation must not destroy the previous batch on failure."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from instructor.batch import BatchProcessor

import pytest

pytestmark = pytest.mark.unit


class Person(BaseModel):
    name: str


def test_failed_serialization_preserves_the_previous_batch() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        path.write_text("previous batch\n")
        processor = BatchProcessor("openai/test-model", Person)

        with pytest.raises(TypeError):
            processor.create_batch_from_messages(
                [
                    [{"role": "user", "content": "Ada"}],
                    [{"role": "user", "content": object()}],
                ],
                str(path),
            )

        assert path.read_text() == "previous batch\n"
        # No temporary file is left behind in the destination directory.
        assert list(Path(directory).glob(".batch-*")) == []


def test_successful_creation_replaces_the_previous_batch_atomically() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        path.write_text("previous batch\n")
        processor = BatchProcessor("openai/test-model", Person)

        processor.create_batch_from_messages(
            [
                [{"role": "user", "content": "Ada"}],
                [{"role": "user", "content": "Bob"}],
            ],
            str(path),
        )

        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert all(json.loads(line)["custom_id"] for line in lines)
        assert list(Path(directory).glob(".batch-*")) == []
