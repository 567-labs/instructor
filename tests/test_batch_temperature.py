"""Anthropic batch sampling parameters match the provider's optional fields."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from instructor.batch.processor import BatchProcessor
from instructor.batch.request import BatchRequest


class Person(BaseModel):
    name: str


def request(temperature: float | None = 0.1) -> BatchRequest[Person]:
    return BatchRequest[Person](
        custom_id="person-1",
        messages=[{"role": "user", "content": "Extract Ada"}],
        response_model=Person,
        model="test-model",
        temperature=temperature,
    )


def test_anthropic_omits_none_temperature() -> None:
    assert "temperature" not in request(None).to_anthropic_format()["params"]


@pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0])
def test_anthropic_preserves_numeric_temperature(temperature: float) -> None:
    assert (
        request(temperature).to_anthropic_format()["params"]["temperature"]
        == temperature
    )


def test_anthropic_preserves_default_temperature() -> None:
    batch_request = BatchRequest(
        custom_id="person-1",
        messages=[{"role": "user", "content": "Extract Ada"}],
        response_model=Person,
        model="test-model",
    )
    assert batch_request.to_anthropic_format()["params"]["temperature"] == 0.1


@pytest.mark.parametrize("file_backed", [False, True])
def test_public_anthropic_batch_omits_none_temperature(
    file_backed: bool, tmp_path: Path
) -> None:
    processor = BatchProcessor("anthropic/test-model", Person)
    path = tmp_path / "batch.jsonl" if file_backed else None

    output = processor.create_batch_from_messages(
        [[{"role": "user", "content": "Extract Ada"}]],
        file_path=str(path) if path is not None else None,
        temperature=None,
    )

    if path is not None:
        content = path.read_text(encoding="utf-8")
    else:
        assert isinstance(output, io.BytesIO)
        content = output.getvalue().decode("utf-8")
    row = json.loads(content)
    assert row["custom_id"] == "request-0"
    assert "temperature" not in row["params"]
    assert row["params"]["max_tokens"] == 1000


def test_openai_temperature_serialization_is_unchanged() -> None:
    assert request(None).to_openai_format()["body"]["temperature"] is None
