"""Tests for instructor.cli.files."""

from openai.types import FileObject


def test_generate_file_table_uses_attribute_access() -> None:
    from instructor.cli.files import generate_file_table

    file = FileObject(
        id="file-abc123",
        bytes=1024,
        created_at=1700000000,
        filename="training.jsonl",
        object="file",
        purpose="fine-tune",
        status="processed",
    )

    table = generate_file_table([file])

    rendered_first_column = table.columns[0]._cells
    assert list(rendered_first_column) == ["file-abc123"]
