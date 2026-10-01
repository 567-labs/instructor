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


def test_upload_stops_when_file_processing_fails(monkeypatch, tmp_path) -> None:
    import pytest
    import typer

    from instructor.cli import files

    path = tmp_path / "train.jsonl"
    path.write_text("{}")

    class FakeFiles:
        def create(self, **_kwargs):
            return FileObject(
                id="file-abc123",
                bytes=2,
                created_at=1700000000,
                filename="train.jsonl",
                object="file",
                purpose="fine-tune",
                status="uploaded",
            )

        def retrieve(self, _file_id):
            return type("F", (), {"status": "error"})()

    class FakeClient:
        files = FakeFiles()

    monkeypatch.setattr(files, "client", FakeClient())
    monkeypatch.setattr(files.time, "sleep", lambda _s: pytest.fail("kept polling"))

    with pytest.raises(typer.Exit) as exc:
        files.upload(str(path), "fine-tune", 0)
    assert exc.value.exit_code == 1


def test_status_stops_when_file_processing_fails(monkeypatch) -> None:
    import pytest
    import typer

    from instructor.cli import files

    monkeypatch.setattr(files, "get_file_status", lambda _id: "error")
    monkeypatch.setattr(files.time, "sleep", lambda _s: pytest.fail("kept polling"))

    with pytest.raises(typer.Exit):
        files.status("file-abc123")
