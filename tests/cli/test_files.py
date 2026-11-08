"""Tests for instructor.cli.files."""

from types import SimpleNamespace

import pytest
from openai.types import FileObject


def _unexpected_poll(_seconds):
    raise AssertionError("Polling continued after a terminal file error")


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


def test_list_displays_files_from_all_pages(monkeypatch) -> None:
    from openai import OpenAI
    from typer.testing import CliRunner

    from instructor.cli import files
    from tests.openai_http import Client, MockTransport, Request, Response

    cursors = []

    def respond(request: Request) -> Response:
        assert request.url.path == "/v1/files"
        cursor = request.url.params.get("after")
        cursors.append(cursor)
        first_page = cursor is None
        return Response(
            200,
            json={
                "object": "list",
                "has_more": first_page,
                "data": [
                    {
                        "id": "file-new" if first_page else "file-old",
                        "bytes": 1,
                        "created_at": 1700000001 if first_page else 1700000000,
                        "filename": "training.jsonl",
                        "object": "file",
                        "purpose": "fine-tune",
                        "status": "processed",
                    }
                ],
            },
        )

    with OpenAI(
        api_key="test-cli-pagination",
        base_url="https://example.test/v1",
        http_client=Client(transport=MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(files, "client", client)
        result = CliRunner().invoke(files.app, ["list"])

    assert result.exit_code == 0, result.output
    assert cursors == [None, "file-new"]
    assert "file-new" in result.output
    assert "file-old" in result.output


def test_upload_stops_when_file_processing_fails(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr(files.time, "sleep", _unexpected_poll)

    with pytest.raises(typer.Exit) as exc:
        files.upload(str(path), "fine-tune", 0)
    assert exc.value.exit_code == 1


def test_status_stops_when_file_processing_fails(monkeypatch) -> None:
    import typer

    from instructor.cli import files

    monkeypatch.setattr(files, "get_file_status", lambda _id: "error")
    monkeypatch.setattr(files.time, "sleep", _unexpected_poll)

    with pytest.raises(typer.Exit):
        files.status("file-abc123")


@pytest.mark.parametrize("failed", ["train", "validation"])
def test_failed_file_prevents_fine_tuning_job(monkeypatch, tmp_path, failed):
    import typer
    from instructor.cli import jobs

    train = tmp_path / "train"
    validation = tmp_path / "validation"
    train.write_text("{}")
    validation.write_text("{}")
    ids = iter(("train", "validation"))
    files = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(id=next(ids)))
    monkeypatch.setattr(jobs, "client", SimpleNamespace(files=files))
    monkeypatch.setattr(
        jobs, "get_file_status", lambda fid: "error" if fid == failed else "processed"
    )
    monkeypatch.setattr(jobs.time, "sleep", _unexpected_poll)
    with pytest.raises(typer.Exit) as exc:
        jobs.create_from_file(
            str(train), "test-model", 0, None, None, None, str(validation), None
        )
    assert exc.value.exit_code == 1
