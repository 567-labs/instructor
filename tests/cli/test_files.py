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


def _openai_client(handler):
    from openai import OpenAI

    from tests.openai_http import Client, MockTransport

    return OpenAI(api_key="test", http_client=Client(transport=MockTransport(handler)))


def _file_json(file_id: str, created_at: int) -> dict:
    return {
        "id": file_id,
        "bytes": 1,
        "created_at": created_at,
        "filename": f"{file_id}.jsonl",
        "object": "file",
        "purpose": "fine-tune",
        "status": "processed",
    }


def test_download_writes_raw_file_bytes(monkeypatch, tmp_path) -> None:
    from instructor.cli import files
    from tests.openai_http import Response

    payload = b'{"a": 1}\n\xff\xfe binary'
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return Response(200, content=payload)

    monkeypatch.setattr(files, "client", _openai_client(handler))
    output = tmp_path / "out.jsonl"

    files.download("file-1", str(output))

    assert output.read_bytes() == payload
    assert paths == ["/v1/files/file-1/content"]


def test_get_files_reads_every_page(monkeypatch) -> None:
    from instructor.cli import files
    from tests.openai_http import Response

    def handler(request):
        if request.url.params.get("after") == "file-old":
            body = {
                "object": "list",
                "data": [_file_json("file-new", 3)],
                "has_more": False,
            }
        else:
            body = {
                "object": "list",
                "data": [_file_json("file-mid", 2), _file_json("file-old", 1)],
                "has_more": True,
            }
        return Response(200, json=body)

    monkeypatch.setattr(files, "client", _openai_client(handler))

    assert [f.id for f in files.get_files()] == ["file-new", "file-mid", "file-old"]


def test_delete_exits_nonzero_on_failure(monkeypatch) -> None:
    import typer

    from instructor.cli import files

    def fail(_file_id):
        raise RuntimeError("not found")

    monkeypatch.setattr(
        files, "client", SimpleNamespace(files=SimpleNamespace(delete=fail))
    )

    with pytest.raises(typer.Exit) as exc:
        files.delete("file-1")
    assert exc.value.exit_code == 1


def test_delete_succeeds_without_exit(monkeypatch) -> None:
    from instructor.cli import files

    deleted = []
    monkeypatch.setattr(
        files, "client", SimpleNamespace(files=SimpleNamespace(delete=deleted.append))
    )

    files.delete("file-1")

    assert deleted == ["file-1"]


def test_cancel_job_exits_nonzero_on_failure(monkeypatch) -> None:
    import typer

    from instructor.cli import jobs

    def fail(_job_id):
        raise RuntimeError("already finished")

    fake = SimpleNamespace(
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(cancel=fail))
    )
    monkeypatch.setattr(jobs, "client", fake)

    with pytest.raises(typer.Exit) as exc:
        jobs.cancel("ftjob-1")
    assert exc.value.exit_code == 1
