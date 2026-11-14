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


def test_download_file_with_openai_client(monkeypatch, tmp_path) -> None:
    import httpx
    from openai import OpenAI

    from instructor.cli import files

    test_bytes = b'{"prompt": "hello", "completion": "world"}\n'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files/file-abc123/content"
        return httpx.Response(200, content=test_bytes)

    client = OpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(files, "client", client)

    output_path = tmp_path / "downloaded.jsonl"
    files.download("file-abc123", str(output_path))

    assert output_path.read_bytes() == test_bytes


def test_download_file_with_fake_client(monkeypatch, tmp_path) -> None:
    from instructor.cli import files

    class FakeFiles:
        def content(self, _file_id: str):
            return b"raw bytes content"

    class FakeClient:
        files = FakeFiles()

    monkeypatch.setattr(files, "client", FakeClient())

    output_path = tmp_path / "fake_downloaded.bin"
    files.download("file-fake123", str(output_path))

    assert output_path.read_bytes() == b"raw bytes content"
