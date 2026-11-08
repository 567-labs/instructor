"""Tests for instructor.cli.files."""

import http.server
import threading
from contextlib import contextmanager
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


@contextmanager
def _loopback_file_server(payload: bytes):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/files/file-repro/content":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    ("payload", "filename"),
    [
        (b"", "empty.bin"),
        (b'{"messages": [{"role": "user", "content": "hi"}]}\n', "data.jsonl"),
        (b"\x00\x01\x02\xff\xfe\xaa\x55", "binary.bin"),
    ],
)
def test_download_file_with_real_sdk_loopback(
    monkeypatch, tmp_path, payload: bytes, filename: str
) -> None:
    import openai

    from instructor.cli import files

    output_path = tmp_path / filename
    with _loopback_file_server(payload) as base_url:
        real_client = openai.OpenAI(base_url=base_url, api_key="local-repro-only")
        monkeypatch.setattr(files, "client", real_client)
        files.download("file-repro", str(output_path))
        assert output_path.exists()
        assert output_path.read_bytes() == payload
