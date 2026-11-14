import json
import os
import subprocess
import threading
import tempfile
import sys

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

line = (
    json.dumps(
        {
            "custom_id": "request-1",
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "fixture"},
                },
            },
        }
    )
    + "\n"
)


@pytest.mark.parametrize("command", ["download-file", "results"])
@pytest.mark.parametrize(
    "mode,existing",
    [
        ("success", False),
        ("success", True),
        ("malformed", False),
        ("malformed", True),
        ("pending", True),
    ],
)
def test_cli_export(command: str, mode: str, existing: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="instructor-cli-export-") as temp:
        target = Path(temp) / "output.jsonl"
        if existing:
            target.write_text("previous-result\n")
        seen: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                seen.append(self.path)
                if self.path == "/results":
                    body = (
                        line
                        + (
                            "not-json\n"
                            if mode == "malformed"
                            else line.replace("request-1", "request-2")
                        )
                    ).encode()
                    content = "application/x-jsonl"
                else:
                    body = json.dumps(
                        {
                            "id": "msgbatch_test",
                            "type": "message_batch",
                            "processing_status": (
                                "in_progress" if mode == "pending" else "ended"
                            ),
                            "request_counts": {
                                "processing": 0,
                                "succeeded": 0,
                                "errored": 1,
                                "canceled": 0,
                                "expired": 0,
                            },
                            "ended_at": None,
                            "created_at": "2026-01-01T00:00:00Z",
                            "expires_at": "2026-01-02T00:00:00Z",
                            "archived_at": None,
                            "cancel_initiated_at": None,
                            "results_url": f"http://127.0.0.1:{server.server_port}/results",
                        }
                    ).encode()
                    content = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}
            )
            thread.start()
            env = {
                **{
                    key: os.environ[key]
                    for key in (
                        "PATH",
                        "SYSTEMROOT",
                        "WINDIR",
                        "PATHEXT",
                        "TEMP",
                        "TMP",
                    )
                    if key in os.environ
                },
                "HOME": temp,
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHON_DOTENV_DISABLED": "1",
                "AWS_EC2_METADATA_DISABLED": "true",
                "ANTHROPIC_API_KEY": "local-only-test",
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                "NO_PROXY": "127.0.0.1,localhost",
                "COLUMNS": "200",
            }
            args = (
                ["--download-file-path", str(target), "--provider", "anthropic"]
                if command == "download-file"
                else ["--output-file", str(target), "--model", "anthropic/test"]
            )
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "instructor.cli.cli",
                        "batch",
                        command,
                        "--batch-id",
                        "msgbatch_test",
                        *args,
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=25,
                )
            finally:
                server.shutdown()
                thread.join()
        assert seen == (
            ["/v1/messages/batches/msgbatch_test"]
            if mode == "pending"
            else [
                "/v1/messages/batches/msgbatch_test",
                "/v1/messages/batches/msgbatch_test",
                "/results",
            ]
        )
        assert result.returncode == 0  # Existing CLI exit semantics, not fixed here.
        if mode != "success":
            assert "Error " in result.stdout
            assert "Results saved" not in result.stdout
            if existing:
                assert target.read_text() == "previous-result\n"
            else:
                assert not target.exists()
        else:
            exported = [json.loads(row) for row in target.read_text().splitlines()]
            assert len(exported) == 2
            assert exported[1]["custom_id"] == "request-2"
            assert exported[0]["custom_id"] == "request-1"
            assert exported[0]["result"]["type"] == "errored"
            assert exported[0]["result"]["error"]["error"]["message"] == "fixture"
            assert "Error " not in result.stdout
            if command == "results":
                assert "Results saved" in result.stdout
        assert sorted(p.name for p in Path(temp).iterdir()) == (
            ["output.jsonl"] if target.exists() else []
        )


@pytest.mark.parametrize("command", ["download-file", "results"])
def test_mistral_still_delegates(
    command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner
    from instructor.cli.batch import app
    from instructor.batch import providers

    calls: list[tuple[str, str]] = []

    class Provider:
        def download_results(self, batch_id: str, path: str) -> None:
            calls.append((batch_id, path))
            Path(path).write_text("fixture\n")

    def provider(name: str) -> Provider:
        assert name == "mistral"
        return Provider()

    monkeypatch.setattr(providers, "get_provider", provider)
    output = tmp_path / "out.jsonl"
    args = (
        ["--download-file-path", str(output), "--provider", "mistral"]
        if command == "download-file"
        else ["--output-file", str(output), "--model", "mistral/test"]
    )
    result = CliRunner().invoke(app, [command, "--batch-id", "batch-test", *args])
    assert result.exit_code == 0, result.output
    assert calls == [("batch-test", str(output))]
    assert output.read_text() == "fixture\n"
