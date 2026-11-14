"""Exercise CLI exit codes through the real SDK and loopback HTTP requests."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.mark.parametrize("status", [200, 401, 404])
@pytest.mark.parametrize(
    "arguments,method,path,success_payload,action",
    [
        (
            ["files", "delete", "file-test"],
            "DELETE",
            "/v1/files/file-test",
            {"id": "file-test", "object": "file", "deleted": True},
            "deleting file",
        ),
        (
            ["jobs", "cancel", "ftjob-test"],
            "POST",
            "/v1/fine_tuning/jobs/ftjob-test/cancel",
            {
                "id": "ftjob-test",
                "object": "fine_tuning.job",
                "created_at": 0,
                "error": None,
                "fine_tuned_model": None,
                "finished_at": None,
                "hyperparameters": {"n_epochs": "auto"},
                "model": "local-model",
                "organization_id": "org-local",
                "result_files": [],
                "seed": 0,
                "status": "cancelled",
                "trained_tokens": None,
                "training_file": "file-test",
                "validation_file": None,
            },
            "cancelling job",
        ),
    ],
)
def test_mutating_command_exit_codes(
    arguments: list[str],
    method: str,
    path: str,
    success_payload: dict[str, object],
    action: str,
    status: int,
) -> None:
    requests: list[tuple[str, str]] = []
    payload = (
        success_payload
        if status == 200
        else {
            "error": {"message": "Local API failure", "type": "invalid_request_error"}
        }
    )

    class Handler(BaseHTTPRequestHandler):
        def respond(self) -> None:
            requests.append((self.command, self.path))
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_DELETE = respond
        do_POST = respond

        def log_message(self, format: str, *args: object) -> None:
            pass

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "ANTHROPIC_", "AZURE_OPENAI_"))
    }
    environment.update(
        OPENAI_API_KEY="local-only-cli-test",
        PYTHONIOENCODING="utf-8",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        COLUMNS="200",
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        environment["OPENAI_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "instructor.cli.cli", *arguments],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
            )
        finally:
            server.shutdown()
            thread.join()

    assert requests == [(method, path)]
    assert result.returncode == (0 if status == 200 else 1), (
        result.stdout + result.stderr
    )
    if status == 200:
        assert "successfully!" in result.stdout
    else:
        assert f"Error {action}" in result.stdout
        assert "successfully!" not in result.stdout
