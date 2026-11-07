"""Exercise file downloads through the CLI and a real SDK HTTP request."""

import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest


@pytest.mark.parametrize(
    "payload",
    [b"", '{"answer": "你好"}\n'.encode(), b"\x00\xff\xfe\r\n\x80"],
    ids=["empty", "utf8-jsonl", "binary"],
)
def test_file_download_preserves_bytes(payload: bytes, tmp_path: Path) -> None:
    requests: list[tuple[str, str]] = []

    class FileHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.command, self.path))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "ANTHROPIC_", "AZURE_OPENAI_"))
        and key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    output = tmp_path / "download.bin"
    with ThreadingHTTPServer(("127.0.0.1", 0), FileHandler) as server:
        environment.update(
            OPENAI_API_KEY="local-download-test-only",
            OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1/",
        )
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "instructor.cli.cli",
                    "files",
                    "download",
                    "file-download-test",
                    str(output),
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert requests == [("GET", "/v1/files/file-download-test/content")]
    assert output.read_bytes() == payload
    assert "downloaded successfully" in result.stdout
