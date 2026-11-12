"""Exercise public PDF uploads through the native SDK and a local HTTP server."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

# This endpoint override is verified against the locked SDK (2.8.0). Older
# releases such as 1.5.0 lack it; never send synthetic credentials to Google.
pytest.importorskip("google.genai", minversion="2.8.0")


def _pdf_bytes() -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 20 20] /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]
    result = b"%PDF-1.4\n"
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        result += f"{offset:010d} 00000 n \n".encode()
    return (
        result
        + (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )


@pytest.mark.parametrize(
    "states,budget,expected_polls,expected_error",
    [
        (("FAILED",), 3, 0, "GenAI file processing failed"),
        (("FAILED",), 0, 0, "GenAI file processing failed"),
        (("PROCESSING", "FAILED"), 3, 1, "GenAI file processing failed"),
        (("PROCESSING", "FAILED"), 1, 1, "GenAI file processing failed"),
        (("ACTIVE",), 0, 0, None),
        (("PROCESSING", "PROCESSING", "ACTIVE"), 3, 2, None),
        (("PROCESSING",), 3, 3, "Max retries reached"),
    ],
)
def test_pdf_file_polling(
    tmp_path: Path,
    states: tuple[str, ...],
    budget: int,
    expected_polls: int,
    expected_error: str | None,
) -> None:
    polls = 0
    uploads: list[bytes] = []
    pdf = _pdf_bytes()

    def file_response(state: str) -> dict:
        return {
            "name": "files/test-pdf",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/test-pdf",
            "mimeType": "application/pdf",
            "state": state,
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def respond(self, data: dict, headers: dict[str, str] | None = None) -> None:
            body = json.dumps(data).encode()
            self.send_response(200)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/upload-body":
                uploads.append(body)
                self.respond(
                    {"file": file_response(states[0])},
                    {"x-goog-upload-status": "final"},
                )
            else:
                self.respond({}, {"x-goog-upload-url": f"{endpoint}/upload-body"})

        def do_GET(self) -> None:
            nonlocal polls
            polls += 1
            self.respond(file_response(states[min(polls, len(states) - 1)]))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "test.pdf"
    path.write_bytes(pdf)
    env = os.environ.copy()
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ):
        env.pop(key, None)
    env.update(
        GOOGLE_GEMINI_BASE_URL=endpoint,
        GEMINI_API_KEY="local-synthetic-key",
        PYTHON_DOTENV_DISABLED="1",
    )
    code = """
import json, sys
from instructor.processing.multimodal import PDFWithGenaiFile
try:
    result = PDFWithGenaiFile.from_new_genai_file(sys.argv[1], retry_delay=0, max_retries=int(sys.argv[2]))
    print(json.dumps({"mime": result.media_type, "source": str(result.source)}))
except Exception as error:
    print(json.dumps({"error": str(error), "type": type(error).__name__}))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(path), str(budget)],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    result = json.loads(completed.stdout)
    assert uploads == [pdf]
    assert polls == expected_polls
    if expected_error is None:
        assert result == {
            "mime": "application/pdf",
            "source": "https://generativelanguage.googleapis.com/v1beta/files/test-pdf",
        }
    else:
        assert expected_error in result["error"]
        if expected_error == "GenAI file processing failed":
            assert result["type"] == "ValueError"
