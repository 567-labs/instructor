"""Loopback HTTP server shared by offline SDK contract tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class JSONHandler(BaseHTTPRequestHandler):
    """Quiet request handler with JSON body helpers."""

    def read_json(self) -> Any:
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    def send_body(
        self,
        body: bytes,
        status: int = 200,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The client timed out or was cancelled first.

    def send_json(
        self, payload: Any, status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_body(body, status, headers=headers)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
        pass


@contextmanager
def serve(
    handler: type[BaseHTTPRequestHandler], *, daemon_threads: bool = True
) -> Generator[str, None, None]:
    """Serve ``handler`` on an ephemeral loopback port and yield its base URL.

    A short poll interval keeps ``shutdown()`` from waiting the default 0.5s.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = daemon_threads
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
