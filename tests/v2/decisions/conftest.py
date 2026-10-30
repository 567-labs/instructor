from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any

import pytest


@dataclass
class DecisionEndpoint:
    url: str = ""
    response: Any = field(default_factory=dict)
    response_body: bytes | None = None
    status: int = 200
    calls: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def decision_endpoint() -> Iterator[DecisionEndpoint]:
    endpoint = DecisionEndpoint()

    class Handler(BaseHTTPRequestHandler):
        def parse_request(self) -> bool:
            if not super().parse_request():
                return False
            self.record = {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": None,
            }
            endpoint.calls.append(self.record)
            return True

        def do_POST(self) -> None:
            self.record["body"] = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            body = (
                json.dumps(endpoint.response).encode()
                if endpoint.response_body is None
                else endpoint.response_body
            )
            self.send_response(endpoint.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
