from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import json
from typing import Any

import pytest

from tests._http import JSONHandler, serve


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

    class Handler(JSONHandler):
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
            self.record["body"] = self.read_json()
            body = endpoint.response_body
            if body is None:
                body = json.dumps(endpoint.response).encode()
            self.send_json(body, endpoint.status)

    with serve(Handler) as url:
        endpoint.url = url
        yield endpoint
