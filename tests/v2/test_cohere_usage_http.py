"""Public native Cohere V2 usage contracts through a real localhost endpoint."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, cast

import pytest
from pydantic import BaseModel

if __name__ != "__main__":
    pytest.importorskip("cohere")
import instructor


class Item(BaseModel):
    value: int


class Server(ThreadingHTTPServer):
    requests: list[dict[str, Any]]
    scenario: str


class Endpoint(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        server = cast(Server, self.server)
        server.requests.append(
            json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        )
        attempt = len(server.requests)
        valid = server.scenario == "success" or attempt == (
            3 if server.scenario == "missing_later" else 2
        )
        usage: dict[str, Any] = {
            "tokens": {"input_tokens": 100, "output_tokens": 20},
            "billed_units": {
                "input_tokens": 10,
                "output_tokens": 5,
                "search_units": 999,
            },
            "cached_tokens": 70,
            "total_tokens": 99999,
            "audit": {"ids": [attempt]},
        }
        if (server.scenario == "missing_first" and attempt == 1) or (
            server.scenario == "missing_later" and attempt == 2
        ):
            usage.pop("tokens")
        payload = {
            "id": "local",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"value": 7 if valid else "invalid"}),
                    }
                ],
            },
            "usage": usage,
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize(
    "scenario,budget",
    [
        ("ordinary", None),
        ("ordinary", 50),
        ("ordinary", 120),
        ("ordinary", 500),
        ("missing_first", None),
        ("missing_first", 500),
        ("missing_later", None),
        ("missing_later", 500),
        ("success", 50),
    ],
)
def test_native_usage_public(async_: bool, scenario: str, budget: int | None) -> None:
    # Fresh SDK import means a host test suite that already imported Cohere
    # cannot capture the wrong CO_API_URL. No SDK objects are patched.
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2])
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, __file__, str(int(async_)), scenario, str(budget)],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_public(
    endpoint: Server, async_: bool, scenario: str, budget: int | None
) -> None:
    assert (
        Path(instructor.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[2])
    )
    endpoint.requests = []
    endpoint.scenario = scenario
    raw: list[Any] = []
    before: list[dict[str, Any]] = []
    snapshots: list[Any] = []

    async def run() -> None:
        client: Any = instructor.from_provider(
            "cohere/command-local",
            api_key="local-not-secret",
            async_client=async_,
            mode=instructor.Mode.MD_JSON,
        )

        def response_hook(response: Any) -> None:
            raw.append(response.usage)
            before.append(copy.deepcopy(response.usage.model_dump()))

        client.on("completion:response", response_hook)
        client.on("completion:usage", lambda usage, **_kwargs: snapshots.append(usage))
        try:
            kwargs: dict[str, Any] = dict(
                response_model=Item,
                messages=[{"role": "user", "content": "Return value"}],
                max_retries=3,
                token_budget=budget,
            )
            expected_error = budget is not None and (
                scenario.startswith("missing")
                or (scenario == "ordinary" and budget <= 120)
            )
            try:
                result = (
                    await client.create(**kwargs) if async_ else client.create(**kwargs)
                )
            except Exception as exc:
                assert expected_error, repr(exc)
                assert type(exc).__name__ == (
                    "TokenUsageUnavailableError"
                    if scenario.startswith("missing")
                    else "TokenBudgetExceeded"
                )
                assert len(endpoint.requests) == (
                    2 if scenario == "missing_later" else 1
                )
            else:
                assert not expected_error
                assert result.value == 7
                assert len(endpoint.requests) == (
                    1
                    if scenario == "success"
                    else 3
                    if scenario == "missing_later"
                    else 2
                )
                if scenario.startswith("missing"):
                    assert not hasattr(result, "_total_usage")
                else:
                    total = result._total_usage
                    assert type(total).__module__.startswith("cohere.")
                    assert total.tokens.input_tokens == 100 * len(endpoint.requests)
                    assert total.tokens.output_tokens == 20 * len(endpoint.requests)
                    assert total.model_fields_set == {"tokens"}
                    assert (
                        total.billed_units is None
                        and getattr(total, "cached_tokens", None) is None
                    )
                    assert not total.model_extra
                    assert result._raw_response.usage is raw[-1]
                    assert total.tokens is not snapshots[-1].tokens
                    assert snapshots[-1].tokens.input_tokens == 100 * len(
                        endpoint.requests
                    )
            assert [usage.model_dump() for usage in raw] == before
            expected_hooks = (
                0
                if scenario == "missing_first"
                else 1
                if scenario == "missing_later"
                else len(endpoint.requests)
            )
            assert len(snapshots) == expected_hooks
            if snapshots:
                assert snapshots[0].tokens is not raw[0].tokens
                assert raw[0].tokens.input_tokens == 100
                if len(snapshots) > 1:
                    assert snapshots[-1].tokens.input_tokens == 200
                    assert snapshots[0].tokens is not snapshots[-1].tokens
        finally:
            if async_:
                await client.client.__aexit__(None, None, None)
            else:
                client.client.__exit__(None, None, None)

    asyncio.run(run())


if __name__ == "__main__":

    def localhost_only(event: str, args: tuple[Any, ...]) -> None:
        if event == "socket.connect":
            address = args[1]
            if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
                raise RuntimeError(
                    "Only localhost connections are permitted by this test"
                )

    sys.addaudithook(localhost_only)
    for key in list(os.environ):
        if key.lower() in ("http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(key)
    server = Server(("127.0.0.1", 0), Endpoint)
    os.environ["CO_API_URL"] = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        _check_public(
            server,
            bool(int(sys.argv[1])),
            sys.argv[2],
            None if sys.argv[3] == "None" else int(sys.argv[3]),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
