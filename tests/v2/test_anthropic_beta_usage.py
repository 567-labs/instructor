"""Ordinary native beta retries account usage; iterative usage stays fail-closed."""

from __future__ import annotations
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, cast
import pytest
from pydantic import BaseModel
import instructor
from instructor.v2.core.errors import TokenBudgetExceeded, TokenUsageUnavailableError
from instructor.v2.core.hooks import Hooks

anthropic = pytest.importorskip("anthropic")


class Result(BaseModel):
    count: int


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize(
    "scenario", ["retry", "limit", "iterative_failure", "iterative_success"]
)
def test_native_beta_retry_usage(
    asynchronous: bool, beta: bool, scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []
    responses: list[Any] = []
    iterative = beta and scenario.startswith("iterative")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            valid = scenario == "iterative_success" or len(requests) > 1
            usage: dict[str, Any] = {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
            }
            if iterative:
                usage.update(
                    input_tokens=0,
                    output_tokens=0,
                    iterations=[
                        {
                            "type": "compaction",
                            "input_tokens": 40,
                            "output_tokens": 5,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        }
                    ],
                )
            payload = {
                "id": "msg_local",
                "type": "message",
                "role": "assistant",
                "model": request["model"],
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_local",
                        "name": "Result",
                        "input": {"count": 3 if valid else "invalid"},
                    }
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": usage,
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    client = cast(
        Any,
        instructor.from_provider(
            "anthropic/claude-local",
            api_key="local-not-secret",
            mode=instructor.Mode.TOOLS,
            beta=beta,
            async_client=asynchronous,
        ),
    )
    hooks = Hooks()
    hooks.on("completion:response", responses.append)

    async def invoke() -> Any:
        value = client.create(
            response_model=Result,
            messages=[{"role": "user", "content": "Extract"}],
            max_tokens=100,
            max_retries=2,
            token_budget=20 if scenario == "limit" else 100,
            hooks=hooks,
        )
        return await value if asynchronous else value

    try:
        if scenario == "limit":
            with pytest.raises(TokenBudgetExceeded) as error:
                asyncio.run(invoke())
            assert error.value.n_attempts == 1
            assert error.value.total_usage.input_tokens == 10
            assert len(requests) == 1
        elif iterative and scenario == "iterative_failure":
            with pytest.raises(TokenUsageUnavailableError):
                asyncio.run(invoke())
            assert len(requests) == 1
            assert (
                responses[0].usage.model_dump()["iterations"][0]["input_tokens"] == 40
            )
            assert responses[0].usage.input_tokens == 0
        else:
            value = asyncio.run(invoke())
            assert value.count == 3
            assert len(requests) == (1 if scenario == "iterative_success" else 2)
            if iterative:
                assert not hasattr(value, "_total_usage")
                assert (
                    responses[0].usage.model_dump()["iterations"][0]["input_tokens"]
                    == 40
                )
            else:
                total = value._total_usage
                assert total.input_tokens == 10 * len(requests)
                assert total.output_tokens == 5 * len(requests)
                assert type(total).__name__ == ("BetaUsage" if beta else "Usage")
                assert value._raw_response.usage.model_dump() == total.model_dump()
    finally:
        if asynchronous:
            asyncio.run(client.client.close())
        else:
            client.client.close()
        server.shutdown()
        server.server_close()
        thread.join()
