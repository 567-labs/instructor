"""Public factory host routing through native xAI clients and loopback TLS gRPC."""

from __future__ import annotations

import asyncio
import json
import inspect
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

if __name__ != "__main__":
    pytest.importorskip("xai_sdk")


class Item(BaseModel):
    count: int


def _channel(client: Any) -> Any:
    """Read the native channel across SDK versions; never replace it."""
    native = client.client
    if hasattr(native, "_api_channel"):
        return native._api_channel
    call = native.chat._stub.GetCompletion
    if hasattr(call, "_thunk"):
        return inspect.getclosurevars(call._thunk).nonlocals["self"]
    return call._python_channel


def _target(client: Any) -> str:
    """Read only: refuse any RPC unless the SDK channel really targets loopback."""
    inner = _channel(client)._channel
    if hasattr(inner, "target"):
        value = inner.target
        return (value() if callable(value) else value).decode()
    return inner._channel.target().decode()


def _run(cert: Path, key: Path, asynchronous: bool, explicit: bool) -> dict[str, Any]:
    # This file runs in a fresh process, before grpc loads its default roots.
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = str(cert)
    os.environ.pop("XAI_MANAGEMENT_KEY", None)
    import grpc
    import instructor
    from xai_sdk.proto import chat_pb2, chat_pb2_grpc, sample_pb2

    assert (
        Path(instructor.__file__).resolve().parents[1]
        == Path(__file__).resolve().parents[2]
    )

    # SDK exposes a runtime-selected protobuf version; keep that native selection.
    chat_proto = cast(Any, chat_pb2)
    sample_proto = cast(Any, sample_pb2)

    service_base = cast(Any, chat_pb2_grpc.ChatServicer)

    class Service(service_base):
        calls = 0

        def GetCompletion(self, _request: Any, context: Any) -> Any:
            self.calls += 1
            assert context.auth_context()["transport_security_type"] == [b"ssl"]
            fields = chat_proto.GetChatCompletionResponse.DESCRIPTOR.fields_by_name
            if "outputs" in fields:
                field, output_type = "outputs", chat_proto.CompletionOutput
            else:
                assert "choices" in fields
                field, output_type = "choices", chat_proto.Choice
            output = output_type(
                index=0,
                finish_reason=sample_proto.REASON_STOP,
                message=chat_proto.CompletionMessage(
                    role=chat_proto.ROLE_ASSISTANT, content='{"count":3}'
                ),
            )
            return chat_proto.GetChatCompletionResponse(id="local", **{field: [output]})

    service = Service()
    with ThreadPoolExecutor(max_workers=2) as pool:
        server = grpc.server(pool)
        chat_pb2_grpc.add_ChatServicer_to_server(service, server)
        credentials = grpc.ssl_server_credentials(
            [(key.read_bytes(), cert.read_bytes())]
        )
        port = server.add_secure_port("127.0.0.1:0", credentials)
        assert port
        server.start()
        host = f"127.0.0.1:{port}"
        options: dict[str, Any] = {"api_host": host} if explicit else {}
        target = ""
        try:
            if asynchronous:

                async def call() -> str:
                    client = instructor.from_provider(
                        "xai/local",
                        api_key="local-not-secret",
                        async_client=True,
                        mode=instructor.Mode.JSON_SCHEMA,
                        **options,
                    )
                    try:
                        actual = _target(client)
                        if not explicit:
                            assert actual.removeprefix("dns:///") == "api.x.ai"
                            return actual
                        assert actual.removeprefix("dns:///") == host, (
                            "Host ignored; refusing RPC"
                        )
                        result = await client.create(
                            response_model=Item,
                            messages=[{"role": "user", "content": "Extract count"}],
                        )
                        assert result.count == 3
                        return actual
                    finally:
                        await _channel(client).close()

                target = asyncio.run(call())
            else:
                client = instructor.from_provider(
                    "xai/local",
                    api_key="local-not-secret",
                    mode=instructor.Mode.JSON_SCHEMA,
                    **options,
                )
                try:
                    target = _target(client)
                    if explicit:
                        assert target.removeprefix("dns:///") == host, (
                            "Host ignored; refusing RPC"
                        )
                        result = client.create(
                            response_model=Item,
                            messages=[{"role": "user", "content": "Extract count"}],
                        )
                        assert result.count == 3
                    else:
                        assert target.removeprefix("dns:///") == "api.x.ai"
                finally:
                    _channel(client).close()
            assert service.calls == int(explicit)
            return {
                "target": target,
                "tls_requests": service.calls,
                "source": instructor.__file__,
            }
        finally:
            server.stop(0).wait()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_public_xai_api_host(
    tmp_path: Path, asynchronous: bool, explicit: bool
) -> None:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip(
            "Local TLS test requires openssl to create an ephemeral certificate"
        )
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    config = tmp_path / "openssl.cnf"
    config.write_text(
        "[req]\nprompt = no\ndistinguished_name = subject\n"
        "x509_extensions = server\n[subject]\nCN = 127.0.0.1\n"
        "[server]\nsubjectAltName = IP:127.0.0.1\n"
    )
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-config",
            str(config),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    for name in (
        "XAI_API_KEY",
        "XAI_MANAGEMENT_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "grpc_proxy",
        "GRPC_PROXY",
    ):
        env.pop(name, None)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(cert),
            str(key),
            str(int(asynchronous)),
            str(int(explicit)),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["tls_requests"] == int(explicit)


if __name__ == "__main__":
    print(
        json.dumps(
            _run(
                Path(sys.argv[1]),
                Path(sys.argv[2]),
                bool(int(sys.argv[3])),
                bool(int(sys.argv[4])),
            )
        )
    )
