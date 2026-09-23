from __future__ import annotations

import ipaddress
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from instructor.v2.core import remote


class FakeSocket:
    def __init__(self, address: str) -> None:
        self.address = address

    def getpeername(self) -> tuple[str, int]:
        return self.address, 443


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        peer: str | None = "93.184.216.34",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks or []
        sock = FakeSocket(peer) if peer else None
        self.raw = SimpleNamespace(_connection=SimpleNamespace(sock=sock))
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int) -> list[bytes]:
        assert chunk_size > 0
        return self.chunks

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append((method, url, kwargs))
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        remote.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                remote.socket.AF_INET,
                remote.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ],
    )


def test_rejects_loopback_before_connecting(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession([])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="non-public"):
        remote.fetch_remote_content(
            "http://127.0.0.1/internal.wav",
            max_bytes=1024,
        )

    assert session.requests == []
    assert session.closed


def test_rejects_hostname_resolving_to_private_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession([])
    monkeypatch.setattr(remote, "_new_session", lambda: session)
    monkeypatch.setattr(
        remote.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                remote.socket.AF_INET,
                remote.socket.SOCK_STREAM,
                6,
                "",
                ("10.0.0.7", 443),
            )
        ],
    )

    with pytest.raises(remote.RemoteFetchError, match="non-public"):
        remote.fetch_remote_content(
            "https://media.example/internal.png",
            max_bytes=1024,
        )

    assert session.requests == []


def test_rejects_redirect_to_private_address(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(
        status_code=302,
        headers={"Location": "http://169.254.169.254/latest/meta-data"},
    )
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="non-public"):
        remote.fetch_remote_content(
            "https://media.example/image.png",
            max_bytes=1024,
        )

    assert len(session.requests) == 1
    assert response.closed


def test_rejects_private_connected_peer_after_public_dns(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(peer="127.0.0.1")
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="non-public"):
        remote.fetch_remote_content(
            "https://media.example/image.png",
            max_bytes=1024,
        )

    assert response.closed


def test_rejects_unverifiable_connected_peer(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(peer=None)
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="verify.*connection peer"):
        remote.fetch_remote_content(
            "https://media.example/image.png",
            max_bytes=1024,
        )

    assert response.closed


def test_rejects_oversized_declared_length(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(headers={"Content-Length": "1025"})
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="1024-byte limit"):
        remote.fetch_remote_content(
            "https://media.example/image.png",
            max_bytes=1024,
        )


def test_rejects_oversized_streamed_body(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(chunks=[b"a" * 700, b"b" * 400])
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="1024-byte limit"):
        remote.fetch_remote_content(
            "https://media.example/image.png",
            max_bytes=1024,
        )


def test_returns_bounded_content_and_normalized_media_type(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(
        headers={
            "Content-Type": "Image/PNG; charset=binary",
            "Content-Length": "7",
        },
        chunks=[b"png", b"data"],
        peer="93.184.216.34",
    )
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    result = remote.fetch_remote_content(
        "https://media.example/image.png",
        max_bytes=1024,
    )

    assert result.content == b"pngdata"
    assert result.content_type == "image/png"
    assert result.url == "https://media.example/image.png"
    assert session.requests[0][2]["allow_redirects"] is False
    assert session.requests[0][2]["stream"] is True
    assert response.closed
    assert session.closed


def test_session_ignores_ambient_proxy_and_credentials() -> None:
    session = remote._new_session()
    try:
        assert session.trust_env is False
    finally:
        session.close()


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("file:///tmp/image.png", "http or https"),
        ("https:///image.png", "hostname"),
        ("https://user:secret@example.com/image.png", "credentials"),
        ("https://example.com:invalid/image.png", "invalid port"),
    ],
)
def test_rejects_malformed_or_credentialed_urls(url: str, message: str) -> None:
    with pytest.raises(remote.RemoteFetchError, match=message):
        remote._validate_public_url(url)


def test_accepts_public_literal_address() -> None:
    remote._validate_public_url("https://93.184.216.34/image.png")


def test_rejects_invalid_address_value() -> None:
    with pytest.raises(remote.RemoteFetchError, match="Invalid remote address"):
        remote._validate_public_address("not-an-address")


@pytest.mark.parametrize(
    ("address", "embedded"),
    [
        ("64:ff9b::a9fe:a9fe", "169.254.169.254"),
        ("64:ff9b::7f00:1", "127.0.0.1"),
        ("64:ff9b::a00:1", "10.0.0.1"),
        ("64:ff9b::c0a8:1", "192.168.0.1"),
        ("64:ff9b:1:a9fe:a9:fe00::", "169.254.169.254"),
        ("::a9fe:a9fe", "169.254.169.254"),
        ("::7f00:1", "127.0.0.1"),
        ("2002:a9fe:a9fe::", "169.254.169.254"),
        ("2002:7f00:1::", "127.0.0.1"),
        ("::ffff:169.254.169.254", "169.254.169.254"),
    ],
)
def test_rejects_ipv6_transition_address_embedding_private_ipv4(
    address: str, embedded: str
) -> None:
    """NAT64, 6to4 and IPv4-compatible wrappers must not smuggle a private IPv4."""
    assert remote._embedded_ipv4(
        ipaddress.IPv6Address(address)
    ) == ipaddress.IPv4Address(embedded)

    with pytest.raises(remote.RemoteFetchError, match="non-public address"):
        remote._validate_public_address(address)

    with pytest.raises(remote.RemoteFetchError, match="non-public address"):
        remote._validate_public_url(f"http://[{address}]/latest/meta-data/")


@pytest.mark.parametrize(
    "address",
    [
        "64:ff9b::8.8.8.8",
        "64:ff9b::5db8:d822",
        "2002:5db8:d822::",
        "2606:4700:4700::1111",
    ],
)
def test_accepts_ipv6_transition_address_embedding_public_ipv4(address: str) -> None:
    """IPv6-only egress through NAT64 keeps working for public destinations."""
    remote._validate_public_address(address)


@pytest.mark.parametrize("result", [socket.gaierror(), []])
def test_rejects_unresolvable_hostname(
    monkeypatch: pytest.MonkeyPatch,
    result: BaseException | list[Any],
) -> None:
    def resolve(*_args: Any, **_kwargs: Any) -> list[Any]:
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(remote.socket, "getaddrinfo", resolve)

    with pytest.raises(remote.RemoteFetchError, match="Unable to resolve"):
        remote._validate_public_url("http://missing.example/image.png")


def test_follows_safe_redirect_and_closes_intermediate_response(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    redirect = FakeResponse(status_code=302, headers={"Location": "/final.png"})
    final = FakeResponse(headers={"Content-Type": "image/png"}, chunks=[b"ok"])
    session = FakeSession([redirect, final])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    result = remote.fetch_remote_content(
        "https://media.example/start.png",
        max_bytes=1024,
    )

    assert result.url == "https://media.example/final.png"
    assert redirect.closed
    assert final.closed


def test_rejects_redirect_without_location(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(status_code=302)
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="missing a location"):
        remote.fetch_remote_content("https://media.example/start", max_bytes=1024)

    assert response.closed


def test_rejects_too_many_redirects(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    responses = [
        FakeResponse(status_code=302, headers={"Location": f"/{index}"})
        for index in range(remote.MAX_REDIRECTS + 1)
    ]
    session = FakeSession(responses)
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="exceeded.*redirects"):
        remote.fetch_remote_content("https://media.example/start", max_bytes=1024)


def test_probes_content_type_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(headers={"Content-Type": "Audio/WAV; rate=44100"})
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    assert remote.probe_remote_content_type("https://media.example/clip") == "audio/wav"
    assert session.requests[0][0] == "HEAD"
    assert response.closed
    assert session.closed


def test_probe_closes_session_when_url_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession([])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError):
        remote.probe_remote_content_type("http://127.0.0.1/clip")

    assert session.closed


@pytest.mark.parametrize("declared_length", ["invalid", "-1"])
def test_rejects_invalid_declared_length(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
    declared_length: str,
) -> None:
    response = FakeResponse(headers={"Content-Length": declared_length})
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    with pytest.raises(remote.RemoteFetchError, match="invalid Content-Length"):
        remote.fetch_remote_content("https://media.example/image", max_bytes=1024)


def test_ignores_empty_stream_chunks_and_missing_content_type(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,  # noqa: ARG001
) -> None:
    response = FakeResponse(chunks=[b"", b"ok"])
    session = FakeSession([response])
    monkeypatch.setattr(remote, "_new_session", lambda: session)

    result = remote.fetch_remote_content(
        "https://media.example/image",
        max_bytes=1024,
    )

    assert result.content == b"ok"
    assert result.content_type is None


def test_rejects_nonpositive_download_limit() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        remote.fetch_remote_content("https://media.example/image", max_bytes=0)
