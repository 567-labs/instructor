"""Safe, bounded HTTP fetching for remote multimodal content."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import NewConnectionError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_REDIRECTS = 5
_CHUNK_SIZE = 64 * 1024
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}

# IPv6 formats that carry an IPv4 destination inside them (RFC 6052, RFC 4291)
_NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL_USE = ipaddress.ip_network("64:ff9b:1::/48")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
_IPV4_COMPATIBLE_EXCLUDED = frozenset(
    {ipaddress.IPv6Address("::"), ipaddress.IPv6Address("::1")}
)


class RemoteFetchError(ValueError):
    """Raised when a remote media request is unsafe or exceeds its limits."""


@dataclass(frozen=True)
class RemoteContent:
    """Validated content returned by :func:`fetch_remote_content`."""

    content: bytes
    content_type: str | None
    url: str


def _normalized_content_type(value: str | None) -> str | None:
    if value is None:
        return None
    return value.split(";", 1)[0].strip().lower() or None


def _embedded_ipv4(address: ipaddress.IPv6Address) -> tuple[ipaddress.IPv4Address, ...]:
    """IPv4 addresses an IPv6 transition format carries inside it.

    ``is_global`` judges the wrapper, not the destination, so a transition address
    embedding a private or link-local IPv4 is reported as globally routable. On a
    host with NAT64/6to4/Teredo routing the traffic still reaches the embedded IPv4.
    """
    embedded: list[ipaddress.IPv4Address] = []

    if address.ipv4_mapped is not None:
        embedded.append(address.ipv4_mapped)
    if address.sixtofour is not None:
        embedded.append(address.sixtofour)
    if address.teredo is not None:
        embedded.extend(address.teredo)

    packed = int(address)
    if address in _NAT64_WELL_KNOWN:
        embedded.append(ipaddress.IPv4Address(packed & 0xFFFFFFFF))
    elif address in _NAT64_LOCAL_USE:
        # RFC 6052 splits the address around the reserved u-octet at bits 64-71
        embedded.append(
            ipaddress.IPv4Address(
                (((packed >> 64) & 0xFFFF) << 16) | ((packed >> 40) & 0xFFFF)
            )
        )
    elif address in _IPV4_COMPATIBLE and address not in _IPV4_COMPATIBLE_EXCLUDED:
        embedded.append(ipaddress.IPv4Address(packed & 0xFFFFFFFF))

    return tuple(embedded)


def _validate_public_address(address: str) -> None:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError as exc:
        raise RemoteFetchError(f"Invalid remote address: {address}") from exc

    candidates: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...] = (parsed,)
    if isinstance(parsed, ipaddress.IPv6Address):
        candidates += _embedded_ipv4(parsed)

    for candidate in candidates:
        if not candidate.is_global or candidate.is_multicast:
            raise RemoteFetchError(
                f"Remote media URL resolves to a non-public address: {address}"
            )


def _validate_public_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise RemoteFetchError("Remote media URL contains an invalid port") from exc

    if parsed.scheme not in {"http", "https"}:
        raise RemoteFetchError("Remote media URL must use http or https")
    if not parsed.hostname:
        raise RemoteFetchError("Remote media URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise RemoteFetchError("Remote media URL must not include credentials")

    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        _validate_public_address(parsed.hostname)
        return

    effective_port = port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {
            str(sockaddr[0])
            for _, _, _, _, sockaddr in socket.getaddrinfo(
                parsed.hostname,
                effective_port,
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise RemoteFetchError(
            f"Unable to resolve remote media hostname: {parsed.hostname}"
        ) from exc

    if not addresses:
        raise RemoteFetchError(
            f"Unable to resolve remote media hostname: {parsed.hostname}"
        )
    for address in addresses:
        _validate_public_address(address)


def _response_peer_address(response: requests.Response) -> str | None:
    """Best-effort extraction of the connected peer from urllib3 internals."""
    raw: Any = response.raw
    connection = getattr(raw, "_connection", None)
    sock = getattr(connection, "sock", None)
    if sock is None:
        fp = getattr(raw, "_fp", None)
        inner_fp = getattr(fp, "fp", None)
        raw_fp = getattr(inner_fp, "raw", None)
        sock = getattr(raw_fp, "_sock", None)
    if sock is None:
        return None
    peer = sock.getpeername()
    return str(peer[0]) if peer else None


def _validate_connected_peer(response: requests.Response) -> None:
    address = _response_peer_address(response)
    if address is None:
        raise RemoteFetchError("Unable to verify the remote media connection peer")
    _validate_public_address(address)


class _PublicHTTPConnection(HTTPConnection):
    def _new_conn(self) -> socket.socket:
        """Resolve once, check every address, then connect to an exact sockaddr.

        Keep the connection's host unchanged for Host, SNI and certificate
        verification. Every new socket (including reconnects) uses this policy.
        """
        try:
            addresses = socket.getaddrinfo(
                self.host, self.port, type=socket.SOCK_STREAM
            )
            for _, _, _, _, address in addresses:
                _validate_public_address(str(address[0]))

            last_error: OSError | None = None
            for family, socktype, proto, _, address in addresses:
                sock = socket.socket(family, socktype, proto)
                try:
                    if isinstance(self.timeout, (int, float)) or self.timeout is None:
                        sock.settimeout(self.timeout)
                    for option in self.socket_options or []:
                        sock.setsockopt(*option)
                    if self.source_address:
                        sock.bind(self.source_address)
                    sock.connect(address)
                    return sock
                except OSError as exc:
                    last_error = exc
                    sock.close()
            raise OSError("Unable to connect to a public media address") from last_error
        except OSError as exc:
            raise NewConnectionError(self, str(exc)) from exc


class _PublicHTTPSConnection(_PublicHTTPConnection, HTTPSConnection):
    """HTTPS keeps urllib3's TLS handling with the public-only socket factory."""


class _PublicHTTPPool(HTTPConnectionPool):
    ConnectionCls = _PublicHTTPConnection


class _PublicHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _PublicHTTPSConnection


class _PublicHTTPAdapter(HTTPAdapter):
    def init_poolmanager(
        self, connections: int, maxsize: int, block: bool = False, **kwargs: Any
    ) -> None:
        super().init_poolmanager(connections, maxsize, block=block, **kwargs)
        # Do not mutate urllib3's process-global pool-class map.
        self.poolmanager.pool_classes_by_scheme = {
            "http": _PublicHTTPPool,
            "https": _PublicHTTPSPool,
        }


def _new_session() -> requests.Session:
    session = requests.Session()
    # Media URLs must not inherit user credentials or route through ambient proxies.
    session.trust_env = False
    session.mount("http://", _PublicHTTPAdapter())
    session.mount("https://", _PublicHTTPAdapter())
    session.headers["User-Agent"] = "instructor-remote-media/1"
    return session


def _request_with_redirects(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: int | float,
) -> tuple[requests.Response, str]:
    current_url = url
    for _ in range(MAX_REDIRECTS + 1):
        _validate_public_url(current_url)
        response = session.request(
            method,
            current_url,
            allow_redirects=False,
            stream=True,
            timeout=timeout,
        )
        try:
            _validate_connected_peer(response)
            if response.status_code not in _REDIRECT_STATUSES:
                response.raise_for_status()
                return response, current_url

            location = response.headers.get("Location")
            if not location:
                raise RemoteFetchError("Remote media redirect is missing a location")
            next_url = urljoin(current_url, location)
            _validate_public_url(next_url)
            current_url = next_url
        except Exception:
            response.close()
            raise
        response.close()

    raise RemoteFetchError(f"Remote media URL exceeded {MAX_REDIRECTS} redirects")


def probe_remote_content_type(
    url: str,
    *,
    timeout: int | float = 30,
) -> str | None:
    """Return the content type of a safe remote URL without downloading its body."""
    session = _new_session()
    response: requests.Response | None = None
    try:
        response, _ = _request_with_redirects(session, "HEAD", url, timeout=timeout)
        return _normalized_content_type(response.headers.get("Content-Type"))
    finally:
        if response is not None:
            response.close()
        session.close()


def fetch_remote_content(
    url: str,
    *,
    max_bytes: int,
    timeout: int | float = 30,
) -> RemoteContent:
    """Fetch public HTTP(S) content while enforcing redirects and decoded size."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    session = _new_session()
    response: requests.Response | None = None
    try:
        response, final_url = _request_with_redirects(
            session,
            "GET",
            url,
            timeout=timeout,
        )
        declared_length = response.headers.get("Content-Length")
        if declared_length is not None:
            try:
                parsed_length = int(declared_length)
            except ValueError as exc:
                raise RemoteFetchError(
                    "Remote media response has an invalid Content-Length"
                ) from exc
            if parsed_length < 0:
                raise RemoteFetchError(
                    "Remote media response has an invalid Content-Length"
                )
            if parsed_length > max_bytes:
                raise RemoteFetchError(
                    f"Remote media exceeds the {max_bytes}-byte limit"
                )

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise RemoteFetchError(
                    f"Remote media exceeds the {max_bytes}-byte limit"
                )
            chunks.append(chunk)

        return RemoteContent(
            content=b"".join(chunks),
            content_type=_normalized_content_type(response.headers.get("Content-Type")),
            url=final_url,
        )
    finally:
        if response is not None:
            response.close()
        session.close()
