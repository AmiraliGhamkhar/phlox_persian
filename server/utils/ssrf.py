"""SSRF guard for user-supplied URLs fetched server-side.

Endpoints such as ``/api/config/validate-url`` and the vision-capability probe
fetch arbitrary URLs configured by the user. Local/LAN model servers (Ollama,
llama.cpp) are a legitimate target, so loopback and RFC1918 ranges stay
allowed — but cloud metadata (link-local), broadcast, reserved and multicast
targets are blocked. Scheme is restricted to http/https.

Beyond the static ``validate_fetch_url`` check, this module ships a
DNS-rebinding-safe transport: every request is resolved **once**, validated,
and then *connected to the validated IP* (with the original hostname carried
in the Host header and, for TLS, in the SNI extension). A hostname whose DNS
changes between the check and the fetch therefore cannot redirect the request
onto a blocked target (API7:2023 SSRF / A01:2025).

``build_guarded_http_client()`` returns an ``httpx.AsyncClient`` using that
transport, so callers only need to swap their client construction.
"""

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = {"http", "https"}


@dataclass(frozen=True)
class ResolvedTarget:
    """A validated, normalised fetch target."""

    scheme: str
    host: str  # original hostname (no brackets for IPv6)
    port: int
    ips: tuple[str, ...]  # validated addresses, in resolver order
    path: str
    query: str

    @property
    def host_header(self) -> str:
        """Value for the HTTP Host header (brackets IPv6 literals)."""
        host_for_header = f"[{self.host}]" if ":" in self.host else self.host
        if self.port in (80, 443):
            return host_for_header
        return f"{host_for_header}:{self.port}"

    def url_for_ip(self, ip: str) -> str:
        """URL that connects directly to ``ip`` but keeps the original path."""
        ip_for_url = f"[{ip}]" if ":" in ip else ip
        path = self.path or "/"
        if self.query:
            path = f"{path}?{self.query}"
        return f"{self.scheme}://{ip_for_url}:{self.port}{path}"


def _blocked_reason(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a reason string if this resolved address must not be fetched."""
    # Loopback and private ranges are intentionally allowed (local model
    # servers are the primary use case). Everything below is not.
    if ip.is_link_local:  # 169.254.0.0/16, fe80::/10 — cloud metadata lives here
        return "link-local addresses are blocked (cloud metadata)"
    if ip.is_unspecified:  # 0.0.0.0 / ::
        return "unspecified address"
    if ip.is_multicast:
        return "multicast address"
    if ip.is_reserved:
        return "reserved address"
    if ip.is_loopback:
        return None  # explicitly allowed
    return None


def _parse_and_validate(url: str) -> ResolvedTarget:
    """Parse ``url`` and resolve/validate its host in one shot."""
    if not url or not url.strip():
        raise ValueError("URL is empty")

    try:
        parts = urlsplit(url.strip())
    except ValueError as e:
        raise ValueError("URL could not be parsed") from e

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("Only http:// and https:// URLs are allowed")

    host = parts.hostname
    if not host:
        raise ValueError("URL has no host")

    # Reject obviously non-domain hosts early (credentials in URL, etc.)
    if "@" in (parts.netloc or ""):
        raise ValueError("URLs with embedded credentials are not allowed")

    port = parts.port or (443 if parts.scheme.lower() == "https" else 80)

    try:
        addr_infos = socket_getaddrinfo(host, port)
    except OSError as e:
        raise ValueError(f"Host could not be resolved: {host}") from e

    ips: list[str] = []
    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:  # pragma: no cover - getaddrinfo returns IPs
            continue
        reason = _blocked_reason(ip)
        if reason:
            raise ValueError(f"Blocked: {host} resolves to {ip} ({reason})")
        if sockaddr[0] not in ips:
            ips.append(sockaddr[0])

    if not ips:
        raise ValueError(f"Host could not be resolved: {host}")

    return ResolvedTarget(
        scheme=parts.scheme.lower(),
        host=host,
        port=port,
        ips=tuple(ips),
        path=parts.path,
        query=parts.query,
    )


def socket_getaddrinfo(host: str, port: int | None = None):
    """Resolve ``host`` (kept as a tiny indirection for testability)."""
    return socket.getaddrinfo(host, port)


def resolve_validated_target(url: str) -> ResolvedTarget:
    """Resolve ``url`` once and validate every address it maps to.

    Raises ``ValueError`` for blocked schemes/targets. The returned target
    carries the validated IPs so the caller can connect directly to one of
    them instead of trusting a second DNS resolution (DNS-rebinding fix).
    """
    return _parse_and_validate(url)


def validate_fetch_url(url: str) -> None:
    """Raise ValueError if the URL must not be fetched by the server.

    Checks the scheme and resolves the host, rejecting destinations whose
    resolved addresses are link-local/unspecified/multicast/reserved.
    Callers convert ValueError into a 4xx response.
    """
    _parse_and_validate(url)


async def _pinned_request(
    request: httpx.Request, target: ResolvedTarget, ip: str
) -> httpx.Request:
    """Rewrite ``request`` to connect to ``ip``, preserving host/SNI."""
    headers = dict(request.headers)
    headers.setdefault("Host", target.host_header)
    extensions = dict(request.extensions or {})
    if target.scheme == "https":
        extensions["sni_hostname"] = target.host

    try:
        content = await request.aread()
    except httpx.StreamConsumed:  # pragma: no cover - fresh requests only
        content = None
    return httpx.Request(
        request.method,
        target.url_for_ip(ip),
        headers=headers,
        content=content,
        extensions=extensions,
    )


class GuardedPinnedTransport(httpx.AsyncBaseTransport):
    """httpx transport that pins every request to a validated resolved IP.

    Resolution happens once per request; the connection goes to that exact
    IP, so rebinding the hostname afterwards has no effect. The original
    hostname is preserved in the Host header and (for https) as the TLS SNI
    name, so certificates and virtual hosts keep working.
    """

    def __init__(self) -> None:
        self._inner: httpx.AsyncClient | None = None

    async def _get_inner(self) -> httpx.AsyncClient:
        if self._inner is None:
            self._inner = httpx.AsyncClient(follow_redirects=False)
        return self._inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        target = await asyncio.to_thread(resolve_validated_target, str(request.url))

        client = await self._get_inner()
        last_error: Exception | None = None
        for ip in target.ips:
            pinned = await _pinned_request(request, target, ip)
            try:
                return await client.send(pinned, stream=True)
            except httpx.TransportError as error:  # try next validated IP
                last_error = error
                logger.debug(
                    "Guarded fetch to %s via %s failed: %s",
                    target.host,
                    ip,
                    error,
                )
        if last_error is not None:
            raise last_error
        raise httpx.ConnectError(f"No reachable address for {target.host}")

    async def aclose(self) -> None:
        if self._inner is not None:
            await self._inner.aclose()
            self._inner = None


def build_guarded_http_client(**kwargs) -> httpx.AsyncClient:
    """Create an ``httpx.AsyncClient`` whose transport pins to validated IPs.

    Any caller-supplied kwargs (timeout, headers, follow_redirects, ...) are
    passed through; ``follow_redirects`` defaults to False so a validated
    host cannot bounce the request elsewhere.
    """
    kwargs.setdefault("follow_redirects", False)
    kwargs["transport"] = GuardedPinnedTransport()
    return httpx.AsyncClient(**kwargs)
