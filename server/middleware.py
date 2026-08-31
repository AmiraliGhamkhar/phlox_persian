"""FastAPI middleware classes."""

import asyncio
import ipaddress
import logging
import os
import secrets
import time
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Centralized path skip rules - add new React routes here
PUBLIC_PATHS = {"/", "/health", "/version", "/favicon.ico"}
REACT_ROUTES = {
    "/new-note",
    "/settings",
    "/rag",
    "/clinic-summary",
    "/outstanding-jobs",
}
STATIC_EXTENSIONS = (
    ".js",
    ".css",
    ".png",
    ".ico",
    ".svg",
    ".woff",
    ".woff2",
    ".webp",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ttf",
    ".eot",
    ".otf",
    ".map",
)


def should_skip_middleware(path: str, *, check_api: bool = False) -> bool:
    """Check if path should skip auth/rate-limiting middleware.

    Args:
        path: The request path to check
        check_api: If True, also skip non-API paths (for rate limiting)

    Returns:
        True if the path should skip middleware checks
    """
    # Public paths (health checks, etc.)
    if path in PUBLIC_PATHS:
        return True

    # Static assets (check /assets/ prefix and common extensions)
    if path.startswith("/assets/"):
        return True
    if not path.startswith("/api/") and any(path.endswith(ext) for ext in STATIC_EXTENSIONS):
        return True

    # React routes (SPA pages)
    if path in REACT_ROUTES or path.startswith("/patient"):
        return True

    # For rate limiting: skip non-API paths entirely
    return bool(check_api and not path.startswith("/api/"))


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP belongs to a private network (Docker/Localhost)."""
    try:
        return ipaddress.ip_address(ip_str).is_private
    except ValueError:
        return False


def _is_trusted_proxy(ip_str: str) -> bool:
    """Decide whether ``ip_str`` may set forwarded headers.

    When TRUSTED_PROXY_CIDRS is configured, only those networks are trusted.
    Otherwise any private-range peer is trusted (previous behaviour, kept for
    backwards compatibility with existing deployments).
    """
    from server.constants import TRUSTED_PROXY_NETWORKS

    if TRUSTED_PROXY_NETWORKS:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        return any(ip in network for network in TRUSTED_PROXY_NETWORKS)
    return _is_private_ip(ip_str)


# Hostnames the API always accepts (loopback/desktop + tests).
DEFAULT_ALLOWED_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "[::1]", "testserver"}
# Origins embedded in desktop shells (Tauri webview fetches).
SHELL_ORIGIN_HOSTNAMES = {"localhost"}


def _get_allowed_hostnames() -> set[str]:
    """Compute the Host-header allowlist from config."""
    from server.constants import ALLOWED_HOSTS, ALLOWED_ORIGINS

    allowed = set(DEFAULT_ALLOWED_HOSTNAMES) | set(ALLOWED_HOSTS)
    for origin in ALLOWED_ORIGINS:
        if origin == "*":
            continue
        try:
            host = urlsplit(origin if "//" in origin else f"//{origin}").hostname
        except ValueError:
            continue
        if host:
            allowed.add(host.lower())
    return allowed


def _hostname_of_host_header(host_header: str) -> str:
    """Extract the lowercase hostname from a Host header (handles [::1]:port)."""
    host = host_header.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    return host.split(":", 1)[0]


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a correlation id to every request.

    Echoes a safe client-supplied ``X-Request-Id`` or generates one. The value
    is stored on ``request.state`` and in a ContextVar so AI logs can include
    it without changing response JSON envelopes.
    """

    async def dispatch(self, request, call_next):
        from server.utils.request_context import (
            normalize_request_id,
            reset_request_id,
            set_request_id,
        )

        request_id = normalize_request_id(request.headers.get("x-request-id")) or secrets.token_hex(
            8
        )
        request.state.request_id = request_id
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            reset_request_id(token)


class HostValidationMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Host or Origin does not belong to this deployment.

    Two protections:
    - Host allowlist: defeats DNS-rebinding (an attacker page at evil.com that
      rebinds to 127.0.0.1 sends ``Host: evil.com``, which is not allowed).
    - Origin check on state-changing methods: defeats cross-site form POSTs to
      multipart endpoints, which browsers send without a CORS preflight.

    Requests without an Origin header (server-to-server clients, the desktop
    shell's HTTP plugin, healthchecks) pass the Origin check.
    """

    UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    async def dispatch(self, request, call_next):
        from server.constants import IS_TESTING

        if IS_TESTING:
            return await call_next(request)

        host_header = request.headers.get("host", "")
        allowed = _get_allowed_hostnames()

        if _hostname_of_host_header(host_header) not in allowed:
            logger.warning("Rejected request with disallowed Host header: %r", host_header)
            return JSONResponse(
                status_code=421,
                content={
                    "detail": "This host is not served by Phlox. "
                    "Set ALLOWED_HOSTS/ALLOWED_ORIGINS for your deployment."
                },
            )

        if request.method in self.UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and not self._origin_allowed(origin, allowed, host_header):
                logger.warning("Rejected %s with disallowed Origin: %r", request.method, origin)
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin request rejected"},
                )

        return await call_next(request)

    @staticmethod
    def _origin_allowed(origin: str, allowed: set[str], host_header: str) -> bool:
        from server.constants import ALLOWED_ORIGINS

        if origin == "null":
            return False
        if origin in ALLOWED_ORIGINS:
            return True
        try:
            parts = urlsplit(origin if "//" in origin else f"//{origin}")
            origin_host = (parts.hostname or "").lower()
        except ValueError:
            return False
        if not origin_host:
            return False
        if origin_host in allowed or origin_host in SHELL_ORIGIN_HOSTNAMES:
            return True
        # Same-host origin is always fine.
        return origin_host == _hostname_of_host_header(host_header)


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject API requests whose declared body exceeds the size caps.

    Works off the Content-Length header (fast, before any auth work). Bodies
    without Content-Length (chunked) are capped by read-limited uploads in the
    handlers themselves.
    """

    async def dispatch(self, request, call_next):
        from server.utils.request_limits import DEFAULT_API_BODY_LIMIT, TRANSCRIBE_API_BODY_LIMIT

        path = request.url.path
        if path.startswith("/api/"):
            limit = (
                TRANSCRIBE_API_BODY_LIMIT
                if path.startswith("/api/transcribe")
                else DEFAULT_API_BODY_LIMIT
            )
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > limit:
                logger.warning("Rejected oversized request to %s (%s bytes)", path, content_length)
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"},
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Restrict resources to same origin, allow inline scripts for React
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        return response


class TrustedProxyMiddleware(BaseHTTPMiddleware):
    """Extract real client IP from X-Forwarded-For header if from trusted proxy.

    Only trusts X-Forwarded-For when the direct connection comes from a
    trusted proxy — a CIDR listed in TRUSTED_PROXY_CIDRS when configured, or
    (backwards-compatible default) any private-range peer. This prevents
    clients from spoofing the header directly and from rotating rate-limit
    buckets or polluting the audit trail with forged IPs.
    """

    async def dispatch(self, request, call_next):
        client_host = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("x-forwarded-for")

        # Only trust X-Forwarded-For if the direct connection is a trusted proxy
        if forwarded_for and client_host != "unknown" and _is_trusted_proxy(client_host):
            # Take the first IP in the chain (original client)
            request.state.client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Fall back to the actual connecting IP
            request.state.client_ip = client_host

        return await call_next(request)


class LocalTokenMiddleware(BaseHTTPMiddleware):
    """Verify local request token on all API requests.


    This middleware protects the API from unauthorized access by other
    applications running on the same machine. Only requests with a valid
    Authorization: Bearer <token> header are allowed.
    """

    async def dispatch(self, request, call_next):
        from server.constants import is_docker_runtime
        from server.utils.local_request_token import get_request_token

        path = request.url.path

        # Skip middleware checks for public/static/React routes
        if should_skip_middleware(path):
            return await call_next(request)

        # In Docker mode, skip token validation. Re-evaluated per request
        # (not the import-time IS_DOCKER constant) so the skip can never be
        # triggered by a stale flag in a non-container runtime.
        if is_docker_runtime():
            logger.debug(f"Auth skipped - Docker mode (path: {path})")
            return await call_next(request)

        # Bare-metal web development (`npm run dev` sets PHLOX_DEV_BOOT=1)
        # has no Tauri IPC channel to deliver a token, so the same skip
        # applies — explicit opt-in only, same fail-closed philosophy.
        if os.environ.get("PHLOX_DEV_BOOT") == "1":
            logger.debug(f"Auth skipped - dev boot mode (path: {path})")
            return await call_next(request)

        # Get expected token
        expected_token = get_request_token()
        if not expected_token:
            logger.error(f"Auth fail-closed - no request token set (path: {path})")
            return JSONResponse(
                status_code=503,
                content={"detail": "Service not initialized"},
            )

        # Verify Authorization header. Browser WebSocket clients cannot set
        # custom headers, so the live transcription handshake carries the
        # token as a second ``Sec-WebSocket-Protocol`` subprotocol instead of
        # a query parameter. Query strings get written verbatim to uvicorn
        # access logs (and the desktop app log on disk), so a ``?token=``
        # handshake would persist the credential in log files (A09:2025).
        # The legacy ``?token=`` path is intentionally NOT accepted anymore.
        auth_header = request.headers.get("Authorization", "")
        provided_token = ""
        if auth_header.startswith("Bearer "):
            provided_token = auth_header[7:]  # remove "Bearer " prefix
        elif path.startswith("/api/transcribe/live"):
            offered = request.headers.get("Sec-WebSocket-Protocol", "") or ""
            protocols = [p.strip() for p in offered.split(",") if p.strip()]
            if "phlox-live" in protocols:
                for protocol in protocols:
                    if protocol != "phlox-live" and protocol:
                        provided_token = protocol
                        break

        if not provided_token:
            logger.debug(f"Missing Bearer header for {path}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )
        if not secrets.compare_digest(provided_token, expected_token):
            # No token material is logged — even a prefix could help a guesser.
            logger.warning(f"Invalid request token for {path}")
            return JSONResponse(status_code=403, content={"detail": "Invalid request token"})

        return await call_next(request)


class ProxyAuthMiddleware(BaseHTTPMiddleware):
    """Validate requests against proxy-passed user headers.

    For use with Authelia, Traefik, Caddy, etc. that pass authenticated
    user identity via headers after performing authentication.

    Only trusts the auth header when the direct connection is from a private IP
    (e.g., a reverse proxy on the same Docker network). This prevents clients
        from spoofing the header directly.
    """

    async def dispatch(self, request, call_next):
        from server.constants import (
            PROXY_AUTH_ALLOWED_USERS,
            PROXY_AUTH_ENABLED,
            PROXY_AUTH_USER_HEADER,
        )

        # Skip if disabled
        if not PROXY_AUTH_ENABLED:
            return await call_next(request)

        path = request.url.path

        # Skip middleware checks for public/static/React routes
        if should_skip_middleware(path):
            return await call_next(request)

        # Only trust auth header if coming from a trusted proxy (private IP)
        client_host = request.client.host if request.client else "unknown"
        if client_host == "unknown" or not _is_private_ip(client_host):
            # Direct connection from public IP - reject or fall through
            # Since proxy auth is enabled, we require the header
            logger.warning(f"Proxy auth header received from non-private IP: {client_host}")
            return JSONResponse(status_code=401, content={"detail": "Authentication required"})

        # Get user from header
        user = request.headers.get(PROXY_AUTH_USER_HEADER)

        if not user:
            return JSONResponse(status_code=401, content={"detail": "Authentication required"})

        if PROXY_AUTH_ALLOWED_USERS and user not in PROXY_AUTH_ALLOWED_USERS:
            logger.warning(f"Access denied for user: {user}")
            return JSONResponse(status_code=403, content={"detail": "Access denied"})

        # Store user for downstream use and bind it to the request context so
        # background tasks (pending-action confirmations, patient access)
        # can enforce object-level authorization (API1:2023).
        request.state.user = user
        from server.utils.request_context import (
            reset_request_actor,
            set_request_actor,
        )

        actor_token = set_request_actor(user)
        try:
            response = await call_next(request)
        finally:
            reset_request_actor(actor_token)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit API requests to prevent abuse and data exfiltration.

    Token-bucket algorithm with in-memory storage: each (client, endpoint)
    bucket starts with ``rate * burst`` tokens, refills continuously at
    ``rate / 60`` tokens per second, and consumes one token per request. This
    gives a genuine short burst allowance while the sustained rate stays at
    ``rate`` requests per minute (the previous sliding-window code effectively
    allowed ``rate * burst`` per minute permanently).
    """

    # Endpoint-specific limits: (requests_per_minute, burst_multiplier)
    RATE_LIMITS = {
        "/api/transcribe": (10, 2),
        "/api/chat": (30, 2),
        "/api/rag": (20, 2),
        "/api/config": (30, 2),
        "/api/templates": (30, 2),
        "/api/letter": (30, 2),
        "/api/dashboard": (30, 2),
    }
    DEFAULT_LIMIT = (60, 2)  # requests_per_minute, burst_multiplier

    # Patient endpoints need special handling
    PATIENT_LIST_LIMIT = (10, 2)  # Prevents bulk enumeration
    PATIENT_DETAIL_LIMIT = (20, 2)  # Normal browsing allowed

    WINDOW_SECONDS = 60

    # Buckets: (client_ip, endpoint) -> [tokens, last_refill_ts]
    # Class-level so the scheduler's cleanup job sees the same state.
    _buckets: dict[tuple[str, str], list[float]] = {}
    _lock = asyncio.Lock()

    def _get_limit_for_path(self, path: str) -> tuple[int, int]:
        """Get rate limit for a given path."""
        from server.constants import IS_DOCKER, RATE_LIMIT_DESKTOP_MULTIPLIER

        # Check for patient list vs detail
        if path == "/api/note/list" or path == "/api/note/list/":
            rate, burst = self.PATIENT_LIST_LIMIT
        elif path.startswith("/api/note/"):
            rate, burst = self.PATIENT_DETAIL_LIMIT
        else:
            # Check other endpoints
            matched = False
            for prefix, limit in self.RATE_LIMITS.items():
                if path.startswith(prefix):
                    rate, burst = limit
                    matched = True
                    break
            if not matched:
                rate, burst = self.DEFAULT_LIMIT

        if not IS_DOCKER:
            rate = rate * RATE_LIMIT_DESKTOP_MULTIPLIER

        return rate, burst

    def _get_endpoint_key(self, path: str) -> str:
        """Get endpoint key for rate limiting (groups related paths)."""
        if path.startswith("/api/note/") and not path.startswith("/api/note/list"):
            # Group all individual note requests
            return "/api/note/detail"
        for prefix in self.RATE_LIMITS:
            if path.startswith(prefix):
                return prefix
        return "/api/default"

    @classmethod
    async def cleanup_all_zombie_ips(cls):
        """Background task to clean up buckets that never returned.

        Called periodically by the scheduler to prevent memory accumulation
        from port scanners or one-off requests.
        """
        now = time.time()
        stale_keys = [
            key
            for key, (_tokens, last_refill) in cls._buckets.items()
            if now - last_refill > cls.WINDOW_SECONDS * 2
        ]
        for key in stale_keys:
            cls._buckets.pop(key, None)

        if stale_keys:
            logger.debug("Cleaned up %d stale rate-limit buckets", len(stale_keys))

    async def dispatch(self, request, call_next):
        from server.constants import RATE_LIMIT_ENABLED

        # Skip if rate limiting is disabled
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)

        path = request.url.path

        # Skip middleware checks for public/static/React routes, and non-API paths
        if should_skip_middleware(path, check_api=True):
            return await call_next(request)

        # Get client IP (set by TrustedProxyMiddleware)
        client_ip = getattr(request.state, "client_ip", "unknown")

        # Get rate limit for this endpoint
        rate_limit, burst_multiplier = self._get_limit_for_path(path)
        endpoint = self._get_endpoint_key(path)

        now = time.time()
        capacity = float(rate_limit * burst_multiplier)
        refill_per_second = rate_limit / self.WINDOW_SECONDS
        key = (client_ip, endpoint)

        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = [capacity, now]
                self._buckets[key] = bucket

            # Refill tokens accrued since the last request.
            elapsed = max(0.0, now - bucket[1])
            bucket[0] = min(capacity, bucket[0] + elapsed * refill_per_second)
            bucket[1] = now

            if bucket[0] >= 1.0:
                bucket[0] -= 1.0
                allowed = True
            else:
                allowed = False

            remaining = int(bucket[0])
            retry_after = int(max(1, (1.0 - bucket[0]) / refill_per_second)) if not allowed else 0

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_ip} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "retry_after": retry_after,
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + retry_after)),
                    "Retry-After": str(retry_after),
                },
            )

        # Process request and add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + self.WINDOW_SECONDS))

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Record every API request to the audit_log table.

    Runs after TrustedProxy (so ``client_ip`` is populated) and wraps the auth
    middlewares, so authenticated requests, auth denials, and rate-limited
    responses are all recorded. Stores request identifiers only — never bodies
    or PHI content. Audit failures never propagate: logging is best-effort.
    """

    async def dispatch(self, request, call_next):
        from server.database.repositories.audit import log_event

        path = request.url.path

        # Only audit real API traffic. Skip:
        #  - the audit endpoints themselves (write-on-read loop)
        #  - the frontend config-status poller (fires every ~15s, would dominate
        #    the log and drown out real access events)
        if not path.startswith("/api/") or path.startswith("/api/audit"):
            return await call_next(request)
        if path == "/api/config/status" and request.method == "GET":
            return await call_next(request)

        actor = getattr(request.state, "user", "local")
        client_ip = getattr(request.state, "client_ip", None)

        start = time.monotonic()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            # Request never produced a response; record as 500 and re-raise.
            # Offloaded to a worker thread: the audit write takes the global DB
            # lock and must not block the event loop on every API request.
            await asyncio.to_thread(
                log_event,
                method=request.method,
                path=path,
                status=500,
                actor=actor,
                client_ip=client_ip,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
            raise

        await asyncio.to_thread(
            log_event,
            method=request.method,
            path=path,
            status=status,
            actor=actor,
            client_ip=client_ip,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
        return response
