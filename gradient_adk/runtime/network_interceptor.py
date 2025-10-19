from __future__ import annotations
import threading
from typing import Set
import httpx, requests


class NetworkInterceptor:
    """
    Minimal, counter-only interceptor.
    - Tracks endpoint patterns to watch
    - Increments a monotonic counter on every matching request (duplicates allowed)
    - Exposes snapshot_token() and hits_since(token)
    """

    def __init__(self):
        self._tracked_endpoints: Set[str] = set()
        self._hit_count: int = 0
        self._lock = threading.Lock()
        self._active = False
        # originals
        self._original_httpx_request = None
        self._original_httpx_send = None
        self._original_httpx_sync_request = None
        self._original_httpx_sync_send = None
        self._original_requests_request = None

    def add_endpoint_pattern(self, pattern: str) -> None:
        with self._lock:
            self._tracked_endpoints.add(pattern)

    def remove_endpoint_pattern(self, pattern: str) -> None:
        with self._lock:
            self._tracked_endpoints.discard(pattern)

    def snapshot_token(self) -> int:
        """Get the current counter value to diff later."""
        with self._lock:
            return self._hit_count

    def hits_since(self, token: int) -> int:
        """Return how many matching requests happened since token."""
        with self._lock:
            return max(0, self._hit_count - token)

    def clear_hits(self) -> None:
        """Optional: reset the counter (e.g., at request start)."""
        with self._lock:
            self._hit_count = 0

    def start_intercepting(self) -> None:
        if self._active:
            return

        # store originals
        self._original_httpx_request = httpx.AsyncClient.request
        self._original_httpx_send = httpx.AsyncClient.send
        self._original_httpx_sync_request = httpx.Client.request
        self._original_httpx_sync_send = httpx.Client.send
        self._original_requests_request = requests.Session.request

        # patch httpx (async)
        async def intercepted_httpx_send(self_client, request, **kwargs):
            _global_interceptor._record_request(str(request.url))
            return await _global_interceptor._original_httpx_send(
                self_client, request, **kwargs
            )

        def intercepted_httpx_request(self_client, method, url, **kwargs):
            _global_interceptor._record_request(str(url))
            return _global_interceptor._original_httpx_request(
                self_client, method, url, **kwargs
            )

        # patch httpx (sync)
        def intercepted_httpx_sync_send(self_client, request, **kwargs):
            _global_interceptor._record_request(str(request.url))
            return _global_interceptor._original_httpx_sync_send(
                self_client, request, **kwargs
            )

        def intercepted_httpx_sync_request(self_client, method, url, **kwargs):
            _global_interceptor._record_request(str(url))
            return _global_interceptor._original_httpx_sync_request(
                self_client, method, url, **kwargs
            )

        # patch requests
        def intercepted_requests_request(self_session, method, url, **kwargs):
            _global_interceptor._record_request(str(url))
            return _global_interceptor._original_requests_request(
                self_session, method, url, **kwargs
            )

        httpx.AsyncClient.send = intercepted_httpx_send
        httpx.AsyncClient.request = intercepted_httpx_request
        httpx.Client.send = intercepted_httpx_sync_send
        httpx.Client.request = intercepted_httpx_sync_request
        requests.Session.request = intercepted_requests_request

        self._active = True

    def stop_intercepting(self) -> None:
        if not self._active:
            return
        # restore originals
        if self._original_httpx_request:
            httpx.AsyncClient.request = self._original_httpx_request
        if self._original_httpx_send:
            httpx.AsyncClient.send = self._original_httpx_send
        if self._original_httpx_sync_request:
            httpx.Client.request = self._original_httpx_sync_request
        if self._original_httpx_sync_send:
            httpx.Client.send = self._original_httpx_sync_send
        if self._original_requests_request:
            requests.Session.request = self._original_requests_request
        self._active = False

    # ---- internal ----
    def _record_request(self, url: str) -> None:
        with self._lock:
            for pattern in self._tracked_endpoints:
                if pattern in url:
                    self._hit_count += 1
                    break


# Global instance
_global_interceptor = NetworkInterceptor()


def get_network_interceptor() -> NetworkInterceptor:
    return _global_interceptor


def setup_digitalocean_interception() -> None:
    intr = get_network_interceptor()
    intr.add_endpoint_pattern("inference.do-ai.run")
    intr.add_endpoint_pattern("inference.do-ai-test.run")
    intr.start_intercepting()
