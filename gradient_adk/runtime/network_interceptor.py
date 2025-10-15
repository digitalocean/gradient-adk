"""
Network request interceptor for detecting LLM endpoint calls.

This module provides a shared utility for intercepting HTTP requests
to detect calls to specific endpoints (like DigitalOcean inference APIs)
that should be classified as LLM operations rather than generic tool calls.
"""

from __future__ import annotations

import threading
from typing import Set, Optional, Callable
from unittest.mock import patch
import httpx
import requests


class NetworkInterceptor:
    """
    A thread-safe network request interceptor that tracks calls to specific endpoints.

    This interceptor monkey-patches httpx and requests libraries to detect
    when HTTP requests are made to specified endpoint patterns.
    """

    def __init__(self):
        self._tracked_endpoints: Set[str] = set()
        self._detected_endpoints: Set[str] = set()
        self._lock = threading.Lock()
        self._active = False
        self._original_httpx_request = None
        self._original_httpx_send = None
        self._original_httpx_sync_request = None
        self._original_httpx_sync_send = None
        self._original_requests_request = None

    def add_endpoint_pattern(self, pattern: str) -> None:
        """Add an endpoint pattern to track."""
        with self._lock:
            self._tracked_endpoints.add(pattern)

    def remove_endpoint_pattern(self, pattern: str) -> None:
        """Remove an endpoint pattern from tracking."""
        with self._lock:
            self._tracked_endpoints.discard(pattern)

    def clear_detected(self) -> None:
        """Clear the set of detected endpoints."""
        with self._lock:
            self._detected_endpoints.clear()

    def was_endpoint_called(self, pattern: str) -> bool:
        """Check if a specific endpoint pattern was called."""
        with self._lock:
            return any(pattern in detected for detected in self._detected_endpoints)

    def get_detected_endpoints(self) -> Set[str]:
        """Get a copy of all detected endpoints."""
        with self._lock:
            return self._detected_endpoints.copy()

    def start_intercepting(self) -> None:
        """Start intercepting network requests."""
        if self._active:
            return

        # Store original methods
        self._original_httpx_request = httpx.AsyncClient.request
        self._original_httpx_send = httpx.AsyncClient.send
        self._original_httpx_sync_request = httpx.Client.request
        self._original_httpx_sync_send = httpx.Client.send
        self._original_requests_request = requests.Session.request

        # Monkey patch httpx AsyncClient.request
        def intercepted_httpx_request(self_client, method, url, **kwargs):
            _global_interceptor._record_request(str(url))
            return _global_interceptor._original_httpx_request(
                self_client, method, url, **kwargs
            )

        # Monkey patch httpx AsyncClient.send (this is what gradient SDK uses)
        async def intercepted_httpx_send(self_client, request, **kwargs):
            _global_interceptor._record_request(str(request.url))
            return await _global_interceptor._original_httpx_send(
                self_client, request, **kwargs
            )

        # Monkey patch httpx sync Client.request
        def intercepted_httpx_sync_request(self_client, method, url, **kwargs):
            _global_interceptor._record_request(str(url))
            return _global_interceptor._original_httpx_sync_request(
                self_client, method, url, **kwargs
            )

        # Monkey patch httpx sync Client.send
        def intercepted_httpx_sync_send(self_client, request, **kwargs):
            _global_interceptor._record_request(str(request.url))
            return _global_interceptor._original_httpx_sync_send(
                self_client, request, **kwargs
            )

        # Monkey patch requests
        def intercepted_requests_request(self_session, method, url, **kwargs):
            _global_interceptor._record_request(str(url))
            return _global_interceptor._original_requests_request(
                self_session, method, url, **kwargs
            )

        httpx.AsyncClient.request = intercepted_httpx_request
        httpx.AsyncClient.send = intercepted_httpx_send
        httpx.Client.request = intercepted_httpx_sync_request
        httpx.Client.send = intercepted_httpx_sync_send
        requests.Session.request = intercepted_requests_request

        self._active = True

    def stop_intercepting(self) -> None:
        """Stop intercepting network requests."""
        if not self._active:
            return

        # Restore original methods
        if self._original_httpx_request:
            httpx.AsyncClient.request = self._original_httpx_request
        if hasattr(self, "_original_httpx_send") and self._original_httpx_send:
            httpx.AsyncClient.send = self._original_httpx_send
        if (
            hasattr(self, "_original_httpx_sync_request")
            and self._original_httpx_sync_request
        ):
            httpx.Client.request = self._original_httpx_sync_request
        if (
            hasattr(self, "_original_httpx_sync_send")
            and self._original_httpx_sync_send
        ):
            httpx.Client.send = self._original_httpx_sync_send
        if self._original_requests_request:
            requests.Session.request = self._original_requests_request

        self._active = False

    def _record_request(self, url: str) -> None:
        """Record a request if it matches any tracked patterns."""
        with self._lock:
            for pattern in self._tracked_endpoints:
                if pattern in url:
                    self._detected_endpoints.add(url)
                    break


# Global instance for shared use across instrumentors
_global_interceptor = NetworkInterceptor()


def get_network_interceptor() -> NetworkInterceptor:
    """Get the global network interceptor instance."""
    return _global_interceptor


def setup_digitalocean_interception() -> None:
    """Setup interception for DigitalOcean inference endpoints."""
    interceptor = get_network_interceptor()
    interceptor.add_endpoint_pattern("inference.do-ai.run")
    interceptor.add_endpoint_pattern("inference.do-ai-test.run")
    interceptor.start_intercepting()
