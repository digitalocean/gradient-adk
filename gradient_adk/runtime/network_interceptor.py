from __future__ import annotations
import threading
import json
from typing import Set, List, Dict, Any, Optional
import httpx, requests


class CapturedRequest:
    """Represents a captured HTTP request/response."""

    def __init__(
        self,
        request_payload: Optional[Dict[str, Any]] = None,
        response_payload: Optional[Dict[str, Any]] = None,
    ):
        self.request_payload = request_payload
        self.response_payload = response_payload


class NetworkInterceptor:
    """
    Generic network interceptor.
    - Tracks endpoint patterns to watch
    - Increments a monotonic counter on every matching request
    - Captures request/response payloads for tracked endpoints
    - Exposes snapshot_token() and hits_since(token)
    """

    def __init__(self):
        self._tracked_endpoints: Set[str] = set()
        self._hit_count: int = 0
        self._captured_requests: List[CapturedRequest] = (
            []
        )  # Capture request/response pairs
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

    def get_captured_requests_since(self, token: int) -> List[CapturedRequest]:
        """Get all captured requests that happened since the given token."""
        with self._lock:
            # Return captured requests from index token onwards
            if token < len(self._captured_requests):
                return self._captured_requests[token:]
            return []

    def clear_hits(self) -> None:
        """Optional: reset the counter (e.g., at request start)."""
        with self._lock:
            self._hit_count = 0
            self._captured_requests.clear()

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
            url_str = str(request.url)
            request_payload = _global_interceptor._extract_request_payload(request)
            _global_interceptor._record_request(url_str, request_payload)

            response = await _global_interceptor._original_httpx_send(
                self_client, request, **kwargs
            )

            # Don't read response body for streaming responses - it would buffer the entire stream!
            # Check if this is a streaming response by looking at headers or response type
            is_streaming = (
                response.headers.get("transfer-encoding") == "chunked" or
                "text/event-stream" in response.headers.get("content-type", "") or
                hasattr(response, "aiter_bytes") or
                hasattr(response, "aiter_lines")
            )
            
            if not is_streaming:
                response_payload = await _global_interceptor._extract_response_payload(
                    response
                )
                _global_interceptor._record_response(url_str, response_payload)
            # For streaming responses, we skip payload capture to preserve streaming behavior

            return response

        def intercepted_httpx_request(self_client, method, url, **kwargs):
            url_str = str(url)
            request_payload = _global_interceptor._extract_request_payload_from_kwargs(
                kwargs
            )
            _global_interceptor._record_request(url_str, request_payload)

            response = _global_interceptor._original_httpx_request(
                self_client, method, url, **kwargs
            )

            # Note: For async request method, we can't easily await response content
            # The send method is more reliable for capturing responses
            return response

        # patch httpx (sync)
        def intercepted_httpx_sync_send(self_client, request, **kwargs):
            url_str = str(request.url)
            request_payload = _global_interceptor._extract_request_payload(request)
            _global_interceptor._record_request(url_str, request_payload)

            response = _global_interceptor._original_httpx_sync_send(
                self_client, request, **kwargs
            )

            response_payload = _global_interceptor._extract_response_payload_sync(
                response
            )
            _global_interceptor._record_response(url_str, response_payload)

            return response

        def intercepted_httpx_sync_request(self_client, method, url, **kwargs):
            url_str = str(url)
            request_payload = _global_interceptor._extract_request_payload_from_kwargs(
                kwargs
            )
            _global_interceptor._record_request(url_str, request_payload)

            response = _global_interceptor._original_httpx_sync_request(
                self_client, method, url, **kwargs
            )

            return response

        # patch requests
        def intercepted_requests_request(self_session, method, url, **kwargs):
            url_str = str(url)
            request_payload = _global_interceptor._extract_request_payload_from_kwargs(
                kwargs
            )
            _global_interceptor._record_request(url_str, request_payload)

            response = _global_interceptor._original_requests_request(
                self_session, method, url, **kwargs
            )

            response_payload = (
                _global_interceptor._extract_response_payload_from_requests(response)
            )
            _global_interceptor._record_response(url_str, response_payload)

            return response

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
    def _is_tracked_url(self, url: str) -> bool:
        """Check if URL matches any tracked endpoint pattern."""
        for pattern in self._tracked_endpoints:
            if pattern in url:
                return True
        return False

    def _record_request(
        self, url: str, request_payload: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            if self._is_tracked_url(url):
                self._hit_count += 1
                # Create a new captured request record
                self._captured_requests.append(
                    CapturedRequest(request_payload=request_payload)
                )

    def _record_response(
        self, url: str, response_payload: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            if self._is_tracked_url(url) and self._captured_requests:
                # Update the most recent captured request with response
                self._captured_requests[-1].response_payload = response_payload

    def _extract_request_payload(self, request) -> Optional[Dict[str, Any]]:
        """Extract JSON payload from httpx Request object."""
        try:
            if hasattr(request, "content"):
                content = request.content
                if isinstance(content, bytes):
                    return json.loads(content.decode("utf-8"))
        except Exception:
            pass
        return None

    def _extract_request_payload_from_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract JSON payload from request kwargs."""
        try:
            if "json" in kwargs:
                return kwargs["json"]
            if "data" in kwargs:
                data = kwargs["data"]
                if isinstance(data, (str, bytes)):
                    return json.loads(data)
                return data
            if "content" in kwargs:
                content = kwargs["content"]
                if isinstance(content, bytes):
                    return json.loads(content.decode("utf-8"))
        except Exception:
            pass
        return None

    async def _extract_response_payload(self, response) -> Optional[Dict[str, Any]]:
        """Extract JSON payload from httpx async Response object."""
        try:
            # Read the response content
            content = await response.aread()
            if content:
                return json.loads(content.decode("utf-8"))
        except Exception:
            pass
        return None

    def _extract_response_payload_sync(self, response) -> Optional[Dict[str, Any]]:
        """Extract JSON payload from httpx sync Response object."""
        try:
            content = response.read()
            if content:
                return json.loads(content.decode("utf-8"))
        except Exception:
            pass
        return None

    def _extract_response_payload_from_requests(
        self, response
    ) -> Optional[Dict[str, Any]]:
        """Extract JSON payload from requests Response object."""
        try:
            return response.json()
        except Exception:
            pass
        return None


# Global instance
_global_interceptor = NetworkInterceptor()


def get_network_interceptor() -> NetworkInterceptor:
    return _global_interceptor


def setup_digitalocean_interception() -> None:
    intr = get_network_interceptor()
    intr.add_endpoint_pattern("inference.do-ai.run")
    intr.add_endpoint_pattern("inference.do-ai-test.run")
    intr.start_intercepting()
