"""Fixtures for A2A tests.

Ensures httpx methods are not monkey-patched by the network interceptor,
which can interfere with FastAPI TestClient.
"""

import httpx
import pytest

# Save pristine httpx methods at import time (before any test can patch them)
_pristine_client_send = httpx.Client.send
_pristine_client_request = httpx.Client.request
_pristine_async_send = httpx.AsyncClient.send
_pristine_async_request = httpx.AsyncClient.request


@pytest.fixture(autouse=True)
def _restore_httpx():
    """Restore pristine httpx methods around each test.

    The network_interceptor module patches httpx globally for request capture.
    This fixture ensures TestClient works correctly regardless of test ordering.
    """
    httpx.Client.send = _pristine_client_send
    httpx.Client.request = _pristine_client_request
    httpx.AsyncClient.send = _pristine_async_send
    httpx.AsyncClient.request = _pristine_async_request
    yield
    httpx.Client.send = _pristine_client_send
    httpx.Client.request = _pristine_client_request
    httpx.AsyncClient.send = _pristine_async_send
    httpx.AsyncClient.request = _pristine_async_request
