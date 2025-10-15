import pytest
import httpx
import requests

from gradient_adk.runtime.network_interceptor import (
    get_network_interceptor,
    setup_digitalocean_interception,
    NetworkInterceptor,
)


@pytest.fixture(autouse=True)
def reset_interceptor():
    """
    Ensure global interceptor is clean & unpatched before/after each test.
    """
    interceptor = get_network_interceptor()
    # Hard reset before
    interceptor.stop_intercepting()
    with interceptor._lock:  # ok to touch internals in unit tests
        interceptor._tracked_endpoints.clear()
        interceptor._detected_endpoints.clear()
    yield
    # Hard reset after
    interceptor.stop_intercepting()
    with interceptor._lock:
        interceptor._tracked_endpoints.clear()
        interceptor._detected_endpoints.clear()


@pytest.fixture
def stub_http_clients(monkeypatch):
    """
    Patch httpx/requests *original* methods with local stubs so that when the
    interceptor saves "original" methods, it saves these harmless stubs
    (no real network!).
    """

    async def async_request_stub(self, method, url, **kwargs):
        return httpx.Response(200, request=httpx.Request(method, url))

    async def async_send_stub(self, request, **kwargs):
        return httpx.Response(200, request=request)

    def sync_request_stub(self, method, url, **kwargs):
        return httpx.Response(200, request=httpx.Request(method, url))

    def sync_send_stub(self, request, **kwargs):
        return httpx.Response(200, request=request)

    def requests_request_stub(self, method, url, **kwargs):
        class _R:
            status_code = 200
            text = "ok"

            def __init__(self, u):
                self.url = u

        return _R(url)

    # Patch BEFORE start_intercepting so these become the "originals"
    monkeypatch.setattr(httpx.AsyncClient, "request", async_request_stub)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send_stub)
    monkeypatch.setattr(httpx.Client, "request", sync_request_stub)
    monkeypatch.setattr(httpx.Client, "send", sync_send_stub)
    monkeypatch.setattr(requests.Session, "request", requests_request_stub)

    return {
        "async_request_stub": async_request_stub,
        "async_send_stub": async_send_stub,
        "sync_request_stub": sync_request_stub,
        "sync_send_stub": sync_send_stub,
        "requests_request_stub": requests_request_stub,
    }


def test_add_and_remove_endpoint_patterns():
    ni = get_network_interceptor()
    assert isinstance(ni, NetworkInterceptor)

    with ni._lock:
        assert "foo" not in ni._tracked_endpoints

    ni.add_endpoint_pattern("foo")
    with ni._lock:
        assert "foo" in ni._tracked_endpoints

    ni.remove_endpoint_pattern("foo")
    with ni._lock:
        assert "foo" not in ni._tracked_endpoints


def test_clear_and_copy_of_detected_set():
    ni = get_network_interceptor()
    ni.add_endpoint_pattern("example.com")

    # Simulate detection
    ni._record_request("https://example.com/v1/models")
    assert ni.was_endpoint_called("example.com") is True

    # get_detected_endpoints returns a COPY
    s1 = ni.get_detected_endpoints()
    s1.add("https://fake/should_not_leak")
    s2 = ni.get_detected_endpoints()
    assert "https://fake/should_not_leak" not in s2

    ni.clear_detected()
    assert ni.get_detected_endpoints() == set()
    assert ni.was_endpoint_called("example.com") is False


def test_was_endpoint_called_substring_semantics():
    ni = get_network_interceptor()
    ni.add_endpoint_pattern("api.foo.com")

    ni._record_request("https://api.foo.com/resource/123?x=y")
    assert ni.was_endpoint_called("api.foo.com") is True
    # Substring semantics: pattern 'resource/123' appears in the detected URL
    assert ni.was_endpoint_called("resource/123") is True
    # Non-matching substring
    assert ni.was_endpoint_called("not-present") is False


@pytest.mark.asyncio
async def test_httpx_async_request_is_recorded_and_restored(stub_http_clients):
    ni = get_network_interceptor()
    ni.add_endpoint_pattern("example.com")

    # Keep references to the stubbed methods so we can verify restore
    original_async_req = httpx.AsyncClient.request

    ni.start_intercepting()
    try:
        # After start, method should have been wrapped (thus different object)
        assert httpx.AsyncClient.request is not original_async_req

        async with httpx.AsyncClient() as client:
            resp = await client.get("https://example.com/test")
            assert resp.status_code == 200

        assert ni.was_endpoint_called("example.com") is True
    finally:
        ni.stop_intercepting()

    # After stop, original stub should be restored
    assert httpx.AsyncClient.request is original_async_req


def test_httpx_sync_request_is_recorded_and_restored(stub_http_clients):
    ni = get_network_interceptor()
    ni.add_endpoint_pattern("sync.example.com")

    original_sync_req = httpx.Client.request
    ni.start_intercepting()
    try:
        assert httpx.Client.request is not original_sync_req
        with httpx.Client() as client:
            resp = client.get("https://sync.example.com/ok")
            assert resp.status_code == 200
        assert ni.was_endpoint_called("sync.example.com") is True
    finally:
        ni.stop_intercepting()
    assert httpx.Client.request is original_sync_req


def test_requests_session_is_recorded_and_restored(stub_http_clients):
    ni = get_network_interceptor()
    ni.add_endpoint_pattern("requests.example.com")

    original_req = requests.Session.request
    ni.start_intercepting()
    try:
        assert requests.Session.request is not original_req
        s = requests.Session()
        r = s.get("https://requests.example.com/hello")
        assert getattr(r, "status_code", None) == 200
        assert ni.was_endpoint_called("requests.example.com") is True
    finally:
        ni.stop_intercepting()
    assert requests.Session.request is original_req


def test_start_stop_idempotent(stub_http_clients):
    ni = get_network_interceptor()
    ni.add_endpoint_pattern("idempotent.test")

    # Calling start twice should not double-wrap or blow up
    ni.start_intercepting()
    first_async_req = httpx.AsyncClient.request
    ni.start_intercepting()
    second_async_req = httpx.AsyncClient.request
    assert first_async_req is second_async_req

    # Calling stop twice should be safe and restore originals
    ni.stop_intercepting()
    after_first_stop = httpx.AsyncClient.request
    ni.stop_intercepting()
    after_second_stop = httpx.AsyncClient.request
    assert after_first_stop is after_second_stop  # still restored


@pytest.mark.asyncio
async def test_setup_digitalocean_interception_records(stub_http_clients):
    ni = get_network_interceptor()

    # Verify patterns are empty before
    with ni._lock:
        assert len(ni._tracked_endpoints) == 0

    # setup_digitalocean_interception should add patterns and start intercepting
    setup_digitalocean_interception()

    try:
        with ni._lock:
            # Contains both prod and test inference patterns
            assert any("inference.do-ai.run" in p for p in ni._tracked_endpoints)
            assert any("inference.do-ai-test.run" in p for p in ni._tracked_endpoints)

        async with httpx.AsyncClient() as client:
            r = await client.get("https://inference.do-ai.run/v1/models")
            assert r.status_code == 200

        assert ni.was_endpoint_called("inference.do-ai.run") is True
    finally:
        ni.stop_intercepting()
