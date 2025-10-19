import asyncio
import threading
from typing import List
import pytest
import httpx
import requests

from gradient_adk.runtime.network_interceptor import (
    NetworkInterceptor,
    get_network_interceptor,
    setup_digitalocean_interception,
)


@pytest.fixture(autouse=True)
def reset_global_interceptor():
    """
    Ensure a clean singleton between tests:
      - stop intercepting (restores patched methods)
      - clear patterns and hits
    """
    intr = get_network_interceptor()
    try:
        intr.stop_intercepting()
    finally:
        # brute-force cleanup of internal state
        intr.clear_hits()
        # Not public, but safe for tests: nuke patterns set
        with intr._lock:
            intr._tracked_endpoints.clear()
            intr._original_httpx_request = None
            intr._original_httpx_send = None
            intr._original_httpx_sync_request = None
            intr._original_httpx_sync_send = None
            intr._original_requests_request = None
            intr._active = False
    yield
    # Best-effort cleanup even if a test failed mid-way
    try:
        intr.stop_intercepting()
    except Exception:
        pass
    intr.clear_hits()
    with intr._lock:
        intr._tracked_endpoints.clear()


@pytest.fixture
def intr() -> NetworkInterceptor:
    return get_network_interceptor()


@pytest.fixture
def stub_httpx_sync(monkeypatch):
    """
    Before start_intercepting(), replace httpx.Client.request/send with harmless stubs.
    This ensures start_intercepting() captures these stubs as 'originals'.
    """

    def _req(self, method, url, **kwargs):
        # minimal stub response
        return httpx.Response(200, request=httpx.Request(method, url))

    def _send(self, request, **kwargs):
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.Client, "request", _req, raising=True)
    monkeypatch.setattr(httpx.Client, "send", _send, raising=True)
    return _req, _send


@pytest.fixture
def stub_httpx_async(monkeypatch):
    async def _areq(self, method, url, **kwargs):
        return httpx.Response(200, request=httpx.Request(method, url))

    async def _asend(self, request, **kwargs):
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "request", _areq, raising=True)
    monkeypatch.setattr(httpx.AsyncClient, "send", _asend, raising=True)
    return _areq, _asend


@pytest.fixture
def stub_requests(monkeypatch):
    def _req(self, method, url, **kwargs):
        r = requests.Response()
        r.status_code = 200
        r.url = url
        r.request = requests.Request(method=method, url=url).prepare()
        return r

    monkeypatch.setattr(requests.Session, "request", _req, raising=True)
    return _req


def test_add_remove_patterns_and_hits(intr, stub_httpx_sync):
    intr.add_endpoint_pattern("match.me")
    intr.add_endpoint_pattern("also")
    intr.remove_endpoint_pattern("also")

    token0 = intr.snapshot_token()
    assert token0 == 0
    assert intr.hits_since(token0) == 0

    intr._record_request("http://foo.com/nope")
    intr._record_request("https://match.me/path")
    intr._record_request("http://bar/match.me?q=1")
    intr._record_request("https://no/again")

    # Only 2 URLs contained "match.me"
    assert intr.hits_since(token0) == 2

    token1 = intr.snapshot_token()
    intr._record_request("http://baz/match.me")
    assert intr.hits_since(token1) == 1

    intr.clear_hits()
    assert intr.snapshot_token() == 0
    assert intr.hits_since(0) == 0


def test_httpx_sync_interception_counts(intr, stub_httpx_sync):
    intr.add_endpoint_pattern("api.test")
    intr.start_intercepting()

    # Ensure methods are patched now
    c = httpx.Client()
    # Non-matching URL
    resp = c.get("http://example.com")
    assert resp.status_code == 200
    # Matching URL -> increments counter
    resp = c.get("http://api.test/resource")
    assert resp.status_code == 200

    assert intr.hits_since(0) == 1

    # Stop and ensure originals restored (i.e., not our interceptors anymore)
    orig_req, orig_send = stub_httpx_sync
    intr.stop_intercepting()

    assert httpx.Client.request is orig_req
    assert httpx.Client.send is orig_send


def test_requests_interception_counts(intr, stub_requests):
    intr.add_endpoint_pattern("billing.svc")
    intr.start_intercepting()

    s = requests.Session()

    # miss
    r1 = s.get("https://example.com")
    assert r1.status_code == 200

    # hit
    r2 = s.post("https://billing.svc/process")
    assert r2.status_code == 200

    assert intr.hits_since(0) == 1

    # Stop, originals restored
    orig_req = stub_requests
    intr.stop_intercepting()
    assert requests.Session.request is orig_req


def test_idempotent_start_stop(intr, stub_httpx_sync, stub_httpx_async, stub_requests):
    intr.add_endpoint_pattern("x")
    intr.start_intercepting()
    # Calling again shouldn't crash or re-wrap
    intr.start_intercepting()

    # And stop twice shouldn't crash
    intr.stop_intercepting()
    intr.stop_intercepting()


def test_thread_safety_recording(intr):
    intr.add_endpoint_pattern("hit.me")

    # We call the private method here to stress the lock deterministically
    urls: List[str] = [
        "http://hit.me/a",
        "http://hit.me/b",
        "http://nope",
        "http://hit.me/c",
        "http://still-no",
    ]

    # expect 3 hits per thread
    def worker():
        for u in urls:
            intr._record_request(u)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 3 hits * 10 threads
    assert intr.hits_since(0) == 30


def test_setup_digitalocean_interception(
    intertr_cleanup_guard=reset_global_interceptor,
):
    """
    Verify that:
      - patterns for prod & test inference domains are added
      - intercepting is started
      - a matching call increments the counter
    """
    setup_digitalocean_interception()
    intr = get_network_interceptor()

    # quick sanity: active and has patterns
    with intr._lock:
        patterns = set(intr._tracked_endpoints)
        active = intr._active
    assert active
    assert "inference.do-ai.run" in patterns
    assert "inference.do-ai-test.run" in patterns

    # We can't guarantee stubs for httpx/requests here (setup_* doesn't stub),
    # so we just exercise the internal recorder to confirm patterns work:
    intr._record_request("https://inference.do-ai.run/v1/chat")
    assert intr.hits_since(0) == 1
