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
    create_adk_user_agent_hook,
    RequestHook,
)


@pytest.fixture(autouse=True)
def reset_global_interceptor():
    """
    Ensure a clean singleton between tests:
      - stop intercepting (restores patched methods)
      - clear patterns, hits, and hooks
    """
    intr = get_network_interceptor()
    try:
        intr.stop_intercepting()
    finally:
        # brute-force cleanup of internal state
        intr.clear_hits()
        # Not public, but safe for tests: nuke patterns set and hooks
        with intr._lock:
            intr._tracked_endpoints.clear()
            intr._request_hooks.clear()
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
        intr._request_hooks.clear()


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


# ---- Request Hook Tests ----


def test_add_request_hook(intr):
    """Test that hooks can be registered."""
    assert len(intr._request_hooks) == 0

    def my_hook(url: str, headers: dict) -> dict:
        headers["X-Custom"] = "value"
        return headers

    intr.add_request_hook(my_hook)
    assert len(intr._request_hooks) == 1
    assert intr._request_hooks[0] is my_hook


def test_apply_request_hooks_empty(intr):
    """Test applying hooks when none are registered."""
    result = intr._apply_request_hooks("http://example.com", {"Existing": "header"})
    assert result == {"Existing": "header"}


def test_apply_request_hooks_single(intr):
    """Test applying a single hook."""

    def add_custom_header(url: str, headers: dict) -> dict:
        headers["X-Custom"] = "added"
        return headers

    intr.add_request_hook(add_custom_header)
    result = intr._apply_request_hooks("http://example.com", {"Existing": "header"})
    assert result == {"Existing": "header", "X-Custom": "added"}


def test_apply_request_hooks_multiple(intr):
    """Test applying multiple hooks in order."""

    def hook1(url: str, headers: dict) -> dict:
        headers["X-First"] = "1"
        return headers

    def hook2(url: str, headers: dict) -> dict:
        headers["X-Second"] = "2"
        return headers

    intr.add_request_hook(hook1)
    intr.add_request_hook(hook2)

    result = intr._apply_request_hooks("http://example.com", {})
    assert result == {"X-First": "1", "X-Second": "2"}


def test_apply_request_hooks_with_none_headers(intr):
    """Test that hooks handle None headers gracefully."""

    def add_header(url: str, headers: dict) -> dict:
        headers["X-Added"] = "value"
        return headers

    intr.add_request_hook(add_header)
    result = intr._apply_request_hooks("http://example.com", None)
    assert result == {"X-Added": "value"}


def test_apply_request_hooks_error_handling(intr):
    """Test that a failing hook doesn't break other hooks."""

    def failing_hook(url: str, headers: dict) -> dict:
        raise RuntimeError("Hook failed!")

    def working_hook(url: str, headers: dict) -> dict:
        headers["X-Works"] = "yes"
        return headers

    intr.add_request_hook(failing_hook)
    intr.add_request_hook(working_hook)

    # Should not raise, and working_hook should still apply
    result = intr._apply_request_hooks("http://example.com", {})
    assert result == {"X-Works": "yes"}


# ---- ADK User-Agent Hook Tests ----


def test_create_adk_user_agent_hook_matching_url():
    """Test that the hook modifies User-Agent for matching URLs."""
    hook = create_adk_user_agent_hook(
        version="1.2.3", url_patterns=["api.example.com", "api.test.com"]
    )

    headers = {"User-Agent": "MyClient/1.0"}
    result = hook("https://api.example.com/v1/chat", headers)

    assert result["User-Agent"] == "MyClient/1.0 adk-1.2.3"


def test_create_adk_user_agent_hook_non_matching_url():
    """Test that the hook doesn't modify User-Agent for non-matching URLs."""
    hook = create_adk_user_agent_hook(version="1.2.3", url_patterns=["api.example.com"])

    headers = {"User-Agent": "MyClient/1.0"}
    result = hook("https://other.domain.com/v1/chat", headers)

    # Should be unchanged
    assert result["User-Agent"] == "MyClient/1.0"


def test_create_adk_user_agent_hook_no_existing_user_agent():
    """Test that the hook works when there's no existing User-Agent."""
    hook = create_adk_user_agent_hook(version="2.0.0", url_patterns=["api.test"])

    headers = {}
    result = hook("https://api.test/endpoint", headers)

    assert result["User-Agent"] == "adk-2.0.0"


def test_create_adk_user_agent_hook_with_deployment_uuid(monkeypatch):
    """Test that the hook includes deployment UUID when available."""
    monkeypatch.setenv("AGENT_WORKSPACE_DEPLOYMENT_UUID", "deploy-abc-123")

    hook = create_adk_user_agent_hook(version="1.0.0", url_patterns=["api.example"])

    headers = {"User-Agent": "Client/1.0"}
    result = hook("https://api.example/v1", headers)

    assert result["User-Agent"] == "Client/1.0 adk-1.0.0/deploy-abc-123"


def test_create_adk_user_agent_hook_without_deployment_uuid(monkeypatch):
    """Test that the hook works without deployment UUID."""
    monkeypatch.delenv("AGENT_WORKSPACE_DEPLOYMENT_UUID", raising=False)

    hook = create_adk_user_agent_hook(version="1.0.0", url_patterns=["api.example"])

    headers = {"User-Agent": "Client/1.0"}
    result = hook("https://api.example/v1", headers)

    assert result["User-Agent"] == "Client/1.0 adk-1.0.0"


def test_create_adk_user_agent_hook_lowercase_user_agent():
    """Test that the hook handles lowercase 'user-agent' header."""
    hook = create_adk_user_agent_hook(version="1.0.0", url_patterns=["api.test"])

    # Some HTTP libraries use lowercase headers
    headers = {"user-agent": "LowercaseClient/1.0"}
    result = hook("https://api.test/endpoint", headers)

    # Should add to the existing value
    assert result["User-Agent"] == "LowercaseClient/1.0 adk-1.0.0"


# ---- Integration Tests for Hooks with Interception ----


def test_hooks_applied_during_httpx_sync_request(intr, stub_httpx_sync):
    """Test that hooks are applied when making httpx sync requests."""
    captured_headers = {}

    def capture_hook(url: str, headers: dict) -> dict:
        headers["X-Captured"] = "yes"
        captured_headers.update(headers)
        return headers

    intr.add_endpoint_pattern("api.test")
    intr.add_request_hook(capture_hook)
    intr.start_intercepting()

    c = httpx.Client()
    c.get("http://api.test/resource", headers={"Original": "header"})

    # Hook should have been called and added the header
    assert "X-Captured" in captured_headers
    assert captured_headers["X-Captured"] == "yes"
    assert captured_headers["Original"] == "header"


def test_hooks_applied_during_requests_session(intr, stub_requests):
    """Test that hooks are applied when making requests.Session calls."""
    captured_headers = {}

    def capture_hook(url: str, headers: dict) -> dict:
        headers["X-From-Hook"] = "hooked"
        captured_headers.update(headers)
        return headers

    intr.add_endpoint_pattern("billing.svc")
    intr.add_request_hook(capture_hook)
    intr.start_intercepting()

    s = requests.Session()
    s.post("https://billing.svc/process", headers={"Content-Type": "application/json"})

    assert captured_headers.get("X-From-Hook") == "hooked"


def test_setup_digitalocean_interception_registers_ua_hook():
    """Test that setup_digitalocean_interception registers the User-Agent hook."""
    setup_digitalocean_interception()
    intr = get_network_interceptor()

    # Should have at least one hook registered
    assert len(intr._request_hooks) >= 1

    # Test that the hook modifies headers for DO inference URLs
    headers = {"User-Agent": "TestClient/1.0"}
    result = intr._apply_request_hooks("https://inference.do-ai.run/v1/chat", headers)

    # Should have appended adk-{version}
    assert "adk-" in result["User-Agent"]
    assert result["User-Agent"].startswith("TestClient/1.0 adk-")
