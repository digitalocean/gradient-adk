import asyncio
import pytest
from fastapi.testclient import TestClient

from gradient_adk.decorator import entrypoint, run_server, RequestContext
import gradient_adk.decorator as entrypoint_mod


# ---------------------------
# Shared fixtures & helpers
# ---------------------------


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch):
    """
    - Make capture_graph() a no-op (prevent side effects at import)
    - Provide a controllable get_tracker()
    """
    tracker = TrackerDouble()
    monkeypatch.setattr(entrypoint_mod, "get_tracker", lambda: tracker, raising=True)
    monkeypatch.setattr(entrypoint_mod, "logger", LoggerDouble(), raising=True)
    return tracker


class TrackerDouble:
    """Mock tracker for testing."""

    def __init__(self):
        self.started = []
        self.ended = []
        self.closed = False
        self._req = {}
        self._session_id = None

    def on_request_start(self, name, inputs, is_evaluation=False, session_id=None):
        self.started.append((name, inputs, is_evaluation, session_id))
        self._req = {"entrypoint": name, "inputs": inputs}
        self._session_id = session_id

    def on_request_end(self, outputs=None, error=None):
        self.ended.append((outputs, error))
        self._req["outputs"] = outputs
        self._req["error"] = error

    async def _submit(self):
        """Simulate async submission."""
        await asyncio.sleep(0)

    async def aclose(self):
        """Simulate async close."""
        await asyncio.sleep(0)
        self.closed = True


class LoggerDouble:
    """Mock logger for testing."""

    def __init__(self):
        self.errors = []

    def error(self, msg, **kwargs):
        self.errors.append((msg, kwargs))


# ----------------------------------
# Unit tests for the entrypoint deco
# ----------------------------------


def test_rejects_functions_with_wrong_arity():
    """Test that functions with wrong number of parameters are rejected."""

    # Too few params
    def too_few():
        pass

    with pytest.raises(ValueError) as ex:
        entrypoint(too_few)
    assert "must accept (data) or (data, context)" in str(ex.value)

    # Too many params
    def too_many(a, b, c):
        pass

    with pytest.raises(ValueError) as ex:
        entrypoint(too_many)
    assert "must accept (data) or (data, context)" in str(ex.value)


def test_health_endpoint_and_fastapi_app_exposure(patch_helpers):
    """Test that health endpoint works and fastapi_app is exposed."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"ok": True}

    # The decorator exposes `fastapi_app` in the caller's module
    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["entrypoint"] == "handler"

    # No tracker calls yet (only /health)
    assert tracker.started == []
    assert tracker.ended == []


def test_run_endpoint_non_streaming_sync_two_params(patch_helpers):
    """Test non-streaming sync handler with (data, context) signature."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        assert isinstance(context, RequestContext)
        assert context.session_id is None  # No Session-Id header provided
        return {"echo": data}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post("/run", json={"a": 1})
        assert r.status_code == 200
        assert r.json() == {"echo": {"a": 1}}

    # Tracker observed start and end
    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and tracker.ended[-1] == ({"echo": {"a": 1}}, None)


def test_run_endpoint_non_streaming_sync_one_param(patch_helpers):
    """Test non-streaming sync handler with (data) signature."""
    tracker = patch_helpers

    @entrypoint
    def handler(data):
        return {"echo": data}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post("/run", json={"a": 1})
        assert r.status_code == 200
        assert r.json() == {"echo": {"a": 1}}

    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and tracker.ended[-1] == ({"echo": {"a": 1}}, None)


@pytest.mark.asyncio
async def test_run_endpoint_non_streaming_async_two_params(patch_helpers):
    """Test non-streaming async handler with (data, context) signature."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data, context):
        await asyncio.sleep(0)
        return {"sum": data.get("x", 0) + data.get("y", 0)}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post("/run", json={"x": 2, "y": 3})
        assert r.status_code == 200
        assert r.json() == {"sum": 5}

    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and tracker.ended[-1] == ({"sum": 5}, None)


@pytest.mark.asyncio
async def test_run_endpoint_non_streaming_async_one_param(patch_helpers):
    """Test non-streaming async handler with (data) signature."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data):
        await asyncio.sleep(0)
        return {"sum": data.get("x", 0) + data.get("y", 0)}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post("/run", json={"x": 2, "y": 3})
        assert r.status_code == 200
        assert r.json() == {"sum": 5}

    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and tracker.ended[-1] == ({"sum": 5}, None)


def test_run_endpoint_invalid_json_returns_400(patch_helpers):
    """Test that invalid JSON returns 400."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"ok": True}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post(
            "/run", data="not-json", headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 400
        # No on_request_start should be called (it happens after JSON parse)
        assert tracker.started == []
        assert tracker.ended == []


def test_run_endpoint_handler_error_returns_500_and_tracks_error(patch_helpers):
    """Test that handler errors return 500 and are tracked."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        raise RuntimeError("boom")

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post("/run", json={"a": 1})
        assert r.status_code == 500
        assert r.json()["detail"] == "Internal server error"

    # started then ended with error
    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and tracker.ended[-1][0] is None
    assert tracker.ended[-1][1]  # error message present


def test_run_endpoint_streaming_async_generator_two_params(patch_helpers):
    """Test streaming with async generator (data, context) signature."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data, context):
        chunks = ["hello", " ", "world"]
        for chunk in chunks:
            yield chunk

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        with client.stream("POST", "/run", json={"p": 1}) as resp:
            assert resp.status_code == 200
            # Read the full stream by iterating
            body = "".join(chunk for chunk in resp.iter_text())
            assert body == "hello world"

    # Tracker should have been called
    assert tracker.started and tracker.started[-1][0] == "handler"
    # For streaming, tracking happens internally via _StreamingIteratorWithTracking
    # The tracker's on_request_end is not called for async generators


def test_run_endpoint_streaming_async_generator_one_param(patch_helpers):
    """Test streaming with async generator (data) signature."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data):
        chunks = ["a", "b", "c"]
        for chunk in chunks:
            yield chunk

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        with client.stream("POST", "/run", json={"p": 1}) as resp:
            assert resp.status_code == 200
            body = "".join(chunk for chunk in resp.iter_text())
            assert body == "abc"

    assert tracker.started and tracker.started[-1][0] == "handler"


def test_run_endpoint_streaming_with_dict_chunks(patch_helpers):
    """Test streaming with dict chunks (JSON serialized)."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data):
        yield {"type": "status", "message": "started"}
        yield {"type": "data", "value": 42}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        with client.stream("POST", "/run", json={"p": 1}) as resp:
            assert resp.status_code == 200
            body = "".join(chunk for chunk in resp.iter_text())
            # Dicts are JSON serialized
            assert '{"type": "status"' in body
            assert '{"type": "data"' in body

    assert tracker.started and tracker.started[-1][0] == "handler"


def test_run_endpoint_streaming_error_handling(patch_helpers):
    """Test that streaming errors are handled gracefully."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data):
        yield "chunk1"
        raise RuntimeError("stream error")
        yield "chunk2"  # Never reached

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        # Streaming exceptions propagate as connection errors
        with pytest.raises(Exception):
            with client.stream("POST", "/run", json={"p": 1}) as resp:
                assert resp.status_code == 200
                # Iterating triggers the generator exception
                for _ in resp.iter_bytes():
                    pass

    # Tracker should have been called
    assert tracker.started and tracker.started[-1][0] == "handler"


def test_run_endpoint_streaming_with_bytes(patch_helpers):
    """Test streaming with bytes chunks."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data):
        yield b"hello"
        yield b" "
        yield b"world"

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        with client.stream("POST", "/run", json={"p": 1}) as resp:
            assert resp.status_code == 200
            body = "".join(chunk for chunk in resp.iter_text())
            assert body == "hello world"

    assert tracker.started and tracker.started[-1][0] == "handler"


def test_run_endpoint_streaming_skips_none(patch_helpers):
    """Test that None chunks are skipped in streaming."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data):
        yield "a"
        yield None  # Should be skipped
        yield "b"

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        with client.stream("POST", "/run", json={"p": 1}) as resp:
            assert resp.status_code == 200
            body = "".join(chunk for chunk in resp.iter_text())
            assert body == "ab"  # None skipped

    assert tracker.started and tracker.started[-1][0] == "handler"


def test_evaluation_mode_tracking(patch_helpers):
    """Test that evaluation mode is tracked correctly."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"result": "ok"}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post("/run", json={"test": 1}, headers={"evaluation-id": "eval-123"})
        assert r.status_code == 200

    # Check that is_evaluation was passed correctly
    assert tracker.started
    assert tracker.started[-1][2] is True  # is_evaluation flag


def test_shutdown_event_calls_tracker_aclose(patch_helpers):
    """Test that shutdown event calls tracker aclose."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"ok": True}

    fastapi_app = globals()["fastapi_app"]
    # TestClient triggers startup/shutdown events when used as a context manager
    with TestClient(fastapi_app) as client:
        assert client.get("/health").status_code == 200
    # After context exit, shutdown event should have run
    assert tracker.closed is True


def test_run_server_invokes_uvicorn(monkeypatch):
    """Test that run_server invokes uvicorn correctly."""
    calls = {}

    def fake_run(fastapi_app, host, port, **kwargs):
        calls["fastapi_app"] = fastapi_app
        calls["host"] = host
        calls["port"] = port
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entrypoint_mod.uvicorn, "run", fake_run, raising=True)

    # Build a tiny app via the decorator to pass into run_server
    @entrypoint
    def handler(data, context):
        return {"ok": True}

    fastapi_app = globals()["fastapi_app"]

    # Call run_server with custom args and ensure uvicorn.run is invoked accordingly
    run_server(fastapi_app, host="127.0.0.1", port=9999, reload=True, log_level="debug")

    assert calls["fastapi_app"] is fastapi_app
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9999
    assert calls["kwargs"]["reload"] is True
    assert calls["kwargs"]["log_level"] == "debug"


# ----------------------------------
# Tests for RequestContext and session_id
# ----------------------------------


def test_request_context_dataclass():
    """Test that RequestContext is a proper dataclass with expected fields."""
    # Default values
    ctx = RequestContext()
    assert ctx.session_id is None

    # With session_id
    ctx = RequestContext(session_id="test-session-123")
    assert ctx.session_id == "test-session-123"


def test_session_id_header_passed_to_context_sync(patch_helpers):
    """Test that Session-Id header is passed to context for sync handler."""
    tracker = patch_helpers
    captured_context = {}

    @entrypoint
    def handler(data, context):
        captured_context["context"] = context
        return {"session_id": context.session_id}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post(
            "/run",
            json={"test": 1},
            headers={"Session-Id": "my-session-abc"},
        )
        assert r.status_code == 200
        assert r.json() == {"session_id": "my-session-abc"}

    # Verify the captured context
    assert isinstance(captured_context["context"], RequestContext)
    assert captured_context["context"].session_id == "my-session-abc"


def test_session_id_header_case_insensitive(patch_helpers):
    """Test that Session-Id header is case-insensitive (lowercase works)."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"session_id": context.session_id}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        # Test with lowercase header name
        r = client.post(
            "/run",
            json={"test": 1},
            headers={"session-id": "lowercase-session"},
        )
        assert r.status_code == 200
        assert r.json() == {"session_id": "lowercase-session"}


@pytest.mark.asyncio
async def test_session_id_header_passed_to_context_async(patch_helpers):
    """Test that Session-Id header is passed to context for async handler."""
    tracker = patch_helpers

    @entrypoint
    async def handler(data, context):
        await asyncio.sleep(0)
        return {"session_id": context.session_id}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post(
            "/run",
            json={"test": 1},
            headers={"Session-Id": "async-session-xyz"},
        )
        assert r.status_code == 200
        assert r.json() == {"session_id": "async-session-xyz"}


def test_session_id_header_passed_to_streaming_handler(patch_helpers):
    """Test that Session-Id header is passed to context for streaming handler."""
    tracker = patch_helpers
    captured_session_id = {}

    @entrypoint
    async def handler(data, context):
        captured_session_id["value"] = context.session_id
        yield f"session:{context.session_id}"

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        with client.stream(
            "POST",
            "/run",
            json={"test": 1},
            headers={"Session-Id": "streaming-session"},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(chunk for chunk in resp.iter_text())
            assert body == "session:streaming-session"

    assert captured_session_id["value"] == "streaming-session"


def test_session_id_none_when_header_not_provided(patch_helpers):
    """Test that session_id is None when header is not provided."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {
            "has_context": context is not None,
            "session_id": context.session_id,
        }

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post("/run", json={"test": 1})
        assert r.status_code == 200
        result = r.json()
        assert result["has_context"] is True
        assert result["session_id"] is None


def test_session_id_empty_string_when_header_is_empty(patch_helpers):
    """Test that session_id is empty string when header value is empty."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"session_id": context.session_id}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post(
            "/run",
            json={"test": 1},
            headers={"Session-Id": ""},
        )
        assert r.status_code == 200
        assert r.json() == {"session_id": ""}


def test_session_id_passed_to_tracker(patch_helpers):
    """Test that session_id is passed to the tracker for tracing."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"ok": True}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post(
            "/run",
            json={"test": 1},
            headers={"Session-Id": "trace-session-456"},
        )
        assert r.status_code == 200

    # Verify that session_id was passed to the tracker
    assert tracker.started
    # started tuple is (name, inputs, is_evaluation, session_id)
    assert tracker.started[-1][3] == "trace-session-456"
    assert tracker._session_id == "trace-session-456"


def test_session_id_none_passed_to_tracker_when_no_header(patch_helpers):
    """Test that session_id is None in tracker when header is not provided."""
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"ok": True}

    fastapi_app = globals()["fastapi_app"]
    with TestClient(fastapi_app) as client:
        r = client.post("/run", json={"test": 1})
        assert r.status_code == 200

    # Verify that session_id is None when header not provided
    assert tracker.started
    assert tracker.started[-1][3] is None
    assert tracker._session_id is None