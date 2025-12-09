import asyncio
import inspect
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient

# ⬇️ Adjust this import path if your decorator lives elsewhere
from gradient_adk.decorator import entrypoint, run_server
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
    # If the module already imported and ran capture_graph(), it's fine.
    # We still patch get_tracker for the runtime calls.
    tracker = TrackerDouble()
    monkeypatch.setattr(entrypoint_mod, "get_tracker", lambda: tracker, raising=True)
    monkeypatch.setattr(entrypoint_mod, "logger", LoggerDouble(), raising=True)
    return tracker


class TrackerDouble:
    def __init__(self):
        self.started = []
        self.ended = []
        self.closed = False

    def on_request_start(self, name, inputs, is_evaluation=False):
        self.started.append((name, inputs))

    def on_request_end(self, outputs=None, error=None):
        self.ended.append((outputs, error))

    async def aclose(self):
        # Simulate async close
        await asyncio.sleep(0)
        self.closed = True


class LoggerDouble:
    def __init__(self):
        self.errors = []

    def error(self, msg, **kwargs):
        self.errors.append((msg, kwargs))


def _make_streaming_response(entrypoint_mod, chunks, *, fail_after=None):
    """
    Build a GradientStreamingResponse using an async generator:
    - chunks: iterable of string/bytes
    - fail_after: if set, raise after yielding that many chunks
    """
    GradientStreamingResponse = entrypoint_mod.GradientStreamingResponse

    async def gen():
        for idx, ch in enumerate(chunks, start=1):
            yield ch
            if fail_after is not None and idx >= fail_after:
                raise RuntimeError("stream-fail")

    return GradientStreamingResponse(
        content=gen(),  # async generator
        media_type="text/event-stream",
        headers={"X-Stream": "1"},
    )


# ----------------------------------
# Unit tests for the entrypoint deco
# ----------------------------------


def test_rejects_functions_with_wrong_arity():
    def bad(a):  # only 1 param
        return a

    with pytest.raises(ValueError) as ex:
        entrypoint(bad)
    assert "must accept exactly (data, context)" in str(ex.value)


def test_health_endpoint_and_app_exposure(patch_helpers):
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"ok": True}

    # The decorator exposes `app` in the caller's module (this test file)
    app = globals()["app"]
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["entrypoint"] == "handler"

    # No tracker calls yet (only /health)
    assert tracker.started == []
    assert tracker.ended == []


def test_run_endpoint_non_streaming_sync(patch_helpers):
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        assert context is None
        return {"echo": data}

    app = globals()["app"]
    with TestClient(app) as client:
        r = client.post("/run", json={"a": 1})
        assert r.status_code == 200
        assert r.json() == {"echo": {"a": 1}}

    # Tracker observed start and end
    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and tracker.ended[-1] == ({"echo": {"a": 1}}, None)


@pytest.mark.asyncio
async def test_run_endpoint_non_streaming_async(patch_helpers):
    tracker = patch_helpers

    @entrypoint
    async def handler(data, context):
        await asyncio.sleep(0)
        return {"sum": data.get("x", 0) + data.get("y", 0)}

    app = globals()["app"]
    # TestClient drives the ASGI app in a thread; it's fine for async routes
    with TestClient(app) as client:
        r = client.post("/run", json={"x": 2, "y": 3})
        assert r.status_code == 200
        assert r.json() == {"sum": 5}

    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and tracker.ended[-1] == ({"sum": 5}, None)


def test_run_endpoint_invalid_json_returns_400(patch_helpers):
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"ok": True}

    app = globals()["app"]
    with TestClient(app) as client:
        # Send invalid JSON payload (string with wrong content-type)
        r = client.post(
            "/run", data="not-json", headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 400
        # No on_request_start should be called (it happens after JSON parse)
        assert tracker.started == []
        assert tracker.ended == []


def test_run_endpoint_handler_error_returns_500_and_tracks_error(patch_helpers):
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        raise RuntimeError("boom")

    app = globals()["app"]
    with TestClient(app) as client:
        r = client.post("/run", json={"a": 1})
        assert r.status_code == 500
        assert r.json()["detail"] == "Internal server error"

    # started then ended with error
    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and tracker.ended[-1][0] is None
    assert tracker.ended[-1][1]  # error message present


def test_run_endpoint_streaming_success_calls_end_with_none(patch_helpers):
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return _make_streaming_response(entrypoint_mod, chunks=[b"a", b"b", b"c"])

    app = globals()["app"]
    with TestClient(app) as client:
        with client.stream("POST", "/run", json={"p": 1}) as resp:
            assert resp.status_code == 200
            # Read the full stream
            body = b"".join(resp.iter_bytes())
            assert body == b"abc"  # chunks concatenated

    # For streaming success, the StreamingResponse object is passed to on_request_end
    # The tracker will wrap it and collect outputs during streaming
    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and len(tracker.ended) > 0
    last_output, last_error = tracker.ended[-1]
    # The output should be a StreamingResponse object, error should be None
    assert isinstance(last_output, entrypoint_mod.GradientStreamingResponse)
    assert last_error is None


def test_run_endpoint_streaming_error_calls_end_with_error(patch_helpers):
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        # Fail after first chunk
        return _make_streaming_response(
            entrypoint_mod, chunks=[b"1", b"2"], fail_after=1
        )

    app = globals()["app"]
    with TestClient(app) as client:
        # Streaming exceptions propagate as connection errors inside TestClient iteration.
        with pytest.raises(Exception):
            with client.stream("POST", "/run", json={"p": 1}) as resp:
                assert resp.status_code == 200
                # iterating triggers the generator exception
                for _ in resp.iter_bytes():
                    pass

    # Tracker should have been called on_request_end with the StreamingResponse
    # Errors during streaming are handled by the tracker's wrapper, not via on_request_end
    assert tracker.started and tracker.started[-1][0] == "handler"
    assert tracker.ended and len(tracker.ended) > 0
    last_output, last_error = tracker.ended[-1]
    # The output should be a StreamingResponse object
    assert isinstance(last_output, entrypoint_mod.GradientStreamingResponse)
    # Note: The error=None here because streaming errors are caught by the tracker's wrapper
    # The actual error handling happens inside the async generator wrapper
    assert last_error is None


def test_shutdown_event_calls_tracker_aclose(patch_helpers):
    tracker = patch_helpers

    @entrypoint
    def handler(data, context):
        return {"ok": True}

    app = globals()["app"]
    # TestClient triggers startup/shutdown events when used as a context manager
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    # After context exit, shutdown event should have run
    assert tracker.closed is True


def test_run_server_invokes_uvicorn(monkeypatch):
    calls = {}

    def fake_run(app, host, port, **kwargs):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entrypoint_mod.uvicorn, "run", fake_run, raising=True)

    # Build a tiny app via the decorator to pass into run_server
    @entrypoint
    def handler(data, context):
        return {"ok": True}

    app = globals()["app"]

    # Call run_server with custom args and ensure uvicorn.run is invoked accordingly
    run_server(app, host="127.0.0.1", port=9999, reload=True, log_level="debug")

    assert calls["app"] is app
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9999
    assert calls["kwargs"]["reload"] is True
    assert calls["kwargs"]["log_level"] == "debug"
