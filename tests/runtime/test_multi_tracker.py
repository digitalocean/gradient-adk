from __future__ import annotations

from typing import Any, Optional

import pytest

from gradient_adk.runtime.digitalocean_tracker import DigitalOceanTracesTracker
from gradient_adk.runtime.multi_tracker import MultiTracker


class LegacyTrackerDouble(DigitalOceanTracesTracker):
    def __init__(self, trace_id: Optional[str]) -> None:
        self.trace_id = trace_id
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def on_request_start(self, *args, **kwargs):
        self.calls.append(("on_request_start", args, kwargs))

    def on_request_end(self, *args, **kwargs):
        self.calls.append(("on_request_end", args, kwargs))

    def on_node_start(self, *args, **kwargs):
        self.calls.append(("on_node_start", args, kwargs))

    def on_node_end(self, *args, **kwargs):
        self.calls.append(("on_node_end", args, kwargs))

    def on_node_error(self, *args, **kwargs):
        self.calls.append(("on_node_error", args, kwargs))

    async def submit_and_get_trace_id(self):
        self.calls.append(("submit_and_get_trace_id", (), {}))
        return self.trace_id

    async def aclose(self):
        self.calls.append(("aclose", (), {}))


class TrackerDouble:
    def __init__(self, trace_id: Optional[str]) -> None:
        self.trace_id = trace_id
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def on_request_start(self, *args, **kwargs):
        self.calls.append(("on_request_start", args, kwargs))

    def on_request_end(self, *args, **kwargs):
        self.calls.append(("on_request_end", args, kwargs))

    def on_node_start(self, *args, **kwargs):
        self.calls.append(("on_node_start", args, kwargs))

    def on_node_end(self, *args, **kwargs):
        self.calls.append(("on_node_end", args, kwargs))

    def on_node_error(self, *args, **kwargs):
        self.calls.append(("on_node_error", args, kwargs))

    async def submit_and_get_trace_id(self):
        self.calls.append(("submit_and_get_trace_id", (), {}))
        return self.trace_id

    async def aclose(self):
        self.calls.append(("aclose", (), {}))


@pytest.mark.asyncio
async def test_multi_tracker_prefers_legacy_trace_id():
    legacy = LegacyTrackerDouble("legacy-trace")
    otlp = TrackerDouble("otlp-trace")
    tracker = MultiTracker([legacy, otlp])

    trace_id = await tracker.submit_and_get_trace_id()
    assert trace_id == "legacy-trace"


@pytest.mark.asyncio
async def test_multi_tracker_falls_back_to_otlp_trace_id():
    legacy = LegacyTrackerDouble(None)
    otlp = TrackerDouble("otlp-trace")
    tracker = MultiTracker([legacy, otlp])

    trace_id = await tracker.submit_and_get_trace_id()
    assert trace_id == "otlp-trace"


def test_multi_tracker_fanout_calls_all_trackers():
    legacy = LegacyTrackerDouble("legacy-trace")
    otlp = TrackerDouble("otlp-trace")
    tracker = MultiTracker([legacy, otlp])

    tracker.on_request_start(
        "agent",
        {"input": "hello"},
        is_evaluation=True,
        session_id="session-1",
        parent_context="parent",
        evaluation_run_uuid="eval-1",
    )
    tracker.on_request_end(outputs={"done": True}, error=None)

    assert legacy.calls[0][0] == "on_request_start"
    assert otlp.calls[0][0] == "on_request_start"
    assert legacy.calls[0][2]["evaluation_run_uuid"] == "eval-1"
    assert otlp.calls[0][2]["parent_context"] == "parent"
    assert legacy.calls[1][0] == "on_request_end"
    assert otlp.calls[1][0] == "on_request_end"
