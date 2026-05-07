"""Lifecycle and exception-fidelity tests for OTLPTracesTracker."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased

from gradient_adk.runtime.interfaces import NodeExecution
from gradient_adk.runtime.otel_setup import build_resource_attributes
from gradient_adk.runtime.otlp_tracker import OTLPTracesTracker


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _patch_tracer_provider(monkeypatch, exporter: InMemorySpanExporter) -> None:
    def fake_create_tracer_provider(
        *,
        agent_workspace_name: str,
        agent_deployment_name: str,
        team_id: str | None = None,
        org_id: str | None = None,
    ) -> TracerProvider:
        provider = TracerProvider(
            resource=Resource.create(
                build_resource_attributes(
                    agent_workspace_name=agent_workspace_name,
                    agent_deployment_name=agent_deployment_name,
                    team_id=team_id,
                    org_id=org_id,
                )
            ),
            sampler=ParentBased(root=ALWAYS_ON),
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return provider

    monkeypatch.setattr(
        "gradient_adk.runtime.otlp_tracker.create_tracer_provider",
        fake_create_tracer_provider,
    )


def test_init_does_not_call_digitalocean_api(monkeypatch):
    """The tracker must not perform any blocking DO API lookup at construction.

    The collector (AAPP-1160) is the canonical source for tenant attribution
    in production; the SDK only honors env vars and falls back to the
    collector. Calling the DO API at __init__ time would block app startup.
    """
    exporter = InMemorySpanExporter()
    _patch_tracer_provider(monkeypatch, exporter)
    monkeypatch.delenv("DIGITALOCEAN_TEAM_ID", raising=False)
    monkeypatch.delenv("DIGITALOCEAN_ORG_ID", raising=False)

    client = MagicMock()
    client.list_agent_workspaces = MagicMock()
    client.aclose = MagicMock()

    OTLPTracesTracker(
        client=client,
        agent_workspace_name="workspace-a",
        agent_deployment_name="prod",
    )

    client.list_agent_workspaces.assert_not_called()


def test_on_node_error_records_original_exception_type(monkeypatch):
    """The original exception (class + traceback) must reach record_exception.

    Wrapping into a fresh RuntimeError loses both the class and the original
    traceback, which makes downstream alerting/triaging much harder.
    """
    exporter = InMemorySpanExporter()
    _patch_tracer_provider(monkeypatch, exporter)
    monkeypatch.setenv("DIGITALOCEAN_CONVERSATION_LOGS_ENABLED", "true")

    tracker = OTLPTracesTracker(
        client=None,
        agent_workspace_name="workspace-a",
        agent_deployment_name="prod",
    )
    tracker.on_request_start("agent_entrypoint", {"prompt": "hello"})

    node = NodeExecution(
        node_id="tool-err",
        node_name="failing_tool",
        framework="custom",
        start_time=_now(),
        inputs={"x": 1},
        metadata={"is_tool_call": True},
    )
    tracker.on_node_start(node)

    class CustomFailure(ValueError):
        pass

    raised = CustomFailure("kaboom")
    tracker.on_node_error(node, raised)
    tracker.on_request_end(None, "kaboom")

    tool_span = next(
        span for span in exporter.get_finished_spans() if span.name == "failing_tool"
    )
    assert tool_span.status.status_code.name == "ERROR"
    exception_events = [
        event for event in tool_span.events if event.name == "exception"
    ]
    assert exception_events, "expected record_exception to emit an 'exception' event"
    attrs = exception_events[0].attributes
    assert attrs["exception.type"].endswith("CustomFailure")
    assert attrs["exception.message"] == "kaboom"


def test_logs_enabled_is_snapshotted_at_request_start(monkeypatch):
    """Once a request starts, mid-request env flips must not change redaction.

    Redaction is per-request: the flag is read once at on_request_start and
    cached on the per-request state. Mid-request env mutation (e.g. by other
    threads or by a buggy library) must not partially redact a single trace.
    """
    exporter = InMemorySpanExporter()
    _patch_tracer_provider(monkeypatch, exporter)
    monkeypatch.setenv("DIGITALOCEAN_CONVERSATION_LOGS_ENABLED", "true")

    tracker = OTLPTracesTracker(
        client=None,
        agent_workspace_name="workspace-a",
        agent_deployment_name="prod",
    )
    tracker.on_request_start("agent_entrypoint", {"prompt": "hello"})

    monkeypatch.setenv("DIGITALOCEAN_CONVERSATION_LOGS_ENABLED", "false")

    node = NodeExecution(
        node_id="tool-1",
        node_name="weather_tool",
        framework="custom",
        start_time=_now(),
        inputs={"city": "nyc"},
        metadata={"is_tool_call": True},
    )
    tracker.on_node_start(node)
    tracker.on_node_end(node, {"result": "sunny"})
    tracker.on_request_end({"done": True}, None)

    tool_span = next(
        span for span in exporter.get_finished_spans() if span.name == "weather_tool"
    )
    event_names = {event.name for event in tool_span.events}
    assert "gen_ai.input.messages" in event_names
    assert "gen_ai.output.messages" in event_names
    assert tool_span.attributes.get("gen_ai.input.redacted") is None
    assert tool_span.attributes.get("gen_ai.output.redacted") is None
