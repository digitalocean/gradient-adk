from __future__ import annotations

from datetime import datetime, timezone

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased

from gradient_adk.runtime.interfaces import NodeExecution
from gradient_adk.runtime.otel_setup import (
    DEFAULT_MAX_EVENT_BYTES,
    build_resource_attributes,
)
from gradient_adk.runtime.otlp_tracker import OTLPTracesTracker


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_payload_events_are_capped_to_max_bytes(monkeypatch):
    exporter = InMemorySpanExporter()

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
    monkeypatch.setenv("DIGITALOCEAN_CONVERSATION_LOGS_ENABLED", "true")

    tracker = OTLPTracesTracker(
        client=None,
        agent_workspace_name="workspace-a",
        agent_deployment_name="prod",
    )
    huge_value = "x" * (1024 * 1024)
    tracker.on_request_start("agent_entrypoint", {"prompt": huge_value})
    node = NodeExecution(
        node_id="tool-1",
        node_name="tool_call",
        framework="custom",
        start_time=_now(),
        inputs={"prompt": huge_value},
        metadata={"is_tool_call": True},
    )

    tracker.on_node_start(node)
    tracker.on_node_end(node, {"result": huge_value})
    tracker.on_request_end({"response": huge_value}, None)

    tool_span = next(span for span in exporter.get_finished_spans() if span.name == "tool_call")
    payload = tool_span.events[0].attributes["content"]
    assert payload.endswith("...<truncated>")
    assert len(payload.encode("utf-8")) <= DEFAULT_MAX_EVENT_BYTES
