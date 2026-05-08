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
from gradient_adk.runtime.otel_setup import build_resource_attributes
from gradient_adk.runtime.otlp_tracker import OTLPTracesTracker


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_redaction_omits_payload_events_and_sets_markers(monkeypatch):
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
    monkeypatch.setenv("DIGITALOCEAN_CONVERSATION_LOGS_ENABLED", "false")

    tracker = OTLPTracesTracker(
        client=None,
        agent_workspace_name="workspace-a",
        agent_deployment_name="prod",
    )
    tracker.on_request_start("agent_entrypoint", {"prompt": "secret"})
    node = NodeExecution(
        node_id="llm-1",
        node_name="call_model",
        framework="custom",
        start_time=_now(),
        inputs={"messages": [{"role": "user", "content": "secret"}]},
        metadata={
            "is_llm_call": True,
            "llm_request_payload": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "secret"}],
            },
            "llm_response_payload": {
                "choices": [{"message": {"role": "assistant", "content": "redacted"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        },
    )

    tracker.on_node_start(node)
    tracker.on_node_end(node, {"content": "redacted"})
    tracker.on_request_end({"response": "redacted"}, None)

    spans = exporter.get_finished_spans()
    llm_span = next(span for span in spans if span.name == "call_model")
    root_span = next(span for span in spans if span.name == "agent_entrypoint")

    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 3
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 2
    assert llm_span.attributes["gen_ai.input.redacted"] is True
    assert llm_span.attributes["gen_ai.output.redacted"] is True
    assert root_span.attributes["gen_ai.input.redacted"] is True
    assert root_span.attributes["gen_ai.output.redacted"] is True
    assert [event.name for event in llm_span.events] == []
    assert [event.name for event in root_span.events] == []
