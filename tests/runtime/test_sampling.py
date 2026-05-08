from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState

from gradient_adk.runtime.network_interceptor import create_trace_context_hook
from gradient_adk.runtime.otel_setup import build_resource_attributes
from gradient_adk.runtime.otlp_tracker import OTLPTracesTracker


def _provider(exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create(
            build_resource_attributes(
                agent_workspace_name="workspace-a",
                agent_deployment_name="prod",
            )
        ),
        sampler=ParentBased(root=ALWAYS_ON),
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def _remote_parent_context(sampled: bool):
    trace_flags = TraceFlags(0x01 if sampled else 0x00)
    span_context = SpanContext(
        trace_id=int("1234567890abcdef1234567890abcdef", 16),
        span_id=int("1234567890abcdef", 16),
        is_remote=True,
        trace_flags=trace_flags,
        trace_state=TraceState(),
    )
    return trace.set_span_in_context(NonRecordingSpan(span_context))


def test_parent_based_sampling_respects_remote_unsampled_parent(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(
        "gradient_adk.runtime.otlp_tracker.create_tracer_provider",
        lambda **_: _provider(exporter),
    )
    tracker = OTLPTracesTracker(
        client=None,
        agent_workspace_name="workspace-a",
        agent_deployment_name="prod",
    )

    tracker.on_request_start(
        "agent_entrypoint",
        {"prompt": "hello"},
        parent_context=_remote_parent_context(sampled=False),
    )
    headers = create_trace_context_hook(["api.openai.com"])(
        "https://api.openai.com/v1/chat/completions", {}
    )
    tracker.on_request_end({"response": "ok"}, None)

    assert not exporter.get_finished_spans()
    assert headers["traceparent"].endswith("-00")


def test_parent_based_sampling_respects_remote_sampled_parent(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(
        "gradient_adk.runtime.otlp_tracker.create_tracer_provider",
        lambda **_: _provider(exporter),
    )
    tracker = OTLPTracesTracker(
        client=None,
        agent_workspace_name="workspace-a",
        agent_deployment_name="prod",
    )

    tracker.on_request_start(
        "agent_entrypoint",
        {"prompt": "hello"},
        parent_context=_remote_parent_context(sampled=True),
    )
    headers = create_trace_context_hook(["api.openai.com"])(
        "https://api.openai.com/v1/chat/completions", {}
    )
    tracker.on_request_end({"response": "ok"}, None)

    assert len(exporter.get_finished_spans()) == 1
    assert headers["traceparent"].endswith("-01")
