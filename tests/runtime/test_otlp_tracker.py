from __future__ import annotations

from datetime import datetime, timezone

import pytest
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


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracker(monkeypatch, exporter: InMemorySpanExporter) -> OTLPTracesTracker:
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
    monkeypatch.setenv("DIGITALOCEAN_TEAM_ID", "123")
    monkeypatch.setenv("DIGITALOCEAN_ORG_ID", "456")
    monkeypatch.setenv("DIGITALOCEAN_CONVERSATION_LOGS_ENABLED", "true")
    return OTLPTracesTracker(
        client=None,
        agent_workspace_name="workspace-a",
        agent_deployment_name="prod",
    )


def test_llm_span_sets_semconv_and_resource_attributes(
    tracker: OTLPTracesTracker, exporter: InMemorySpanExporter
):
    tracker.on_request_start(
        "agent_entrypoint",
        {"prompt": "hello"},
        session_id="session-123",
        evaluation_run_uuid="eval-456",
    )
    node = NodeExecution(
        node_id="llm-1",
        node_name="call_model",
        framework="custom",
        start_time=_now(),
        inputs={"messages": [{"role": "user", "content": "hello"}]},
        metadata={
            "is_llm_call": True,
            "llm_url": "https://api.openai.com/v1/chat/completions",
            "llm_request_payload": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "name": "search"}],
            },
            "llm_response_payload": {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
            "time_to_first_token_ns": 100_000_000,
        },
    )

    tracker.on_node_start(node)
    tracker.on_node_end(node, {"content": "hi"})
    tracker.on_request_end({"response": "hi"}, None)

    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {"agent_entrypoint", "call_model"}

    llm_span = next(span for span in spans if span.name == "call_model")
    root_span = next(span for span in spans if span.name == "agent_entrypoint")

    assert llm_span.attributes["gen_ai.operation.name"] == "chat"
    assert llm_span.attributes["gen_ai.system"] == "openai"
    assert llm_span.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert llm_span.attributes["gen_ai.request.temperature"] == 0.2
    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 10
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 4
    assert llm_span.attributes["gen_ai.usage.total_tokens"] == 14
    assert llm_span.attributes["gen_ai.server.time_to_first_token"] == 0.1
    assert {event.name for event in llm_span.events} >= {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.request.tools",
    }

    assert root_span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root_span.attributes["gen_ai.conversation.id"] == "session-123"
    assert root_span.attributes["gen_ai.evaluation.run_uuid"] == "eval-456"
    assert root_span.resource.attributes["service.name"] == "adk-agent"
    assert root_span.resource.attributes["do.trace.visibility"] == "customer"
    assert root_span.resource.attributes["team_id"] == "123"
    assert root_span.resource.attributes["org_id"] == "456"


def test_tool_and_retriever_spans_map_expected_attributes(
    tracker: OTLPTracesTracker, exporter: InMemorySpanExporter
):
    tracker.on_request_start("agent_entrypoint", {"query": "weather"})

    tool_node = NodeExecution(
        node_id="tool-1",
        node_name="weather_tool",
        framework="custom",
        start_time=_now(),
        inputs={"city": "nyc"},
        metadata={"is_tool_call": True, "tool_call_id": "call-123"},
    )
    retriever_node = NodeExecution(
        node_id="retriever-1",
        node_name="knowledge_base",
        framework="custom",
        start_time=_now(),
        inputs={"query": "latest weather"},
        metadata={"is_retriever_call": True},
    )

    tracker.on_node_start(tool_node)
    tracker.on_node_end(tool_node, {"result": "sunny"})
    tracker.on_node_start(retriever_node)
    tracker.on_node_end(retriever_node, {"hits": 2})
    tracker.on_request_end({"done": True}, None)

    spans = exporter.get_finished_spans()
    tool_span = next(span for span in spans if span.name == "weather_tool")
    retriever_span = next(span for span in spans if span.name == "knowledge_base")

    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.name"] == "weather_tool"
    assert tool_span.attributes["gen_ai.tool.call.id"] == "call-123"
    assert retriever_span.attributes["gen_ai.operation.name"] == "retrieval"
    assert retriever_span.attributes["gen_ai.retrieval.query"] == "latest weather"


def test_error_span_records_error_status(
    tracker: OTLPTracesTracker, exporter: InMemorySpanExporter
):
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
    tracker.on_node_error(node, RuntimeError("boom"))
    tracker.on_request_end(None, "boom")

    spans = exporter.get_finished_spans()
    tool_span = next(span for span in spans if span.name == "failing_tool")
    root_span = next(span for span in spans if span.name == "agent_entrypoint")

    assert tool_span.status.status_code.name == "ERROR"
    assert root_span.status.status_code.name == "ERROR"
