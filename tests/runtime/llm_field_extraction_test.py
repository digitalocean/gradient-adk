"""Tests for LLM field extraction in spans."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from gradient_adk.runtime.interfaces import NodeExecution
from gradient_adk.runtime.digitalocean_tracker import DigitalOceanTracesTracker
from gradient_adk.digital_ocean_api.models import TraceSpanType


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def create_mock_client():
    """Create a mock AsyncDigitalOceanGenAI client."""
    client = AsyncMock()
    client.create_traces = AsyncMock(return_value=MagicMock(trace_uuids=["test-uuid"]))
    client.aclose = AsyncMock()
    return client


class TestLLMFieldExtraction:
    """Tests for extracting LLM-specific fields from captured API payloads."""

    def test_extract_model_from_request_payload(self):
        """Test that model is extracted from the request payload."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        # Create a node execution with LLM request payload in metadata
        node = NodeExecution(
            node_id="test-node-1",
            node_name="llm:test-model",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={"messages": [{"role": "user", "content": "Hello"}]},
            outputs={"content": "Hi there!"},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                "llm_response_payload": {
                    "choices": [{"message": {"content": "Hi there!"}}],
                },
            },
        )

        span = tracker._to_span(node)

        assert span.type == TraceSpanType.TRACE_SPAN_TYPE_LLM
        assert span.llm is not None
        assert span.llm.model == "gpt-4o-mini"

    def test_llm_input_only_contains_messages(self):
        """Test that LLM span input only contains messages, not the full request payload."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        node = NodeExecution(
            node_id="test-node-messages",
            node_name="llm:gpt-4",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},  # Original inputs
            outputs={"content": "Hi!"},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {
                    "model": "gpt-4",
                    "messages": messages,
                    "temperature": 0.7,
                    "stream": False,  # Should NOT appear in span input
                    "max_tokens": 100,  # Should NOT appear in span input
                },
            },
        )

        span = tracker._to_span(node)

        # Span input should be {"messages": [...]} (wrapped in dict for protobuf Struct)
        # Should NOT include model, temperature, stream, max_tokens, etc.
        assert span.input == {"messages": messages}
        assert "model" not in span.input
        assert "temperature" not in span.input
        assert "stream" not in span.input
        assert "max_tokens" not in span.input

    def test_llm_output_only_contains_choices(self):
        """Test that LLM span output only contains choices, not the full response payload."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ]

        node = NodeExecution(
            node_id="test-node-output",
            node_name="llm:gpt-4",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},  # Original outputs
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {"model": "gpt-4", "messages": []},
                "llm_response_payload": {
                    "id": "chatcmpl-123",  # Should NOT appear in span output
                    "object": "chat.completion",  # Should NOT appear in span output
                    "created": 1234567890,  # Should NOT appear in span output
                    "model": "gpt-4",  # Should NOT appear in span output
                    "choices": choices,
                    "usage": {  # Should NOT appear in span output
                        "prompt_tokens": 10,
                        "completion_tokens": 15,
                        "total_tokens": 25,
                    },
                },
            },
        )

        span = tracker._to_span(node)

        # Span output should be {"choices": [...]} (wrapped in dict for protobuf Struct)
        # Should NOT include id, object, created, model, usage, etc.
        assert span.output == {"choices": choices}
        assert "id" not in span.output
        assert "object" not in span.output
        assert "created" not in span.output
        assert "model" not in span.output
        assert "usage" not in span.output

    def test_extract_tools_from_request_payload(self):
        """Test that tools are extracted from the request payload."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

        node = NodeExecution(
            node_id="test-node-2",
            node_name="llm:gpt-4",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={"messages": []},
            outputs={"content": ""},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {
                    "model": "gpt-4",
                    "messages": [],
                    "tools": tools,
                },
            },
        )

        span = tracker._to_span(node)

        assert span.llm is not None
        assert span.llm.tools is not None
        assert len(span.llm.tools) == 2
        assert span.llm.tools[0]["type"] == "function"
        assert span.llm.tools[0]["function"]["name"] == "get_weather"
        assert span.llm.tools[1]["function"]["name"] == "search_web"

    def test_extract_temperature_from_request_payload(self):
        """Test that temperature is extracted from the request payload."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        node = NodeExecution(
            node_id="test-node-3",
            node_name="llm:test",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "messages": [],
                },
            },
        )

        span = tracker._to_span(node)

        assert span.llm is not None
        assert span.llm.temperature == 0.7

    def test_extract_token_counts_from_response_payload(self):
        """Test that token counts are extracted from the response payload."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        node = NodeExecution(
            node_id="test-node-4",
            node_name="llm:test",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {"model": "gpt-4", "messages": []},
                "llm_response_payload": {
                    "choices": [{"message": {"content": "Hello!"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            },
        )

        span = tracker._to_span(node)

        assert span.llm is not None
        assert span.llm.num_input_tokens == 10
        assert span.llm.num_output_tokens == 5
        assert span.llm.total_tokens == 15

    def test_extract_time_to_first_token(self):
        """Test that time_to_first_token_ns is extracted from metadata."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        ttft_ns = 123456789  # ~123ms

        node = NodeExecution(
            node_id="test-node-5",
            node_name="llm:streaming-test",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={"content": "Streamed response"},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {"model": "gpt-4", "messages": []},
                "llm_response_payload": {},
                "time_to_first_token_ns": ttft_ns,
            },
        )

        span = tracker._to_span(node)

        assert span.llm is not None
        assert span.llm.time_to_first_token_ns == ttft_ns

    def test_all_fields_together(self):
        """Test that all LLM fields are extracted together correctly."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        tools = [
            {
                "type": "function",
                "function": {"name": "test_tool", "description": "A test tool"},
            }
        ]

        node = NodeExecution(
            node_id="test-node-6",
            node_name="llm:complete-test",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {
                    "model": "claude-3-opus",
                    "temperature": 0.5,
                    "tools": tools,
                    "messages": [{"role": "user", "content": "Test"}],
                },
                "llm_response_payload": {
                    "content": [{"type": "text", "text": "Response"}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                },
                "time_to_first_token_ns": 500000000,  # 500ms
            },
        )

        span = tracker._to_span(node)

        assert span.type == TraceSpanType.TRACE_SPAN_TYPE_LLM
        assert span.llm is not None
        assert span.llm.model == "claude-3-opus"
        assert span.llm.temperature == 0.5
        assert span.llm.tools == tools
        assert span.llm.num_input_tokens == 100
        assert span.llm.num_output_tokens == 50
        assert span.llm.total_tokens == 150
        assert span.llm.time_to_first_token_ns == 500000000

    def test_missing_llm_payloads_graceful_fallback(self):
        """Test graceful handling when LLM payloads are missing."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        # No llm_request_payload or llm_response_payload
        node = NodeExecution(
            node_id="test-node-7",
            node_name="llm:fallback-test",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},
            metadata={"is_llm_call": True},
        )

        span = tracker._to_span(node)

        assert span.type == TraceSpanType.TRACE_SPAN_TYPE_LLM
        assert span.llm is not None
        # Model should fallback to node name
        assert span.llm.model == "fallback-test"
        # Optional fields should be None
        assert span.llm.tools is None
        assert span.llm.temperature is None
        assert span.llm.num_input_tokens is None
        assert span.llm.num_output_tokens is None
        assert span.llm.total_tokens is None
        assert span.llm.time_to_first_token_ns is None

    def test_empty_usage_object_in_response(self):
        """Test handling of empty usage object in response."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        node = NodeExecution(
            node_id="test-node-8",
            node_name="llm:empty-usage",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {"model": "test"},
                "llm_response_payload": {"usage": {}},
            },
        )

        span = tracker._to_span(node)

        assert span.llm is not None
        assert span.llm.num_input_tokens is None
        assert span.llm.num_output_tokens is None
        assert span.llm.total_tokens is None

    def test_model_fallback_priority(self):
        """Test model extraction fallback priority: request > metadata > node_name."""
        client = create_mock_client()
        tracker = DigitalOceanTracesTracker(
            client=client,
            agent_workspace_name="test-workspace",
            agent_deployment_name="test-deployment",
        )

        # Case 1: Request payload has model
        node1 = NodeExecution(
            node_id="test-1",
            node_name="llm:node-model",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},
            metadata={
                "is_llm_call": True,
                "model_name": "metadata-model",
                "llm_request_payload": {"model": "request-model"},
            },
        )
        span1 = tracker._to_span(node1)
        assert span1.llm.model == "request-model"

        # Case 2: No request model, use metadata
        node2 = NodeExecution(
            node_id="test-2",
            node_name="llm:node-model",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},
            metadata={
                "is_llm_call": True,
                "model_name": "metadata-model",
                "llm_request_payload": {},
            },
        )
        span2 = tracker._to_span(node2)
        assert span2.llm.model == "metadata-model"

        # Case 3: No request or metadata model, use node_name
        node3 = NodeExecution(
            node_id="test-3",
            node_name="llm:node-model",
            framework="langgraph",
            start_time=_utc(),
            end_time=_utc(),
            inputs={},
            outputs={},
            metadata={
                "is_llm_call": True,
                "llm_request_payload": {},
            },
        )
        span3 = tracker._to_span(node3)
        assert span3.llm.model == "node-model"