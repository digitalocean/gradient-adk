from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from gradient_adk.runtime.digitalocean_tracker import DigitalOceanTracesTracker
from gradient_adk.runtime.interfaces import NodeExecution
from gradient_adk.digital_ocean_api import (
    CreateTracesInput,
    CreateTracesOutput,
    TraceSpanType,
)
from gradient_adk.streaming import StreamingResponse, ServerSentEventsResponse


@pytest.fixture
def mock_client():
    """Mock AsyncDigitalOceanGenAI client."""
    client = AsyncMock()
    client.create_traces = AsyncMock(
        return_value=CreateTracesOutput(trace_uuids=["trace-123"])
    )
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def tracker(mock_client):
    """Create a tracker instance with mocked client."""
    return DigitalOceanTracesTracker(
        client=mock_client,
        agent_workspace_name="test-workspace",
        agent_deployment_name="test-deployment",
    )


class TestRequestLifecycle:
    """Test the basic request lifecycle: start, node execution, end."""

    @pytest.mark.asyncio
    async def test_simple_request_with_single_node(self, tracker, mock_client):
        """Test a simple request with one node execution."""
        # Start request
        tracker.on_request_start("my_agent", {"query": "hello"}, is_evaluation=False)

        # Execute a node
        node = NodeExecution(
            node_id="node-1",
            node_name="process",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"query": "hello"},
        )
        tracker.on_node_start(node)
        tracker.on_node_end(node, {"response": "world"})

        # End request
        tracker.on_request_end(outputs={"result": "success"}, error=None)

        # Wait for async submission
        await tracker.aclose()

        # Verify API was called
        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        assert isinstance(call_args, CreateTracesInput)
        assert call_args.agent_workspace_name == "test-workspace"
        assert call_args.agent_deployment_name == "test-deployment"
        assert len(call_args.traces) == 1

        trace = call_args.traces[0]
        assert trace.name == "my_agent"
        assert trace.input == {"query": "hello"}
        assert trace.output == {"result": "success"}
        assert len(trace.spans) == 1

        span = trace.spans[0]
        assert span.name == "process"
        assert span.input == {"query": "hello"}
        assert span.output == {"response": "world"}
        assert span.type == TraceSpanType.TRACE_SPAN_TYPE_TOOL

    @pytest.mark.asyncio
    async def test_request_with_multiple_nodes(self, tracker, mock_client):
        """Test a request with multiple node executions."""
        tracker.on_request_start("multi_node_agent", {"input": "test"})

        # First node
        node1 = NodeExecution(
            node_id="node-1",
            node_name="fetch",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"input": "test"},
        )
        tracker.on_node_start(node1)
        tracker.on_node_end(node1, {"data": "fetched"})

        # Second node
        node2 = NodeExecution(
            node_id="node-2",
            node_name="process",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"data": "fetched"},
        )
        tracker.on_node_start(node2)
        tracker.on_node_end(node2, {"result": "processed"})

        tracker.on_request_end(outputs={"final": "result"}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert len(trace.spans) == 2
        assert trace.spans[0].name == "fetch"
        assert trace.spans[1].name == "process"

    @pytest.mark.asyncio
    async def test_request_with_error(self, tracker, mock_client):
        """Test handling of request-level errors."""
        tracker.on_request_start("error_agent", {"input": "test"})
        tracker.on_request_end(outputs=None, error="Something went wrong")
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert trace.output["error"] == "Something went wrong"


class TestNodeExecution:
    """Test node execution tracking."""

    @pytest.mark.asyncio
    async def test_node_with_dict_inputs_outputs(self, tracker, mock_client):
        """Test that dict inputs/outputs are preserved as-is."""
        tracker.on_request_start("agent", {})

        node = NodeExecution(
            node_id="node-1",
            node_name="dict_node",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"key": "value", "nested": {"a": 1}},
        )
        tracker.on_node_start(node)
        tracker.on_node_end(node, {"output_key": "output_value", "count": 42})
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        span = call_args.traces[0].spans[0]
        assert span.input == {"key": "value", "nested": {"a": 1}}
        assert span.output == {"output_key": "output_value", "count": 42}

    @pytest.mark.asyncio
    async def test_node_with_primitive_inputs_outputs(self, tracker, mock_client):
        """Test that primitive inputs/outputs are wrapped."""
        tracker.on_request_start("agent", {})

        node = NodeExecution(
            node_id="node-1",
            node_name="primitive_node",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs="string input",
        )
        tracker.on_node_start(node)
        tracker.on_node_end(node, 123)
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        span = call_args.traces[0].spans[0]
        assert span.input == {"input": "string input"}
        assert span.output == {"output": 123}

    @pytest.mark.asyncio
    async def test_node_with_none_outputs(self, tracker, mock_client):
        """Test handling of None outputs."""
        tracker.on_request_start("agent", {})

        node = NodeExecution(
            node_id="node-1",
            node_name="none_node",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={},
        )
        tracker.on_node_start(node)
        tracker.on_node_end(node, None)
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        span = call_args.traces[0].spans[0]
        assert span.output == {"output": None}

    @pytest.mark.asyncio
    async def test_node_error_handling(self, tracker, mock_client):
        """Test node error tracking."""
        tracker.on_request_start("agent", {})

        node = NodeExecution(
            node_id="node-1",
            node_name="error_node",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"test": "data"},
        )
        tracker.on_node_start(node)
        tracker.on_node_error(node, ValueError("Test error"))
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        span = call_args.traces[0].spans[0]
        assert span.output["error"] == "Test error"


class TestSpanTypes:
    """Test span type classification based on metadata."""

    @pytest.mark.asyncio
    async def test_llm_span_type(self, tracker, mock_client):
        """Test that nodes with is_llm_call metadata get LLM span type."""
        tracker.on_request_start("agent", {})

        node = NodeExecution(
            node_id="node-1",
            node_name="llm_call",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"prompt": "test"},
            metadata={"is_llm_call": True},
        )
        tracker.on_node_start(node)
        tracker.on_node_end(node, {"response": "generated"})
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        span = call_args.traces[0].spans[0]
        assert span.type == TraceSpanType.TRACE_SPAN_TYPE_LLM

    @pytest.mark.asyncio
    async def test_retriever_span_type(self, tracker, mock_client):
        """Test that nodes with is_retriever_call metadata get RETRIEVER span type."""
        tracker.on_request_start("agent", {})

        node = NodeExecution(
            node_id="node-1",
            node_name="retriever_call",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"query": "search"},
            metadata={"is_retriever_call": True},
        )
        tracker.on_node_start(node)
        tracker.on_node_end(node, {"documents": []})
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        span = call_args.traces[0].spans[0]
        assert span.type == TraceSpanType.TRACE_SPAN_TYPE_RETRIEVER

    @pytest.mark.asyncio
    async def test_tool_span_type_default(self, tracker, mock_client):
        """Test that nodes without special metadata get TOOL span type."""
        tracker.on_request_start("agent", {})

        node = NodeExecution(
            node_id="node-1",
            node_name="tool_call",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"arg": "value"},
        )
        tracker.on_node_start(node)
        tracker.on_node_end(node, {"result": "done"})
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        span = call_args.traces[0].spans[0]
        assert span.type == TraceSpanType.TRACE_SPAN_TYPE_TOOL


class TestEvaluationMode:
    """Test evaluation-specific behavior."""

    @pytest.mark.asyncio
    async def test_evaluation_mode_returns_trace_id(self, tracker, mock_client):
        """Test that evaluation mode returns trace_id synchronously."""
        tracker.on_request_start("agent", {}, is_evaluation=True)
        tracker.on_request_end(outputs={"result": "test"}, error=None)

        trace_id = await tracker.submit_and_get_trace_id()
        assert trace_id == "trace-123"

    @pytest.mark.asyncio
    async def test_evaluation_mode_does_not_fire_and_forget(self, tracker, mock_client):
        """Test that evaluation mode doesn't fire-and-forget (waits for explicit submit)."""
        tracker.on_request_start("agent", {}, is_evaluation=True)
        tracker.on_request_end(outputs={"result": "test"}, error=None)

        # Should not have called API yet
        assert not mock_client.create_traces.called

        # Explicit submit
        await tracker.submit_and_get_trace_id()
        assert mock_client.create_traces.called


class TestTopLevelIO:
    """Test top-level input/output normalization."""

    @pytest.mark.asyncio
    async def test_dict_inputs_preserved(self, tracker, mock_client):
        """Test that dict inputs are preserved at top level."""
        tracker.on_request_start("agent", {"key1": "val1", "key2": 42})
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert trace.input == {"key1": "val1", "key2": 42}

    @pytest.mark.asyncio
    async def test_primitive_inputs_wrapped(self, tracker, mock_client):
        """Test that primitive inputs are wrapped at top level."""
        tracker.on_request_start("agent", "string input")
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert trace.input == {"input": "string input"}

    @pytest.mark.asyncio
    async def test_none_inputs_become_empty_dict(self, tracker, mock_client):
        """Test that None inputs become empty dict."""
        tracker.on_request_start("agent", None)
        tracker.on_request_end(outputs={}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert trace.input == {}

    @pytest.mark.asyncio
    async def test_dict_outputs_preserved(self, tracker, mock_client):
        """Test that dict outputs are preserved at top level."""
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs={"status": "success", "count": 5}, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert trace.output == {"status": "success", "count": 5}

    @pytest.mark.asyncio
    async def test_primitive_outputs_wrapped(self, tracker, mock_client):
        """Test that primitive outputs are wrapped at top level."""
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs="success", error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert trace.output == {"result": "success"}

    @pytest.mark.asyncio
    async def test_none_outputs_become_empty_dict(self, tracker, mock_client):
        """Test that None outputs become empty dict."""
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=None, error=None)
        await tracker.aclose()

        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert trace.output == {}


class TestErrorHandling:
    """Test error handling and robustness."""

    @pytest.mark.asyncio
    async def test_api_error_does_not_raise(self, tracker, mock_client):
        """Test that API errors don't propagate."""
        mock_client.create_traces.side_effect = Exception("API error")

        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs={}, error=None)

        # Should not raise
        await tracker.aclose()

    @pytest.mark.asyncio
    async def test_multiple_requests_reset_state(self, tracker, mock_client):
        """Test that multiple requests properly reset internal state."""
        # First request
        tracker.on_request_start("agent1", {"first": "request"})
        node1 = NodeExecution(
            node_id="node-1",
            node_name="node1",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={},
        )
        tracker.on_node_start(node1)
        tracker.on_node_end(node1, {})
        tracker.on_request_end(outputs={"first": "output"}, error=None)
        await tracker.aclose()

        # Second request
        tracker.on_request_start("agent2", {"second": "request"})
        node2 = NodeExecution(
            node_id="node-2",
            node_name="node2",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={},
        )
        tracker.on_node_start(node2)
        tracker.on_node_end(node2, {})
        tracker.on_request_end(outputs={"second": "output"}, error=None)
        await tracker.aclose()

        # Verify second request has correct data
        assert mock_client.create_traces.call_count == 2
        second_call_args = mock_client.create_traces.call_args[0][0]
        trace = second_call_args.traces[0]
        assert trace.name == "agent2"
        assert trace.input == {"second": "request"}
        assert trace.output == {"second": "output"}
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "node2"


class TestStreamingSupport:
    """Test streaming response handling."""

    @pytest.mark.asyncio
    async def test_streaming_response_collected(self, tracker, mock_client):
        """Test that streaming responses are collected."""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        response = StreamingResponse(content=mock_stream())
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=response, error=None)

        # Consume the stream
        chunks = []
        async for chunk in response.content:
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2", "chunk3"]

        # Wait for submission
        await asyncio.sleep(0.1)

        # Verify the collected output
        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        assert trace.output == {"result": "chunk1chunk2chunk3"}
