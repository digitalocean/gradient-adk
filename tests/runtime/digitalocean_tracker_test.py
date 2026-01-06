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


class TestSessionId:
    """Test session_id handling in traces."""

    @pytest.mark.asyncio
    async def test_session_id_passed_to_create_traces(self, tracker, mock_client):
        """Test that session_id is passed to CreateTracesInput."""
        tracker.on_request_start(
            "agent", {"input": "test"}, is_evaluation=False, session_id="sess-abc-123"
        )
        tracker.on_request_end(outputs={"result": "ok"}, error=None)
        await tracker.aclose()

        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        assert isinstance(call_args, CreateTracesInput)
        assert call_args.session_id == "sess-abc-123"

    @pytest.mark.asyncio
    async def test_session_id_none_when_not_provided(self, tracker, mock_client):
        """Test that session_id is None when not provided."""
        tracker.on_request_start("agent", {"input": "test"}, is_evaluation=False)
        tracker.on_request_end(outputs={"result": "ok"}, error=None)
        await tracker.aclose()

        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        assert call_args.session_id is None

    @pytest.mark.asyncio
    async def test_session_id_preserved_across_request(self, tracker, mock_client):
        """Test that session_id is preserved throughout the request lifecycle."""
        tracker.on_request_start(
            "agent", {"input": "test"}, is_evaluation=True, session_id="eval-session"
        )

        node = NodeExecution(
            node_id="node-1",
            node_name="process",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"test": "data"},
        )
        tracker.on_node_start(node)
        tracker.on_node_end(node, {"result": "done"})
        tracker.on_request_end(outputs={"final": "result"}, error=None)

        # Use submit_and_get_trace_id for evaluation mode
        await tracker.submit_and_get_trace_id()

        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        assert call_args.session_id == "eval-session"


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
    async def test_fastapi_streaming_response_collected(self, tracker, mock_client):
        """Test that FastAPIStreamingResponse streams are collected."""
        from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        response = FastAPIStreamingResponse(mock_stream(), media_type="text/plain")
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=response, error=None)

        # The tracker wraps the iterator, so we need to consume the wrapped one
        # Access the iterator from the response (may have been replaced by tracker)
        iterator = getattr(response, "body_iterator", None) or getattr(response, "content", None)
        if iterator:
            chunks = []
            async for chunk in iterator:
                chunks.append(chunk)
            assert chunks == ["chunk1", "chunk2", "chunk3"]

        # Wait for async submission to complete
        await asyncio.sleep(0.1)

        # Verify the collected output was submitted
        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        # The output should be the concatenated chunks
        output_str = str(trace.output.get("result", trace.output))
        assert "chunk1" in output_str
        assert "chunk2" in output_str
        assert "chunk3" in output_str

    @pytest.mark.asyncio
    async def test_streaming_with_string_chunks(self, tracker, mock_client):
        """Test streaming with string chunks."""
        from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

        async def mock_stream():
            yield "hello"
            yield " "
            yield "world"

        response = FastAPIStreamingResponse(mock_stream(), media_type="text/plain")
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=response, error=None)

        # Initially outputs should be None
        assert tracker._req.get("outputs") is None

        # Consume stream - tracker wraps it, so we get the wrapped iterator
        iterator = getattr(response, "body_iterator", None) or getattr(response, "content", None)
        if iterator:
            chunks = []
            async for chunk in iterator:
                chunks.append(chunk)
            assert chunks == ["hello", " ", "world"]

        # Wait for async submission
        await asyncio.sleep(0.1)

        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        # Output should contain concatenated strings
        output_str = str(trace.output.get("result", trace.output))
        assert "hello" in output_str
        assert "world" in output_str
        # Verify it was collected in tracker
        assert tracker._req.get("outputs") is not None
        assert "hello" in str(tracker._req.get("outputs"))

    @pytest.mark.asyncio
    async def test_streaming_with_bytes_chunks(self, tracker, mock_client):
        """Test streaming with bytes chunks."""
        from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

        async def mock_stream():
            yield b"hello"
            yield b" "
            yield b"world"

        response = FastAPIStreamingResponse(mock_stream(), media_type="text/plain")
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=response, error=None)

        # Consume stream - bytes should be yielded as-is, but collected as decoded strings
        iterator = getattr(response, "body_iterator", None) or getattr(response, "content", None)
        if iterator:
            chunks = []
            async for chunk in iterator:
                chunks.append(chunk)
            # Chunks are yielded as bytes
            assert chunks == [b"hello", b" ", b"world"]

        await asyncio.sleep(0.1)

        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        # Bytes should be decoded to strings in the collected output
        output_str = str(trace.output.get("result", trace.output))
        assert "hello" in output_str
        assert "world" in output_str

    @pytest.mark.asyncio
    async def test_streaming_with_dict_chunks(self, tracker, mock_client):
        """Test streaming with dict chunks."""
        from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
        import json

        async def mock_stream():
            yield {"type": "status", "message": "started"}
            yield {"type": "data", "value": 42}

        response = FastAPIStreamingResponse(mock_stream(), media_type="text/plain")
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=response, error=None)

        # Consume stream - dicts are yielded as-is
        iterator = getattr(response, "body_iterator", None) or getattr(response, "content", None)
        if iterator:
            chunks = []
            async for chunk in iterator:
                chunks.append(chunk)
            assert len(chunks) == 2
            assert chunks[0] == {"type": "status", "message": "started"}

        await asyncio.sleep(0.1)

        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        # Dicts should be converted to strings in collected output
        output_str = str(trace.output.get("result", trace.output))
        assert "status" in output_str or "started" in output_str
        assert "data" in output_str or "42" in output_str

    @pytest.mark.asyncio
    async def test_streaming_skips_none_chunks(self, tracker, mock_client):
        """Test that None chunks are skipped during streaming."""
        from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

        async def mock_stream():
            yield "chunk1"
            yield None
            yield "chunk2"

        response = FastAPIStreamingResponse(mock_stream(), media_type="text/plain")
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=response, error=None)

        # Consume stream - None should be skipped by tracker's collecting_iter
        iterator = getattr(response, "body_iterator", None) or getattr(response, "content", None)
        if iterator:
            chunks = []
            async for chunk in iterator:
                chunks.append(chunk)
            # None is skipped, so we only get 2 chunks
            assert chunks == ["chunk1", "chunk2"]

        await asyncio.sleep(0.1)

        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        output_str = str(trace.output.get("result", trace.output))
        assert "chunk1" in output_str
        assert "chunk2" in output_str
        # None should not be in collected output
        assert "None" not in output_str or "chunk1chunk2" in output_str

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, tracker, mock_client):
        """Test that streaming errors are handled and tracked."""
        from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

        async def mock_stream():
            yield "chunk1"
            raise RuntimeError("stream error")
            yield "chunk2"  # Never reached

        response = FastAPIStreamingResponse(mock_stream(), media_type="text/plain")
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=response, error=None)

        # Consume stream - error should be caught by tracker's collecting_iter
        iterator = getattr(response, "body_iterator", None) or getattr(response, "content", None)
        if iterator:
            chunks = []
            try:
                async for chunk in iterator:
                    chunks.append(chunk)
            except RuntimeError as e:
                # Error is re-raised after collection
                assert str(e) == "stream error"
            assert chunks == ["chunk1"]  # Only first chunk collected

        await asyncio.sleep(0.1)

        # Should still submit with error
        assert mock_client.create_traces.called
        call_args = mock_client.create_traces.call_args[0][0]
        trace = call_args.traces[0]
        # Error should be in output
        assert "error" in trace.output or "stream error" in str(trace.output)
        # Verify error was set in tracker
        assert tracker._req.get("error") is not None
        assert "stream error" in str(tracker._req.get("error"))

    @pytest.mark.asyncio
    async def test_streaming_with_raw_async_iterator(self, tracker, mock_client):
        """Test streaming with raw async iterator (not wrapped in FastAPIStreamingResponse).
        
        Note: Raw async iterators are detected by the tracker, but since they can't be
        replaced in place, the tracker sets outputs to None initially. The actual collection
        would need to happen through the wrapped iterator, which isn't accessible for raw iterators.
        This test verifies the tracker handles raw iterators gracefully without crashing.
        """
        async def mock_stream():
            yield "raw"
            yield "chunk"

        stream = mock_stream()
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=stream, error=None)

        # The tracker detects raw async iterators and sets outputs to None (streaming mode)
        # However, since raw iterators can't be replaced in place, the wrapper isn't used
        assert tracker._req.get("outputs") is None

        # Consume the original stream (tracker can't wrap it, so we use original)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        assert chunks == ["raw", "chunk"]

        # Wait a bit - for raw iterators, the tracker's wrapper isn't accessible
        # so outputs remains None (this is expected behavior)
        await asyncio.sleep(0.1)

        # The tracker detected it as streaming (outputs is None)
        # For raw iterators, the tracker can't easily collect since it can't replace the iterator
        # This is acceptable - the main use case is FastAPIStreamingResponse which can be wrapped
        assert tracker._req.get("outputs") is None

    @pytest.mark.asyncio
    async def test_streaming_outputs_set_to_none_initially(self, tracker, mock_client):
        """Test that streaming outputs are set to None initially, then filled after streaming."""
        from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

        async def mock_stream():
            yield "test"

        response = FastAPIStreamingResponse(mock_stream(), media_type="text/plain")
        tracker.on_request_start("agent", {})
        tracker.on_request_end(outputs=response, error=None)

        # Initially outputs should be None (will be filled after streaming)
        assert tracker._req.get("outputs") is None

        # Consume stream
        iterator = getattr(response, "body_iterator", None) or getattr(response, "content", None)
        if iterator:
            async for _ in iterator:
                pass

        await asyncio.sleep(0.1)

        # After streaming, outputs should be filled
        assert tracker._req.get("outputs") is not None
        assert "test" in str(tracker._req.get("outputs"))