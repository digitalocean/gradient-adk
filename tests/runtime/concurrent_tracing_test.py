"""Tests for concurrent request handling in the tracing system.

These tests verify that when multiple requests are processed simultaneously,
each request maintains proper isolation of:
- Request state (inputs, outputs, errors)
- Span tracking (live and done lists)
- Trace submission (correct spans per trace)
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

from gradient_adk.runtime.digitalocean_tracker import (
    DigitalOceanTracesTracker,
    RequestState,
    get_current_request_state,
    set_current_request_state,
    reset_request_state,
)
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
    # Return unique trace IDs for each call
    call_count = 0
    
    async def create_traces_mock(req):
        nonlocal call_count
        call_count += 1
        return CreateTracesOutput(trace_uuids=[f"trace-{call_count}"])
    
    client.create_traces = AsyncMock(side_effect=create_traces_mock)
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


class TestRequestStateIsolation:
    """Test that request state is properly isolated between concurrent requests."""

    def test_request_state_creation(self):
        """Test that RequestState is created with unique IDs."""
        state1 = RequestState()
        state2 = RequestState()
        
        assert state1.request_id != state2.request_id
        assert state1.req is not state2.req
        assert state1.live is not state2.live
        assert state1.done is not state2.done

    def test_context_var_isolation(self):
        """Test that context variables properly isolate state."""
        # Initially no state
        assert get_current_request_state() is None
        
        # Set state
        state1 = RequestState()
        token1 = set_current_request_state(state1)
        assert get_current_request_state() is state1
        
        # Create nested state
        state2 = RequestState()
        token2 = set_current_request_state(state2)
        assert get_current_request_state() is state2
        
        # Reset to previous
        reset_request_state(token2)
        assert get_current_request_state() is state1
        
        # Reset to None
        reset_request_state(token1)
        assert get_current_request_state() is None

    @pytest.mark.asyncio
    async def test_on_request_start_creates_isolated_state(self, tracker, mock_client):
        """Test that on_request_start creates isolated state for each request."""
        # Start first request
        token1 = tracker.on_request_start("agent1", {"input": "req1"})
        state1 = get_current_request_state()
        
        assert state1 is not None
        assert state1.req["entrypoint"] == "agent1"
        assert state1.req["inputs"] == {"input": "req1"}
        
        # Start second request (simulating concurrent request)
        token2 = tracker.on_request_start("agent2", {"input": "req2"})
        state2 = get_current_request_state()
        
        assert state2 is not None
        assert state2 is not state1
        assert state2.req["entrypoint"] == "agent2"
        assert state2.req["inputs"] == {"input": "req2"}
        
        # State 1's data should be unchanged
        assert state1.req["entrypoint"] == "agent1"
        
        # Clean up
        reset_request_state(token2)
        reset_request_state(token1)


class TestConcurrentSpanTracking:
    """Test that spans are tracked correctly under concurrent requests."""

    @pytest.mark.asyncio
    async def test_spans_isolated_between_requests(self, tracker, mock_client):
        """Test that spans are properly isolated between concurrent requests."""
        # Start request 1
        token1 = tracker.on_request_start("agent1", {"query": "hello"})
        
        # Add span to request 1
        node1 = NodeExecution(
            node_id="node-1-a",
            node_name="process_a",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"query": "hello"},
        )
        tracker.on_node_start(node1)
        tracker.on_node_end(node1, {"response": "world"})
        
        # Save state1 reference
        state1 = get_current_request_state()
        
        # Simulate concurrent request 2 starting
        token2 = tracker.on_request_start("agent2", {"query": "test"})
        state2 = get_current_request_state()
        
        # Add span to request 2
        node2 = NodeExecution(
            node_id="node-2-a",
            node_name="process_b",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
            inputs={"query": "test"},
        )
        tracker.on_node_start(node2)
        tracker.on_node_end(node2, {"response": "result"})
        
        # Verify request 1 still has only its spans
        assert len(state1.done) == 1
        assert state1.done[0].node_name == "process_a"
        
        # Verify request 2 has only its spans
        assert len(state2.done) == 1
        assert state2.done[0].node_name == "process_b"
        
        # Clean up
        reset_request_state(token2)
        reset_request_state(token1)

    @pytest.mark.asyncio
    async def test_multiple_spans_per_request_isolated(self, tracker, mock_client):
        """Test multiple spans per request remain isolated."""
        # Start two concurrent requests
        token1 = tracker.on_request_start("agent1", {"input": "1"})
        state1 = get_current_request_state()
        
        token2 = tracker.on_request_start("agent2", {"input": "2"})
        state2 = get_current_request_state()
        
        # Add spans to request 1
        for i in range(3):
            node = NodeExecution(
                node_id=f"node-1-{i}",
                node_name=f"span1_{i}",
                framework="langgraph",
                start_time=datetime.now(timezone.utc),
            )
            # Temporarily switch to state1 to add spans
            temp_token = set_current_request_state(state1)
            tracker.on_node_start(node)
            tracker.on_node_end(node, {"result": f"output1_{i}"})
            reset_request_state(temp_token)
        
        # Add spans to request 2
        for i in range(2):
            node = NodeExecution(
                node_id=f"node-2-{i}",
                node_name=f"span2_{i}",
                framework="langgraph",
                start_time=datetime.now(timezone.utc),
            )
            tracker.on_node_start(node)
            tracker.on_node_end(node, {"result": f"output2_{i}"})
        
        # Verify correct span counts
        assert len(state1.done) == 3
        assert len(state2.done) == 2
        
        # Verify span names
        assert all(s.node_name.startswith("span1_") for s in state1.done)
        assert all(s.node_name.startswith("span2_") for s in state2.done)
        
        # Clean up
        reset_request_state(token2)
        reset_request_state(token1)


class TestConcurrentTraceSubmission:
    """Test that trace submission works correctly with concurrent requests."""

    @pytest.mark.asyncio
    async def test_concurrent_trace_submissions(self, tracker, mock_client):
        """Test that concurrent requests submit separate traces with correct data."""
        submissions = []
        
        # Capture all submissions
        async def capture_create_traces(req):
            submissions.append(req)
            return CreateTracesOutput(trace_uuids=[f"trace-{len(submissions)}"])
        
        mock_client.create_traces = AsyncMock(side_effect=capture_create_traces)
        
        # Process two requests concurrently
        async def process_request(name: str, input_data: dict, span_count: int):
            token = tracker.on_request_start(name, input_data)
            state = get_current_request_state()
            
            for i in range(span_count):
                node = NodeExecution(
                    node_id=f"{name}-node-{i}",
                    node_name=f"{name}_span_{i}",
                    framework="langgraph",
                    start_time=datetime.now(timezone.utc),
                )
                tracker.on_node_start(node)
                tracker.on_node_end(node, {"i": i})
            
            state.req["outputs"] = {"result": f"{name}_output"}
            await tracker._submit_state(state)
            reset_request_state(token)
        
        # Run concurrently
        await asyncio.gather(
            process_request("agent1", {"q": "query1"}, 2),
            process_request("agent2", {"q": "query2"}, 3),
        )
        
        # Verify we got 2 separate submissions
        assert len(submissions) == 2
        
        # Find each submission by name
        sub1 = next((s for s in submissions if s.traces[0].name == "agent1"), None)
        sub2 = next((s for s in submissions if s.traces[0].name == "agent2"), None)
        
        assert sub1 is not None
        assert sub2 is not None
        
        # Verify correct span counts
        assert len(sub1.traces[0].spans) == 2
        assert len(sub2.traces[0].spans) == 3
        
        # Verify span names match their trace
        for span in sub1.traces[0].spans:
            assert span.name.startswith("agent1_")
        
        for span in sub2.traces[0].spans:
            assert span.name.startswith("agent2_")

    @pytest.mark.asyncio
    async def test_evaluation_mode_isolation(self, tracker, mock_client):
        """Test that evaluation mode flag is isolated per request."""
        # Start evaluation request
        token1 = tracker.on_request_start(
            "eval_agent", {"input": "eval"}, is_evaluation=True
        )
        state1 = get_current_request_state()
        
        # Start normal request
        token2 = tracker.on_request_start(
            "normal_agent", {"input": "normal"}, is_evaluation=False
        )
        state2 = get_current_request_state()
        
        # Verify isolation
        assert state1.is_evaluation is True
        assert state2.is_evaluation is False
        
        # Clean up
        reset_request_state(token2)
        reset_request_state(token1)

    @pytest.mark.asyncio
    async def test_session_id_isolation(self, tracker, mock_client):
        """Test that session IDs are isolated per request."""
        token1 = tracker.on_request_start(
            "agent1", {}, session_id="session-abc"
        )
        state1 = get_current_request_state()
        
        token2 = tracker.on_request_start(
            "agent2", {}, session_id="session-xyz"
        )
        state2 = get_current_request_state()
        
        # Verify isolation
        assert state1.session_id == "session-abc"
        assert state2.session_id == "session-xyz"
        
        # Clean up
        reset_request_state(token2)
        reset_request_state(token1)


class TestConcurrentErrorHandling:
    """Test error handling in concurrent scenarios."""

    @pytest.mark.asyncio
    async def test_error_isolated_per_request(self, tracker, mock_client):
        """Test that errors are isolated to their specific request."""
        token1 = tracker.on_request_start("agent1", {})
        state1 = get_current_request_state()
        
        # Add node with error
        node1 = NodeExecution(
            node_id="error-node",
            node_name="failing_node",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
        )
        tracker.on_node_start(node1)
        tracker.on_node_error(node1, ValueError("Test error"))
        
        # Start another request
        token2 = tracker.on_request_start("agent2", {})
        state2 = get_current_request_state()
        
        # Add successful node
        node2 = NodeExecution(
            node_id="success-node",
            node_name="success_node",
            framework="langgraph",
            start_time=datetime.now(timezone.utc),
        )
        tracker.on_node_start(node2)
        tracker.on_node_end(node2, {"success": True})
        
        # Verify error is isolated
        assert len(state1.done) == 1
        assert state1.done[0].error == "Test error"
        
        assert len(state2.done) == 1
        assert state2.done[0].error is None
        
        # Clean up
        reset_request_state(token2)
        reset_request_state(token1)


class TestNetworkInterceptorConcurrency:
    """Test network interceptor behavior under concurrent requests."""

    def test_captured_request_has_unique_id(self):
        """Test that CapturedRequest objects have unique IDs."""
        from gradient_adk.runtime.network_interceptor import CapturedRequest
        
        req1 = CapturedRequest(url="http://test.com")
        req2 = CapturedRequest(url="http://test.com")
        
        assert req1.request_id != req2.request_id

    def test_response_correlation_by_request_id(self):
        """Test that responses can be correlated by request ID."""
        from gradient_adk.runtime.network_interceptor import NetworkInterceptor
        
        interceptor = NetworkInterceptor()
        interceptor.add_endpoint_pattern("test.com")
        
        # Record two requests
        id1 = interceptor._record_request(
            "http://test.com/api1",
            {"payload": "request1"}
        )
        id2 = interceptor._record_request(
            "http://test.com/api2",
            {"payload": "request2"}
        )
        
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2
        
        # Record responses out of order
        interceptor._record_response(
            "http://test.com/api2",
            {"result": "response2"},
            request_id=id2,
        )
        interceptor._record_response(
            "http://test.com/api1",
            {"result": "response1"},
            request_id=id1,
        )
        
        # Verify correct correlation
        captured = interceptor.get_captured_requests_since(0)
        assert len(captured) == 2
        
        # Find by request ID
        req1 = next(c for c in captured if c.request_id == id1)
        req2 = next(c for c in captured if c.request_id == id2)
        
        assert req1.response_payload == {"result": "response1"}
        assert req2.response_payload == {"result": "response2"}

    def test_response_fallback_to_url_matching(self):
        """Test response correlation falls back to URL matching when no request_id."""
        from gradient_adk.runtime.network_interceptor import NetworkInterceptor
        
        interceptor = NetworkInterceptor()
        interceptor.add_endpoint_pattern("test.com")
        
        # Record request
        interceptor._record_request(
            "http://test.com/api",
            {"payload": "data"}
        )
        
        # Record response without request_id (legacy behavior)
        interceptor._record_response(
            "http://test.com/api",
            {"result": "data"},
        )
        
        captured = interceptor.get_captured_requests_since(0)
        assert len(captured) == 1
        assert captured[0].response_payload == {"result": "data"}


class TestRaceConditionPrevention:
    """Test that race conditions are prevented in concurrent scenarios."""

    @pytest.mark.asyncio
    async def test_no_span_leakage_between_requests(self, tracker, mock_client):
        """Test that spans don't leak between concurrent requests."""
        submissions = []
        
        async def capture_create_traces(req):
            submissions.append(req)
            return CreateTracesOutput(trace_uuids=["trace-1"])
        
        mock_client.create_traces = AsyncMock(side_effect=capture_create_traces)
        
        async def process_request(name: str, delay_before_spans: float):
            token = tracker.on_request_start(name, {"name": name})
            state = get_current_request_state()
            
            # Delay to interleave with other request
            await asyncio.sleep(delay_before_spans)
            
            # Add span
            node = NodeExecution(
                node_id=f"{name}-node",
                node_name=f"{name}_span",
                framework="langgraph",
                start_time=datetime.now(timezone.utc),
            )
            tracker.on_node_start(node)
            
            # Small delay
            await asyncio.sleep(0.01)
            
            tracker.on_node_end(node, {"name": name})
            state.req["outputs"] = {"name": name}
            await tracker._submit_state(state)
            reset_request_state(token)
        
        # Run with interleaved timing
        await asyncio.gather(
            process_request("A", 0.0),
            process_request("B", 0.005),
        )
        
        # Verify each submission has exactly one span with correct name
        assert len(submissions) == 2
        
        for sub in submissions:
            trace = sub.traces[0]
            assert len(trace.spans) == 1
            # Span name should match trace name
            expected_span_name = f"{trace.name}_span"
            assert trace.spans[0].name == expected_span_name

    @pytest.mark.asyncio
    async def test_rapid_concurrent_requests(self, tracker, mock_client):
        """Test many rapid concurrent requests maintain isolation."""
        NUM_REQUESTS = 20
        submissions = []
        
        async def capture_create_traces(req):
            submissions.append(req)
            return CreateTracesOutput(trace_uuids=["trace-1"])
        
        mock_client.create_traces = AsyncMock(side_effect=capture_create_traces)
        
        async def process_request(request_id: int):
            name = f"agent_{request_id}"
            token = tracker.on_request_start(name, {"id": request_id})
            state = get_current_request_state()
            
            # Add a unique number of spans based on request_id
            span_count = (request_id % 3) + 1
            for i in range(span_count):
                node = NodeExecution(
                    node_id=f"{name}-node-{i}",
                    node_name=f"{name}_span_{i}",
                    framework="langgraph",
                    start_time=datetime.now(timezone.utc),
                )
                tracker.on_node_start(node)
                tracker.on_node_end(node, {"id": request_id, "i": i})
            
            state.req["outputs"] = {"id": request_id}
            await tracker._submit_state(state)
            reset_request_state(token)
        
        # Run all requests concurrently
        await asyncio.gather(*[process_request(i) for i in range(NUM_REQUESTS)])
        
        # Verify we got all submissions
        assert len(submissions) == NUM_REQUESTS
        
        # Verify each trace has correct number of spans
        for sub in submissions:
            trace = sub.traces[0]
            # Extract request ID from name
            request_id = int(trace.name.split("_")[1])
            expected_span_count = (request_id % 3) + 1
            assert len(trace.spans) == expected_span_count, (
                f"Request {request_id} expected {expected_span_count} spans, "
                f"got {len(trace.spans)}"
            )


class TestBackwardCompatibility:
    """Test backward compatibility with legacy usage patterns."""

    @pytest.mark.asyncio
    async def test_legacy_usage_without_context(self, mock_client):
        """Test that legacy usage (without context management) still works."""
        tracker = DigitalOceanTracesTracker(
            client=mock_client,
            agent_workspace_name="test-ws",
            agent_deployment_name="test-dep",
        )
        
        # Clear any existing context
        token = set_current_request_state(None)
        
        try:
            # Use legacy pattern (directly manipulating instance vars)
            tracker._req = {"entrypoint": "legacy_agent", "inputs": {"test": True}}
            tracker._done = []
            tracker._is_evaluation = False
            tracker._session_id = "legacy-session"
            
            # Add node using instance vars
            node = NodeExecution(
                node_id="legacy-node",
                node_name="legacy_span",
                framework="test",
                start_time=datetime.now(timezone.utc),
            )
            tracker._live[node.node_id] = node
            tracker._live.pop(node.node_id)
            node.end_time = datetime.now(timezone.utc)
            node.outputs = {"legacy": True}
            tracker._done.append(node)
            
            tracker._req["outputs"] = {"result": "success"}
            
            # Submit using legacy method
            await tracker._submit()
            
            # Verify submission
            assert mock_client.create_traces.called
            call_args = mock_client.create_traces.call_args[0][0]
            assert call_args.traces[0].name == "legacy_agent"
            assert len(call_args.traces[0].spans) == 1
        finally:
            reset_request_state(token)