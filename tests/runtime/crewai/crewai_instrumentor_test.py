"""Unit and integration tests for the CrewAI instrumentor."""

import pytest
from unittest.mock import MagicMock, patch

# Skip all tests if crewai is not installed
pytest.importorskip("crewai")

from gradient_adk.runtime.crewai.crewai_instrumentor import (
    CrewAIInstrumentor,
    _freeze,
    _get_captured_payloads_with_type,
    _transform_kbaas_response,
    _get_current_agent,
    _push_agent,
    _pop_agent,
    AgentContext,
    _mk_exec,
    _agent_stack,
    _agent_stack_lock,
)


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def tracker():
    """Mock tracker with on_node_start/end/error methods."""
    t = MagicMock()
    t.on_node_start = MagicMock()
    t.on_node_end = MagicMock()
    t.on_node_error = MagicMock()
    return t


@pytest.fixture
def interceptor():
    """Mock network interceptor."""
    intr = MagicMock()
    intr.snapshot_token.return_value = 42
    intr.hits_since.return_value = 0
    return intr


@pytest.fixture(autouse=True)
def patch_interceptor(interceptor):
    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_network_interceptor",
        return_value=interceptor,
    ):
        yield interceptor


@pytest.fixture(autouse=True)
def clear_agent_stack():
    """Clear the agent stack before and after each test."""
    with _agent_stack_lock:
        _agent_stack.clear()
    yield
    with _agent_stack_lock:
        _agent_stack.clear()


@pytest.fixture
def instrumentor(tracker):
    """Installed CrewAI instrumentor."""
    inst = CrewAIInstrumentor()
    inst.install(tracker)
    yield inst
    inst.uninstall()


@pytest.fixture
def mock_agent():
    """Mock CrewAI Agent for event testing."""
    agent = MagicMock()
    agent.role = "Researcher"
    agent.goal = "Find information"
    return agent


@pytest.fixture
def mock_task():
    """Mock CrewAI Task for event testing."""
    task = MagicMock()
    task.description = "Research the topic"
    return task


@pytest.fixture
def mock_tool():
    """Mock CrewAI Tool for event testing."""
    tool = MagicMock()
    tool.name = "search_tool"
    return tool


# -----------------------------
# Instrumentor Lifecycle Tests
# -----------------------------


def test_install_sets_installed_flag(tracker):
    """Test that install sets the _installed flag."""
    inst = CrewAIInstrumentor()
    assert not inst._installed

    inst.install(tracker)
    assert inst._installed

    inst.uninstall()
    assert not inst._installed


def test_install_is_idempotent(tracker):
    """Test that calling install twice is a no-op."""
    inst = CrewAIInstrumentor()

    inst.install(tracker)
    first_tracker = inst._tracker

    inst.install(tracker)
    assert inst._tracker is first_tracker

    inst.uninstall()


def test_uninstall_without_install_is_safe():
    """Test that uninstalling before installing doesn't raise."""
    inst = CrewAIInstrumentor()
    inst.uninstall()
    assert not inst._installed


def test_is_installed_property(tracker):
    """Test is_installed returns correct boolean state."""
    inst = CrewAIInstrumentor()

    assert not inst.is_installed()

    inst.install(tracker)
    assert inst.is_installed()

    inst.uninstall()
    assert not inst.is_installed()


# -----------------------------
# Helper Function Tests - _freeze
# -----------------------------


def test_freeze_handles_primitives():
    """Test _freeze with primitive types."""
    assert _freeze(None) is None
    assert _freeze("string") == "string"
    assert _freeze(42) == 42
    assert _freeze(3.14) == 3.14
    assert _freeze(True) is True
    assert _freeze(False) is False


def test_freeze_handles_dict():
    """Test _freeze with dictionaries."""
    result = _freeze({"key": "value", "nested": {"a": 1}})
    assert result == {"key": "value", "nested": {"a": 1}}


def test_freeze_handles_list():
    """Test _freeze with lists."""
    result = _freeze([1, 2, 3, {"x": "y"}])
    assert result == [1, 2, 3, {"x": "y"}]


def test_freeze_handles_tuple():
    """Test _freeze with tuples."""
    result = _freeze((1, 2, 3))
    assert result == [1, 2, 3]


def test_freeze_handles_set():
    """Test _freeze with sets."""
    result = _freeze({1, 2, 3})
    assert sorted(result) == [1, 2, 3]


def test_freeze_handles_pydantic_model():
    """Test _freeze with Pydantic models."""
    try:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = _freeze(model)
        assert result == {"name": "test", "value": 42}
    except ImportError:
        pytest.skip("Pydantic not installed")


def test_freeze_handles_unknown_object():
    """Test _freeze falls back to repr for unknown objects."""

    class CustomObject:
        def __repr__(self):
            return "CustomObject()"

    result = _freeze(CustomObject())
    assert result == "CustomObject()"


# -----------------------------
# Helper Function Tests - _transform_kbaas_response
# -----------------------------


def test_transform_kbaas_response_converts_text_content():
    """Test that text_content is converted to page_content."""
    response = {
        "results": [
            {"metadata": {"source": "doc1.pdf"}, "text_content": "Document content."}
        ],
        "total_results": 1,
    }

    transformed = _transform_kbaas_response(response)

    assert isinstance(transformed, list)
    assert len(transformed) == 1
    assert transformed[0]["page_content"] == "Document content."
    assert "text_content" not in transformed[0]


def test_transform_kbaas_response_handles_hierarchical():
    """Test hierarchical KB with parent_chunk_text."""
    response = {
        "results": [
            {
                "text_content": "Embedded chunk.",
                "parent_chunk_text": "Full parent context.",
            }
        ],
    }

    transformed = _transform_kbaas_response(response)

    assert transformed[0]["page_content"] == "Full parent context."
    assert transformed[0]["embedded_content"] == "Embedded chunk."
    assert "parent_chunk_text" not in transformed[0]
    assert "text_content" not in transformed[0]


def test_transform_kbaas_response_handles_none():
    """Test that None response is handled gracefully."""
    assert _transform_kbaas_response(None) is None


def test_transform_kbaas_response_handles_non_dict():
    """Test that non-dict responses are returned as-is."""
    assert _transform_kbaas_response("string response") == "string response"
    assert _transform_kbaas_response(123) == 123


# -----------------------------
# Helper Function Tests - _get_captured_payloads_with_type
# -----------------------------


def test_get_captured_payloads_with_type_inference_url():
    """Test _get_captured_payloads_with_type identifies inference URLs."""
    mock_intr = MagicMock()
    mock_captured = MagicMock()
    mock_captured.url = "https://inference.do-ai.run/v1/chat"
    mock_captured.request_payload = {"messages": []}
    mock_captured.response_payload = {"choices": []}

    mock_intr.get_captured_requests_since.return_value = [mock_captured]

    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_request_captured_list",
        return_value=None,
    ):
        req, resp, is_llm, is_retriever = _get_captured_payloads_with_type(mock_intr, 0)

    assert req == {"messages": []}
    assert resp == {"choices": []}
    assert is_llm is True
    assert is_retriever is False


def test_get_captured_payloads_with_type_kbaas_url():
    """Test _get_captured_payloads_with_type identifies KBaaS URLs."""
    mock_intr = MagicMock()
    mock_captured = MagicMock()
    mock_captured.url = "https://kbaas.do-ai.run/retrieve"
    mock_captured.request_payload = {"query": "test"}
    mock_captured.response_payload = {"results": []}

    mock_intr.get_captured_requests_since.return_value = [mock_captured]

    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_request_captured_list",
        return_value=None,
    ):
        req, resp, is_llm, is_retriever = _get_captured_payloads_with_type(mock_intr, 0)

    assert req == {"query": "test"}
    assert resp == {"results": []}
    assert is_llm is False
    assert is_retriever is True


def test_get_captured_payloads_with_type_no_captures():
    """Test _get_captured_payloads_with_type when no requests captured."""
    mock_intr = MagicMock()
    mock_intr.get_captured_requests_since.return_value = []

    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_request_captured_list",
        return_value=None,
    ):
        req, resp, is_llm, is_retriever = _get_captured_payloads_with_type(mock_intr, 0)

    assert req is None
    assert resp is None
    assert is_llm is False
    assert is_retriever is False


# -----------------------------
# Context Management Tests
# -----------------------------


def test_agent_context_push_pop():
    """Test agent context push and pop operations."""
    # Create and push an agent context
    node = _mk_exec("test_agent", {})
    ctx = AgentContext(node=node, agent_role="TestAgent")

    _push_agent(ctx)
    assert _get_current_agent() is ctx

    # Pop and verify
    popped = _pop_agent()
    assert popped is ctx
    assert _get_current_agent() is None


def test_agent_context_stack():
    """Test nested agent contexts."""
    # Create two contexts
    node1 = _mk_exec("agent1", {})
    ctx1 = AgentContext(node=node1, agent_role="Agent1")

    node2 = _mk_exec("agent2", {})
    ctx2 = AgentContext(node=node2, agent_role="Agent2")

    # Push both
    _push_agent(ctx1)
    assert _get_current_agent() is ctx1

    _push_agent(ctx2)
    assert _get_current_agent() is ctx2

    # Pop in reverse order
    assert _pop_agent() is ctx2
    assert _get_current_agent() is ctx1

    assert _pop_agent() is ctx1
    assert _get_current_agent() is None


def test_pop_empty_stack_returns_none():
    """Test that popping from an empty stack returns None."""
    assert _pop_agent() is None


# -----------------------------
# Span Name Generation Tests
# -----------------------------


def test_agent_span_name_uses_role():
    """Test that agent spans use the role in the name."""
    node = _mk_exec("agent:Researcher", {"task": "Research AI"})
    assert node.node_name == "agent:Researcher"
    assert node.framework == "crewai"


def test_llm_span_name_uses_model():
    """Test that LLM spans use the model name."""
    node = _mk_exec("llm:gpt-4", {"messages": []})
    assert node.node_name == "llm:gpt-4"


def test_tool_span_name_uses_tool_name():
    """Test that tool spans use the tool name directly."""
    node = _mk_exec("search_tool", {"query": "test"})
    assert node.node_name == "search_tool"


# -----------------------------
# Integration Tests - Event Handling
# -----------------------------


def test_agent_execution_started_creates_context(tracker, instrumentor, mock_agent, mock_task):
    """Test that AgentExecutionStartedEvent creates an agent context."""
    from crewai.events import crewai_event_bus, AgentExecutionStartedEvent

    # Create and emit event
    event = MagicMock(spec=AgentExecutionStartedEvent)
    event.agent = mock_agent
    event.task = mock_task

    # Emit the event manually
    crewai_event_bus.emit(event, event)

    # Verify context was created
    ctx = _get_current_agent()
    assert ctx is not None
    assert ctx.agent_role == "Researcher"
    assert ctx.node.node_name == "agent:Researcher"

    # Clean up
    _pop_agent()


def test_agent_execution_completed_reports_span(tracker, instrumentor, mock_agent, mock_task):
    """Test that AgentExecutionCompletedEvent reports the span to tracker."""
    from crewai.events import crewai_event_bus, AgentExecutionStartedEvent, AgentExecutionCompletedEvent

    # Start event
    start_event = MagicMock(spec=AgentExecutionStartedEvent)
    start_event.agent = mock_agent
    start_event.task = mock_task
    crewai_event_bus.emit(start_event, start_event)

    # Complete event
    complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
    complete_event.output = "Research completed successfully"
    crewai_event_bus.emit(complete_event, complete_event)

    # Verify tracker was called
    assert tracker.on_node_start.call_count >= 1
    assert tracker.on_node_end.call_count >= 1


def test_agent_execution_error_reports_error(tracker, instrumentor, mock_agent, mock_task):
    """Test that AgentExecutionErrorEvent reports error to tracker."""
    from crewai.events import crewai_event_bus, AgentExecutionStartedEvent, AgentExecutionErrorEvent

    # Start event
    start_event = MagicMock(spec=AgentExecutionStartedEvent)
    start_event.agent = mock_agent
    start_event.task = mock_task
    crewai_event_bus.emit(start_event, start_event)

    # Error event
    error_event = MagicMock(spec=AgentExecutionErrorEvent)
    error_event.error = "Something went wrong"
    crewai_event_bus.emit(error_event, error_event)

    # Verify tracker.on_node_error was called
    assert tracker.on_node_error.call_count >= 1


def test_llm_call_creates_sub_span(tracker, instrumentor, mock_agent, mock_task):
    """Test that LLM calls create sub-spans under the agent context."""
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        LLMCallStartedEvent,
        LLMCallCompletedEvent,
    )

    # Start agent
    start_event = MagicMock(spec=AgentExecutionStartedEvent)
    start_event.agent = mock_agent
    start_event.task = mock_task
    crewai_event_bus.emit(start_event, start_event)

    # LLM call
    llm_start = MagicMock(spec=LLMCallStartedEvent)
    llm_start.model = "gpt-4"
    llm_start.messages = [{"role": "user", "content": "Hello"}]
    crewai_event_bus.emit(llm_start, llm_start)

    llm_complete = MagicMock(spec=LLMCallCompletedEvent)
    llm_complete.response = "Hi there!"
    crewai_event_bus.emit(llm_complete, llm_complete)

    # Complete agent
    complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
    complete_event.output = "Done"
    crewai_event_bus.emit(complete_event, complete_event)

    # Verify tracker was called and span has sub_spans
    assert tracker.on_node_start.call_count >= 1

    # Find the agent span
    for call in tracker.on_node_start.call_args_list:
        span = call[0][0]
        if span.metadata.get("is_workflow"):
            sub_spans = span.metadata.get("sub_spans", [])
            assert len(sub_spans) >= 1
            # Check for LLM sub-span
            llm_spans = [s for s in sub_spans if "llm:" in s.node_name]
            assert len(llm_spans) >= 1
            break


def test_tool_call_creates_sub_span(tracker, instrumentor, mock_agent, mock_task, mock_tool):
    """Test that tool calls create sub-spans under the agent context."""
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
    )

    # Start agent
    start_event = MagicMock(spec=AgentExecutionStartedEvent)
    start_event.agent = mock_agent
    start_event.task = mock_task
    crewai_event_bus.emit(start_event, start_event)

    # Tool call
    tool_start = MagicMock(spec=ToolUsageStartedEvent)
    tool_start.tool_name = "search_tool"
    tool_start.tool_input = {"query": "AI trends"}
    crewai_event_bus.emit(tool_start, tool_start)

    tool_finish = MagicMock(spec=ToolUsageFinishedEvent)
    tool_finish.output = "Search results..."
    crewai_event_bus.emit(tool_finish, tool_finish)

    # Complete agent
    complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
    complete_event.output = "Done"
    crewai_event_bus.emit(complete_event, complete_event)

    # Verify tracker was called and span has sub_spans
    assert tracker.on_node_start.call_count >= 1

    # Find the agent span
    for call in tracker.on_node_start.call_args_list:
        span = call[0][0]
        if span.metadata.get("is_workflow"):
            sub_spans = span.metadata.get("sub_spans", [])
            assert len(sub_spans) >= 1
            # Check for tool sub-span
            tool_spans = [s for s in sub_spans if s.node_name == "search_tool"]
            assert len(tool_spans) >= 1
            # Verify tool metadata
            assert tool_spans[0].metadata.get("is_tool_call") is True
            break


def test_multiple_agents_create_separate_spans(tracker, instrumentor):
    """Test that multiple agents create separate workflow spans."""
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
    )

    # First agent
    agent1 = MagicMock()
    agent1.role = "Researcher"

    task1 = MagicMock()
    task1.description = "Research task"

    start1 = MagicMock(spec=AgentExecutionStartedEvent)
    start1.agent = agent1
    start1.task = task1
    crewai_event_bus.emit(start1, start1)

    complete1 = MagicMock(spec=AgentExecutionCompletedEvent)
    complete1.output = "Research done"
    crewai_event_bus.emit(complete1, complete1)

    # Second agent
    agent2 = MagicMock()
    agent2.role = "Writer"

    task2 = MagicMock()
    task2.description = "Write task"

    start2 = MagicMock(spec=AgentExecutionStartedEvent)
    start2.agent = agent2
    start2.task = task2
    crewai_event_bus.emit(start2, start2)

    complete2 = MagicMock(spec=AgentExecutionCompletedEvent)
    complete2.output = "Writing done"
    crewai_event_bus.emit(complete2, complete2)

    # Verify two workflow spans were created
    workflow_spans = []
    for call in tracker.on_node_start.call_args_list:
        span = call[0][0]
        if span.metadata.get("is_workflow"):
            workflow_spans.append(span)

    assert len(workflow_spans) >= 2

    # Verify distinct agent names
    agent_names = {s.node_name for s in workflow_spans}
    assert "agent:Researcher" in agent_names
    assert "agent:Writer" in agent_names


def test_span_framework_is_crewai(tracker, instrumentor, mock_agent, mock_task):
    """Test that all spans have framework set to 'crewai'."""
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
    )

    # Execute agent
    start_event = MagicMock(spec=AgentExecutionStartedEvent)
    start_event.agent = mock_agent
    start_event.task = mock_task
    crewai_event_bus.emit(start_event, start_event)

    complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
    complete_event.output = "Done"
    crewai_event_bus.emit(complete_event, complete_event)

    # Verify framework
    for call in tracker.on_node_start.call_args_list:
        span = call[0][0]
        assert span.framework == "crewai"


def test_span_timing_populated(tracker, instrumentor, mock_agent, mock_task):
    """Test that spans have start_time and end_time populated."""
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
    )

    # Execute agent
    start_event = MagicMock(spec=AgentExecutionStartedEvent)
    start_event.agent = mock_agent
    start_event.task = mock_task
    crewai_event_bus.emit(start_event, start_event)

    complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
    complete_event.output = "Done"
    crewai_event_bus.emit(complete_event, complete_event)

    # Verify timing
    for call in tracker.on_node_start.call_args_list:
        span = call[0][0]
        if span.metadata.get("is_workflow"):
            assert span.start_time is not None
            assert span.end_time is not None
            assert span.start_time <= span.end_time


def test_tool_span_preserves_input_output(tracker, instrumentor, mock_agent, mock_task):
    """Test that tool spans preserve the original tool input/output."""
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
    )

    # Start agent
    start_event = MagicMock(spec=AgentExecutionStartedEvent)
    start_event.agent = mock_agent
    start_event.task = mock_task
    crewai_event_bus.emit(start_event, start_event)

    # Tool call with specific input
    tool_start = MagicMock(spec=ToolUsageStartedEvent)
    tool_start.tool_name = "search_tool"
    tool_start.tool_input = {"query": "AI news"}
    crewai_event_bus.emit(tool_start, tool_start)

    # Tool finish with specific output
    tool_finish = MagicMock(spec=ToolUsageFinishedEvent)
    tool_finish.output = {"results": ["result1", "result2"]}
    crewai_event_bus.emit(tool_finish, tool_finish)

    # Complete agent
    complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
    complete_event.output = "Done"
    crewai_event_bus.emit(complete_event, complete_event)

    # Find the tool span and verify input/output
    for call in tracker.on_node_start.call_args_list:
        span = call[0][0]
        if span.metadata.get("is_workflow"):
            sub_spans = span.metadata.get("sub_spans", [])
            tool_spans = [s for s in sub_spans if s.node_name == "search_tool"]
            if tool_spans:
                tool_span = tool_spans[0]
                # Verify input contains the tool input
                assert tool_span.inputs == {"query": "AI news"}
                # Verify output is the tool result
                assert tool_span.outputs == {"results": ["result1", "result2"]}
                break


# -----------------------------
# Network Interceptor Classification Tests
# -----------------------------


def test_network_interceptor_classifies_llm_call(tracker, mock_agent, mock_task):
    """Test that network interceptor's is_llm classification is authoritative."""
    from gradient_adk.runtime.crewai.crewai_instrumentor import CrewAIInstrumentor
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        LLMCallStartedEvent,
        LLMCallCompletedEvent,
    )

    # Create mock interceptor that returns LLM classification
    mock_interceptor = MagicMock()
    mock_interceptor.snapshot_token.return_value = 0
    
    # Mock captured request to inference URL
    mock_captured = MagicMock()
    mock_captured.url = "https://inference.do-ai.run/v1/chat/completions"
    mock_captured.request_payload = {"messages": [{"role": "user", "content": "test"}]}
    mock_captured.response_payload = {"choices": [{"message": {"content": "response"}}]}
    mock_interceptor.get_captured_requests_since.return_value = [mock_captured]

    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_network_interceptor",
        return_value=mock_interceptor,
    ):
        with patch(
            "gradient_adk.runtime.crewai.crewai_instrumentor.get_request_captured_list",
            return_value=[mock_captured],  # Simulate captured request
        ):
            inst = CrewAIInstrumentor()
            inst.install(tracker)

            try:
                # Start agent
                start_event = MagicMock(spec=AgentExecutionStartedEvent)
                start_event.agent = mock_agent
                start_event.task = mock_task
                crewai_event_bus.emit(start_event, start_event)

                # LLM call
                llm_start = MagicMock(spec=LLMCallStartedEvent)
                llm_start.model = "gpt-4"
                llm_start.messages = []
                crewai_event_bus.emit(llm_start, llm_start)

                llm_complete = MagicMock(spec=LLMCallCompletedEvent)
                llm_complete.response = "Response"
                crewai_event_bus.emit(llm_complete, llm_complete)

                # Complete agent
                complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
                complete_event.output = "Done"
                crewai_event_bus.emit(complete_event, complete_event)

                # Find and verify LLM span
                for call in tracker.on_node_start.call_args_list:
                    span = call[0][0]
                    if span.metadata.get("is_workflow"):
                        sub_spans = span.metadata.get("sub_spans", [])
                        llm_spans = [s for s in sub_spans if s.metadata.get("is_llm_call")]
                        assert len(llm_spans) >= 1
                        # Verify API payloads were stored
                        assert llm_spans[0].metadata.get("llm_request_payload") is not None
                        break
            finally:
                inst.uninstall()


def test_explicit_tool_stays_tool_even_with_llm_calls(tracker, mock_agent, mock_task):
    """Test that explicit tool spans stay as tools even if LLM HTTP calls happen during execution.
    
    In CrewAI, tool execution may internally trigger LLM calls (e.g., to format
    input/output). These should not reclassify the tool span as an LLM span.
    """
    from gradient_adk.runtime.crewai.crewai_instrumentor import CrewAIInstrumentor
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
    )

    # Mock captured request to inference URL (LLM call during tool execution)
    mock_captured = MagicMock()
    mock_captured.url = "https://inference.do-ai.run/v1/chat/completions"
    mock_captured.request_payload = {"messages": [{"role": "user", "content": "process tool result"}]}
    mock_captured.response_payload = {"choices": [{"message": {"content": "processed"}}]}

    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_network_interceptor",
        return_value=MagicMock(),
    ):
        with patch(
            "gradient_adk.runtime.crewai.crewai_instrumentor.get_request_captured_list",
            return_value=[mock_captured],
        ):
            inst = CrewAIInstrumentor()
            inst.install(tracker)

            try:
                # Start agent
                start_event = MagicMock(spec=AgentExecutionStartedEvent)
                start_event.agent = mock_agent
                start_event.task = mock_task
                crewai_event_bus.emit(start_event, start_event)

                # Tool execution (even though LLM HTTP calls happen, it stays as tool)
                tool_start = MagicMock(spec=ToolUsageStartedEvent)
                tool_start.tool_name = "my_tool"
                tool_start.tool_input = {"arg": "value"}
                crewai_event_bus.emit(tool_start, tool_start)

                tool_finish = MagicMock(spec=ToolUsageFinishedEvent)
                tool_finish.output = "tool result"
                crewai_event_bus.emit(tool_finish, tool_finish)

                # Complete agent
                complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
                complete_event.output = "Done"
                crewai_event_bus.emit(complete_event, complete_event)

                # Find and verify tool span stayed as tool
                for call in tracker.on_node_start.call_args_list:
                    span = call[0][0]
                    if span.metadata.get("is_workflow"):
                        sub_spans = span.metadata.get("sub_spans", [])
                        tool_spans = [s for s in sub_spans if s.node_name == "my_tool"]
                        assert len(tool_spans) >= 1
                        # Should still be classified as tool (explicit tool is preserved)
                        assert tool_spans[0].metadata.get("is_tool_call") is True
                        # Should NOT be reclassified as LLM
                        assert tool_spans[0].metadata.get("is_llm_call") is None
                        # Tool should preserve its original input/output
                        assert tool_spans[0].inputs == {"arg": "value"}
                        assert tool_spans[0].outputs == "tool result"
                        break
            finally:
                inst.uninstall()


def test_tool_calling_external_api_stays_tool(tracker, mock_agent, mock_task):
    """Test that tools calling external APIs (not our inference/kbaas) stay as tools."""
    from gradient_adk.runtime.crewai.crewai_instrumentor import CrewAIInstrumentor
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
    )

    # Mock captured request to external API (Serper)
    mock_captured = MagicMock()
    mock_captured.url = "https://google.serper.dev/search"  # External API
    mock_captured.request_payload = {"q": "AI news"}
    mock_captured.response_payload = {"organic": [{"title": "result"}]}

    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_network_interceptor",
        return_value=MagicMock(),
    ):
        with patch(
            "gradient_adk.runtime.crewai.crewai_instrumentor.get_request_captured_list",
            return_value=[mock_captured],
        ):
            inst = CrewAIInstrumentor()
            inst.install(tracker)

            try:
                # Start agent
                start_event = MagicMock(spec=AgentExecutionStartedEvent)
                start_event.agent = mock_agent
                start_event.task = mock_task
                crewai_event_bus.emit(start_event, start_event)

                # Tool calling external API
                tool_start = MagicMock(spec=ToolUsageStartedEvent)
                tool_start.tool_name = "serper_search"
                tool_start.tool_input = {"search_query": "AI news"}
                crewai_event_bus.emit(tool_start, tool_start)

                tool_finish = MagicMock(spec=ToolUsageFinishedEvent)
                tool_finish.output = {"organic": [{"title": "result"}]}
                crewai_event_bus.emit(tool_finish, tool_finish)

                # Complete agent
                complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
                complete_event.output = "Done"
                crewai_event_bus.emit(complete_event, complete_event)

                # Find and verify tool span still has is_tool_call
                for call in tracker.on_node_start.call_args_list:
                    span = call[0][0]
                    if span.metadata.get("is_workflow"):
                        sub_spans = span.metadata.get("sub_spans", [])
                        tool_spans = [s for s in sub_spans if s.node_name == "serper_search"]
                        assert len(tool_spans) >= 1
                        # Should still be classified as tool (external API doesn't override)
                        assert tool_spans[0].metadata.get("is_tool_call") is True
                        # Should NOT be classified as LLM or retriever
                        assert tool_spans[0].metadata.get("is_llm_call") is None
                        assert tool_spans[0].metadata.get("is_retriever_call") is None
                        # Tool should preserve its original input/output
                        assert tool_spans[0].inputs == {"search_query": "AI news"}
                        break
            finally:
                inst.uninstall()


def test_llm_span_uses_api_payloads(tracker, mock_agent, mock_task):
    """Test that LLM spans use captured API payloads for input/output."""
    from gradient_adk.runtime.crewai.crewai_instrumentor import CrewAIInstrumentor
    from crewai.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        LLMCallStartedEvent,
        LLMCallCompletedEvent,
    )

    # Mock captured request with specific payloads
    mock_captured = MagicMock()
    mock_captured.url = "https://inference.do-ai.run/v1/chat/completions"
    mock_captured.request_payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello API"}],
    }
    mock_captured.response_payload = {
        "choices": [{"message": {"content": "Hello from API"}}],
    }

    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_network_interceptor",
        return_value=MagicMock(),
    ):
        with patch(
            "gradient_adk.runtime.crewai.crewai_instrumentor.get_request_captured_list",
            return_value=[mock_captured],
        ):
            inst = CrewAIInstrumentor()
            inst.install(tracker)

            try:
                # Start agent
                start_event = MagicMock(spec=AgentExecutionStartedEvent)
                start_event.agent = mock_agent
                start_event.task = mock_task
                crewai_event_bus.emit(start_event, start_event)

                # LLM call
                llm_start = MagicMock(spec=LLMCallStartedEvent)
                llm_start.model = "gpt-4"
                llm_start.messages = [{"role": "user", "content": "Original messages"}]
                crewai_event_bus.emit(llm_start, llm_start)

                llm_complete = MagicMock(spec=LLMCallCompletedEvent)
                llm_complete.response = "Original response"
                crewai_event_bus.emit(llm_complete, llm_complete)

                # Complete agent
                complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
                complete_event.output = "Done"
                crewai_event_bus.emit(complete_event, complete_event)

                # Find LLM span and verify it uses API payloads
                for call in tracker.on_node_start.call_args_list:
                    span = call[0][0]
                    if span.metadata.get("is_workflow"):
                        sub_spans = span.metadata.get("sub_spans", [])
                        llm_spans = [s for s in sub_spans if s.metadata.get("is_llm_call")]
                        if llm_spans:
                            llm_span = llm_spans[0]
                            # Input should be API request payload
                            assert llm_span.inputs == mock_captured.request_payload
                            # Output should be API response payload
                            assert llm_span.outputs == mock_captured.response_payload
                            break
            finally:
                inst.uninstall()
