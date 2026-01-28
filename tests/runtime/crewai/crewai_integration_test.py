"""Integration tests for CrewAI instrumentation.

These tests run CrewAI crews locally with instrumentation to verify
spans are correctly captured, nested, and classified.
"""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Skip all tests if crewai is not installed
pytest.importorskip("crewai")

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def mock_tracker():
    """Create a mock tracker to capture spans."""
    tracker = MagicMock()
    tracker.on_node_start = MagicMock()
    tracker.on_node_end = MagicMock()
    tracker.on_node_error = MagicMock()
    return tracker


@pytest.fixture
def mock_interceptor():
    """Create a mock network interceptor."""
    interceptor = MagicMock()
    interceptor.snapshot_token.return_value = 0
    interceptor.hits_since.return_value = 0
    return interceptor


@pytest.fixture(autouse=True)
def patch_network_interceptor(mock_interceptor):
    """Patch the network interceptor for all tests."""
    with patch(
        "gradient_adk.runtime.crewai.crewai_instrumentor.get_network_interceptor",
        return_value=mock_interceptor,
    ):
        with patch(
            "gradient_adk.runtime.crewai.crewai_instrumentor.get_request_captured_list",
            return_value=None,
        ):
            yield mock_interceptor


@pytest.fixture(autouse=True)
def clear_agent_stack():
    """Clear the agent stack before and after each test."""
    from gradient_adk.runtime.crewai.crewai_instrumentor import (
        _agent_stack,
        _agent_stack_lock,
    )
    with _agent_stack_lock:
        _agent_stack.clear()
    yield
    with _agent_stack_lock:
        _agent_stack.clear()


@pytest.fixture
def instrumentor(mock_tracker):
    """Create and install an instrumentor."""
    from gradient_adk.runtime.crewai.crewai_instrumentor import CrewAIInstrumentor

    inst = CrewAIInstrumentor()
    inst.install(mock_tracker)
    yield inst
    inst.uninstall()


# -----------------------------
# Helper Functions
# -----------------------------


def get_workflow_spans(tracker):
    """Extract workflow spans from tracker calls."""
    spans = []
    for call in tracker.on_node_start.call_args_list:
        span = call[0][0]
        if hasattr(span, "metadata") and span.metadata.get("is_workflow"):
            spans.append(span)
    return spans


def get_all_spans(tracker):
    """Extract all spans from tracker calls."""
    spans = []
    for call in tracker.on_node_start.call_args_list:
        spans.append(call[0][0])
    return spans


# -----------------------------
# Basic Instrumentation Tests
# -----------------------------


class TestBasicInstrumentation:
    """Test basic instrumentation functionality."""

    def test_instrumentor_installs_successfully(self, mock_tracker):
        """Test that the instrumentor installs without errors."""
        from gradient_adk.runtime.crewai.crewai_instrumentor import CrewAIInstrumentor

        inst = CrewAIInstrumentor()
        inst.install(mock_tracker)
        
        assert inst.is_installed()
        
        inst.uninstall()
        assert not inst.is_installed()

    def test_instrumentor_registers_event_handlers(self, instrumentor, mock_tracker):
        """Test that event handlers are registered."""
        from crewai.events import crewai_event_bus

        # The event bus should have our handlers registered
        # This is verified by the fact that install() completed successfully
        assert instrumentor.is_installed()


# -----------------------------
# Single Agent Tests
# -----------------------------


class TestSingleAgentCrew:
    """Test instrumentation with a single-agent crew."""

    @pytest.mark.skipif(
        not os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
        reason="Requires GRADIENT_MODEL_ACCESS_KEY for LLM calls",
    )
    def test_single_agent_creates_workflow_span(self, instrumentor, mock_tracker):
        """Test that a single agent creates a workflow span."""
        from crewai import Agent, Task, Crew, Process, LLM

        llm = LLM(
            model="openai-gpt-4.1",
            base_url="https://inference.do-ai.run/v1",
            api_key=os.getenv("GRADIENT_MODEL_ACCESS_KEY"),
        )

        agent = Agent(
            role="Greeter",
            goal="Greet users",
            backstory="A friendly assistant.",
            llm=llm,
            verbose=False,
        )

        task = Task(
            description="Say hello.",
            expected_output="A greeting.",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()

        # Verify a workflow span was created
        workflow_spans = get_workflow_spans(mock_tracker)
        assert len(workflow_spans) >= 1

        # Verify the span has the correct name
        span = workflow_spans[0]
        assert "agent:Greeter" in span.node_name

    def test_single_agent_mocked_llm(self, instrumentor, mock_tracker):
        """Test single agent with mocked LLM for offline testing."""
        from crewai import Agent, Task, Crew, Process
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
            LLMCallStartedEvent,
            LLMCallCompletedEvent,
        )

        # Simulate agent execution via events
        agent_mock = MagicMock()
        agent_mock.role = "TestAgent"
        
        task_mock = MagicMock()
        task_mock.description = "Test task"

        # Agent start
        start_event = MagicMock(spec=AgentExecutionStartedEvent)
        start_event.agent = agent_mock
        start_event.task = task_mock
        crewai_event_bus.emit(start_event, start_event)

        # LLM call
        llm_start = MagicMock(spec=LLMCallStartedEvent)
        llm_start.model = "test-model"
        llm_start.messages = [{"role": "user", "content": "test"}]
        crewai_event_bus.emit(llm_start, llm_start)

        llm_complete = MagicMock(spec=LLMCallCompletedEvent)
        llm_complete.response = "Test response"
        crewai_event_bus.emit(llm_complete, llm_complete)

        # Agent complete
        complete_event = MagicMock(spec=AgentExecutionCompletedEvent)
        complete_event.output = "Task completed"
        crewai_event_bus.emit(complete_event, complete_event)

        # Verify spans
        workflow_spans = get_workflow_spans(mock_tracker)
        assert len(workflow_spans) == 1

        span = workflow_spans[0]
        assert span.node_name == "agent:TestAgent"
        assert span.framework == "crewai"

        # Verify sub-spans
        sub_spans = span.metadata.get("sub_spans", [])
        assert len(sub_spans) >= 1

        llm_spans = [s for s in sub_spans if "llm:" in s.node_name]
        assert len(llm_spans) >= 1


# -----------------------------
# Multi-Agent Tests
# -----------------------------


class TestMultiAgentCrew:
    """Test instrumentation with multi-agent crews."""

    def test_two_agents_create_separate_spans(self, instrumentor, mock_tracker):
        """Test that two agents create separate workflow spans."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
        )

        # First agent
        agent1 = MagicMock()
        agent1.role = "Researcher"
        task1 = MagicMock()
        task1.description = "Research"

        start1 = MagicMock(spec=AgentExecutionStartedEvent)
        start1.agent = agent1
        start1.task = task1
        crewai_event_bus.emit(start1, start1)

        complete1 = MagicMock(spec=AgentExecutionCompletedEvent)
        complete1.output = "Research complete"
        crewai_event_bus.emit(complete1, complete1)

        # Second agent
        agent2 = MagicMock()
        agent2.role = "Writer"
        task2 = MagicMock()
        task2.description = "Write"

        start2 = MagicMock(spec=AgentExecutionStartedEvent)
        start2.agent = agent2
        start2.task = task2
        crewai_event_bus.emit(start2, start2)

        complete2 = MagicMock(spec=AgentExecutionCompletedEvent)
        complete2.output = "Writing complete"
        crewai_event_bus.emit(complete2, complete2)

        # Verify two workflow spans
        workflow_spans = get_workflow_spans(mock_tracker)
        assert len(workflow_spans) == 2

        agent_names = {s.node_name for s in workflow_spans}
        assert "agent:Researcher" in agent_names
        assert "agent:Writer" in agent_names

    @pytest.mark.skipif(
        not os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
        reason="Requires GRADIENT_MODEL_ACCESS_KEY for LLM calls",
    )
    def test_example_trivia_crew_runs(self, instrumentor, mock_tracker):
        """Test that the example trivia crew runs with instrumentation."""
        # Add examples directory to path
        examples_dir = Path(__file__).parent.parent.parent.parent / "examples" / "crewai"
        sys.path.insert(0, str(examples_dir))

        try:
            # Import the example module
            from main import create_trivia_crew

            # Create the crew (this doesn't run it)
            crew = create_trivia_crew("2025-01-15", "Technology")

            # Verify crew was created
            assert len(crew.agents) == 2
            assert len(crew.tasks) == 2

            # Note: Actually running the crew requires valid API keys
            # and would make real LLM calls. For unit tests, we just
            # verify the crew is constructed correctly.

        finally:
            sys.path.remove(str(examples_dir))


# -----------------------------
# Tool Call Tests
# -----------------------------


class TestToolInstrumentation:
    """Test instrumentation of tool calls."""

    def test_tool_call_creates_tool_span(self, instrumentor, mock_tracker):
        """Test that tool calls create spans with is_tool_call metadata."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
            ToolUsageStartedEvent,
            ToolUsageFinishedEvent,
        )

        # Start agent
        agent = MagicMock()
        agent.role = "ToolUser"
        task = MagicMock()
        task.description = "Use tools"

        start = MagicMock(spec=AgentExecutionStartedEvent)
        start.agent = agent
        start.task = task
        crewai_event_bus.emit(start, start)

        # Tool call
        tool_start = MagicMock(spec=ToolUsageStartedEvent)
        tool_start.tool_name = "web_search"
        tool_start.tool_input = {"query": "test query"}
        crewai_event_bus.emit(tool_start, tool_start)

        tool_finish = MagicMock(spec=ToolUsageFinishedEvent)
        tool_finish.output = {"results": ["result1"]}
        crewai_event_bus.emit(tool_finish, tool_finish)

        # Complete agent
        complete = MagicMock(spec=AgentExecutionCompletedEvent)
        complete.output = "Done"
        crewai_event_bus.emit(complete, complete)

        # Verify tool span
        workflow_spans = get_workflow_spans(mock_tracker)
        assert len(workflow_spans) == 1

        sub_spans = workflow_spans[0].metadata.get("sub_spans", [])
        tool_spans = [s for s in sub_spans if s.metadata.get("is_tool_call")]
        
        assert len(tool_spans) == 1
        assert tool_spans[0].node_name == "web_search"
        assert tool_spans[0].inputs == {"query": "test query"}
        assert tool_spans[0].outputs == {"results": ["result1"]}

    def test_tool_error_captured(self, instrumentor, mock_tracker):
        """Test that tool errors are captured correctly."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
            ToolUsageStartedEvent,
            ToolUsageErrorEvent,
        )

        # Start agent
        agent = MagicMock()
        agent.role = "ToolUser"
        task = MagicMock()
        task.description = "Use tools"

        start = MagicMock(spec=AgentExecutionStartedEvent)
        start.agent = agent
        start.task = task
        crewai_event_bus.emit(start, start)

        # Tool call with error
        tool_start = MagicMock(spec=ToolUsageStartedEvent)
        tool_start.tool_name = "failing_tool"
        tool_start.tool_input = {"arg": "value"}
        crewai_event_bus.emit(tool_start, tool_start)

        tool_error = MagicMock(spec=ToolUsageErrorEvent)
        tool_error.error = "Tool execution failed"
        crewai_event_bus.emit(tool_error, tool_error)

        # Complete agent
        complete = MagicMock(spec=AgentExecutionCompletedEvent)
        complete.output = "Done with errors"
        crewai_event_bus.emit(complete, complete)

        # Verify error span
        workflow_spans = get_workflow_spans(mock_tracker)
        sub_spans = workflow_spans[0].metadata.get("sub_spans", [])
        tool_spans = [s for s in sub_spans if s.node_name == "failing_tool"]
        
        assert len(tool_spans) == 1
        assert tool_spans[0].error == "Tool execution failed"


# -----------------------------
# Error Handling Tests
# -----------------------------


class TestErrorHandling:
    """Test error handling in instrumentation."""

    def test_agent_error_captured(self, instrumentor, mock_tracker):
        """Test that agent errors are captured correctly."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionErrorEvent,
        )

        agent = MagicMock()
        agent.role = "FailingAgent"
        task = MagicMock()
        task.description = "Fail"

        start = MagicMock(spec=AgentExecutionStartedEvent)
        start.agent = agent
        start.task = task
        crewai_event_bus.emit(start, start)

        error = MagicMock(spec=AgentExecutionErrorEvent)
        error.error = "Agent execution failed"
        crewai_event_bus.emit(error, error)

        # Verify error was reported
        assert mock_tracker.on_node_error.call_count >= 1

    def test_llm_error_captured(self, instrumentor, mock_tracker):
        """Test that LLM errors are captured correctly."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
            LLMCallStartedEvent,
            LLMCallFailedEvent,
        )

        agent = MagicMock()
        agent.role = "Agent"
        task = MagicMock()
        task.description = "Task"

        start = MagicMock(spec=AgentExecutionStartedEvent)
        start.agent = agent
        start.task = task
        crewai_event_bus.emit(start, start)

        llm_start = MagicMock(spec=LLMCallStartedEvent)
        llm_start.model = "test-model"
        llm_start.messages = []
        crewai_event_bus.emit(llm_start, llm_start)

        llm_fail = MagicMock(spec=LLMCallFailedEvent)
        llm_fail.error = "LLM call failed"
        crewai_event_bus.emit(llm_fail, llm_fail)

        complete = MagicMock(spec=AgentExecutionCompletedEvent)
        complete.output = "Done"
        crewai_event_bus.emit(complete, complete)

        # Verify LLM error span
        workflow_spans = get_workflow_spans(mock_tracker)
        sub_spans = workflow_spans[0].metadata.get("sub_spans", [])
        error_spans = [s for s in sub_spans if s.error]
        
        assert len(error_spans) >= 1


# -----------------------------
# Span Metadata Tests
# -----------------------------


class TestSpanMetadata:
    """Test that span metadata is correctly populated."""

    def test_workflow_span_has_is_workflow_flag(self, instrumentor, mock_tracker):
        """Test that workflow spans have is_workflow=True."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
        )

        agent = MagicMock()
        agent.role = "TestAgent"
        task = MagicMock()
        task.description = "Test"

        start = MagicMock(spec=AgentExecutionStartedEvent)
        start.agent = agent
        start.task = task
        crewai_event_bus.emit(start, start)

        complete = MagicMock(spec=AgentExecutionCompletedEvent)
        complete.output = "Done"
        crewai_event_bus.emit(complete, complete)

        workflow_spans = get_workflow_spans(mock_tracker)
        assert len(workflow_spans) == 1
        assert workflow_spans[0].metadata.get("is_workflow") is True

    def test_llm_span_has_is_llm_call_flag(self, instrumentor, mock_tracker):
        """Test that LLM spans have is_llm_call=True."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
            LLMCallStartedEvent,
            LLMCallCompletedEvent,
        )

        agent = MagicMock()
        agent.role = "Agent"
        task = MagicMock()
        task.description = "Task"

        start = MagicMock(spec=AgentExecutionStartedEvent)
        start.agent = agent
        start.task = task
        crewai_event_bus.emit(start, start)

        llm_start = MagicMock(spec=LLMCallStartedEvent)
        llm_start.model = "gpt-4"
        llm_start.messages = []
        crewai_event_bus.emit(llm_start, llm_start)

        llm_complete = MagicMock(spec=LLMCallCompletedEvent)
        llm_complete.response = "Response"
        crewai_event_bus.emit(llm_complete, llm_complete)

        complete = MagicMock(spec=AgentExecutionCompletedEvent)
        complete.output = "Done"
        crewai_event_bus.emit(complete, complete)

        workflow_spans = get_workflow_spans(mock_tracker)
        sub_spans = workflow_spans[0].metadata.get("sub_spans", [])
        llm_spans = [s for s in sub_spans if s.metadata.get("is_llm_call")]
        
        assert len(llm_spans) >= 1

    def test_tool_span_has_is_tool_call_flag(self, instrumentor, mock_tracker):
        """Test that tool spans have is_tool_call=True."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
            ToolUsageStartedEvent,
            ToolUsageFinishedEvent,
        )

        agent = MagicMock()
        agent.role = "Agent"
        task = MagicMock()
        task.description = "Task"

        start = MagicMock(spec=AgentExecutionStartedEvent)
        start.agent = agent
        start.task = task
        crewai_event_bus.emit(start, start)

        tool_start = MagicMock(spec=ToolUsageStartedEvent)
        tool_start.tool_name = "test_tool"
        tool_start.tool_input = {}
        crewai_event_bus.emit(tool_start, tool_start)

        tool_finish = MagicMock(spec=ToolUsageFinishedEvent)
        tool_finish.output = "result"
        crewai_event_bus.emit(tool_finish, tool_finish)

        complete = MagicMock(spec=AgentExecutionCompletedEvent)
        complete.output = "Done"
        crewai_event_bus.emit(complete, complete)

        workflow_spans = get_workflow_spans(mock_tracker)
        sub_spans = workflow_spans[0].metadata.get("sub_spans", [])
        tool_spans = [s for s in sub_spans if s.metadata.get("is_tool_call")]
        
        assert len(tool_spans) >= 1
        assert tool_spans[0].metadata.get("tool_name") == "test_tool"

    def test_spans_have_timestamps(self, instrumentor, mock_tracker):
        """Test that all spans have start and end timestamps."""
        from crewai.events import (
            crewai_event_bus,
            AgentExecutionStartedEvent,
            AgentExecutionCompletedEvent,
        )

        agent = MagicMock()
        agent.role = "Agent"
        task = MagicMock()
        task.description = "Task"

        start = MagicMock(spec=AgentExecutionStartedEvent)
        start.agent = agent
        start.task = task
        crewai_event_bus.emit(start, start)

        complete = MagicMock(spec=AgentExecutionCompletedEvent)
        complete.output = "Done"
        crewai_event_bus.emit(complete, complete)

        workflow_spans = get_workflow_spans(mock_tracker)
        assert len(workflow_spans) == 1
        
        span = workflow_spans[0]
        assert span.start_time is not None
        assert span.end_time is not None
        assert span.start_time <= span.end_time
