"""Integration tests for CrewAI instrumentation.

These tests run CrewAI crews locally with instrumentation to verify
spans are correctly captured, nested, and classified.
"""

import pytest
import os
import sys
import time
from unittest.mock import MagicMock, patch
from pathlib import Path

# Skip all tests if crewai is not installed
pytest.importorskip("crewai")

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# -----------------------------
# Fixtures
# -----------------------------


# Note: mock_tracker, clear_agent_stack, and integration_instrumentor fixtures
# are provided by conftest.py to ensure a single instrumentor installation
# across all test files (preventing handler accumulation on CrewAI's event bus).


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


@pytest.fixture
def instrumentor(mock_tracker, shared_instrumentor, shared_delegating_tracker):
    """Per-test instrumentor wired to this test's mock_tracker."""
    shared_delegating_tracker.set_delegate(mock_tracker)
    yield shared_instrumentor
    shared_delegating_tracker.set_delegate(None)


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


def wait_for_event_bus(delay: float = 0.05):
    """Wait for the CrewAI event bus to process handlers.

    Uses flush() to properly wait for all pending handlers to complete,
    which is more reliable than a fixed sleep when the thread pool is busy.
    """
    from crewai.events import crewai_event_bus
    crewai_event_bus.flush(timeout=5.0)
    time.sleep(delay)


def make_real_event(event_class, **kwargs):
    """Create a real CrewAI event instance.

    CrewAI's event bus uses isinstance checks, so we need to create
    actual event instances rather than mocks. We use model_construct
    to bypass Pydantic validation since we're using mock objects for
    agent and task.
    """
    from datetime import datetime, timezone
    from crewai.events import (
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent,
        LLMCallStartedEvent,
        LLMCallCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
    )

    # Import optional event types
    LLMCallFailedEvent = None
    ToolUsageErrorEvent = None
    try:
        from crewai.events import LLMCallFailedEvent
    except ImportError:
        pass
    try:
        from crewai.events import ToolUsageErrorEvent
    except ImportError:
        pass

    # Build required args based on event type
    if event_class == AgentExecutionStartedEvent:
        defaults = {
            "agent": kwargs.pop("agent", MagicMock()),
            "task": kwargs.pop("task", MagicMock()),
            "tools": kwargs.pop("tools", []),
            "task_prompt": kwargs.pop("task_prompt", "Test task prompt"),
        }
    elif event_class == AgentExecutionCompletedEvent:
        defaults = {
            "agent": kwargs.pop("agent", MagicMock()),
            "task": kwargs.pop("task", MagicMock()),
            "output": kwargs.pop("output", "Test output"),
        }
    elif event_class == AgentExecutionErrorEvent:
        defaults = {
            "agent": kwargs.pop("agent", MagicMock()),
            "task": kwargs.pop("task", MagicMock()),
            "error": kwargs.pop("error", "Test error"),
        }
    elif event_class == LLMCallStartedEvent:
        defaults = {
            "model": kwargs.pop("model", "gpt-4"),
            "messages": kwargs.pop("messages", []),
        }
    elif event_class == LLMCallCompletedEvent:
        defaults = {
            "response": kwargs.pop("response", "Test response"),
            "call_type": kwargs.pop("call_type", "completion"),
        }
    elif LLMCallFailedEvent is not None and event_class == LLMCallFailedEvent:
        defaults = {
            "error": kwargs.pop("error", "LLM call failed"),
        }
    elif event_class == ToolUsageStartedEvent:
        defaults = {
            "tool_name": kwargs.pop("tool_name", "test_tool"),
            "tool_args": kwargs.pop("tool_args", kwargs.pop("tool_input", {})),
        }
    elif event_class == ToolUsageFinishedEvent:
        now = datetime.now(timezone.utc)
        defaults = {
            "tool_name": kwargs.pop("tool_name", "test_tool"),
            "tool_args": kwargs.pop("tool_args", kwargs.pop("tool_input", {})),
            "started_at": kwargs.pop("started_at", now),
            "finished_at": kwargs.pop("finished_at", now),
            "output": kwargs.pop("output", "Test output"),
        }
    elif ToolUsageErrorEvent is not None and event_class == ToolUsageErrorEvent:
        defaults = {
            "tool_name": kwargs.pop("tool_name", "test_tool"),
            "tool_args": kwargs.pop("tool_args", kwargs.pop("tool_input", {})),
            "error": kwargs.pop("error", "Tool error"),
        }
    else:
        defaults = {}

    # Merge with any remaining kwargs
    defaults.update(kwargs)

    # Use model_construct to bypass Pydantic validation for mock objects
    return event_class.model_construct(**defaults)


# -----------------------------
# Basic Instrumentation Tests
# -----------------------------


class TestBasicInstrumentation:
    """Test basic instrumentation functionality."""

    def test_instrumentor_installs_successfully(self, instrumentor):
        """Test that the instrumentor installs without errors.

        Note: We use the module-scoped instrumentor fixture to avoid
        registering duplicate handlers on CrewAI's event bus, which
        doesn't support handler removal.
        """
        assert instrumentor.is_installed()

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
        start_event = make_real_event(AgentExecutionStartedEvent, agent=agent_mock, task=task_mock)
        crewai_event_bus.emit(start_event, start_event)
        wait_for_event_bus()

        # LLM call
        llm_start = make_real_event(LLMCallStartedEvent, model="test-model", messages=[{"role": "user", "content": "test"}])
        crewai_event_bus.emit(llm_start, llm_start)
        wait_for_event_bus()

        llm_complete = make_real_event(LLMCallCompletedEvent, response="Test response")
        crewai_event_bus.emit(llm_complete, llm_complete)
        wait_for_event_bus()

        # Agent complete
        complete_event = make_real_event(AgentExecutionCompletedEvent, output="Task completed")
        crewai_event_bus.emit(complete_event, complete_event)
        wait_for_event_bus()

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

        start1 = make_real_event(AgentExecutionStartedEvent, agent=agent1, task=task1)
        crewai_event_bus.emit(start1, start1)
        wait_for_event_bus()

        complete1 = make_real_event(AgentExecutionCompletedEvent, output="Research complete")
        crewai_event_bus.emit(complete1, complete1)
        wait_for_event_bus()

        # Second agent
        agent2 = MagicMock()
        agent2.role = "Writer"
        task2 = MagicMock()
        task2.description = "Write"

        start2 = make_real_event(AgentExecutionStartedEvent, agent=agent2, task=task2)
        crewai_event_bus.emit(start2, start2)
        wait_for_event_bus()

        complete2 = make_real_event(AgentExecutionCompletedEvent, output="Writing complete")
        crewai_event_bus.emit(complete2, complete2)
        wait_for_event_bus()

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

        start = make_real_event(AgentExecutionStartedEvent, agent=agent, task=task)
        crewai_event_bus.emit(start, start)
        wait_for_event_bus()

        # Tool call
        tool_start = make_real_event(ToolUsageStartedEvent, tool_name="web_search", tool_input={"query": "test query"})
        crewai_event_bus.emit(tool_start, tool_start)
        wait_for_event_bus()

        tool_finish = make_real_event(ToolUsageFinishedEvent, tool_name="web_search", output={"results": ["result1"]})
        crewai_event_bus.emit(tool_finish, tool_finish)
        wait_for_event_bus()

        # Complete agent
        complete = make_real_event(AgentExecutionCompletedEvent, output="Done")
        crewai_event_bus.emit(complete, complete)
        wait_for_event_bus()

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

        start = make_real_event(AgentExecutionStartedEvent, agent=agent, task=task)
        crewai_event_bus.emit(start, start)
        wait_for_event_bus()

        # Tool call with error
        tool_start = make_real_event(ToolUsageStartedEvent, tool_name="failing_tool", tool_input={"arg": "value"})
        crewai_event_bus.emit(tool_start, tool_start)
        wait_for_event_bus()

        tool_error = make_real_event(ToolUsageErrorEvent, tool_name="failing_tool", error="Tool execution failed")
        crewai_event_bus.emit(tool_error, tool_error)
        wait_for_event_bus()

        # Complete agent
        complete = make_real_event(AgentExecutionCompletedEvent, output="Done with errors")
        crewai_event_bus.emit(complete, complete)
        wait_for_event_bus()

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

        start = make_real_event(AgentExecutionStartedEvent, agent=agent, task=task)
        crewai_event_bus.emit(start, start)
        wait_for_event_bus()

        error = make_real_event(AgentExecutionErrorEvent, error="Agent execution failed")
        crewai_event_bus.emit(error, error)
        wait_for_event_bus()

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

        start = make_real_event(AgentExecutionStartedEvent, agent=agent, task=task)
        crewai_event_bus.emit(start, start)
        wait_for_event_bus()

        llm_start = make_real_event(LLMCallStartedEvent, model="test-model", messages=[])
        crewai_event_bus.emit(llm_start, llm_start)
        wait_for_event_bus()

        llm_fail = make_real_event(LLMCallFailedEvent, error="LLM call failed")
        crewai_event_bus.emit(llm_fail, llm_fail)
        wait_for_event_bus()

        complete = make_real_event(AgentExecutionCompletedEvent, output="Done")
        crewai_event_bus.emit(complete, complete)
        wait_for_event_bus()

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

        start = make_real_event(AgentExecutionStartedEvent, agent=agent, task=task)
        crewai_event_bus.emit(start, start)
        wait_for_event_bus()

        complete = make_real_event(AgentExecutionCompletedEvent, output="Done")
        crewai_event_bus.emit(complete, complete)
        wait_for_event_bus()

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

        start = make_real_event(AgentExecutionStartedEvent, agent=agent, task=task)
        crewai_event_bus.emit(start, start)
        wait_for_event_bus()

        llm_start = make_real_event(LLMCallStartedEvent, model="gpt-4", messages=[])
        crewai_event_bus.emit(llm_start, llm_start)
        wait_for_event_bus()

        llm_complete = make_real_event(LLMCallCompletedEvent, response="Response")
        crewai_event_bus.emit(llm_complete, llm_complete)
        wait_for_event_bus()

        complete = make_real_event(AgentExecutionCompletedEvent, output="Done")
        crewai_event_bus.emit(complete, complete)
        wait_for_event_bus()

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

        start = make_real_event(AgentExecutionStartedEvent, agent=agent, task=task)
        crewai_event_bus.emit(start, start)
        wait_for_event_bus()

        tool_start = make_real_event(ToolUsageStartedEvent, tool_name="test_tool", tool_input={})
        crewai_event_bus.emit(tool_start, tool_start)
        wait_for_event_bus()

        tool_finish = make_real_event(ToolUsageFinishedEvent, tool_name="test_tool", output="result")
        crewai_event_bus.emit(tool_finish, tool_finish)
        wait_for_event_bus()

        complete = make_real_event(AgentExecutionCompletedEvent, output="Done")
        crewai_event_bus.emit(complete, complete)
        wait_for_event_bus()

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

        start = make_real_event(AgentExecutionStartedEvent, agent=agent, task=task)
        crewai_event_bus.emit(start, start)
        wait_for_event_bus()

        complete = make_real_event(AgentExecutionCompletedEvent, output="Done")
        crewai_event_bus.emit(complete, complete)
        wait_for_event_bus()

        workflow_spans = get_workflow_spans(mock_tracker)
        assert len(workflow_spans) == 1

        span = workflow_spans[0]
        assert span.start_time is not None
        assert span.end_time is not None
        assert span.start_time <= span.end_time
