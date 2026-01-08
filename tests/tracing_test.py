"""Tests for the tracing module's programmatic span functions."""

from unittest.mock import MagicMock, patch

import pytest

from gradient_adk.tracing import add_llm_span, add_tool_span, add_agent_span


class TestAddLlmSpan:
    """Tests for add_llm_span function."""

    def test_no_tracker_does_not_raise(self):
        """Should not raise when no tracker is available."""
        with patch("gradient_adk.tracing.get_tracker", return_value=None):
            add_llm_span(
                name="test_llm",
                input={"prompt": "Hello"},
                output={"response": "Hi"},
            )

    def test_basic_span_creation(self):
        """Should create and submit LLM span with basic fields."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_llm_span(
                name="test_llm",
                input={"prompt": "Hello"},
                output={"response": "Hi"},
            )

        assert mock_tracker.on_node_start.call_count == 1
        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.node_name == "test_llm"
        assert span.inputs == {"prompt": "Hello"}
        assert span.metadata["is_llm_call"] is True

        assert mock_tracker.on_node_end.call_count == 1

    def test_with_model(self):
        """Should include model name in metadata."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_llm_span(
                name="test_llm",
                input={"prompt": "Hello"},
                output={"response": "Hi"},
                model="gpt-4",
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.metadata["model_name"] == "gpt-4"

    def test_with_tokens(self):
        """Should include token counts in metadata."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_llm_span(
                name="test_llm",
                input={"prompt": "Hello"},
                output={"response": "Hi"},
                num_input_tokens=10,
                num_output_tokens=5,
                total_tokens=15,
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        usage = span.metadata["llm_response_payload"]["usage"]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_with_temperature(self):
        """Should include temperature in request payload."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_llm_span(
                name="test_llm",
                input={"prompt": "Hello"},
                output={"response": "Hi"},
                temperature=0.7,
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.metadata["llm_request_payload"]["temperature"] == 0.7

    def test_with_all_optional_fields(self):
        """Should handle all optional fields."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_llm_span(
                name="test_llm",
                input={"prompt": "Hello"},
                output={"response": "Hi"},
                model="gpt-4",
                tools=[{"type": "function"}],
                num_input_tokens=10,
                num_output_tokens=5,
                total_tokens=15,
                temperature=0.7,
                time_to_first_token_ns=100000000,
                duration_ns=500000000,
                metadata={"custom": "data"},
                tags=["production", "test"],
                status_code=200,
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        meta = span.metadata

        assert meta["is_llm_call"] is True
        assert meta["model_name"] == "gpt-4"
        assert meta["time_to_first_token_ns"] == 100000000
        assert meta["duration_ns"] == 500000000
        assert meta["custom_metadata"] == {"custom": "data"}
        assert meta["tags"] == ["production", "test"]
        assert meta["status_code"] == 200


class TestAddToolSpan:
    """Tests for add_tool_span function."""

    def test_no_tracker_does_not_raise(self):
        """Should not raise when no tracker is available."""
        with patch("gradient_adk.tracing.get_tracker", return_value=None):
            add_tool_span(
                name="calculator",
                input={"x": 5, "y": 3},
                output={"result": 8},
            )

    def test_basic_span_creation(self):
        """Should create and submit tool span with basic fields."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_tool_span(
                name="calculator",
                input={"x": 5, "y": 3},
                output={"result": 8},
            )

        assert mock_tracker.on_node_start.call_count == 1
        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.node_name == "calculator"
        assert span.inputs == {"x": 5, "y": 3}
        assert span.metadata["is_tool_call"] is True

        assert mock_tracker.on_node_end.call_count == 1

    def test_with_tool_call_id(self):
        """Should include tool_call_id in metadata."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_tool_span(
                name="calculator",
                input={"x": 5, "y": 3},
                output={"result": 8},
                tool_call_id="call_abc123",
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.metadata["tool_call_id"] == "call_abc123"

    def test_with_all_optional_fields(self):
        """Should handle all optional fields."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_tool_span(
                name="calculator",
                input={"x": 5, "y": 3},
                output={"result": 8},
                tool_call_id="call_abc123",
                duration_ns=1000000,
                metadata={"function": "add"},
                tags=["math"],
                status_code=200,
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        meta = span.metadata

        assert meta["is_tool_call"] is True
        assert meta["tool_call_id"] == "call_abc123"
        assert meta["duration_ns"] == 1000000
        assert meta["custom_metadata"] == {"function": "add"}
        assert meta["tags"] == ["math"]
        assert meta["status_code"] == 200


class TestAddAgentSpan:
    """Tests for add_agent_span function."""

    def test_no_tracker_does_not_raise(self):
        """Should not raise when no tracker is available."""
        with patch("gradient_adk.tracing.get_tracker", return_value=None):
            add_agent_span(
                name="research_agent",
                input={"query": "What is AI?"},
                output={"answer": "AI is..."},
            )

    def test_basic_span_creation(self):
        """Should create and submit agent span with basic fields."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_agent_span(
                name="research_agent",
                input={"query": "What is AI?"},
                output={"answer": "AI is..."},
            )

        assert mock_tracker.on_node_start.call_count == 1
        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.node_name == "research_agent"
        assert span.inputs == {"query": "What is AI?"}
        assert span.metadata["is_agent_call"] is True

        assert mock_tracker.on_node_end.call_count == 1

    def test_with_all_optional_fields(self):
        """Should handle all optional fields."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_agent_span(
                name="research_agent",
                input={"query": "What is AI?"},
                output={"answer": "AI is..."},
                duration_ns=5000000000,
                metadata={"model": "gpt-4"},
                tags=["research"],
                status_code=200,
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        meta = span.metadata

        assert meta["is_agent_call"] is True
        assert meta["duration_ns"] == 5000000000
        assert meta["custom_metadata"] == {"model": "gpt-4"}
        assert meta["tags"] == ["research"]
        assert meta["status_code"] == 200


class TestInputOutputSerialization:
    """Tests for input/output serialization handling."""

    def test_complex_input_is_frozen(self):
        """Should properly serialize complex input objects."""
        mock_tracker = MagicMock()

        class CustomObject:
            def __repr__(self):
                return "CustomObject()"

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_llm_span(
                name="test",
                input=CustomObject(),
                output="response",
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.inputs == "CustomObject()"

    def test_nested_dict_input(self):
        """Should handle nested dictionaries."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_llm_span(
                name="test",
                input={
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ],
                    "config": {"temperature": 0.7},
                },
                output={"response": "result"},
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.inputs["messages"][0]["role"] == "user"
        assert span.inputs["config"]["temperature"] == 0.7

    def test_list_input(self):
        """Should handle list inputs."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_tool_span(
                name="batch_process",
                input=[1, 2, 3, 4, 5],
                output=[2, 4, 6, 8, 10],
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.inputs == [1, 2, 3, 4, 5]

    def test_none_input_output(self):
        """Should handle None inputs and outputs."""
        mock_tracker = MagicMock()

        with patch("gradient_adk.tracing.get_tracker", return_value=mock_tracker):
            add_agent_span(
                name="test",
                input=None,
                output=None,
            )

        span = mock_tracker.on_node_start.call_args[0][0]
        assert span.inputs is None