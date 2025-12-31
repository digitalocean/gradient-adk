"""Tests for gradient_adk.tracing module."""

import asyncio
import dataclasses
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from gradient_adk.tracing import (
    SpanType,
    _freeze,
    _snapshot_args_kwargs,
    _snapshot_output,
    _ensure_meta,
    _create_span,
    _add_span,
    trace_llm,
    trace_retriever,
    trace_tool,
    add_llm_span,
    add_retriever_span,
    add_tool_span,
)
from gradient_adk.runtime.interfaces import NodeExecution


# ---------------------------
# Test Doubles
# ---------------------------


class TrackerDouble:
    """Mock tracker for testing span operations."""

    def __init__(self):
        self.node_starts = []
        self.node_ends = []
        self.node_errors = []

    def on_node_start(self, node: NodeExecution):
        self.node_starts.append(node)

    def on_node_end(self, node: NodeExecution, outputs):
        self.node_ends.append((node, outputs))

    def on_node_error(self, node: NodeExecution, error: BaseException):
        self.node_errors.append((node, error))


# ---------------------------
# Helper Function Tests
# ---------------------------


class TestFreeze:
    """Tests for the _freeze helper function."""

    def test_freeze_primitives(self):
        """Test that primitives are returned as-is."""
        assert _freeze(None) is None
        assert _freeze("hello") == "hello"
        assert _freeze(42) == 42
        assert _freeze(3.14) == 3.14
        assert _freeze(True) is True
        assert _freeze(False) is False

    def test_freeze_dict(self):
        """Test dict serialization."""
        result = _freeze({"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}

    def test_freeze_nested_dict(self):
        """Test nested dict serialization."""
        result = _freeze({"outer": {"inner": {"deep": 123}}})
        assert result == {"outer": {"inner": {"deep": 123}}}

    def test_freeze_list(self):
        """Test list serialization."""
        result = _freeze([1, 2, 3])
        assert result == [1, 2, 3]

    def test_freeze_tuple(self):
        """Test tuple serialization (converts to list)."""
        result = _freeze((1, 2, 3))
        assert result == [1, 2, 3]

    def test_freeze_set(self):
        """Test set serialization (converts to list)."""
        result = _freeze({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_freeze_max_depth(self):
        """Test that max depth is respected."""
        deep = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        result = _freeze(deep, max_depth=2)
        # max_depth=2 means: depth 2 (a), depth 1 (b), depth 0 (c -> max-depth)
        assert result == {"a": {"b": {"c": "<max-depth>"}}}

    def test_freeze_max_items(self):
        """Test that max items is respected for dicts."""
        large_dict = {f"key{i}": i for i in range(10)}
        result = _freeze(large_dict, max_items=3)
        # Should have 3 items + truncated marker
        assert len(result) == 4
        assert result.get("<truncated>") is True

    def test_freeze_max_items_list(self):
        """Test that max items is respected for lists."""
        large_list = list(range(10))
        result = _freeze(large_list, max_items=3)
        assert len(result) == 4
        assert result[-1] == "<truncated>"

    def test_freeze_fallback_repr(self):
        """Test that unknown types fall back to repr()."""

        class CustomClass:
            def __repr__(self):
                return "CustomClass()"

        result = _freeze(CustomClass())
        assert result == "CustomClass()"

    def test_freeze_dataclass(self):
        """Test dataclass serialization."""

        @dataclasses.dataclass
        class Person:
            name: str
            age: int

        result = _freeze(Person(name="Alice", age=30))
        assert result == {"name": "Alice", "age": 30}

    def test_freeze_pydantic_model(self):
        """Test pydantic model serialization."""
        try:
            from pydantic import BaseModel

            class Item(BaseModel):
                name: str
                value: int

            result = _freeze(Item(name="test", value=42))
            assert result == {"name": "test", "value": 42}
        except ImportError:
            pytest.skip("pydantic not installed")


class TestSnapshotArgsKwargs:
    """Tests for the _snapshot_args_kwargs helper function."""

    def test_single_arg_no_kwargs(self):
        """Test with single arg and no kwargs - returns just that arg."""
        result = _snapshot_args_kwargs(("hello",), {})
        assert result == "hello"

    def test_kwargs_only(self):
        """Test with no args and kwargs only - returns just the kwargs."""
        result = _snapshot_args_kwargs((), {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_multiple_args(self):
        """Test with multiple args - returns list."""
        result = _snapshot_args_kwargs(("a", "b", "c"), {})
        assert result == ["a", "b", "c"]

    def test_args_and_kwargs(self):
        """Test with both args and kwargs - returns dict with both."""
        result = _snapshot_args_kwargs(("a",), {"x": 1})
        assert result == {"args": ["a"], "kwargs": {"x": 1}}

    def test_empty_args_and_kwargs(self):
        """Test with empty args and kwargs."""
        result = _snapshot_args_kwargs((), {})
        assert result == []


class TestSnapshotOutput:
    """Tests for the _snapshot_output helper function."""

    def test_snapshot_output_primitive(self):
        """Test output snapshotting with primitives."""
        assert _snapshot_output("result") == "result"
        assert _snapshot_output(42) == 42

    def test_snapshot_output_dict(self):
        """Test output snapshotting with dict."""
        result = _snapshot_output({"key": "value"})
        assert result == {"key": "value"}

    def test_snapshot_output_list(self):
        """Test output snapshotting with list."""
        result = _snapshot_output([1, 2, 3])
        assert result == [1, 2, 3]


class TestEnsureMeta:
    """Tests for the _ensure_meta helper function."""

    def test_ensure_meta_creates_dict(self):
        """Test that _ensure_meta creates a metadata dict if none exists."""
        node = NodeExecution(
            node_id="123",
            node_name="test",
            framework="custom",
            start_time=datetime.now(timezone.utc),
        )
        assert node.metadata is None

        meta = _ensure_meta(node)
        assert isinstance(meta, dict)
        assert node.metadata == meta

    def test_ensure_meta_returns_existing(self):
        """Test that _ensure_meta returns existing metadata."""
        node = NodeExecution(
            node_id="123",
            node_name="test",
            framework="custom",
            start_time=datetime.now(timezone.utc),
            metadata={"existing": "data"},
        )

        meta = _ensure_meta(node)
        assert meta == {"existing": "data"}


class TestCreateSpan:
    """Tests for the _create_span helper function."""

    def test_create_span_basic(self):
        """Test basic span creation."""
        span = _create_span("my_span", {"input": "data"})

        assert span.node_name == "my_span"
        assert span.framework == "custom"
        assert span.inputs == {"input": "data"}
        assert span.start_time is not None
        assert span.node_id is not None

    def test_create_span_uuid_uniqueness(self):
        """Test that each span gets a unique ID."""
        span1 = _create_span("span1", {})
        span2 = _create_span("span2", {})

        assert span1.node_id != span2.node_id


# ---------------------------
# Decorator Tests
# ---------------------------


class TestTraceLlmDecorator:
    """Tests for the @trace_llm decorator."""

    def test_trace_llm_async_function(self):
        """Test @trace_llm with async function."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_llm("my_llm_call")
                async def call_llm(prompt: str) -> str:
                    return f"Response to: {prompt}"

                result = asyncio.run(call_llm("Hello"))

        assert result == "Response to: Hello"
        assert len(tracker.node_starts) == 1
        assert len(tracker.node_ends) == 1

        started_node = tracker.node_starts[0]
        assert started_node.node_name == "my_llm_call"
        assert started_node.metadata.get("is_llm_call") is True

    def test_trace_llm_sync_function(self):
        """Test @trace_llm with sync function."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_llm("sync_llm")
                def call_llm(prompt: str) -> str:
                    return f"Response: {prompt}"

                result = call_llm("Test")

        assert result == "Response: Test"
        assert len(tracker.node_starts) == 1
        assert tracker.node_starts[0].metadata.get("is_llm_call") is True

    def test_trace_llm_uses_function_name_when_no_name_provided(self):
        """Test that function name is used when no custom name is provided."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_llm()
                async def my_custom_llm_function(prompt: str) -> str:
                    return "response"

                asyncio.run(my_custom_llm_function("test"))

        assert tracker.node_starts[0].node_name == "my_custom_llm_function"

    def test_trace_llm_async_generator(self):
        """Test @trace_llm with async generator (streaming)."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_llm("streaming_llm")
                async def stream_llm(prompt: str):
                    yield "Hello"
                    yield " "
                    yield "World"

                async def consume():
                    chunks = []
                    async for chunk in stream_llm("test"):
                        chunks.append(chunk)
                    return chunks

                result = asyncio.run(consume())

        assert result == ["Hello", " ", "World"]
        assert len(tracker.node_starts) == 1
        assert len(tracker.node_ends) == 1

        # Check collected content
        ended_node, outputs = tracker.node_ends[0]
        assert outputs == {"content": "Hello World"}

    def test_trace_llm_error_handling(self):
        """Test that errors are tracked correctly."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0

                @trace_llm("error_llm")
                async def failing_llm(prompt: str) -> str:
                    raise ValueError("LLM error")

                with pytest.raises(ValueError, match="LLM error"):
                    asyncio.run(failing_llm("test"))

        assert len(tracker.node_starts) == 1
        assert len(tracker.node_errors) == 1
        assert isinstance(tracker.node_errors[0][1], ValueError)

    def test_trace_llm_no_tracker(self):
        """Test that function works when no tracker is available."""
        with patch("gradient_adk.tracing.get_tracker", return_value=None):

            @trace_llm("no_tracker")
            async def call_llm(prompt: str) -> str:
                return "response"

            result = asyncio.run(call_llm("test"))

        assert result == "response"


class TestTraceRetrieverDecorator:
    """Tests for the @trace_retriever decorator."""

    def test_trace_retriever_async_function(self):
        """Test @trace_retriever with async function."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_retriever("vector_search")
                async def search(query: str) -> list:
                    return [{"id": 1, "text": "result"}]

                result = asyncio.run(search("test query"))

        assert result == [{"id": 1, "text": "result"}]
        assert len(tracker.node_starts) == 1
        assert tracker.node_starts[0].metadata.get("is_retriever_call") is True

    def test_trace_retriever_sync_function(self):
        """Test @trace_retriever with sync function."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_retriever("db_search")
                def search(query: str) -> list:
                    return [{"id": 1}]

                result = search("test")

        assert result == [{"id": 1}]
        assert tracker.node_starts[0].metadata.get("is_retriever_call") is True


class TestTraceToolDecorator:
    """Tests for the @trace_tool decorator."""

    def test_trace_tool_async_function(self):
        """Test @trace_tool with async function."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_tool("calculator")
                async def add(x: int, y: int) -> int:
                    return x + y

                result = asyncio.run(add(5, 3))

        assert result == 8
        assert len(tracker.node_starts) == 1
        assert tracker.node_starts[0].metadata.get("is_tool_call") is True

    def test_trace_tool_sync_function(self):
        """Test @trace_tool with sync function."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_tool("multiply")
                def multiply(x: int, y: int) -> int:
                    return x * y

                result = multiply(4, 5)

        assert result == 20
        assert tracker.node_starts[0].metadata.get("is_tool_call") is True


class TestDecoratorInputOutputCapture:
    """Tests for input/output capture in decorators."""

    def test_captures_inputs_correctly(self):
        """Test that function inputs are captured correctly."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_tool("test_tool")
                def process(data: dict) -> dict:
                    return {"processed": True}

                process({"key": "value"})

        started_node = tracker.node_starts[0]
        assert started_node.inputs == {"key": "value"}

    def test_captures_outputs_correctly(self):
        """Test that function outputs are captured correctly."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            with patch("gradient_adk.tracing.get_network_interceptor") as mock_interceptor:
                mock_interceptor.return_value.snapshot_token.return_value = 0
                mock_interceptor.return_value.hits_since.return_value = 0

                @trace_tool("test_tool")
                def process(data: dict) -> dict:
                    return {"result": 42}

                process({"input": "data"})

        ended_node, outputs = tracker.node_ends[0]
        assert outputs == {"result": 42}


# ---------------------------
# Function-based Span API Tests
# ---------------------------


class TestAddLlmSpan:
    """Tests for the add_llm_span function."""

    def test_add_llm_span_basic(self):
        """Test basic add_llm_span usage."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            add_llm_span(
                "my_llm",
                inputs={"prompt": "Hello"},
                output="World",
            )

        assert len(tracker.node_starts) == 1
        assert len(tracker.node_ends) == 1

        node = tracker.node_starts[0]
        assert node.node_name == "my_llm"
        assert node.inputs == {"prompt": "Hello"}
        assert node.metadata.get("is_llm_call") is True

        ended_node, outputs = tracker.node_ends[0]
        assert outputs == "World"

    def test_add_llm_span_with_model_name(self):
        """Test add_llm_span with model_name parameter."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            add_llm_span(
                "my_llm",
                inputs={"prompt": "test"},
                output="response",
                model_name="gpt-4",
            )

        node = tracker.node_starts[0]
        assert node.metadata.get("model_name") == "gpt-4"

    def test_add_llm_span_with_ttft_ms(self):
        """Test add_llm_span with ttft_ms parameter."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            add_llm_span(
                "my_llm",
                inputs={"prompt": "test"},
                output="response",
                ttft_ms=150.5,
            )

        node = tracker.node_starts[0]
        assert node.metadata.get("ttft_ms") == 150.5

    def test_add_llm_span_with_extra_metadata(self):
        """Test add_llm_span with extra metadata via kwargs."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            add_llm_span(
                "my_llm",
                inputs={"prompt": "test"},
                output="response",
                model_name="gpt-4",
                ttft_ms=100.0,
                temperature=0.7,
                max_tokens=1000,
            )

        node = tracker.node_starts[0]
        assert node.metadata.get("model_name") == "gpt-4"
        assert node.metadata.get("ttft_ms") == 100.0
        assert node.metadata.get("temperature") == 0.7
        assert node.metadata.get("max_tokens") == 1000

    def test_add_llm_span_no_tracker(self):
        """Test that add_llm_span silently skips when no tracker."""
        with patch("gradient_adk.tracing.get_tracker", return_value=None):
            # Should not raise
            add_llm_span("my_llm", inputs={}, output="test")


class TestAddRetrieverSpan:
    """Tests for the add_retriever_span function."""

    def test_add_retriever_span_basic(self):
        """Test basic add_retriever_span usage."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            add_retriever_span(
                "vector_search",
                inputs={"query": "test"},
                output=[{"id": 1}, {"id": 2}],
            )

        assert len(tracker.node_starts) == 1
        node = tracker.node_starts[0]
        assert node.node_name == "vector_search"
        assert node.metadata.get("is_retriever_call") is True

    def test_add_retriever_span_with_extra_metadata(self):
        """Test add_retriever_span with extra metadata."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            add_retriever_span(
                "vector_search",
                inputs={"query": "test"},
                output=[{"id": 1}],
                num_results=1,
                similarity_threshold=0.8,
            )

        node = tracker.node_starts[0]
        assert node.metadata.get("num_results") == 1
        assert node.metadata.get("similarity_threshold") == 0.8

    def test_add_retriever_span_no_tracker(self):
        """Test that add_retriever_span silently skips when no tracker."""
        with patch("gradient_adk.tracing.get_tracker", return_value=None):
            add_retriever_span("search", inputs={}, output=[])


class TestAddToolSpan:
    """Tests for the add_tool_span function."""

    def test_add_tool_span_basic(self):
        """Test basic add_tool_span usage."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            add_tool_span(
                "calculator",
                inputs={"x": 5, "y": 3},
                output=8,
            )

        assert len(tracker.node_starts) == 1
        node = tracker.node_starts[0]
        assert node.node_name == "calculator"
        assert node.metadata.get("is_tool_call") is True

        ended_node, outputs = tracker.node_ends[0]
        assert outputs == 8

    def test_add_tool_span_with_extra_metadata(self):
        """Test add_tool_span with extra metadata."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            add_tool_span(
                "calculator",
                inputs={"x": 5, "y": 3},
                output=8,
                execution_time_ms=12.5,
                tool_version="1.0.0",
            )

        node = tracker.node_starts[0]
        assert node.metadata.get("execution_time_ms") == 12.5
        assert node.metadata.get("tool_version") == "1.0.0"

    def test_add_tool_span_no_tracker(self):
        """Test that add_tool_span silently skips when no tracker."""
        with patch("gradient_adk.tracing.get_tracker", return_value=None):
            add_tool_span("tool", inputs={}, output=None)


class TestInternalAddSpan:
    """Tests for the internal _add_span function."""

    def test_add_span_llm_type(self):
        """Test _add_span with LLM span type."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            _add_span("test", {"input": 1}, "output", SpanType.LLM)

        assert tracker.node_starts[0].metadata.get("is_llm_call") is True

    def test_add_span_tool_type(self):
        """Test _add_span with TOOL span type."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            _add_span("test", {"input": 1}, "output", SpanType.TOOL)

        assert tracker.node_starts[0].metadata.get("is_tool_call") is True

    def test_add_span_retriever_type(self):
        """Test _add_span with RETRIEVER span type."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            _add_span("test", {"input": 1}, "output", SpanType.RETRIEVER)

        assert tracker.node_starts[0].metadata.get("is_retriever_call") is True

    def test_add_span_with_extra_metadata(self):
        """Test _add_span with extra metadata."""
        tracker = TrackerDouble()

        with patch("gradient_adk.tracing.get_tracker", return_value=tracker):
            _add_span(
                "test",
                {"input": 1},
                "output",
                SpanType.LLM,
                extra_metadata={"custom_key": "custom_value"},
            )

        node = tracker.node_starts[0]
        assert node.metadata.get("custom_key") == "custom_value"
        assert node.metadata.get("is_llm_call") is True


class TestSpanType:
    """Tests for the SpanType enum."""

    def test_span_type_values(self):
        """Test SpanType enum values."""
        assert SpanType.LLM.value == "llm"
        assert SpanType.TOOL.value == "tool"
        assert SpanType.RETRIEVER.value == "retriever"