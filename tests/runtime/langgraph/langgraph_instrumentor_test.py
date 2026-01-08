import pytest
import os
from unittest.mock import MagicMock, patch
from langgraph.graph import StateGraph

from gradient_adk.runtime.langgraph.langgraph_instrumentor import (
    LangGraphInstrumentor,
    WRAPPED_FLAG,
    _transform_kbaas_response,
    _get_captured_payloads_with_type,
)

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def tracker():
    t = MagicMock()
    t.on_node_start = MagicMock()
    t.on_node_end = MagicMock()
    t.on_node_error = MagicMock()
    return t


@pytest.fixture
def interceptor():
    intr = MagicMock()
    intr.snapshot_token.return_value = 42
    intr.hits_since.return_value = 0
    return intr


@pytest.fixture(autouse=True)
def patch_interceptor(interceptor):
    with patch(
        "gradient_adk.runtime.langgraph.langgraph_instrumentor.get_network_interceptor",
        return_value=interceptor,
    ):
        yield interceptor


@pytest.fixture
def instrumentor(tracker):
    inst = LangGraphInstrumentor()
    inst.install(tracker)
    return inst


def make_graph():
    # LangGraph graphs pass a single "state" dict to nodes
    return StateGraph(dict)


def _compile_singleton_graph(graph: StateGraph, node_name: str):
    """Make this node both the entry and finish point, then compile."""
    graph.set_entry_point(node_name)
    graph.set_finish_point(node_name)
    return graph.compile()


def test_install_monkeypatches_stategraph(tracker):
    inst = LangGraphInstrumentor()
    old_add = StateGraph.add_node
    old_compile = StateGraph.compile

    inst.install(tracker)

    # We only patch add_node now
    assert StateGraph.add_node is not old_add
    assert StateGraph.compile is old_compile
    assert inst._installed

    # Idempotency: calling install again shouldn't change anything further
    inst.install(tracker)
    assert StateGraph.add_node is not old_add
    assert StateGraph.compile is old_compile


def test_sync_node_wrapped_and_tracked(tracker, instrumentor, interceptor):
    def node(state: dict):
        # return a mapping -> instrumentor should use it as canonical output
        return {"x": state["x"] + 1}

    g = make_graph()
    g.add_node("inc", node)

    # Optional: verify underlying callable has WRAPPED_FLAG (if discoverable)
    spec = g.nodes["inc"]
    func_obj = getattr(getattr(spec, "runnable", spec), "func", None)
    if func_obj is not None:
        assert getattr(func_obj, WRAPPED_FLAG, False) is True

    app = _compile_singleton_graph(g, "inc")
    result = app.invoke({"x": 1})
    assert result == {"x": 2}

    tracker.on_node_start.assert_called_once()
    tracker.on_node_end.assert_called_once()
    tracker.on_node_error.assert_not_called()
    interceptor.hits_since.assert_called_once_with(42)


@pytest.mark.asyncio
async def test_async_node_wrapped_and_tracked(tracker, instrumentor):
    async def node(state: dict):
        return {"y": state["y"] * 2}

    g = make_graph()
    g.add_node("double", node)
    app = _compile_singleton_graph(g, "double")

    out = await app.ainvoke({"y": 5})
    assert out == {"y": 10}

    tracker.on_node_start.assert_called_once()
    tracker.on_node_end.assert_called_once()
    tracker.on_node_error.assert_not_called()


def test_runnable_invoke_wrapped_and_tracked(tracker, instrumentor):
    class R:
        def invoke(self, state: dict):
            return {"ok": True, "v": state.get("v", 0) + 1}

    g = make_graph()
    g.add_node("runnable", R())
    app = _compile_singleton_graph(g, "runnable")

    res = app.invoke({"v": 10})
    assert res == {"ok": True, "v": 11}
    tracker.on_node_start.assert_called_once()
    tracker.on_node_end.assert_called_once()


@pytest.mark.asyncio
async def test_runnable_ainvoke_wrapped_and_tracked(tracker, instrumentor):
    class R:
        async def ainvoke(self, state: dict):
            return {"ok": True, "w": state.get("w", 0) + 3}

    g = make_graph()
    g.add_node("runnable", R())
    app = _compile_singleton_graph(g, "runnable")

    res = await app.ainvoke({"w": 2})
    assert res == {"ok": True, "w": 5}
    tracker.on_node_start.assert_called_once()
    tracker.on_node_end.assert_called_once()


def test_error_calls_on_node_error(tracker, instrumentor):
    def bad(state: dict):
        raise ValueError("boom")

    g = make_graph()
    g.add_node("bad", bad)
    app = _compile_singleton_graph(g, "bad")

    with pytest.raises(ValueError):
        app.invoke({})

    tracker.on_node_error.assert_called_once()
    tracker.on_node_end.assert_not_called()


def test_llm_hit_sets_metadata(tracker, instrumentor, interceptor):
    interceptor.hits_since.return_value = 5  # simulate an LLM call

    def node(state: dict):
        return {"r": 1}

    g = make_graph()
    g.add_node("llm", node)
    app = _compile_singleton_graph(g, "llm")

    app.invoke({})
    # NodeExecution record is arg0 to on_node_end
    exec_rec = tracker.on_node_end.call_args[0][0]
    assert exec_rec.metadata.get("is_llm_call") is True


def test_already_wrapped_function_is_not_rewrapped(tracker, instrumentor):
    def f(state: dict):
        return {"ok": 1}

    # Mark as already wrapped before add_node
    setattr(f, WRAPPED_FLAG, True)

    g = make_graph()
    g.add_node("noop", f)
    spec = g.nodes["noop"]

    # If we can see the base func, it should be exactly 'f'
    func_obj = getattr(getattr(spec, "runnable", spec), "func", None)
    if func_obj is not None:
        assert func_obj is f

    app = _compile_singleton_graph(g, "noop")
    assert app.invoke({}) == {"ok": 1}


# -----------------------------
# KBaaS Response Transformation Tests
# -----------------------------


def test_transform_kbaas_response_converts_text_content_to_page_content():
    """Test that text_content is converted to page_content in results."""
    response = {
        "results": [
            {
                "metadata": {"source": "doc1.pdf", "page": 1},
                "text_content": "This is the document content.",
            },
            {
                "metadata": {"source": "doc2.pdf", "page": 2},
                "text_content": "Another document chunk.",
            },
        ],
        "total_results": 2,
    }

    transformed = _transform_kbaas_response(response)

    # Should return a list (array) directly, not a dict
    assert isinstance(transformed, list)
    assert len(transformed) == 2

    # Check that text_content was converted to page_content
    assert "text_content" not in transformed[0]
    assert "text_content" not in transformed[1]
    assert transformed[0]["page_content"] == "This is the document content."
    assert transformed[1]["page_content"] == "Another document chunk."

    # Check that metadata is preserved
    assert transformed[0]["metadata"]["source"] == "doc1.pdf"
    assert transformed[1]["metadata"]["source"] == "doc2.pdf"


def test_transform_kbaas_response_handles_empty_results():
    """Test that empty results list is handled correctly."""
    response = {"results": [], "total_results": 0}

    transformed = _transform_kbaas_response(response)

    # Should return an empty list
    assert isinstance(transformed, list)
    assert transformed == []


def test_transform_kbaas_response_preserves_items_without_text_content():
    """Test that items without text_content are preserved unchanged."""
    response = {
        "results": [
            {"metadata": {"source": "doc1.pdf"}, "text_content": "Has text content."},
            {
                "metadata": {"source": "doc2.pdf"},
                "page_content": "Already has page_content.",
            },
            {
                "metadata": {"source": "doc3.pdf"}
                # No text_content or page_content
            },
        ],
        "total_results": 3,
    }

    transformed = _transform_kbaas_response(response)

    # Should return a list
    assert isinstance(transformed, list)
    assert len(transformed) == 3

    # First item should be converted
    assert transformed[0]["page_content"] == "Has text content."
    assert "text_content" not in transformed[0]

    # Second item should be unchanged (already has page_content)
    assert transformed[1]["page_content"] == "Already has page_content."

    # Third item should be unchanged (no text_content)
    assert "page_content" not in transformed[2]
    assert "text_content" not in transformed[2]


def test_transform_kbaas_response_handles_none():
    """Test that None response is handled gracefully."""
    assert _transform_kbaas_response(None) is None


def test_transform_kbaas_response_handles_non_dict():
    """Test that non-dict responses are returned as-is."""
    assert _transform_kbaas_response("string response") == "string response"
    assert _transform_kbaas_response(123) == 123
    assert _transform_kbaas_response(["list", "response"]) == ["list", "response"]


def test_transform_kbaas_response_handles_missing_results_key():
    """Test that response without results key returns empty list."""
    response = {"other_key": "value"}
    transformed = _transform_kbaas_response(response)
    # When "results" key is missing, get() returns [], so we get empty list
    assert transformed == []


def test_transform_kbaas_response_hierarchical_kb_with_parent_chunk():
    """Test hierarchical KB: parent_chunk_text becomes page_content, text_content becomes embedded_content."""
    response = {
        "results": [
            {
                "metadata": {"source": "doc1.pdf", "page": 1},
                "text_content": "This is the embedded chunk.",
                "parent_chunk_text": "This is the full parent context with more information.",
            },
            {
                "metadata": {"source": "doc2.pdf", "page": 2},
                "text_content": "Another embedded chunk.",
                "parent_chunk_text": "Another parent context.",
            },
        ],
        "total_results": 2,
    }

    transformed = _transform_kbaas_response(response)

    # Should return a list
    assert isinstance(transformed, list)
    assert len(transformed) == 2

    # parent_chunk_text should become page_content
    assert (
        transformed[0]["page_content"]
        == "This is the full parent context with more information."
    )
    assert transformed[1]["page_content"] == "Another parent context."

    # text_content should become embedded_content
    assert transformed[0]["embedded_content"] == "This is the embedded chunk."
    assert transformed[1]["embedded_content"] == "Another embedded chunk."

    # Original keys should be removed
    assert "parent_chunk_text" not in transformed[0]
    assert "parent_chunk_text" not in transformed[1]
    assert "text_content" not in transformed[0]
    assert "text_content" not in transformed[1]

    # Metadata should be preserved
    assert transformed[0]["metadata"]["source"] == "doc1.pdf"


def test_transform_kbaas_response_hierarchical_kb_parent_only():
    """Test hierarchical KB with parent_chunk_text but no text_content."""
    response = {
        "results": [
            {
                "metadata": {"source": "doc1.pdf"},
                "parent_chunk_text": "Parent context only.",
            }
        ],
        "total_results": 1,
    }

    transformed = _transform_kbaas_response(response)

    assert isinstance(transformed, list)
    assert len(transformed) == 1
    assert transformed[0]["page_content"] == "Parent context only."
    assert "embedded_content" not in transformed[0]
    assert "parent_chunk_text" not in transformed[0]


def test_transform_kbaas_response_mixed_results():
    """Test mixed results: some with parent_chunk_text, some with only text_content."""
    response = {
        "results": [
            {
                "metadata": {"source": "hierarchical.pdf"},
                "text_content": "Embedded chunk.",
                "parent_chunk_text": "Full parent context.",
            },
            {
                "metadata": {"source": "standard.pdf"},
                "text_content": "Standard KB chunk.",
            },
            {
                "metadata": {"source": "empty.pdf"}
                # No content fields
            },
        ],
        "total_results": 3,
    }

    transformed = _transform_kbaas_response(response)

    assert isinstance(transformed, list)
    assert len(transformed) == 3

    # First item: hierarchical (has parent_chunk_text)
    assert transformed[0]["page_content"] == "Full parent context."
    assert transformed[0]["embedded_content"] == "Embedded chunk."

    # Second item: standard (only text_content)
    assert transformed[1]["page_content"] == "Standard KB chunk."
    assert "embedded_content" not in transformed[1]

    # Third item: no content fields
    assert "page_content" not in transformed[2]
    assert "embedded_content" not in transformed[2]


# -----------------------------
# Retriever Call Detection Tests
# -----------------------------


def test_retriever_hit_sets_metadata(tracker, interceptor):
    """Test that KBaaS calls set is_retriever_call metadata instead of is_llm_call."""
    # Create a mock captured request for KBaaS
    mock_captured = MagicMock()
    mock_captured.url = "https://kbaas.do-ai.run/v1/retrieve"
    mock_captured.request_payload = {"query": "test query"}
    mock_captured.response_payload = {
        "results": [{"text_content": "doc content", "metadata": {}}],
        "total_results": 1,
    }

    interceptor.hits_since.return_value = 1
    interceptor.get_captured_requests_since.return_value = [mock_captured]

    inst = LangGraphInstrumentor()
    inst.install(tracker)

    def node(state: dict):
        return {"r": 1}

    g = make_graph()
    g.add_node("retriever", node)
    app = _compile_singleton_graph(g, "retriever")

    app.invoke({})

    # NodeExecution record is arg0 to on_node_end
    exec_rec = tracker.on_node_end.call_args[0][0]
    assert exec_rec.metadata.get("is_retriever_call") is True
    assert (
        exec_rec.metadata.get("is_llm_call") is None
        or exec_rec.metadata.get("is_llm_call") is False
    )


def test_retriever_response_is_transformed(tracker, interceptor):
    """Test that KBaaS responses have text_content converted to page_content."""
    # Create a mock captured request for KBaaS
    mock_captured = MagicMock()
    mock_captured.url = "https://kbaas.do-ai.run/v1/retrieve"
    mock_captured.request_payload = {"query": "test query"}
    mock_captured.response_payload = {
        "results": [
            {
                "text_content": "Document content here",
                "metadata": {"source": "test.pdf"},
            }
        ],
        "total_results": 1,
    }

    interceptor.hits_since.return_value = 1
    interceptor.get_captured_requests_since.return_value = [mock_captured]

    inst = LangGraphInstrumentor()
    inst.install(tracker)

    def node(state: dict):
        return {"r": 1}

    g = make_graph()
    g.add_node("retriever", node)
    app = _compile_singleton_graph(g, "retriever")

    app.invoke({})

    # Check the output payload passed to on_node_end
    out_payload = tracker.on_node_end.call_args[0][1]

    # The response should be a list (array) with page_content instead of text_content
    assert isinstance(out_payload, list)
    assert len(out_payload) == 1
    assert out_payload[0]["page_content"] == "Document content here"
    assert "text_content" not in out_payload[0]


def test_inference_call_still_sets_llm_metadata(tracker, interceptor):
    """Test that inference calls still set is_llm_call metadata."""
    # Create a mock captured request for inference
    mock_captured = MagicMock()
    mock_captured.url = "https://inference.do-ai.run/v1/chat/completions"
    mock_captured.request_payload = {"messages": [{"role": "user", "content": "Hello"}]}
    mock_captured.response_payload = {"choices": [{"message": {"content": "Hi!"}}]}

    interceptor.hits_since.return_value = 1
    interceptor.get_captured_requests_since.return_value = [mock_captured]

    inst = LangGraphInstrumentor()
    inst.install(tracker)

    def node(state: dict):
        return {"r": 1}

    g = make_graph()
    g.add_node("llm", node)
    app = _compile_singleton_graph(g, "llm")

    app.invoke({})

    # NodeExecution record is arg0 to on_node_end
    exec_rec = tracker.on_node_end.call_args[0][0]
    assert exec_rec.metadata.get("is_llm_call") is True
    assert (
        exec_rec.metadata.get("is_retriever_call") is None
        or exec_rec.metadata.get("is_retriever_call") is False
    )


def test_get_captured_payloads_with_type_inference_url():
    """Test _get_captured_payloads_with_type correctly identifies inference URLs."""
    mock_intr = MagicMock()
    mock_captured = MagicMock()
    mock_captured.url = "https://inference.do-ai.run/v1/chat"
    mock_captured.request_payload = {"messages": []}
    mock_captured.response_payload = {"choices": []}

    mock_intr.get_captured_requests_since.return_value = [mock_captured]

    req, resp, is_llm, is_retriever = _get_captured_payloads_with_type(mock_intr, 0)

    assert req == {"messages": []}
    assert resp == {"choices": []}
    assert is_llm is True
    assert is_retriever is False


def test_get_captured_payloads_with_type_kbaas_url():
    """Test _get_captured_payloads_with_type correctly identifies KBaaS URLs."""
    mock_intr = MagicMock()
    mock_captured = MagicMock()
    mock_captured.url = "https://kbaas.do-ai.run/retrieve"
    mock_captured.request_payload = {"query": "test"}
    mock_captured.response_payload = {"results": []}

    mock_intr.get_captured_requests_since.return_value = [mock_captured]

    req, resp, is_llm, is_retriever = _get_captured_payloads_with_type(mock_intr, 0)

    assert req == {"query": "test"}
    assert resp == {"results": []}
    assert is_llm is False
    assert is_retriever is True


def test_get_captured_payloads_with_type_no_captures():
    """Test _get_captured_payloads_with_type when no requests captured."""
    mock_intr = MagicMock()
    mock_intr.get_captured_requests_since.return_value = []

    req, resp, is_llm, is_retriever = _get_captured_payloads_with_type(mock_intr, 0)

    assert req is None
    assert resp is None
    assert is_llm is False
    assert is_retriever is False
