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
from gradient_adk.runtime.langgraph.helpers import (
    _is_langgraph_instrumentation_disabled,
    _is_langgraph_available,
    DISABLE_LANGGRAPH_INSTRUMENTOR_ENV,
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
