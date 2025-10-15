from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

import gradient_adk.runtime.langgraph.langgraph_instrumentor as mod


@dataclass
class DummySpan:
    metadata: Dict[str, Any] = field(default_factory=dict)


class DummyTracker:
    """Minimal duck-typed ExecutionTracker for tests."""

    def __init__(self) -> None:
        self.starts: List[Dict[str, Any]] = []
        self.ends: List[Dict[str, Any]] = []

    def start_node_execution(
        self,
        *,
        node_id: str,
        node_name: str,
        framework: str,
        inputs: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> DummySpan:
        self.starts.append(
            {
                "node_id": node_id,
                "node_name": node_name,
                "framework": framework,
                "inputs": inputs,
                "metadata": dict(metadata),
            }
        )
        return DummySpan(metadata={})

    def end_node_execution(self, span: DummySpan, *, outputs: Any) -> None:
        self.ends.append({"span": span, "outputs": outputs})


class DummyInterceptor:
    def __init__(self, endpoints: Optional[List[str]] = None) -> None:
        self._endpoints = endpoints or []
        self.cleared = 0

    def clear_detected(self):
        self.cleared += 1

    def get_detected_endpoints(self) -> List[str]:
        return list(self._endpoints)


class Runnable:
    def __init__(self, func=None, afunc=None, _func=None, _afunc=None):
        if func is not None:
            self.func = func
        if afunc is not None:
            self.afunc = afunc
        if _func is not None:
            self._func = _func
        if _afunc is not None:
            self._afunc = _afunc


class NodeSpec:
    def __init__(self, runnable: Any = None, func=None, _func=None):
        if runnable is not None:
            self.runnable = runnable
        if func is not None:
            self.func = func
        if _func is not None:
            self._func = _func


class GraphWithNodes:
    def __init__(self, nodes: Dict[str, Any]):
        self.nodes = nodes


class GraphWith_Nodes:
    def __init__(self, nodes: Dict[str, Any]):
        self._nodes = nodes


class NestedGraph:
    def __init__(self, nodes: Dict[str, Any]):
        self.graph = GraphWithNodes(nodes)


# ----------------------------
# Pytest fixtures
# ----------------------------


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    # Ensure clean singleton per test
    monkeypatch.setattr(mod, "_singleton", None, raising=False)


@pytest.fixture
def tracker():
    return DummyTracker()


@pytest.fixture
def noop_setup_interception(monkeypatch):
    called = {"count": 0}

    def _setup():
        called["count"] += 1

    monkeypatch.setattr(mod, "setup_digitalocean_interception", _setup)
    return called


@pytest.fixture
def interceptor_empty(monkeypatch):
    dummy = DummyInterceptor([])
    monkeypatch.setattr(mod, "get_network_interceptor", lambda: dummy)
    return dummy


@pytest.fixture
def interceptor_llm(monkeypatch):
    dummy = DummyInterceptor(["https://api.openai.com/v1/chat/completions"])
    monkeypatch.setattr(mod, "get_network_interceptor", lambda: dummy)
    return dummy


# ----------------------------
# Unit tests
# ----------------------------


def test_install_and_uninstall(tracker, noop_setup_interception):
    inst = mod.install(tracker)
    assert isinstance(inst, mod.LangGraphInstrumentor)
    assert inst.is_installed() is True
    assert noop_setup_interception["count"] == 1

    # idempotent install
    inst2 = mod.install(tracker)
    assert inst2 is inst
    assert inst.is_installed() is True
    assert noop_setup_interception["count"] == 1  # still once

    inst.uninstall()
    assert inst.is_installed() is False


def test_wraps_runnable_func_and_tracks_sync(
    tracker, interceptor_empty, noop_setup_interception
):
    called = {"echo": 0}

    def echo(x, y=0):
        called["echo"] += 1
        return x + y

    g = GraphWithNodes({"add": NodeSpec(runnable=Runnable(func=echo))})

    mod.install(tracker)
    mod.attach_to_graph(g)

    wrapped = g.nodes["add"].runnable.func
    assert callable(wrapped)
    assert getattr(wrapped, "__gradient_wrapped__", False) is True

    result = wrapped(2, y=3)
    assert result == 5
    assert called["echo"] == 1

    # One start + end; framework/langgraph recorded; inputs serialized
    assert len(tracker.starts) == 1
    assert len(tracker.ends) == 1
    start = tracker.starts[0]
    assert start["framework"] == "langgraph"
    assert start["node_name"] == "add"
    assert start["inputs"]["args"] == [2]
    assert start["inputs"]["kwargs"]["y"] == 3

    # is_llm_call metadata set by wrapper; with empty interceptor, should be False
    assert tracker.ends[0]["span"].metadata.get("is_llm_call") is False
    assert tracker.ends[0]["outputs"] == 5

    # Idempotent attach: calling attach again must NOT double-wrap
    mod.attach_to_graph(g)
    wrapped_again = g.nodes["add"].runnable.func
    assert wrapped_again is wrapped  # identity unchanged

    # Call once more and ensure only one additional start/end (not two)
    wrapped(1, y=1)
    assert len(tracker.starts) == 2
    assert len(tracker.ends) == 2


@pytest.mark.asyncio
async def test_wraps_runnable_afunc_and_tracks_async(
    tracker, interceptor_llm, noop_setup_interception
):
    async def times2(n: int) -> int:
        return n * 2

    g = GraphWithNodes({"dbl": NodeSpec(runnable=Runnable(afunc=times2))})

    mod.install(tracker)
    mod.attach_to_graph(g)

    wrapped = g.nodes["dbl"].runnable.afunc
    assert asyncio.iscoroutinefunction(wrapped)
    assert getattr(wrapped, "__gradient_wrapped__", False) is True

    out = await wrapped(7)
    assert out == 14

    assert len(tracker.starts) == 1
    assert len(tracker.ends) == 1
    # LLM endpoints detected → metadata flag True
    assert tracker.ends[0]["span"].metadata.get("is_llm_call") is True
    assert tracker.ends[0]["outputs"] == 14


def test_wraps_nodespec_func_field(tracker, interceptor_empty):
    def square(n: int) -> int:
        return n * n

    g = GraphWithNodes({"sq": NodeSpec(func=square)})
    mod.install(tracker)
    mod.attach_to_graph(g)

    wrapped = g.nodes["sq"].func
    assert getattr(wrapped, "__gradient_wrapped__", False) is True
    assert wrapped(4) == 16
    assert len(tracker.starts) == 1
    assert len(tracker.ends) == 1
    assert tracker.ends[0]["outputs"] == 16


def test_wraps_direct_callable_in_nodes_dict(tracker, interceptor_empty):
    def shout(s: str) -> str:
        return s.upper()

    g = GraphWithNodes({"shout": shout})
    mod.install(tracker)
    mod.attach_to_graph(g)

    wrapped = g.nodes["shout"]
    assert getattr(wrapped, "__gradient_wrapped__", False) is True
    assert wrapped("hey") == "HEY"
    assert len(tracker.starts) == 1
    assert len(tracker.ends) == 1
    assert tracker.ends[0]["outputs"] == "HEY"


def test_finds__nodes_and_nested_graph_variants(tracker, interceptor_empty):
    def inc(n: int) -> int:
        return n + 1

    def dec(n: int) -> int:
        return n - 1

    g1 = GraphWith_Nodes({"inc": NodeSpec(runnable=Runnable(func=inc))})
    g2 = NestedGraph({"dec": NodeSpec(runnable=Runnable(func=dec))})

    mod.install(tracker)

    mod.attach_to_graph(g1)
    mod.attach_to_graph(g2)

    assert (
        getattr(g1._nodes["inc"].runnable.func, "__gradient_wrapped__", False) is True
    )
    assert (
        getattr(g2.graph.nodes["dec"].runnable.func, "__gradient_wrapped__", False)
        is True
    )

    assert g1._nodes["inc"].runnable.func(1) == 2
    assert g2.graph.nodes["dec"].runnable.func(2) == 1

    # 2 nodes executed → two spans
    assert len(tracker.starts) == 2
    assert len(tracker.ends) == 2


def test_error_path_records_error_output(tracker, interceptor_empty):
    def boom():
        raise ValueError("kaboom")

    g = GraphWithNodes({"bad": NodeSpec(runnable=Runnable(func=boom))})
    mod.install(tracker)
    mod.attach_to_graph(g)

    with pytest.raises(ValueError, match="kaboom"):
        g.nodes["bad"].runnable.func()

    assert len(tracker.starts) == 1
    assert len(tracker.ends) == 1
    # Error is serialized into outputs
    assert isinstance(tracker.ends[0]["outputs"], dict)
    assert "error" in tracker.ends[0]["outputs"]
    assert "kaboom" in tracker.ends[0]["outputs"]["error"]


def test_inputs_serialization_truncates_large_dict(tracker, interceptor_empty):
    def takes_big(d: Dict[str, int]) -> int:
        return sum(d.values())

    big = {f"k{i}": i for i in range(25)}  # >10 entries
    g = GraphWithNodes({"big": NodeSpec(runnable=Runnable(func=takes_big))})

    mod.install(tracker)
    mod.attach_to_graph(g)

    out = g.nodes["big"].runnable.func(big)
    assert out == sum(big.values())

    inputs = tracker.starts[0]["inputs"]
    # args[0] is serialized dict with truncation marker
    assert isinstance(inputs["args"][0], dict)
    assert inputs["args"][0].get("__truncated__") is True
