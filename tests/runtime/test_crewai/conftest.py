"""Shared fixtures for CrewAI instrumentation tests.

CrewAI's event bus doesn't support unregistering handlers, so we must install
the instrumentor exactly once across ALL test files in this package. This
conftest provides a session-scoped instrumentor that both the unit tests and
integration tests share, using a DelegatingTracker to swap the per-test tracker.
"""

import pytest
from unittest.mock import MagicMock

from gradient_adk.runtime.crewai.crewai_instrumentor import (
    CrewAIInstrumentor,
    _agent_stack,
    _agent_stack_lock,
)


class DelegatingTracker:
    """Tracker that delegates all calls to a swappable underlying tracker.

    This allows a single instrumentor installation to serve all tests.
    Each test sets its own mock tracker as the delegate.
    """

    def __init__(self):
        self._delegate = None

    def set_delegate(self, delegate):
        self._delegate = delegate

    def on_node_start(self, *args, **kwargs):
        if self._delegate:
            return self._delegate.on_node_start(*args, **kwargs)

    def on_node_end(self, *args, **kwargs):
        if self._delegate:
            return self._delegate.on_node_end(*args, **kwargs)

    def on_node_error(self, *args, **kwargs):
        if self._delegate:
            return self._delegate.on_node_error(*args, **kwargs)


# Single shared delegating tracker and instrumentor for the entire test session.
_shared_delegating_tracker = DelegatingTracker()
_shared_instrumentor = None


@pytest.fixture(scope="session")
def shared_instrumentor():
    """Session-scoped instrumentor — installed exactly once.

    This prevents handler accumulation on CrewAI's event bus.
    """
    global _shared_instrumentor
    inst = CrewAIInstrumentor()
    inst.install(_shared_delegating_tracker)
    _shared_instrumentor = inst
    yield inst
    inst.uninstall()


@pytest.fixture(autouse=True)
def clear_agent_stack():
    """Clear the agent stack before and after each test."""
    try:
        from crewai.events import crewai_event_bus
        crewai_event_bus.flush(timeout=5.0)
    except Exception:
        # If CrewAI isn't available or flush fails, continue cleanup anyway.
        pass
    with _agent_stack_lock:
        _agent_stack.clear()
    yield
    try:
        from crewai.events import crewai_event_bus
        crewai_event_bus.flush(timeout=5.0)
    except Exception:
        pass
    with _agent_stack_lock:
        _agent_stack.clear()
    # Disconnect delegate so stale async handlers don't leak into the next test
    _shared_delegating_tracker.set_delegate(None)


@pytest.fixture
def tracker():
    """Fresh mock tracker for each test (used by instrumentor unit tests)."""
    t = MagicMock()
    t.on_node_start = MagicMock()
    t.on_node_end = MagicMock()
    t.on_node_error = MagicMock()
    return t


@pytest.fixture
def mock_tracker():
    """Fresh mock tracker for each test (used by integration tests)."""
    t = MagicMock()
    t.on_node_start = MagicMock()
    t.on_node_end = MagicMock()
    t.on_node_error = MagicMock()
    return t


@pytest.fixture
def instrumentor(tracker, shared_instrumentor):
    """Per-test instrumentor wired to this test's tracker.

    Reuses the session-scoped instrumentor, just swaps the delegate.
    """
    _shared_delegating_tracker.set_delegate(tracker)
    yield shared_instrumentor
    _shared_delegating_tracker.set_delegate(None)


@pytest.fixture
def shared_delegating_tracker():
    """Expose the shared delegating tracker for test files that need it."""
    return _shared_delegating_tracker
