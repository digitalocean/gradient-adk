"""
EvalRecord — per-request evaluation data collector.

Provides a ContextVar-based store so agent code can attach retrieval context
and tool-call metadata during a request.  The eval runner retrieves the data
after the response completes.

In production (no eval context) ``eval_record()`` returns a lightweight NoOp
that silently discards all writes — zero overhead.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """A single tool invocation recorded during evaluation."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    output: Any = None


@dataclass
class EvalRecord:
    """Mutable bag of evaluation data populated by agent code."""

    retrieval_context: Optional[List[str]] = None
    tool_calls: List[ToolCallRecord] = field(default_factory=list)

    def add_tool_call(
        self, name: str, *, args: Optional[Dict[str, Any]] = None, output: Any = None
    ) -> None:
        self.tool_calls.append(
            ToolCallRecord(name=name, args=args or {}, output=output)
        )

    def add_context(self, chunks: List[str]) -> None:
        """Convenience setter for ``retrieval_context``."""
        self.retrieval_context = chunks


# ---------------------------------------------------------------------------
# NoOp — returned when not inside an eval context
# ---------------------------------------------------------------------------

class _NoOpEvalRecord:
    """Drop-in replacement that silently discards all writes."""

    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D105
        pass

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        if name == "tool_calls":
            return []
        return None

    def add_tool_call(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        pass

    def add_context(self, *args: Any) -> None:  # noqa: D102
        pass


_NOOP = _NoOpEvalRecord()


# ---------------------------------------------------------------------------
# ContextVar + module-level store
# ---------------------------------------------------------------------------

_eval_record_var: contextvars.ContextVar[Optional[EvalRecord]] = contextvars.ContextVar(
    "eval_record", default=None
)

# Keyed by request-id so the runner can retrieve the record after the
# response completes.
_eval_store: Dict[str, EvalRecord] = {}


def _begin_eval_request(request_id: str) -> contextvars.Token:
    """Create a fresh EvalRecord, store it, and set the ContextVar."""
    record = EvalRecord()
    _eval_store[request_id] = record
    return _eval_record_var.set(record)


def _end_eval_request(token: contextvars.Token) -> None:
    """Reset the ContextVar — the record stays in ``_eval_store``."""
    _eval_record_var.reset(token)


def _pop_eval_record(request_id: str) -> Optional[EvalRecord]:
    """Retrieve (and remove) the record the runner needs."""
    return _eval_store.pop(request_id, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def eval_record() -> EvalRecord:
    """Return the current request's EvalRecord, or a silent NoOp.

    Agent authors call this to attach evaluation metadata::

        from gradient_adk import eval_record

        record = eval_record()
        record.retrieval_context = ["chunk1", "chunk2"]
        record.add_tool_call("search", args={"q": "hello"}, output="world")

    Outside an eval context every operation is a harmless no-op.
    """
    rec = _eval_record_var.get()
    if rec is not None:
        return rec
    return _NOOP  # type: ignore[return-value]
