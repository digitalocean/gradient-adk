"""Tests for gradient_adk.evaluation.record."""

from __future__ import annotations

import asyncio

import pytest

from gradient_adk.evaluation.record import (
    EvalRecord,
    ToolCallRecord,
    _NoOpEvalRecord,
    _begin_eval_request,
    _end_eval_request,
    _eval_record_var,
    _eval_store,
    _pop_eval_record,
    eval_record,
)


# ---------------------------------------------------------------------------
# NoOp behaviour
# ---------------------------------------------------------------------------

class TestNoOp:
    def test_eval_record_returns_noop_outside_context(self):
        """eval_record() returns a NoOp when no eval context is active."""
        rec = eval_record()
        assert isinstance(rec, _NoOpEvalRecord)

    def test_noop_discards_setattr(self):
        noop = _NoOpEvalRecord()
        noop.retrieval_context = ["a", "b"]
        assert noop.retrieval_context is None  # still None via __getattr__

    def test_noop_tool_calls_returns_empty_list(self):
        noop = _NoOpEvalRecord()
        assert noop.tool_calls == []

    def test_noop_other_attr_returns_none(self):
        noop = _NoOpEvalRecord()
        assert noop.retrieval_context is None
        assert noop.some_random_attr is None

    def test_noop_add_tool_call_is_silent(self):
        noop = _NoOpEvalRecord()
        noop.add_tool_call("search", args={"q": "hi"}, output="result")
        assert noop.tool_calls == []

    def test_noop_add_context_is_silent(self):
        noop = _NoOpEvalRecord()
        noop.add_context(["chunk1"])
        assert noop.retrieval_context is None


# ---------------------------------------------------------------------------
# EvalRecord behaviour
# ---------------------------------------------------------------------------

class TestEvalRecord:
    def test_defaults(self):
        rec = EvalRecord()
        assert rec.retrieval_context is None
        assert rec.tool_calls == []

    def test_add_tool_call(self):
        rec = EvalRecord()
        rec.add_tool_call("search", args={"q": "hello"}, output="world")
        assert len(rec.tool_calls) == 1
        tc = rec.tool_calls[0]
        assert tc.name == "search"
        assert tc.args == {"q": "hello"}
        assert tc.output == "world"

    def test_add_multiple_tool_calls(self):
        rec = EvalRecord()
        rec.add_tool_call("a")
        rec.add_tool_call("b", args={"x": 1})
        assert len(rec.tool_calls) == 2
        assert rec.tool_calls[0].name == "a"
        assert rec.tool_calls[1].name == "b"

    def test_add_context(self):
        rec = EvalRecord()
        rec.add_context(["chunk1", "chunk2"])
        assert rec.retrieval_context == ["chunk1", "chunk2"]

    def test_add_context_overwrites(self):
        rec = EvalRecord()
        rec.add_context(["a"])
        rec.add_context(["b", "c"])
        assert rec.retrieval_context == ["b", "c"]


# ---------------------------------------------------------------------------
# ContextVar + Store
# ---------------------------------------------------------------------------

class TestStore:
    def setup_method(self):
        # Clean up any leftover state
        _eval_store.clear()

    def test_begin_sets_real_record(self):
        token = _begin_eval_request("req-1")
        try:
            rec = eval_record()
            assert isinstance(rec, EvalRecord)
        finally:
            _end_eval_request(token)

    def test_end_resets_to_noop(self):
        token = _begin_eval_request("req-2")
        _end_eval_request(token)
        rec = eval_record()
        assert isinstance(rec, _NoOpEvalRecord)

    def test_pop_retrieves_and_removes(self):
        token = _begin_eval_request("req-3")
        rec = eval_record()
        rec.add_context(["data"])
        _end_eval_request(token)

        popped = _pop_eval_record("req-3")
        assert popped is not None
        assert popped.retrieval_context == ["data"]

        # Second pop returns None
        assert _pop_eval_record("req-3") is None

    def test_pop_missing_returns_none(self):
        assert _pop_eval_record("nonexistent") is None

    def test_record_persists_writes(self):
        token = _begin_eval_request("req-4")
        rec = eval_record()
        rec.retrieval_context = ["ctx"]
        rec.add_tool_call("fn", args={"a": 1})
        _end_eval_request(token)

        popped = _pop_eval_record("req-4")
        assert popped.retrieval_context == ["ctx"]
        assert len(popped.tool_calls) == 1
        assert popped.tool_calls[0].name == "fn"


# ---------------------------------------------------------------------------
# Async isolation
# ---------------------------------------------------------------------------

class TestAsyncIsolation:
    @pytest.mark.asyncio
    async def test_contextvar_isolates_coroutines(self):
        """Two concurrent coroutines get independent EvalRecords."""
        results = {}

        async def worker(request_id: str, context_val: str):
            token = _begin_eval_request(request_id)
            try:
                rec = eval_record()
                rec.add_context([context_val])
                # Yield control to let other coroutine run
                await asyncio.sleep(0)
                # Should still see our own data
                results[request_id] = eval_record().retrieval_context
            finally:
                _end_eval_request(token)

        await asyncio.gather(
            worker("iso-a", "data-a"),
            worker("iso-b", "data-b"),
        )

        assert results["iso-a"] == ["data-a"]
        assert results["iso-b"] == ["data-b"]

        # Both records retrievable from store
        rec_a = _pop_eval_record("iso-a")
        rec_b = _pop_eval_record("iso-b")
        assert rec_a.retrieval_context == ["data-a"]
        assert rec_b.retrieval_context == ["data-b"]
