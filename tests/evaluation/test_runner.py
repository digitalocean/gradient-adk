"""Tests for gradient_adk.evaluation.runner (mocked DeepEval)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gradient_adk.evaluation.config import EvalConfig
from gradient_adk.evaluation.record import (
    _begin_eval_request,
    _end_eval_request,
    _eval_store,
    _pop_eval_record,
    eval_record,
)
from gradient_adk.evaluation.runner import (
    EvalResults,
    CaseResult,
    _import_agent_app,
)


class TestImportAgentApp:
    def test_import_raises_on_missing_fastapi_app(self, tmp_path, monkeypatch):
        """Module without fastapi_app raises RuntimeError."""
        mod_file = tmp_path / "noapp.py"
        mod_file.write_text("x = 1\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(RuntimeError, match="fastapi_app"):
            _import_agent_app("noapp.py")


class TestEvalRecordIntegration:
    """Test that EvalRecord is populated and retrievable after a request."""

    def test_begin_and_pop(self):
        _eval_store.clear()
        token = _begin_eval_request("test-id")
        rec = eval_record()
        rec.add_context(["chunk"])
        rec.add_tool_call("fn", args={"a": 1})
        _end_eval_request(token)

        popped = _pop_eval_record("test-id")
        assert popped is not None
        assert popped.retrieval_context == ["chunk"]
        assert popped.tool_calls[0].name == "fn"


class TestEvalResultsDataclasses:
    def test_eval_results_defaults(self):
        r = EvalResults()
        assert r.test_case_results == []
        assert r.metric_summaries == []
        assert r.skipped_metrics == []
        assert r.total_time == 0.0

    def test_test_case_result(self):
        tcr = CaseResult(
            input="hello",
            actual_output="world",
            expected_output="world",
            metric_scores={"answer_relevancy": 0.9},
            metric_passed={"answer_relevancy": True},
            completion_time=1.5,
        )
        assert tcr.metric_scores["answer_relevancy"] == 0.9
        assert tcr.completion_time == 1.5
