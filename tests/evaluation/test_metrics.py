"""Tests for gradient_adk.evaluation.metrics."""

from __future__ import annotations

import pytest

from gradient_adk.evaluation.metrics import (
    METRIC_REGISTRY,
    PRESETS,
    MetricSpec,
    SkippedMetric,
    resolve_preset,
    resolve_runnable_metrics,
)


class TestRegistry:
    def test_registry_not_empty(self):
        assert len(METRIC_REGISTRY) > 0

    def test_all_specs_have_required_fields(self):
        for name, spec in METRIC_REGISTRY.items():
            assert spec.name == name
            assert spec.deepeval_class
            assert isinstance(spec.required_fields, frozenset)
            assert isinstance(spec.default_threshold, float)
            assert spec.description


class TestPresets:
    def test_basic_preset(self):
        names = resolve_preset("basic")
        assert "answer_relevancy" in names
        assert "bias" in names
        assert "toxicity" in names

    def test_rag_preset(self):
        names = resolve_preset("rag")
        assert "faithfulness" in names
        assert "contextual_relevancy" in names

    def test_agent_preset(self):
        names = resolve_preset("agent")
        assert "tool_correctness" in names

    def test_all_preset_contains_every_metric(self):
        names = resolve_preset("all")
        assert set(names) == set(METRIC_REGISTRY.keys())

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            resolve_preset("nonexistent")


class TestSkipLogic:
    def test_all_fields_present_runs_all(self):
        names = ["answer_relevancy", "faithfulness"]
        available = {"retrieval_context"}
        runnable, skipped = resolve_runnable_metrics(names, available)
        assert len(runnable) == 2
        assert len(skipped) == 0

    def test_missing_retrieval_context_skips_faithfulness(self):
        names = ["answer_relevancy", "faithfulness"]
        available = set()  # no retrieval_context
        runnable, skipped = resolve_runnable_metrics(names, available)
        assert len(runnable) == 1
        assert runnable[0].name == "answer_relevancy"
        assert len(skipped) == 1
        assert skipped[0].name == "faithfulness"
        assert "retrieval_context" in skipped[0].reason

    def test_skip_reason_is_actionable(self):
        names = ["faithfulness"]
        runnable, skipped = resolve_runnable_metrics(names, set())
        assert "eval_record()" in skipped[0].reason

    def test_skip_reason_for_expected_output(self):
        names = ["contextual_precision"]
        available = {"retrieval_context"}  # missing expected_output
        runnable, skipped = resolve_runnable_metrics(names, available)
        assert len(skipped) == 1
        assert "expected_output" in skipped[0].reason
        assert "dataset CSV" in skipped[0].reason

    def test_tool_correctness_needs_both_fields(self):
        names = ["tool_correctness"]
        # Only tools, no expected_tools
        runnable, skipped = resolve_runnable_metrics(names, {"tool_calls"})
        assert len(skipped) == 1
        assert "expected_tools" in skipped[0].reason

    def test_unknown_metric_is_skipped(self):
        runnable, skipped = resolve_runnable_metrics(["nonexistent"], set())
        assert len(runnable) == 0
        assert len(skipped) == 1
        assert "unknown metric" in skipped[0].reason

    def test_basic_preset_needs_no_special_fields(self):
        """Basic preset metrics have no required_fields."""
        names = resolve_preset("basic")
        runnable, skipped = resolve_runnable_metrics(names, set())
        assert len(skipped) == 0
        assert len(runnable) == len(names)
