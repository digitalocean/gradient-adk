"""
Local evaluation runner — the main orchestrator.

Imports the agent in-process, drives it via ASGI transport, collects
EvalRecord data, constructs DeepEval test cases, and runs the selected
metrics with a configurable judge LLM.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from .config import EvalConfig
from .dataset import DatasetRow, ParsedDataset, parse_dataset
from .metrics import (
    MetricSpec,
    SkippedMetric,
    resolve_preset,
    resolve_runnable_metrics,
)
from .record import _pop_eval_record


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Result of evaluating a single dataset row."""

    input: Any
    actual_output: str
    expected_output: Optional[str] = None
    metric_scores: Dict[str, float] = field(default_factory=dict)
    metric_passed: Dict[str, bool] = field(default_factory=dict)
    completion_time: float = 0.0


@dataclass
class MetricSummary:
    """Aggregate summary for one metric across all test cases."""

    name: str
    avg_score: float
    threshold: float
    passed: bool


@dataclass
class EvalResults:
    """Complete evaluation results."""

    test_case_results: List[CaseResult] = field(default_factory=list)
    metric_summaries: List[MetricSummary] = field(default_factory=list)
    skipped_metrics: List[SkippedMetric] = field(default_factory=list)
    total_time: float = 0.0


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _import_agent_app(entrypoint_file: str) -> Any:
    """Import the agent module and return its ``fastapi_app``."""
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module_name = (
        entrypoint_file.replace(".py", "").replace("/", ".").replace("\\", ".")
    )
    module = importlib.import_module(module_name)

    if not hasattr(module, "fastapi_app"):
        raise RuntimeError(
            f"Module '{module_name}' has no 'fastapi_app' attribute. "
            "Ensure the module uses the @entrypoint decorator."
        )
    return module.fastapi_app


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def run_local_evaluation(
    *,
    entrypoint_file: str,
    config: EvalConfig,
    dataset_path: Path,
    metric_names: Optional[List[str]] = None,
) -> EvalResults:
    """Run a full local evaluation and return structured results.

    Parameters
    ----------
    entrypoint_file:
        Path to the agent module (e.g. ``"main.py"``).
    config:
        Resolved ``EvalConfig``.
    dataset_path:
        Path to the evaluation CSV.
    metric_names:
        Explicit list of metrics.  If *None*, resolved from ``config.preset``.
    """
    import httpx
    from httpx import ASGITransport

    start = time.monotonic()

    # 1. Parse dataset
    parsed: ParsedDataset = parse_dataset(dataset_path)

    # 2. Import agent
    app = _import_agent_app(entrypoint_file)

    # 3. Build ASGI client
    transport = ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")

    # 4. Resolve metric names
    if metric_names is None:
        metric_names = resolve_preset(config.preset)

    # 5. Drive agent for each row and collect outputs + EvalRecords
    raw_results: List[Dict[str, Any]] = []
    try:
        for row in parsed.rows:
            result = await _run_single(client, row, config)
            raw_results.append(result)
    finally:
        await client.aclose()

    # 6. Determine available fields for skip logic
    available_fields: Set[str] = set(parsed.available_columns)
    # Check first EvalRecord (if any) for callback-provided fields
    for r in raw_results:
        rec = r.get("eval_record")
        if rec is not None:
            if rec.retrieval_context:
                available_fields.add("retrieval_context")
            if rec.tool_calls:
                available_fields.add("tool_calls")
            break  # one sample is enough

    # 7. Resolve runnable vs skipped
    runnable_specs, skipped = resolve_runnable_metrics(metric_names, available_fields)

    # 8. Build DeepEval test cases and run metrics
    test_case_results: List[CaseResult] = []
    metric_summaries: List[MetricSummary] = []

    if runnable_specs:
        test_case_results, metric_summaries = await _run_deepeval(
            runnable_specs, raw_results, parsed, config
        )

    elapsed = time.monotonic() - start

    return EvalResults(
        test_case_results=test_case_results,
        metric_summaries=metric_summaries,
        skipped_metrics=skipped,
        total_time=elapsed,
    )


async def _run_single(
    client: "httpx.AsyncClient",
    row: DatasetRow,
    config: EvalConfig,
) -> Dict[str, Any]:
    """Send one dataset row to the agent and collect the response + EvalRecord."""
    request_id = str(uuid4())

    t0 = time.monotonic()
    resp = await client.post(
        "/run",
        json=row.query,
        headers={
            "evaluation-id": "local-eval",
            "x-eval-request-id": request_id,
        },
        timeout=config.timeout,
    )
    completion_time = time.monotonic() - t0

    resp.raise_for_status()

    # Parse response body
    try:
        actual_output = resp.json()
    except Exception:
        actual_output = resp.text

    if isinstance(actual_output, dict):
        # Try to extract a meaningful string
        actual_output = actual_output.get("response", str(actual_output))
    if not isinstance(actual_output, str):
        actual_output = str(actual_output)

    eval_record = _pop_eval_record(request_id)

    return {
        "input": row.query,
        "actual_output": actual_output,
        "expected_output": row.expected_output,
        "expected_context": row.expected_context,
        "expected_tools": row.expected_tools,
        "eval_record": eval_record,
        "completion_time": completion_time,
    }


async def _run_deepeval(
    specs: List[MetricSpec],
    raw_results: List[Dict[str, Any]],
    parsed: ParsedDataset,
    config: EvalConfig,
) -> tuple[List[CaseResult], List[MetricSummary]]:
    """Construct DeepEval objects, run evaluation, return results."""
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        BiasMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        ToolCorrectnessMetric,
        ToxicityMetric,
    )
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase
    from deepeval.test_case.llm_test_case import ToolCall

    _CLASS_MAP = {
        "AnswerRelevancyMetric": AnswerRelevancyMetric,
        "BiasMetric": BiasMetric,
        "ToxicityMetric": ToxicityMetric,
        "FaithfulnessMetric": FaithfulnessMetric,
        "ContextualRelevancyMetric": ContextualRelevancyMetric,
        "ContextualPrecisionMetric": ContextualPrecisionMetric,
        "ContextualRecallMetric": ContextualRecallMetric,
        "ToolCorrectnessMetric": ToolCorrectnessMetric,
    }

    # Build judge model
    judge = _make_judge_model(config)

    # Instantiate metric objects
    metric_objects = []
    for spec in specs:
        cls = _CLASS_MAP.get(spec.deepeval_class)
        if cls is None:
            continue
        threshold = config.thresholds.get(
            spec.name,
            config.thresholds.get("__global__", spec.default_threshold),
        )
        # Per-metric judge override
        metric_judge = judge
        if spec.name in config.metric_judge_models:
            metric_judge = _make_judge_model(
                config, model_override=config.metric_judge_models[spec.name]
            )
        metric_objects.append(cls(threshold=threshold, model=metric_judge))

    if not metric_objects:
        return [], []

    # Build LLMTestCases
    test_cases: List[LLMTestCase] = []
    for r in raw_results:
        kwargs: Dict[str, Any] = {
            "input": str(r["input"]) if not isinstance(r["input"], str) else r["input"],
            "actual_output": r["actual_output"],
        }
        if r.get("expected_output"):
            kwargs["expected_output"] = r["expected_output"]
        if r.get("expected_context"):
            kwargs["context"] = r["expected_context"]

        # Merge callback-provided retrieval context
        rec = r.get("eval_record")
        if rec and rec.retrieval_context:
            kwargs["retrieval_context"] = rec.retrieval_context

        # Tool calls from EvalRecord
        if rec and rec.tool_calls:
            kwargs["tools_called"] = [
                ToolCall(name=tc.name, input_parameters=tc.args, output=tc.output)
                for tc in rec.tool_calls
            ]
        if r.get("expected_tools"):
            kwargs["expected_tools"] = [
                ToolCall(name=t) if isinstance(t, str) else ToolCall(**t)
                for t in r["expected_tools"]
            ]

        test_cases.append(LLMTestCase(**kwargs))

    # Suppress litellm and deepeval verbose logging
    import io as _io
    import logging as _logging
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    for _logger_name in ("LiteLLM", "litellm", "httpx", "deepeval"):
        _logging.getLogger(_logger_name).setLevel(_logging.ERROR)

    # Run DeepEval with display suppressed
    from deepeval.evaluate import DisplayConfig
    from deepeval.evaluate.evaluate import EvaluationResult as _EvalResult

    # Redirect stdout to suppress DeepEval's direct console prints
    import sys as _sys
    _orig_stdout = _sys.stdout
    _sys.stdout = _io.StringIO()
    try:
        eval_result: _EvalResult = deepeval_evaluate(
            test_cases=test_cases,
            metrics=metric_objects,
            display_config=DisplayConfig(
                show_indicator=False,
                print_results=False,
            ),
        )
    finally:
        _sys.stdout = _orig_stdout

    # Build lookup for inverted metrics (lower raw = better)
    _inverted = {s.name for s in specs if s.inverted}

    # Extract per-test-case results from EvaluationResult.test_results
    tc_results: List[CaseResult] = []
    for i, r in enumerate(raw_results):
        tcr = CaseResult(
            input=r["input"],
            actual_output=r["actual_output"],
            expected_output=r.get("expected_output"),
            completion_time=r["completion_time"],
        )
        if i < len(eval_result.test_results):
            tr = eval_result.test_results[i]
            if tr.metrics_data:
                for md in tr.metrics_data:
                    key = md.name.lower().replace(" ", "_")
                    if md.score is not None:
                        # Normalize: for inverted metrics, display 1 - score
                        # so higher is always better from user perspective
                        score = (1.0 - md.score) if key in _inverted else md.score
                        tcr.metric_scores[key] = score
                        tcr.metric_passed[key] = md.success
        tc_results.append(tcr)

    # Aggregate metric summaries
    # Use DeepEval's own pass/fail (it knows which metrics are lower-is-better)
    summaries: List[MetricSummary] = []
    for spec in specs:
        scores = [
            tcr.metric_scores[spec.name]
            for tcr in tc_results
            if spec.name in tcr.metric_scores
        ]
        passes = [
            tcr.metric_passed[spec.name]
            for tcr in tc_results
            if spec.name in tcr.metric_passed
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        threshold = config.thresholds.get(
            spec.name,
            config.thresholds.get("__global__", spec.default_threshold),
        )
        summaries.append(
            MetricSummary(
                name=spec.name,
                avg_score=avg,
                threshold=threshold,
                passed=all(passes) if passes else False,
            )
        )

    return tc_results, summaries


def _metric_obj_name(metric_obj: Any) -> str:
    """Extract the canonical metric name from a DeepEval metric object."""
    # DeepEval metrics have a __name__ or name attribute
    for attr in ("__name__", "name"):
        v = getattr(metric_obj, attr, None)
        if v:
            return str(v).lower().replace(" ", "_")
    return type(metric_obj).__name__.lower().replace("metric", "")


def _make_judge_model(config: EvalConfig, *, model_override: Optional[str] = None) -> Any:
    """Create a DeepEval judge LLM model."""
    from deepeval.models import DeepEvalBaseLLM

    try:
        from deepeval.models import LiteLLMModel

        return LiteLLMModel(
            model=model_override or config.judge_model,
            base_url=config.judge_base_url,
            api_key=config.judge_api_key or "",
        )
    except ImportError:
        # Fallback: use base URL via environment variables for litellm
        os.environ.setdefault("OPENAI_API_BASE", config.judge_base_url)
        os.environ.setdefault("OPENAI_API_KEY", config.judge_api_key or "")
        from deepeval.models import LiteLLMModel

        return LiteLLMModel(model=model_override or config.judge_model)
