"""
Metric registry, presets, and skip-logic for local DeepEval evaluation.

Metrics are only *instantiated* inside the runner (which imports deepeval);
this module is pure stdlib so it can be imported anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass(frozen=True)
class MetricSpec:
    """Declarative description of a single DeepEval metric."""

    name: str
    deepeval_class: str
    required_fields: FrozenSet[str]
    default_threshold: float
    description: str
    # If True, the raw score is inverted (0 = best).  We display 1 - score
    # so that higher is always better from the user's perspective.
    inverted: bool = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, MetricSpec] = {}


def _register(*specs: MetricSpec) -> None:
    for s in specs:
        METRIC_REGISTRY[s.name] = s


_register(
    # --- basic ---
    MetricSpec(
        name="answer_relevancy",
        deepeval_class="AnswerRelevancyMetric",
        required_fields=frozenset(),
        default_threshold=0.5,
        description="Measures how relevant the answer is to the input query.",
    ),
    MetricSpec(
        name="bias",
        deepeval_class="BiasMetric",
        required_fields=frozenset(),
        default_threshold=0.5,
        description="Detects social or cognitive bias in the output.",
        inverted=True,
    ),
    MetricSpec(
        name="toxicity",
        deepeval_class="ToxicityMetric",
        required_fields=frozenset(),
        default_threshold=0.5,
        description="Detects toxic, harmful, or offensive content.",
        inverted=True,
    ),
    # --- rag ---
    MetricSpec(
        name="faithfulness",
        deepeval_class="FaithfulnessMetric",
        required_fields=frozenset({"retrieval_context"}),
        default_threshold=0.5,
        description="Checks whether the answer is faithful to the retrieval context.",
    ),
    MetricSpec(
        name="contextual_relevancy",
        deepeval_class="ContextualRelevancyMetric",
        required_fields=frozenset({"retrieval_context"}),
        default_threshold=0.5,
        description="Measures the relevancy of the retrieved context to the query.",
    ),
    MetricSpec(
        name="contextual_precision",
        deepeval_class="ContextualPrecisionMetric",
        required_fields=frozenset({"retrieval_context", "expected_output"}),
        default_threshold=0.5,
        description="Measures precision of the retrieval context given expected output.",
    ),
    MetricSpec(
        name="contextual_recall",
        deepeval_class="ContextualRecallMetric",
        required_fields=frozenset({"retrieval_context", "expected_output"}),
        default_threshold=0.5,
        description="Measures recall of the retrieval context given expected output.",
    ),
    # --- agent ---
    MetricSpec(
        name="tool_correctness",
        deepeval_class="ToolCorrectnessMetric",
        required_fields=frozenset({"tool_calls", "expected_tools"}),
        default_threshold=0.5,
        description="Checks whether the agent invoked the correct tools.",
    ),
)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, List[str]] = {
    "basic": ["answer_relevancy", "bias", "toxicity"],
    "rag": [
        "faithfulness",
        "contextual_relevancy",
        "contextual_precision",
        "contextual_recall",
    ],
    "agent": ["tool_correctness"],
    "all": list(METRIC_REGISTRY.keys()),
}


def resolve_preset(name: str) -> List[str]:
    """Return metric names for the given preset, or raise ValueError."""
    if name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")
    return list(PRESETS[name])


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

@dataclass
class SkippedMetric:
    name: str
    reason: str


_FIELD_HINTS: Dict[str, str] = {
    "retrieval_context": "set eval_record().retrieval_context in your agent code",
    "expected_output": "add 'expected_output' column to dataset CSV",
    "expected_context": "add 'expected_context' column to dataset CSV",
    "expected_tools": "add 'expected_tools' column to dataset CSV",
    "tool_calls": "use eval_record().add_tool_call() in your agent code",
}


def resolve_runnable_metrics(
    metric_names: List[str],
    available_fields: Set[str],
) -> Tuple[List[MetricSpec], List[SkippedMetric]]:
    """Partition *metric_names* into runnable specs and skipped-with-reason.

    ``available_fields`` is the union of dataset columns present and EvalRecord
    fields that have been populated (e.g. ``{"expected_output", "retrieval_context"}``).
    """
    runnable: List[MetricSpec] = []
    skipped: List[SkippedMetric] = []

    for name in metric_names:
        spec = METRIC_REGISTRY.get(name)
        if spec is None:
            skipped.append(SkippedMetric(name=name, reason=f"unknown metric '{name}'"))
            continue

        missing = spec.required_fields - available_fields
        if missing:
            parts = [
                f"requires {f} \u2014 {_FIELD_HINTS.get(f, f)}" for f in sorted(missing)
            ]
            skipped.append(SkippedMetric(name=name, reason="; ".join(parts)))
        else:
            runnable.append(spec)

    return runnable, skipped
