"""
Terminal formatting for evaluation results.
"""

from __future__ import annotations

from typing import List

from .runner import EvalResults


def format_results(results: EvalResults, *, verbose: bool = False) -> str:
    """Return a formatted string suitable for terminal display."""
    lines: List[str] = []

    lines.append("")
    lines.append("=" * 60)
    lines.append("  Evaluation Results")
    lines.append("=" * 60)

    # ---- Metrics table ----
    if results.metric_summaries:
        lines.append("")
        lines.append(f"  {'Metric':<28} {'Score':>6}  {'Threshold':>10}  {'Result':>6}")
        lines.append("  " + "-" * 56)
        for ms in results.metric_summaries:
            status = "PASS" if ms.passed else "FAIL"
            lines.append(
                f"  {ms.name:<28} {ms.avg_score:>5.2f}  {ms.threshold:>10.2f}  {status:>6}"
            )

    # ---- Skipped metrics ----
    if results.skipped_metrics:
        lines.append("")
        lines.append("  Skipped (insufficient data):")
        for sm in results.skipped_metrics:
            lines.append(f"    {sm.name:<26} {sm.reason}")

    # ---- Per-row breakdown (verbose) ----
    if verbose and results.test_case_results:
        lines.append("")
        lines.append("-" * 60)
        lines.append("  Per-Row Breakdown")
        lines.append("-" * 60)
        for i, tcr in enumerate(results.test_case_results, start=1):
            input_display = str(tcr.input)
            if len(input_display) > 50:
                input_display = input_display[:47] + "..."
            output_display = tcr.actual_output
            if len(output_display) > 70:
                output_display = output_display[:67] + "..."
            lines.append("")
            lines.append(f"  Row {i}: {input_display}")
            lines.append(f"    Output:  {output_display}")
            lines.append(f"    Latency: {tcr.completion_time:.2f}s")
            for name, score in tcr.metric_scores.items():
                passed = tcr.metric_passed.get(name, False)
                status = "PASS" if passed else "FAIL"
                lines.append(f"    {name:<26} {score:>5.2f}  {status}")

    # ---- Summary footer ----
    lines.append("")
    lines.append("=" * 60)
    n_computed = len(results.metric_summaries)
    n_passed = sum(1 for ms in results.metric_summaries if ms.passed)
    n_skipped = len(results.skipped_metrics)
    n_cases = len(results.test_case_results)

    lines.append(f"  {n_passed}/{n_computed} metrics passed across {n_cases} test case(s)")
    if n_skipped:
        lines.append(f"  {n_skipped} metric(s) skipped (missing required data)")
    lines.append(f"  Total time: {results.total_time:.1f}s")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)
