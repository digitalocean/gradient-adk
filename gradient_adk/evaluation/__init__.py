"""
Gradient ADK evaluation module.

Provides EvalRecord for collecting evaluation data within agent code,
and local evaluation support powered by DeepEval.
"""

from .record import eval_record, EvalRecord

__all__ = ["eval_record", "EvalRecord"]
