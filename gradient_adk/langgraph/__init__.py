"""
LangGraph integration for gradient-adk.

This module provides LangGraph-specific functionality for gradient-adk,
including graph instrumentation and monitoring.
"""

from ..runtime.manager import attach_graph

__all__ = [
    "attach_graph",
]
