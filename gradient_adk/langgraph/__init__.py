"""
LangGraph integration for gradient-agents.

This module provides LangGraph-specific functionality for gradient-agents,
including graph instrumentation and monitoring.
"""

from ..runtime.manager import attach_graph

__all__ = [
    "attach_graph",
]
