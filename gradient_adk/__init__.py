"""
Unified Gradient Agent package providing both the SDK (decorator, runtime)
and the CLI (gradient command).
"""

from .decorator import entrypoint
from .tracing import (  # manual tracing decorators
    trace_llm,
    trace_retriever,
    trace_tool,
)

__all__ = [
    "entrypoint",
    "trace_llm",
    "trace_retriever",
    "trace_tool",
]

__version__ = "0.0.5"
