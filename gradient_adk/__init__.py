"""
Unified Gradient Agent package providing both the SDK (decorator, runtime)
and the CLI (gradient command).
"""

from .decorator import entrypoint
from .tracing import (  # manual tracing decorators and functions
    trace_llm,
    trace_retriever,
    trace_tool,
    add_llm_span,
    add_retriever_span,
    add_tool_span,
)

__all__ = [
    "entrypoint",
    # Decorators
    "trace_llm",
    "trace_retriever",
    "trace_tool",
    # Functions
    "add_llm_span",
    "add_retriever_span",
    "add_tool_span",
]

__version__ = "0.0.5"
