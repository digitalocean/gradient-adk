"""
Unified Gradient Agent package providing both the SDK (decorator, runtime)
and the CLI (gradient command).
"""

from .decorator import entrypoint, RequestContext
from .tracing import (
    # Decorators
    trace_llm,
    trace_retriever,
    trace_tool,
    # Programmatic span functions
    add_llm_span,
    add_tool_span,
    add_agent_span,
)

__all__ = [
    "entrypoint",
    "RequestContext",
    # Decorators
    "trace_llm",
    "trace_retriever",
    "trace_tool",
    # Programmatic span functions
    "add_llm_span",
    "add_tool_span",
    "add_agent_span",
]

__version__ = "0.0.5"
