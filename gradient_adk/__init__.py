"""
Unified Gradient Agent package providing both the SDK (decorator, runtime)
and the CLI (gradient command).
"""

from .decorator import entrypoint
from .streaming import (  # streaming utilities
    StreamingResponse,
    JSONStreamingResponse,
    ServerSentEventsResponse,
    stream_json,
    stream_events,
)
from .tracing import (  # manual tracing decorators
    trace_llm,
    trace_retriever,
    trace_tool,
)

__all__ = [
    "entrypoint",
    "StreamingResponse",
    "JSONStreamingResponse",
    "ServerSentEventsResponse",
    "stream_json",
    "stream_events",
    "trace_llm",
    "trace_retriever",
    "trace_tool",
]

__version__ = "0.0.4"
