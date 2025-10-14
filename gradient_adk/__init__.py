"""
Unified Gradient Agent package providing both the SDK (decorator, runtime)
and the CLI (gradient command).
"""

from .decorator import entrypoint, run_server  # SDK exports
from .runtime import get_runtime_manager  # runtime manager accessor
from .streaming import (  # streaming utilities
    StreamingResponse,
    JSONStreamingResponse,
    ServerSentEventsResponse,
    stream_json,
    stream_events,
)

__all__ = [
    "entrypoint",
    "run_server",
    "get_runtime_manager",
    "StreamingResponse",
    "JSONStreamingResponse",
    "ServerSentEventsResponse",
    "stream_json",
    "stream_events",
]

__version__ = "0.2.0"  # bumped for unified package release
