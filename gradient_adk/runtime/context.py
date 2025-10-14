"""
Request context management for tracking execution across frameworks.

This module provides context management for associating all framework
executions with a specific request/entrypoint call.
"""

import contextvars
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

from .interfaces import ExecutionTracker, NodeExecution


@dataclass
class RequestContext:
    """Context for tracking a single request through the system."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entrypoint_name: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate request duration in milliseconds."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None

    @property
    def status(self) -> str:
        """Get the request status."""
        if self.error:
            return "error"
        elif self.end_time:
            return "completed"
        else:
            return "running"


# Context variable to store the current request context
_current_context: contextvars.ContextVar[Optional[RequestContext]] = (
    contextvars.ContextVar("current_request_context", default=None)
)


def get_current_context() -> Optional[RequestContext]:
    """Get the current request context."""
    return _current_context.get()


def start_request_context(
    entrypoint_name: str,
    inputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RequestContext:
    """Start a new request context."""
    context = RequestContext(
        entrypoint_name=entrypoint_name, inputs=inputs, metadata=metadata or {}
    )
    _current_context.set(context)
    return context


def end_request_context(
    outputs: Optional[Any] = None, error: Optional[str] = None
) -> Optional[RequestContext]:
    """End the current request context."""
    context = get_current_context()
    if context:
        context.end_time = datetime.now()
        # Store outputs as-is, don't require dict type
        context.outputs = outputs
        context.error = error
    return context


def clear_request_context() -> None:
    """Clear the current request context."""
    _current_context.set(None)
