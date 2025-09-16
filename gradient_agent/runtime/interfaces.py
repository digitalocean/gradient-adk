"""
Core interfaces for the runtime tracking system.

This module defines the abstract interfaces that framework instrumentors
and execution trackers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
import uuid


@dataclass
class NodeExecution:
    """Represents the execution of a single node/step in a framework."""

    node_id: str
    node_name: str
    framework: str  # e.g., "langgraph", "langchain"
    start_time: datetime
    end_time: Optional[datetime] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate duration in milliseconds if both start and end times are set."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None

    @property
    def status(self) -> str:
        """Get the execution status."""
        if self.error:
            return "error"
        elif self.end_time:
            return "completed"
        else:
            return "running"


class ExecutionTracker(ABC):
    """Abstract interface for tracking execution across frameworks."""

    @abstractmethod
    def start_node_execution(
        self,
        node_id: str,
        node_name: str,
        framework: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NodeExecution:
        """Start tracking a new node execution."""
        pass

    @abstractmethod
    def end_node_execution(
        self,
        node_execution: NodeExecution,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """End tracking for a node execution."""
        pass

    @abstractmethod
    def get_executions(self) -> List[NodeExecution]:
        """Get all tracked executions for the current request."""
        pass

    @abstractmethod
    def clear_executions(self) -> None:
        """Clear all tracked executions."""
        pass


class FrameworkInstrumentor(ABC):
    """Abstract interface for instrumenting specific frameworks."""

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Name of the framework this instrumentor handles."""
        pass

    @abstractmethod
    def install(self, tracker: ExecutionTracker) -> None:
        """Install instrumentation hooks for this framework."""
        pass

    @abstractmethod
    def uninstall(self) -> None:
        """Remove instrumentation hooks for this framework."""
        pass

    @abstractmethod
    def is_installed(self) -> bool:
        """Check if instrumentation is currently installed."""
        pass
