"""
Default implementation of the execution tracker.

This module provides a simple in-memory implementation of the ExecutionTracker
interface that stores node executions in the current request context.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from .interfaces import ExecutionTracker, NodeExecution
from .context import get_current_context


class DefaultExecutionTracker(ExecutionTracker):
    """Default implementation that stores executions in memory."""

    def __init__(self):
        self._executions: List[NodeExecution] = []

    def start_node_execution(
        self,
        node_id: str,
        node_name: str,
        framework: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NodeExecution:
        """Start tracking a new node execution."""
        execution = NodeExecution(
            node_id=node_id,
            node_name=node_name,
            framework=framework,
            start_time=datetime.now(),
            inputs=inputs,
            metadata=metadata or {},
        )

        self._executions.append(execution)

        # Log the start of execution
        context = get_current_context()
        request_id = context.request_id if context else "unknown"
        print(
            f"[RUNTIME] Request {request_id[:8]} | Started {framework} node: {node_name} (id: {node_id})"
        )

        return execution

    def end_node_execution(
        self,
        node_execution: NodeExecution,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """End tracking for a node execution."""
        node_execution.end_time = datetime.now()
        node_execution.outputs = outputs
        node_execution.error = error

        # Log the completion of execution
        context = get_current_context()
        request_id = context.request_id if context else "unknown"
        status = "ERROR" if error else "COMPLETED"
        duration = node_execution.duration_ms or 0

        print(
            f"[RUNTIME] Request {request_id[:8]} | {status} {node_execution.framework} node: {node_execution.node_name} ({duration:.1f}ms)"
        )

        if error:
            print(f"[RUNTIME] Request {request_id[:8]} | Error details: {error}")

    def get_executions(self) -> List[NodeExecution]:
        """Get all tracked executions for the current request."""
        return self._executions.copy()

    def clear_executions(self) -> None:
        """Clear all tracked executions."""
        self._executions.clear()

    def print_summary(self) -> None:
        """Print a summary of all executions."""
        context = get_current_context()
        if not context:
            print("[RUNTIME] No active request context")
            return

        print(f"\n[RUNTIME] === Request Summary ===")
        print(f"[RUNTIME] Request ID: {context.request_id}")
        print(f"[RUNTIME] Entrypoint: {context.entrypoint_name}")
        print(f"[RUNTIME] Status: {context.status}")
        print(f"[RUNTIME] Duration: {context.duration_ms or 0:.1f}ms")
        print(f"[RUNTIME] Node Executions: {len(self._executions)}")

        if self._executions:
            print(f"[RUNTIME] === Node Details ===")
            for i, execution in enumerate(self._executions, 1):
                status = execution.status.upper()
                duration = execution.duration_ms or 0
                print(
                    f"[RUNTIME] {i:2}. [{execution.framework}] {execution.node_name} - {status} ({duration:.1f}ms)"
                )
                if execution.error:
                    print(f"[RUNTIME]     Error: {execution.error}")

        print(f"[RUNTIME] === End Summary ===\n")
