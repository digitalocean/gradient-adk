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

        # Skip individual node logging - only show final summary
        pass

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

        # Skip individual node logging - only show final summary
        pass

    def _is_internal_node(self, node_name: str) -> bool:
        """Check if this is an internal LangGraph node that should be filtered out."""
        internal_patterns = [
            "CompiledStateGraph.invoke",
            "CompiledStateGraph.stream",
            "Pregel.invoke",
            "Pregel.stream",
        ]
        return any(pattern in node_name for pattern in internal_patterns)

    def get_executions(self) -> List[NodeExecution]:
        """Get all tracked executions for the current request."""
        return self._executions.copy()

    def clear_executions(self) -> None:
        """Clear all tracked executions."""
        self._executions.clear()

    def print_summary(self) -> None:
        """Print a concise summary with inputs/outputs and only user-defined nodes."""
        context = get_current_context()
        if not context:
            print("[RUNTIME] No active request context")
            return

        # Filter out internal LangGraph nodes for the summary
        user_executions = [
            exec
            for exec in self._executions
            if not self._is_internal_node(exec.node_name)
        ]

        print(f"\n[RUNTIME] === Request Summary ===")
        print(f"[RUNTIME] Request ID: {context.request_id}")
        print(f"[RUNTIME] Entrypoint: {context.entrypoint_name}")
        print(f"[RUNTIME] Status: {context.status}")
        print(f"[RUNTIME] Duration: {context.duration_ms or 0:.1f}ms")

        # Show request inputs and outputs
        if context.inputs:
            inputs_str = self._format_data(context.inputs, max_length=100)
            print(f"[RUNTIME] Request Input: {inputs_str}")

        if context.outputs:
            outputs_str = self._format_data(context.outputs, max_length=100)
            print(f"[RUNTIME] Request Output: {outputs_str}")

        print(f"[RUNTIME] Node Executions: {len(user_executions)}")

        if user_executions:
            print(f"[RUNTIME] === Node Details ===")
            for i, execution in enumerate(user_executions, 1):
                status = execution.status.upper()
                duration = execution.duration_ms or 0
                print(
                    f"[RUNTIME] {i:2}. [{execution.framework}] {execution.node_name} - {status} ({duration:.1f}ms)"
                )

                # Show node inputs
                if execution.inputs:
                    inputs_str = self._format_data(execution.inputs, max_length=80)
                    print(f"[RUNTIME]     Input: {inputs_str}")

                # Show node outputs or error
                if execution.error:
                    print(f"[RUNTIME]     Error: {execution.error}")
                elif execution.outputs:
                    outputs_str = self._format_data(execution.outputs, max_length=80)
                    print(f"[RUNTIME]     Output: {outputs_str}")

        print(f"[RUNTIME] === End Summary ===\n")

    def _format_data(self, data: Any, max_length: int = 100) -> str:
        """Format data for display, truncating if too long."""
        try:
            data_str = str(data)
            if len(data_str) <= max_length:
                return data_str
            return data_str[: max_length - 3] + "..."
        except:
            return "<unserializable>"
