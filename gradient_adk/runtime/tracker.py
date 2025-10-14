"""
Default implementation of the execution tracker.

This module provides a simple in-memory implementation of the ExecutionTracker
interface that stores node executions in the current request context.
"""

from datetime import datetime
import os
from typing import List, Optional, Dict, Any
import uuid

from gradient_adk.logging import get_logger
from .interfaces import ExecutionTracker, NodeExecution
from .context import get_current_context

logger = get_logger(__name__)


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

    def _is_internal_node(self, execution: NodeExecution) -> bool:
        """Check if this is an internal framework node that should be filtered out.

        Uses metadata to determine if a node is internal, allowing each framework
        instrumentor to mark their own internal nodes appropriately.
        """
        # Check metadata for internal flag (set by instrumentors)
        return execution.metadata.get("internal", False)

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
            logger.debug("No active request context")
            return

        # Filter out internal framework nodes for the summary
        user_executions = [
            exec for exec in self._executions if not self._is_internal_node(exec)
        ]

        # Only show verbose summary if GRADIENT_VERBOSE is enabled
        if not os.getenv("GRADIENT_VERBOSE") == "1":
            return

        logger.info(
            "=== Request Summary ===",
            request_id=context.request_id,
            entrypoint=context.entrypoint_name,
            status=context.status,
            duration_ms=context.duration_ms or 0,
            node_count=len(user_executions),
        )

        # Show request inputs and outputs
        if context.inputs:
            inputs_str = self._format_data(context.inputs, max_length=100)
            logger.debug("Request Input", input=inputs_str)

        if context.outputs:
            outputs_str = self._format_data(context.outputs, max_length=100)
            logger.debug("Request Output", output=outputs_str)

        if user_executions:
            logger.debug("=== Node Details ===")
            for i, execution in enumerate(user_executions, 1):
                status = execution.status.upper()
                duration = execution.duration_ms or 0
                logger.debug(
                    f"{i:2}. [{execution.framework}] {execution.node_name}",
                    status=status,
                    duration_ms=duration,
                )

                # Show node inputs
                if execution.inputs:
                    inputs_str = self._format_data(execution.inputs, max_length=80)
                    logger.debug(
                        "Node Input", node=execution.node_name, input=inputs_str
                    )

                # Show node outputs or error
                if execution.error:
                    logger.debug(
                        "Node Error", node=execution.node_name, error=execution.error
                    )
                elif execution.outputs:
                    outputs_str = self._format_data(execution.outputs, max_length=80)
                    logger.debug(
                        "Node Output", node=execution.node_name, output=outputs_str
                    )

        logger.debug("=== End Summary ===")

    def _format_data(self, data: Any, max_length: int = 100) -> str:
        """Format data for display, truncating if too long."""
        try:
            data_str = str(data)
            if len(data_str) <= max_length:
                return data_str
            return data_str[: max_length - 3] + "..."
        except:
            return "<unserializable>"
