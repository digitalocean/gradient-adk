"""
Runtime manager for coordinating instrumentation and tracking.

This module provides the main runtime manager that coordinates
framework instrumentors and execution tracking.
"""

from typing import List, Optional, Dict, Any
import atexit

from .interfaces import FrameworkInstrumentor, ExecutionTracker
from .tracker import DefaultExecutionTracker
from .langgraph_instrumentor import LangGraphInstrumentor
from .context import start_request_context, end_request_context, get_current_context


class RuntimeManager:
    """Manages framework instrumentation and execution tracking."""

    def __init__(self, tracker: Optional[ExecutionTracker] = None):
        self._tracker = tracker or DefaultExecutionTracker()
        self._instrumentors: List[FrameworkInstrumentor] = []
        self._installed = False

        # Register default instrumentors
        self._register_default_instrumentors()

        # Ensure cleanup on exit
        atexit.register(self.shutdown)

    def _register_default_instrumentors(self) -> None:
        """Register default framework instrumentors."""
        self._instrumentors = [
            LangGraphInstrumentor(),
            # Add more instrumentors here as they're implemented
        ]

    def install_instrumentation(self) -> None:
        """Install all framework instrumentors."""
        if self._installed:
            return

        print("[RUNTIME] Installing framework instrumentation...")

        for instrumentor in self._instrumentors:
            try:
                instrumentor.install(self._tracker)
            except Exception as e:
                print(
                    f"[RUNTIME] Failed to install {instrumentor.framework_name} instrumentation: {e}"
                )

        self._installed = True
        print("[RUNTIME] Framework instrumentation installed")

    def uninstall_instrumentation(self) -> None:
        """Uninstall all framework instrumentors."""
        if not self._installed:
            return

        print("[RUNTIME] Uninstalling framework instrumentation...")

        for instrumentor in self._instrumentors:
            try:
                instrumentor.uninstall()
            except Exception as e:
                print(
                    f"[RUNTIME] Failed to uninstall {instrumentor.framework_name} instrumentation: {e}"
                )

        self._installed = False
        print("[RUNTIME] Framework instrumentation uninstalled")

    def start_request(
        self,
        entrypoint_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start tracking a new request."""
        # Ensure instrumentation is installed
        if not self._installed:
            self.install_instrumentation()

        # Clear any previous executions
        self._tracker.clear_executions()

        # Start request context
        context = start_request_context(entrypoint_name, inputs, metadata)
        print(
            f"[RUNTIME] Started request {context.request_id[:8]} for entrypoint: {entrypoint_name}"
        )

    def end_request(
        self, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None
    ) -> None:
        """End tracking for the current request."""
        context = end_request_context(outputs, error)

        if context:
            status = "ERROR" if error else "COMPLETED"
            duration = context.duration_ms or 0
            print(
                f"[RUNTIME] {status} request {context.request_id[:8]} ({duration:.1f}ms)"
            )

            # Print summary if we're using the default tracker
            if isinstance(self._tracker, DefaultExecutionTracker):
                self._tracker.print_summary()

    def get_tracker(self) -> ExecutionTracker:
        """Get the current execution tracker."""
        return self._tracker

    def shutdown(self) -> None:
        """Shutdown the runtime manager."""
        self.uninstall_instrumentation()


# Global runtime manager instance
_runtime_manager: Optional[RuntimeManager] = None


def get_runtime_manager() -> RuntimeManager:
    """Get the global runtime manager instance."""
    global _runtime_manager
    if _runtime_manager is None:
        _runtime_manager = RuntimeManager()
    return _runtime_manager


def install_runtime() -> None:
    """Install runtime instrumentation globally."""
    manager = get_runtime_manager()
    manager.install_instrumentation()


def uninstall_runtime() -> None:
    """Uninstall runtime instrumentation globally."""
    manager = get_runtime_manager()
    manager.uninstall_instrumentation()
