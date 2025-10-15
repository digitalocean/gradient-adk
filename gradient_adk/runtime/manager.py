"""
Runtime manager for coordinating instrumentation and tracking.

This module provides the main runtime manager that coordinates
framework instrumentors and execution tracking.
"""

from typing import List, Optional, Dict, Any
import atexit

from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI
from gradient_adk.logging import get_logger

from .interfaces import FrameworkInstrumentor, ExecutionTracker
from .tracker import DefaultExecutionTracker
from .digitalocean_tracker import DigitalOceanTracesTracker
from .langgraph.langgraph_instrumentor import LangGraphInstrumentor
from .context import start_request_context, end_request_context, get_current_context

logger = get_logger(__name__)


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

        # Install instrumentation quietly

        for instrumentor in self._instrumentors:
            try:
                instrumentor.install(self._tracker)
            except Exception as e:
                pass  # Suppress error logging

        self._installed = True
        # Instrumentation installed quietly

    def uninstall_instrumentation(self) -> None:
        """Uninstall all framework instrumentors."""
        if not self._installed:
            return

        # Uninstall instrumentation quietly

        for instrumentor in self._instrumentors:
            try:
                instrumentor.uninstall()
            except Exception as e:
                pass  # Suppress error logging

        self._installed = False
        # Instrumentation uninstalled quietly

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
        # Request started quietly

    def end_request(
        self, outputs: Optional[Any] = None, error: Optional[str] = None
    ) -> None:
        """End tracking for the current request."""
        context = end_request_context(outputs, error)

        if context:
            # Trigger trace submission for DigitalOcean tracker when request completes
            from .digitalocean_tracker import DigitalOceanTracesTracker

            if isinstance(self._tracker, DigitalOceanTracesTracker):
                # Create a dummy node execution to trigger trace submission
                from .interfaces import NodeExecution
                from datetime import datetime

                dummy_execution = NodeExecution(
                    node_id="request-complete",
                    node_name="RequestComplete",
                    framework="runtime",
                    start_time=datetime.now(),
                    metadata={"internal": True},
                )
                self._tracker.end_node_execution(dummy_execution, outputs)

            # Print summary if we're using the default tracker
            # if isinstance(self._tracker, DefaultExecutionTracker):
            # self._tracker.print_summary()

    def get_tracker(self) -> ExecutionTracker:
        """Get the current execution tracker."""
        return self._tracker

    def attach_to_graph(self, graph_like: Any) -> None:
        """
        Attach all installed instrumentors that support `attach_to_graph`
        to the provided graph-like object *before* graph.compile().

        Safe to call multiple times (idempotent). Never raises.
        """
        # Ensure instrumentors are installed/ready
        if not self._installed:
            self.install_instrumentation()

        for instrumentor in self._instrumentors:
            attach = getattr(instrumentor, "attach_to_graph", None)
            if callable(attach):
                try:
                    attach(graph_like)
                except Exception:
                    # Never break user code on instrumentation errors
                    pass

    async def run_entrypoint(self, func, data, context):
        """Run an entrypoint function with proper runtime tracking."""
        import inspect

        # Start request tracking
        self.start_request(func.__name__, inputs={"data": data})

        try:
            # Call the function
            if inspect.iscoroutinefunction(func):
                result = await func(data, context)
            else:
                result = func(data, context)

            # End request tracking with success
            self.end_request(outputs=result)
            return result

        except Exception as e:
            # End request tracking with error
            self.end_request(error=str(e))
            raise

    def shutdown(self) -> None:
        """Shutdown the runtime manager."""
        self.uninstall_instrumentation()


# Global runtime manager instance
_runtime_manager: Optional[RuntimeManager] = None


def _looks_like_stream(result: Any) -> bool:
    try:
        content = getattr(result, "content", None)
        return bool(content) and (
            hasattr(content, "__iter__") or hasattr(content, "__aiter__")
        )
    except Exception:
        return False


def _try_auto_configure_traces() -> bool:
    """Try to auto-configure DigitalOcean traces from API token and agent config."""
    import os
    import yaml
    from pathlib import Path

    global _runtime_manager

    logger.info("Attempting to auto-configure DigitalOcean traces")

    # Only need API token from environment
    api_token = os.getenv("DIGITALOCEAN_API_TOKEN")
    if not api_token:
        logger.info("No DIGITALOCEAN_API_TOKEN found in environment")
        return False

    logger.info(
        "Found DIGITALOCEAN_API_TOKEN in environment"
    )  # Load agent configuration from .gradient/agent.yml
    try:
        config_file = Path.cwd() / ".gradient" / "agent.yml"
        if not config_file.exists():
            logger.info(f"Agent config file not found: {config_file}")
            return False

        logger.info(f"Loading agent config from: {config_file}")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        agent_name = config.get("agent_name")
        agent_environment = config.get("agent_environment", "default")

        if not agent_name:
            logger.debug("No agent_name found in config")
            return False

        logger.debug(
            f"Auto-configuring traces for agent: {agent_name}/{agent_environment}"
        )

        # Create DigitalOcean tracker
        do_client = AsyncDigitalOceanGenAI(api_token=api_token)
        tracker = DigitalOceanTracesTracker(
            client=do_client,
            agent_workspace_name=agent_name,
            agent_deployment_name=agent_environment,
            enable_auto_submit=True,
        )

        # Create new runtime manager with the tracker
        _runtime_manager = RuntimeManager(tracker=tracker)
        logger.info(
            f"✅ Auto-configured DigitalOcean traces for {agent_name}/{agent_environment}"
        )
        return True

    except Exception as e:
        logger.error("Failed to auto-configure traces", error=str(e), exc_info=True)
        return False


def get_runtime_manager() -> RuntimeManager:
    """Get the global runtime manager instance."""
    global _runtime_manager
    if _runtime_manager is None:
        # Try to auto-configure DigitalOcean traces
        if not _try_auto_configure_traces():
            # If auto-configuration failed, use default tracker
            _runtime_manager = RuntimeManager()
        else:
            tracker_type = type(_runtime_manager.get_tracker()).__name__
            logger.debug("Using tracker", tracker_type=tracker_type)
    return _runtime_manager


def configure_digitalocean_traces(
    api_token: str,
    agent_workspace_name: Optional[str] = None,
    agent_deployment_name: Optional[str] = None,
    enable_auto_submit: bool = True,
) -> RuntimeManager:
    """Configure runtime to use DigitalOcean traces tracker.

    Args:
        api_token: DigitalOcean API token
        agent_workspace_name: Name of the agent workspace (uses agent config if None)
        agent_deployment_name: Name of the agent deployment (uses agent config if None)
        enable_auto_submit: Whether to auto-submit traces on request end

    Returns:
        The configured runtime manager
    """
    import yaml
    from pathlib import Path

    global _runtime_manager

    # Load from agent config if not provided
    if agent_workspace_name is None or agent_deployment_name is None:
        try:
            config_file = Path.cwd() / ".gradient" / "agent.yml"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)

                if agent_workspace_name is None:
                    agent_workspace_name = config.get("agent_name")
                if agent_deployment_name is None:
                    agent_deployment_name = config.get("agent_environment")
        except Exception:
            pass  # Ignore config loading errors

    # Create DigitalOcean tracker
    do_client = AsyncDigitalOceanGenAI(api_token=api_token)
    tracker = DigitalOceanTracesTracker(
        client=do_client,
        agent_workspace_name=agent_workspace_name,
        agent_deployment_name=agent_deployment_name,
        enable_auto_submit=enable_auto_submit,
    )

    # Create new runtime manager with the tracker
    _runtime_manager = RuntimeManager(tracker=tracker)

    return _runtime_manager


def attach_graph(graph_like: Any) -> None:
    """
    Convenience wrapper so agents can do:
        from gradient_agents.runtime.manager import attach_graph
        attach_graph(graph)
        workflow = graph.compile()
    """
    manager = get_runtime_manager()
    manager.attach_to_graph(graph_like)
