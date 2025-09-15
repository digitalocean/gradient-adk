"""
LangGraph framework instrumentor.

This module provides instrumentation for LangGraph nodes to track their execution
within the runtime system.
"""

import functools
import importlib
from typing import Any, Dict, Optional, Callable
import uuid

from .interfaces import FrameworkInstrumentor, ExecutionTracker
from .context import get_current_context


class LangGraphInstrumentor(FrameworkInstrumentor):
    """Instrumentor for LangGraph framework."""

    def __init__(self):
        self._tracker: Optional[ExecutionTracker] = None
        self._original_functions: Dict[str, Callable] = {}
        self._installed = False

    @property
    def framework_name(self) -> str:
        """Name of the framework this instrumentor handles."""
        return "langgraph"

    def install(self, tracker: ExecutionTracker) -> None:
        """Install instrumentation hooks for LangGraph."""
        if self._installed:
            return

        self._tracker = tracker

        try:
            # Try to import LangGraph modules
            langgraph_core = importlib.import_module("langgraph.graph")

            # Instrument CompiledStateGraph.invoke if it exists
            if hasattr(langgraph_core, "CompiledStateGraph"):
                self._instrument_compiled_graph(langgraph_core.CompiledStateGraph)

            # Also try the state module which may have the compiled graph
            try:
                langgraph_state = importlib.import_module("langgraph.graph.state")
                if hasattr(langgraph_state, "CompiledStateGraph"):
                    self._instrument_compiled_graph(langgraph_state.CompiledStateGraph)
            except ImportError:
                pass

            # Try to instrument other common LangGraph classes
            try:
                langgraph_pregel = importlib.import_module("langgraph.pregel")
                if hasattr(langgraph_pregel, "Pregel"):
                    self._instrument_pregel(langgraph_pregel.Pregel)
            except ImportError:
                pass

            self._installed = True
            print(f"[RUNTIME] Installed {self.framework_name} instrumentation")

        except ImportError:
            print(f"[RUNTIME] LangGraph not found, skipping instrumentation")

    def uninstall(self) -> None:
        """Remove instrumentation hooks for LangGraph."""
        if not self._installed:
            return

        # Restore original functions
        for module_attr, original_func in self._original_functions.items():
            module_name, attr_name = module_attr.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                setattr(module, attr_name, original_func)
            except (ImportError, AttributeError):
                pass

        self._original_functions.clear()
        self._tracker = None
        self._installed = False
        print(f"[RUNTIME] Uninstalled {self.framework_name} instrumentation")

    def is_installed(self) -> bool:
        """Check if instrumentation is currently installed."""
        return self._installed

    def _instrument_compiled_graph(self, compiled_graph_class: type) -> None:
        """Instrument CompiledStateGraph class methods."""
        # Instrument invoke method
        if hasattr(compiled_graph_class, "invoke"):
            original_invoke = compiled_graph_class.invoke
            class_name = compiled_graph_class.__name__
            self._original_functions[f"langgraph.graph.{class_name}.invoke"] = (
                original_invoke
            )

            # Create a closure that captures the instrumentor instance
            instrumentor = self

            def instrumented_invoke(graph_instance, *args, **kwargs):
                return instrumentor._instrument_node_execution(
                    original_invoke,
                    graph_instance,
                    "invoke",
                    f"{class_name}.invoke",
                    *args,
                    **kwargs,
                )

            compiled_graph_class.invoke = instrumented_invoke

        # Instrument stream method if it exists
        if hasattr(compiled_graph_class, "stream"):
            original_stream = compiled_graph_class.stream
            class_name = compiled_graph_class.__name__
            self._original_functions[f"langgraph.graph.{class_name}.stream"] = (
                original_stream
            )

            # Create a closure that captures the instrumentor instance
            instrumentor = self

            def instrumented_stream(graph_instance, *args, **kwargs):
                return instrumentor._instrument_node_execution(
                    original_stream,
                    graph_instance,
                    "stream",
                    f"{class_name}.stream",
                    *args,
                    **kwargs,
                )

            compiled_graph_class.stream = instrumented_stream

    def _instrument_pregel(self, pregel_class: type) -> None:
        """Instrument Pregel class methods."""
        # Instrument invoke method
        if hasattr(pregel_class, "invoke"):
            original_invoke = pregel_class.invoke
            self._original_functions["langgraph.pregel.Pregel.invoke"] = original_invoke

            # Create a closure that captures the instrumentor instance
            instrumentor = self

            def instrumented_invoke(pregel_instance, *args, **kwargs):
                return instrumentor._instrument_node_execution(
                    original_invoke,
                    pregel_instance,
                    "invoke",
                    "Pregel.invoke",
                    *args,
                    **kwargs,
                )

            pregel_class.invoke = instrumented_invoke

    def _instrument_node_execution(
        self,
        original_func: Callable,
        instance: Any,
        method_name: str,
        node_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """Wrap a node execution with tracking."""
        # Only track if we have an active request context
        context = get_current_context()
        if not context or not self._tracker:
            return original_func(instance, *args, **kwargs)

        # Generate a unique node ID for this execution
        node_id = str(uuid.uuid4())

        # Extract inputs for tracking
        inputs = {}
        if args:
            inputs["args"] = args
        if kwargs:
            inputs["kwargs"] = kwargs

        # Start tracking
        execution = self._tracker.start_node_execution(
            node_id=node_id,
            node_name=node_name,
            framework=self.framework_name,
            inputs=inputs,
            metadata={"method": method_name, "class": instance.__class__.__name__},
        )

        try:
            # Execute the original function
            result = original_func(instance, *args, **kwargs)

            # Track successful completion
            outputs = (
                {"result": str(result)[:200]} if result else None
            )  # Truncate for logging
            self._tracker.end_node_execution(execution, outputs=outputs)

            return result

        except Exception as e:
            # Track error
            self._tracker.end_node_execution(execution, error=str(e))
            raise
