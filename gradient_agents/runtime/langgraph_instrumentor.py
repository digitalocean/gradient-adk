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
            # Try to import LangGraph modules - handle different import paths
            langgraph_core = None
            try:
                langgraph_core = importlib.import_module("langgraph.graph")
            except ImportError:
                # Try alternative import path
                langgraph_core = importlib.import_module("langgraph")

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

            # Try to instrument node execution at a deeper level
            try:
                self._instrument_node_functions()
            except Exception as e:
                pass  # Suppress error logging

            # Try to instrument the actual node execution mechanism
            try:
                self._instrument_pregel_node_execution()
            except Exception as e:
                pass  # Suppress error logging

            self._installed = True
            # Suppress installation logging

        except ImportError:
            pass  # Suppress import error logging

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
        # Uninstalled quietly

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

    def _instrument_node_functions(self) -> None:
        """Try to instrument individual node functions by hooking into common execution patterns."""
        try:
            # Try to hook into the node execution mechanism
            langgraph_constants = importlib.import_module("langgraph.constants")
            if hasattr(langgraph_constants, "INVOKE"):
                # This is where individual node functions get called
                print("[LANGGRAPH] Found langgraph constants for node instrumentation")
        except ImportError:
            pass

        # Alternative: try to hook into the runnable interface that nodes often use
        try:
            langchain_core = importlib.import_module("langchain_core.runnables")
            if hasattr(langchain_core, "Runnable"):
                # Many LangGraph nodes inherit from Runnable
                self._instrument_runnable_invoke(langchain_core.Runnable)
        except ImportError:
            pass

    def _instrument_runnable_invoke(self, runnable_class: type) -> None:
        """Instrument Runnable.invoke to catch individual node executions."""
        if hasattr(runnable_class, "invoke"):
            original_invoke = runnable_class.invoke
            self._original_functions["langchain_core.runnables.Runnable.invoke"] = (
                original_invoke
            )

            instrumentor = self

            def instrumented_runnable_invoke(runnable_instance, *args, **kwargs):
                # Get the function name from the runnable instance
                node_name = (
                    getattr(runnable_instance, "name", None)
                    or getattr(
                        runnable_instance, "__class__", type(runnable_instance)
                    ).__name__
                )

                return instrumentor._instrument_node_execution(
                    original_invoke,
                    runnable_instance,
                    "invoke",
                    f"Node.{node_name}",
                    *args,
                    **kwargs,
                )

            runnable_class.invoke = instrumented_runnable_invoke

    def _instrument_pregel_node_execution(self) -> None:
        """Instrument the actual node execution mechanism in Pregel."""
        try:
            # Try to hook into the internal node execution
            langgraph_pregel = importlib.import_module("langgraph.pregel")

            # Look for the actual node execution methods
            if hasattr(langgraph_pregel, "Pregel"):
                pregel_class = langgraph_pregel.Pregel

                # Instrument _tick method which executes individual nodes
                if hasattr(pregel_class, "_tick"):
                    original_tick = pregel_class._tick
                    self._original_functions["langgraph.pregel.Pregel._tick"] = (
                        original_tick
                    )

                    instrumentor = self

                    def instrumented_tick(pregel_instance, *args, **kwargs):
                        # Let the original _tick run but try to capture node details
                        return instrumentor._instrument_tick_execution(
                            original_tick, pregel_instance, *args, **kwargs
                        )

                    pregel_class._tick = instrumented_tick

                # Also try to instrument _atick for async execution
                if hasattr(pregel_class, "_atick"):
                    original_atick = pregel_class._atick
                    self._original_functions["langgraph.pregel.Pregel._atick"] = (
                        original_atick
                    )

                    instrumentor = self

                    async def instrumented_atick(pregel_instance, *args, **kwargs):
                        return await instrumentor._instrument_async_tick_execution(
                            original_atick, pregel_instance, *args, **kwargs
                        )

                    pregel_class._atick = instrumented_atick

        except ImportError:
            pass

        # Alternative approach: try to hook into node execution at a lower level
        try:
            self._instrument_node_callable_execution()
        except Exception as e:
            print(f"[RUNTIME] Could not instrument callable execution: {e}")

        # Try to hook into StateGraph node additions to track node names
        try:
            self._instrument_state_graph_nodes()
        except Exception as e:
            print(f"[RUNTIME] Could not instrument StateGraph nodes: {e}")

    def _instrument_state_graph_nodes(self) -> None:
        """Instrument StateGraph.add_node to track node names and their functions."""
        try:
            langgraph_graph = importlib.import_module("langgraph.graph")

            # Look for StateGraph class
            if hasattr(langgraph_graph, "StateGraph"):
                state_graph_class = langgraph_graph.StateGraph

                # Instrument add_node method
                if hasattr(state_graph_class, "add_node"):
                    original_add_node = state_graph_class.add_node
                    self._original_functions["langgraph.graph.StateGraph.add_node"] = (
                        original_add_node
                    )

                    # Store node mappings for later reference
                    if not hasattr(self, "_node_mappings"):
                        self._node_mappings = {}

                    instrumentor = self

                    def instrumented_add_node(
                        graph_instance, key, action=None, **kwargs
                    ):
                        # Store the mapping of node name to function
                        instrumentor._node_mappings[key] = {
                            "action": action,
                            "graph_instance": graph_instance,
                            "kwargs": kwargs,
                        }

                        # If the action is callable, instrument it directly
                        if callable(action) and hasattr(action, "__call__"):
                            wrapped_action = instrumentor._create_node_wrapper(
                                key, action
                            )
                            return original_add_node(
                                graph_instance, key, wrapped_action, **kwargs
                            )

                        return original_add_node(graph_instance, key, action, **kwargs)

                    state_graph_class.add_node = instrumented_add_node

        except ImportError:
            pass

        # Also try to hook into the execution internals more directly
        try:
            self._instrument_pregel_execution_internals()
        except Exception as e:
            print(f"[RUNTIME] Could not instrument pregel internals: {e}")

    def _instrument_pregel_execution_internals(self) -> None:
        """Try to instrument the internal execution mechanisms of Pregel."""
        try:
            langgraph_pregel = importlib.import_module("langgraph.pregel")

            if hasattr(langgraph_pregel, "Pregel"):
                pregel_class = langgraph_pregel.Pregel

                # Look for internal step execution methods
                for method_name in [
                    "_execute_step",
                    "_aexecute_step",
                    "_step",
                    "_astep",
                ]:
                    if hasattr(pregel_class, method_name):
                        original_method = getattr(pregel_class, method_name)
                        self._original_functions[
                            f"langgraph.pregel.Pregel.{method_name}"
                        ] = original_method

                        if method_name.startswith("_a"):  # async method
                            setattr(
                                pregel_class,
                                method_name,
                                self._create_async_step_wrapper(
                                    method_name, original_method
                                ),
                            )
                        else:  # sync method
                            setattr(
                                pregel_class,
                                method_name,
                                self._create_step_wrapper(method_name, original_method),
                            )

                        print(
                            f"[LANGGRAPH] Instrumented {method_name} for node-level tracking"
                        )

        except ImportError:
            pass

    def _create_step_wrapper(
        self, method_name: str, original_method: Callable
    ) -> Callable:
        """Create a wrapper for step execution methods."""
        instrumentor = self

        def wrapped_step_method(pregel_instance, *args, **kwargs):
            # Try to extract node information from step arguments
            node_name = "unknown_step"
            if args:
                # Look for node identifiers in the arguments
                for arg in args[:3]:  # Check first few args
                    if isinstance(arg, str) and len(arg) < 50:  # Likely a node name
                        node_name = f"Step.{arg}"
                        break
                    elif isinstance(arg, dict) and "node" in arg:
                        node_name = f"Step.{arg['node']}"
                        break
                    elif hasattr(arg, "name"):
                        node_name = f"Step.{arg.name}"
                        break

            return instrumentor._instrument_node_execution(
                original_method,
                pregel_instance,
                method_name,
                node_name,
                *args,
                **kwargs,
            )

        return wrapped_step_method

    def _create_async_step_wrapper(
        self, method_name: str, original_method: Callable
    ) -> Callable:
        """Create a wrapper for async step execution methods."""
        instrumentor = self

        async def wrapped_async_step_method(pregel_instance, *args, **kwargs):
            # Try to extract node information from step arguments
            node_name = "unknown_async_step"
            if args:
                for arg in args[:3]:
                    if isinstance(arg, str) and len(arg) < 50:
                        node_name = f"AsyncStep.{arg}"
                        break
                    elif isinstance(arg, dict) and "node" in arg:
                        node_name = f"AsyncStep.{arg['node']}"
                        break
                    elif hasattr(arg, "name"):
                        node_name = f"AsyncStep.{arg.name}"
                        break

            # Handle async execution with proper instrumentation
            context = get_current_context()
            if not context or not instrumentor._tracker:
                return await original_method(pregel_instance, *args, **kwargs)

            node_id = str(uuid.uuid4())
            import time

            start_time = time.time()

            execution = instrumentor._tracker.start_node_execution(
                node_id=node_id,
                node_name=node_name,
                framework=instrumentor.framework_name,
                inputs={"args": args[:2], "kwargs": kwargs} if args or kwargs else {},
                metadata={
                    "method": method_name,
                    "class": pregel_instance.__class__.__name__,
                },
            )

            try:
                result = await original_method(pregel_instance, *args, **kwargs)

                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                print(
                    f"[LANGGRAPH] Completed async step: {node_name} ({latency_ms:.1f}ms)"
                )
                instrumentor._tracker.end_node_execution(
                    execution, outputs={"result": str(result)[:200]}
                )

                return result

            except Exception as e:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                print(
                    f"[LANGGRAPH] ERROR in async step: {node_name} ({latency_ms:.1f}ms) - {str(e)}"
                )
                instrumentor._tracker.end_node_execution(execution, error=str(e))
                raise

        return wrapped_async_step_method

    def _create_node_wrapper(
        self, node_name: str, original_function: Callable
    ) -> Callable:
        """Create a wrapper for a node function to add instrumentation."""
        instrumentor = self

        def wrapped_node_function(*args, **kwargs):
            # Only track if we have an active request context
            context = get_current_context()
            if not context or not instrumentor._tracker:
                return original_function(*args, **kwargs)

            # Generate a unique node ID for this execution
            node_id = str(uuid.uuid4())

            # Extract inputs for tracking - capture more detail
            inputs = {}
            if args:
                # Convert args to serializable format for logging
                serialized_args = []
                for arg in args:
                    try:
                        if isinstance(arg, (str, int, float, bool, list, dict)):
                            serialized_args.append(arg)
                        else:
                            serialized_args.append(
                                str(arg)[:200]
                            )  # Truncate long objects
                    except:
                        serialized_args.append("<unserializable>")
                inputs["args"] = serialized_args
            if kwargs:
                # Convert kwargs to serializable format
                serialized_kwargs = {}
                for k, v in kwargs.items():
                    try:
                        if isinstance(v, (str, int, float, bool, list, dict)):
                            serialized_kwargs[k] = v
                        else:
                            serialized_kwargs[k] = str(v)[:200]  # Truncate long objects
                    except:
                        serialized_kwargs[k] = "<unserializable>"
                inputs["kwargs"] = serialized_kwargs

            # Start tracking with timestamp
            import time

            start_time = time.time()

            execution = instrumentor._tracker.start_node_execution(
                node_id=node_id,
                node_name=f"Node.{node_name}",
                framework=instrumentor.framework_name,
                inputs=inputs,
                metadata={
                    "method": "invoke",
                    "node_name": node_name,
                    "start_time": start_time,
                },
            )

            try:
                # Execute the original function with the original arguments
                result = original_function(*args, **kwargs)

                # Calculate latency
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                # Track successful completion with detailed outputs
                outputs = None
                if result:
                    try:
                        if isinstance(result, (str, int, float, bool, list, dict)):
                            outputs = {"result": result}
                        else:
                            outputs = {
                                "result": str(result)[:500]
                            }  # More detail for outputs
                    except:
                        outputs = {"result": "<unserializable>"}

                instrumentor._tracker.end_node_execution(execution, outputs=outputs)
                return result

            except Exception as e:
                # Calculate latency for error case too
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                # Track error
                instrumentor._tracker.end_node_execution(execution, error=str(e))
                raise

        # Preserve function metadata
        wrapped_node_function.__name__ = getattr(
            original_function, "__name__", node_name
        )
        wrapped_node_function.__doc__ = getattr(original_function, "__doc__", None)
        wrapped_node_function._original_function = original_function
        wrapped_node_function._node_name = node_name

        return wrapped_node_function

    def _instrument_node_callable_execution(self) -> None:
        """Try to instrument the actual callable execution of nodes."""
        try:
            # Import concurrent.futures to hook into the execution
            import concurrent.futures

            # Store original submit method
            original_submit = concurrent.futures.ThreadPoolExecutor.submit
            self._original_functions["concurrent.futures.ThreadPoolExecutor.submit"] = (
                original_submit
            )

            instrumentor = self

            def instrumented_submit(executor_self, fn, *args, **kwargs):
                # Check if this looks like a LangGraph node execution
                if hasattr(fn, "__name__") and (
                    # Common patterns for LangGraph node functions
                    "node" in fn.__name__.lower()
                    or hasattr(fn, "__self__")
                    and hasattr(fn.__self__, "name")
                ):
                    # Wrap the function to add instrumentation
                    def wrapped_fn(*fn_args, **fn_kwargs):
                        node_name = getattr(fn, "__name__", "unknown_node")
                        if hasattr(fn, "__self__") and hasattr(fn.__self__, "name"):
                            node_name = fn.__self__.name

                        return instrumentor._instrument_node_execution(
                            fn, fn, "invoke", f"Node.{node_name}", *fn_args, **fn_kwargs
                        )

                    return original_submit(executor_self, wrapped_fn, *args, **kwargs)

                return original_submit(executor_self, fn, *args, **kwargs)

            concurrent.futures.ThreadPoolExecutor.submit = instrumented_submit

        except Exception as e:
            print(f"[RUNTIME] Could not instrument ThreadPoolExecutor: {e}")

    def _instrument_tick_execution(
        self, original_tick, pregel_instance, *args, **kwargs
    ):
        """Instrument the _tick method to capture individual node executions."""
        # Try to extract node information from the tick arguments
        node_info = "unknown_tick"
        if args and len(args) > 0:
            # The first argument might contain node information
            first_arg = args[0]
            if hasattr(first_arg, "name"):
                node_info = f"Tick.{first_arg.name}"
            elif isinstance(first_arg, dict) and "node" in first_arg:
                node_info = f"Tick.{first_arg['node']}"

        return self._instrument_node_execution(
            original_tick, pregel_instance, "_tick", node_info, *args, **kwargs
        )

    async def _instrument_async_tick_execution(
        self, original_atick, pregel_instance, *args, **kwargs
    ):
        """Instrument the async _atick method to capture individual node executions."""
        node_info = "unknown_async_tick"
        if args and len(args) > 0:
            first_arg = args[0]
            if hasattr(first_arg, "name"):
                node_info = f"AsyncTick.{first_arg.name}"
            elif isinstance(first_arg, dict) and "node" in first_arg:
                node_info = f"AsyncTick.{first_arg['node']}"

        # For async, we need to handle it differently
        context = get_current_context()
        if not context or not self._tracker:
            return await original_atick(pregel_instance, *args, **kwargs)

        node_id = str(uuid.uuid4())
        inputs = {"args": args, "kwargs": kwargs} if args or kwargs else {}

        import time

        start_time = time.time()

        execution = self._tracker.start_node_execution(
            node_id=node_id,
            node_name=node_info,
            framework=self.framework_name,
            inputs=inputs,
            metadata={"method": "_atick", "class": pregel_instance.__class__.__name__},
        )

        try:
            result = await original_atick(pregel_instance, *args, **kwargs)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            print(f"[LANGGRAPH] Completed async node: {node_info} ({latency_ms:.1f}ms)")
            self._tracker.end_node_execution(
                execution, outputs={"result": str(result)[:200]}
            )

            return result

        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            print(
                f"[LANGGRAPH] ERROR in async node: {node_info} ({latency_ms:.1f}ms) - {str(e)}"
            )
            self._tracker.end_node_execution(execution, error=str(e))
            raise

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

        # Extract inputs for tracking - capture more detail
        inputs = {}
        if args:
            # Convert args to serializable format for logging
            serialized_args = []
            for arg in args:
                try:
                    if isinstance(arg, (str, int, float, bool, list, dict)):
                        serialized_args.append(arg)
                    else:
                        serialized_args.append(str(arg)[:200])  # Truncate long objects
                except:
                    serialized_args.append("<unserializable>")
            inputs["args"] = serialized_args
        if kwargs:
            # Convert kwargs to serializable format
            serialized_kwargs = {}
            for k, v in kwargs.items():
                try:
                    if isinstance(v, (str, int, float, bool, list, dict)):
                        serialized_kwargs[k] = v
                    else:
                        serialized_kwargs[k] = str(v)[:200]  # Truncate long objects
                except:
                    serialized_kwargs[k] = "<unserializable>"
            inputs["kwargs"] = serialized_kwargs

        # Only log for non-internal nodes to reduce noise
        if not any(
            internal in node_name for internal in ["CompiledStateGraph", "Pregel"]
        ):
            print(f"[LANGGRAPH] Starting node: {node_name}")
            print(f"[LANGGRAPH] Node inputs: {inputs}")

        # Start tracking with timestamp
        import time

        start_time = time.time()

        execution = self._tracker.start_node_execution(
            node_id=node_id,
            node_name=node_name,
            framework=self.framework_name,
            inputs=inputs,
            metadata={
                "method": method_name,
                "class": instance.__class__.__name__,
                "start_time": start_time,
            },
        )

        try:
            # Execute the original function
            result = original_func(instance, *args, **kwargs)

            # Calculate latency
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Track successful completion with detailed outputs
            outputs = None
            if result:
                try:
                    if isinstance(result, (str, int, float, bool, list, dict)):
                        outputs = {"result": result}
                    else:
                        outputs = {
                            "result": str(result)[:500]
                        }  # More detail for outputs
                except:
                    outputs = {"result": "<unserializable>"}

            # Only log for non-internal nodes to reduce noise
            if not any(
                internal in node_name for internal in ["CompiledStateGraph", "Pregel"]
            ):
                print(f"[LANGGRAPH] Completed node: {node_name} ({latency_ms:.1f}ms)")
                print(f"[LANGGRAPH] Node outputs: {outputs}")

            self._tracker.end_node_execution(execution, outputs=outputs)

            return result

        except Exception as e:
            # Calculate latency for error case too
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Only log for non-internal nodes to reduce noise
            if not any(
                internal in node_name for internal in ["CompiledStateGraph", "Pregel"]
            ):
                print(
                    f"[LANGGRAPH] ERROR in node: {node_name} ({latency_ms:.1f}ms) - {str(e)}"
                )

            # Track error
            self._tracker.end_node_execution(execution, error=str(e))
            raise
