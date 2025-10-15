from __future__ import annotations

import functools
import importlib
import inspect
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, List

from ..interfaces import FrameworkInstrumentor, ExecutionTracker

from ..network_interceptor import (
    get_network_interceptor,
    setup_digitalocean_interception,
)


class LangGraphInstrumentor(FrameworkInstrumentor):
    """Simple LangGraph instrumentor that wraps nodes before compilation."""

    def __init__(self) -> None:
        self._installed: bool = False
        self._tracker: Optional[ExecutionTracker] = None

    @property
    def framework_name(self) -> str:
        return "langgraph"

    def install(self, tracker: ExecutionTracker) -> None:
        if self._installed:
            return
        self._tracker = tracker

        # Set up network interception
        try:
            setup_digitalocean_interception()
        except Exception:
            pass

        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._tracker = None
        self._installed = False

    def is_installed(self) -> bool:
        return self._installed

    def attach_to_graph(self, graph: Any) -> None:
        """Attach instrumentation to a StateGraph before compilation."""
        if not self._tracker:
            return

        # Find the nodes dictionary in the graph
        nodes_dict = self._find_nodes_dict(graph)
        if not nodes_dict:
            return

        # Wrap each node function
        wrapped_count = 0
        for node_name, node_spec in list(nodes_dict.items()):
            # Handle StateNodeSpec objects (LangGraph's internal structure)
            original_func = None

            # Check if node_spec has a runnable attribute (RunnableCallable)
            if hasattr(node_spec, "runnable"):
                runnable = node_spec.runnable

                # Check for sync function first (prefer func over afunc for sync functions)
                if hasattr(runnable, "func") and callable(runnable.func):
                    original_func = runnable.func
                elif hasattr(runnable, "afunc") and callable(runnable.afunc):
                    # Only use afunc if it's actually an async function
                    afunc = runnable.afunc
                    if inspect.iscoroutinefunction(afunc) or (
                        hasattr(afunc, "func")
                        and inspect.iscoroutinefunction(afunc.func)
                    ):
                        original_func = afunc
                elif hasattr(runnable, "_func") and callable(runnable._func):
                    original_func = runnable._func
                elif hasattr(runnable, "_afunc") and callable(runnable._afunc):
                    original_func = runnable._afunc
                elif callable(runnable):
                    # If the runnable itself is callable, use it
                    original_func = runnable

            # Fallback: check direct function attributes
            if not original_func:
                for attr_name in ["func", "_func"]:
                    if hasattr(node_spec, attr_name):
                        attr_val = getattr(node_spec, attr_name, None)
                        if callable(attr_val) and not isinstance(attr_val, type):
                            original_func = attr_val
                            break

            if (
                not original_func
                and callable(node_spec)
                and not isinstance(node_spec, type)
            ):
                original_func = node_spec

            if not original_func:
                continue

            if getattr(original_func, "__gradient_wrapped__", False):
                continue

            wrapped_func = self._wrap_node_function(node_name, original_func)
            wrapped_func.__gradient_wrapped__ = True

            # Update the StateNodeSpec with the wrapped function
            if hasattr(node_spec, "runnable"):
                runnable = node_spec.runnable
                if hasattr(runnable, "afunc") and runnable.afunc == original_func:
                    runnable.afunc = wrapped_func
                elif hasattr(runnable, "func") and runnable.func == original_func:
                    runnable.func = wrapped_func
                elif hasattr(runnable, "_func") and runnable._func == original_func:
                    runnable._func = wrapped_func
                elif hasattr(runnable, "_afunc") and runnable._afunc == original_func:
                    runnable._afunc = wrapped_func
                else:
                    node_spec.runnable = wrapped_func
            elif hasattr(node_spec, "func"):
                node_spec.func = wrapped_func
            else:
                nodes_dict[node_name] = wrapped_func

            wrapped_count += 1

    def _find_nodes_dict(self, graph: Any) -> Optional[Dict[str, Any]]:
        """Find the nodes dictionary in a StateGraph."""
        # Try graph.nodes first (most common)
        nodes = getattr(graph, "nodes", None)
        if isinstance(nodes, dict):
            return nodes

        # Try graph._nodes
        nodes = getattr(graph, "_nodes", None)
        if isinstance(nodes, dict):
            return nodes

        # Try graph.graph.nodes (if there's a nested graph)
        inner_graph = getattr(graph, "graph", None)
        if inner_graph:
            nodes = getattr(inner_graph, "nodes", None)
            if isinstance(nodes, dict):
                return nodes
            nodes = getattr(inner_graph, "_nodes", None)
            if isinstance(nodes, dict):
                return nodes

        return None

    def _wrap_node_function(self, node_name: str, original_func: Callable) -> Callable:
        """Wrap a node function with execution tracking."""

        if inspect.iscoroutinefunction(original_func):

            @functools.wraps(original_func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_tracking_async(
                    node_name, original_func, args, kwargs
                )

            return async_wrapper
        else:

            @functools.wraps(original_func)
            def sync_wrapper(*args, **kwargs):
                return self._execute_with_tracking(
                    node_name, original_func, args, kwargs, is_async=False
                )

            return sync_wrapper

    def _execute_with_tracking(
        self, node_name: str, func: Callable, args: Tuple, kwargs: Dict, is_async: bool
    ):
        import types

        interceptor = get_network_interceptor()
        if interceptor:
            try:
                interceptor.clear_detected()
            except Exception:
                pass

        node_id = str(uuid.uuid4())
        inputs = self._serialize_inputs(args, kwargs)

        span = None
        if self._tracker:
            try:
                span = self._tracker.start_node_execution(
                    node_id=node_id,
                    node_name=node_name,
                    framework=self.framework_name,
                    inputs=inputs,
                    metadata={},
                )
            except Exception:
                pass

        error = None
        result = None
        will_stream = False  # <<< NEW

        try:
            result = func(*args, **kwargs)

            # Case 1: plain sync generator
            if isinstance(result, types.GeneratorType):
                will_stream = True

                def generator_wrapper():
                    nonlocal error
                    try:
                        for item in result:
                            yield item
                    except Exception as e:
                        error = str(e)
                        raise
                    finally:
                        if span and self._tracker:
                            try:
                                is_llm = False
                                if interceptor:
                                    try:
                                        endpoints = (
                                            interceptor.get_detected_endpoints() or []
                                        )
                                        endpoint_str = " ".join(endpoints)
                                        is_llm = any(
                                            ep in endpoint_str
                                            for ep in [
                                                "inference.do-ai.run",
                                                "inference.do-ai-test.run",
                                                "api.openai.com",
                                                "api.anthropic.com",
                                            ]
                                        )
                                    except Exception:
                                        pass
                                if hasattr(span, "metadata") and isinstance(
                                    span.metadata, dict
                                ):
                                    span.metadata["is_llm_call"] = is_llm
                                self._tracker.end_node_execution(
                                    span,
                                    outputs=(
                                        {"streamed": True, "error": error}
                                        if error
                                        else {"streamed": True}
                                    ),
                                )
                            except Exception:
                                pass

                return generator_wrapper()

            # Case 2: dict containing a sync generator under "_stream_generator"
            import types as _types  # avoid shadowing

            if isinstance(result, dict) and isinstance(
                result.get("_stream_generator"), _types.GeneratorType
            ):
                will_stream = True
                orig_gen = result["_stream_generator"]

                def dict_generator_wrapper():
                    nonlocal error
                    try:
                        for item in orig_gen:
                            yield item
                    except Exception as e:
                        error = str(e)
                        raise
                    finally:
                        if span and self._tracker:
                            try:
                                is_llm = False
                                if interceptor:
                                    try:
                                        endpoints = (
                                            interceptor.get_detected_endpoints() or []
                                        )
                                        endpoint_str = " ".join(endpoints)
                                        is_llm = any(
                                            ep in endpoint_str
                                            for ep in [
                                                "inference.do-ai.run",
                                                "inference.do-ai-test.run",
                                                "api.openai.com",
                                                "api.anthropic.com",
                                            ]
                                        )
                                    except Exception:
                                        pass
                                if hasattr(span, "metadata") and isinstance(
                                    span.metadata, dict
                                ):
                                    span.metadata["is_llm_call"] = is_llm
                                self._tracker.end_node_execution(
                                    span,
                                    outputs=(
                                        {"streamed": True, "error": error}
                                        if error
                                        else {"streamed": True}
                                    ),
                                )
                            except Exception:
                                pass

                # IMPORTANT: return a new dict with the wrapped generator
                return {**result, "_stream_generator": dict_generator_wrapper()}

        except Exception as e:
            error = str(e)
            raise
        finally:
            # Only end here if we did NOT set up a streaming wrapper
            if span and self._tracker and not will_stream:
                try:
                    is_llm = False
                    if interceptor:
                        try:
                            endpoints = interceptor.get_detected_endpoints() or []
                            endpoint_str = " ".join(endpoints)
                            is_llm = any(
                                ep in endpoint_str
                                for ep in [
                                    "inference.do-ai.run",
                                    "inference.do-ai-test.run",
                                    "api.openai.com",
                                    "api.anthropic.com",
                                ]
                            )
                        except Exception:
                            pass
                    if hasattr(span, "metadata") and isinstance(span.metadata, dict):
                        span.metadata["is_llm_call"] = is_llm
                    outputs = (
                        self._serialize_output(result)
                        if error is None
                        else {"error": error}
                    )
                    self._tracker.end_node_execution(span, outputs=outputs)
                except Exception:
                    pass

        return result

    async def _execute_with_tracking_async(
        self, node_name: str, func: Callable, args: Tuple, kwargs: Dict
    ):
        import types

        interceptor = get_network_interceptor()
        if interceptor:
            try:
                interceptor.clear_detected()
            except Exception:
                pass

        node_id = str(uuid.uuid4())
        inputs = self._serialize_inputs(args, kwargs)

        span = None
        if self._tracker:
            try:
                span = self._tracker.start_node_execution(
                    node_id=node_id,
                    node_name=node_name,
                    framework=self.framework_name,
                    inputs=inputs,
                    metadata={},
                )
            except Exception:
                pass

        error = None
        result = None
        will_stream = False  # <<< NEW

        try:
            result = await func(*args, **kwargs)

            # Case 1: plain async generator
            if isinstance(result, types.AsyncGeneratorType):
                will_stream = True

                async def async_generator_wrapper():
                    nonlocal error
                    try:
                        async for item in result:
                            yield item
                    except Exception as e:
                        error = str(e)
                        raise
                    finally:
                        if span and self._tracker:
                            try:
                                is_llm = False
                                if interceptor:
                                    try:
                                        endpoints = (
                                            interceptor.get_detected_endpoints() or []
                                        )
                                        endpoint_str = " ".join(endpoints)
                                        is_llm = any(
                                            ep in endpoint_str
                                            for ep in [
                                                "inference.do-ai.run",
                                                "inference.do-ai-test.run",
                                                "api.openai.com",
                                                "api.anthropic.com",
                                            ]
                                        )
                                    except Exception:
                                        pass
                                if hasattr(span, "metadata") and isinstance(
                                    span.metadata, dict
                                ):
                                    span.metadata["is_llm_call"] = is_llm
                                self._tracker.end_node_execution(
                                    span,
                                    outputs=(
                                        {"streamed": True, "error": error}
                                        if error
                                        else {"streamed": True}
                                    ),
                                )
                            except Exception:
                                pass

                return async_generator_wrapper()

            # Case 2: dict containing an **async** generator under "_stream_generator"
            if isinstance(result, dict) and isinstance(
                result.get("_stream_generator"), types.AsyncGeneratorType
            ):
                will_stream = True
                orig_agen = result["_stream_generator"]

                async def dict_async_generator_wrapper():
                    nonlocal error
                    try:
                        async for item in orig_agen:
                            yield item
                    except Exception as e:
                        error = str(e)
                        raise
                    finally:
                        if span and self._tracker:
                            try:
                                is_llm = False
                                if interceptor:
                                    try:
                                        endpoints = (
                                            interceptor.get_detected_endpoints() or []
                                        )
                                        endpoint_str = " ".join(endpoints)
                                        is_llm = any(
                                            ep in endpoint_str
                                            for ep in [
                                                "inference.do-ai.run",
                                                "inference.do-ai-test.run",
                                                "api.openai.com",
                                                "api.anthropic.com",
                                            ]
                                        )
                                    except Exception:
                                        pass
                                if hasattr(span, "metadata") and isinstance(
                                    span.metadata, dict
                                ):
                                    span.metadata["is_llm_call"] = is_llm
                                self._tracker.end_node_execution(
                                    span,
                                    outputs=(
                                        {"streamed": True, "error": error}
                                        if error
                                        else {"streamed": True}
                                    ),
                                )
                            except Exception:
                                pass

                return {**result, "_stream_generator": dict_async_generator_wrapper()}

        except Exception as e:
            error = str(e)
            raise
        finally:
            # Only end here if we did NOT set up a streaming wrapper
            if span and self._tracker and not will_stream:
                try:
                    is_llm = False
                    if interceptor:
                        try:
                            endpoints = interceptor.get_detected_endpoints() or []
                            endpoint_str = " ".join(endpoints)
                            is_llm = any(
                                ep in endpoint_str
                                for ep in [
                                    "inference.do-ai.run",
                                    "inference.do-ai-test.run",
                                    "api.openai.com",
                                    "api.anthropic.com",
                                ]
                            )
                        except Exception:
                            pass
                    if hasattr(span, "metadata") and isinstance(span.metadata, dict):
                        span.metadata["is_llm_call"] = is_llm
                    outputs = (
                        self._serialize_output(result)
                        if error is None
                        else {"error": error}
                    )
                    self._tracker.end_node_execution(span, outputs=outputs)
                except Exception:
                    pass

        return result

    def _serialize_inputs(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Serialize inputs for tracking."""
        return {
            "args": [self._safe_serialize(arg) for arg in args],
            "kwargs": {k: self._safe_serialize(v) for k, v in kwargs.items()},
        }

    def _serialize_output(self, result: Any) -> Any:
        """Serialize output for tracking."""
        return self._safe_serialize(result)

    def _safe_serialize(self, obj: Any, depth: int = 0) -> Any:
        """Safely serialize an object for tracking, avoiding infinite recursion."""
        if depth > 3:
            return "<max_depth>"

        try:
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [
                    self._safe_serialize(item, depth + 1) for item in obj[:10]
                ]  # Limit list size
            elif isinstance(obj, dict):
                result = {}
                for i, (k, v) in enumerate(obj.items()):
                    if i >= 10:  # Limit dict size
                        result["__truncated__"] = True
                        break
                    result[str(k)] = self._safe_serialize(v, depth + 1)
                return result
            else:
                # For complex objects, just return their string representation
                str_repr = str(obj)
                return str_repr[:200] + "..." if len(str_repr) > 200 else str_repr
        except Exception:
            return "<unserializable>"


# ---- Singleton management for convenience ------------------------------------
_singleton: Optional[LangGraphInstrumentor] = None


def install(tracker: ExecutionTracker) -> LangGraphInstrumentor:
    """Install the global LangGraph instrumentor."""
    global _singleton
    if _singleton is None:
        _singleton = LangGraphInstrumentor()
    _singleton.install(tracker)
    return _singleton


def attach_to_graph(graph: Any) -> None:
    """Attach the global instrumentor to a graph. Requires install() to be called first."""
    if _singleton is None or not _singleton.is_installed():
        return
    _singleton.attach_to_graph(graph)
