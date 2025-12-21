"""Tracing decorators for manual span tracking.

These decorators allow developers to instrument their custom agent functions
with the same kind of tracing automatically provided for some other frameworks.

Example usage:
    from gradient_adk import entrypoint, trace_llm, trace_tool, trace_retriever

    @trace_retriever("fetch_data")
    async def fetch_data(query: str) -> dict:
        # Your retrieval logic here
        return {"data": "..."}

    @trace_llm("call_model")
    async def call_model(prompt: str) -> str:
        # LLM call - will be marked as LLM span
        return "response"

    @trace_tool("calculate")
    async def calculate(x: int, y: int) -> int:
        # Tool call
        return x + y

    @entrypoint
    async def my_agent(input: dict, context: dict):
        data = await fetch_data(input["query"])
        result = await calculate(5, 10)
        response = await call_model(data["prompt"])
        return {"response": response}
"""

from __future__ import annotations

import functools
import inspect
import uuid
import json
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from .runtime.interfaces import NodeExecution
from .runtime.langgraph.helpers import get_tracker
from .runtime.network_interceptor import get_network_interceptor

F = TypeVar("F", bound=Callable[..., Any])


class SpanType(Enum):
    """Types of spans that can be traced."""

    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _freeze(obj: Any, max_depth: int = 3, max_items: int = 100) -> Any:
    """Create a JSON-serializable snapshot of arbitrary Python objects."""
    if max_depth < 0:
        return "<max-depth>"

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Dict-like
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                out["<truncated>"] = True
                break
            out[str(k)] = _freeze(v, max_depth - 1, max_items)
        return out

    # Sequences
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        out = []
        for i, v in enumerate(seq):
            if i >= max_items:
                out.append("<truncated>")
                break
            out.append(_freeze(v, max_depth - 1, max_items))
        return out

    # Pydantic models
    try:
        from pydantic import BaseModel

        if isinstance(obj, BaseModel):
            return _freeze(obj.model_dump(), max_depth - 1, max_items)
    except Exception:
        pass

    # Dataclasses
    try:
        import dataclasses

        if dataclasses.is_dataclass(obj):
            return _freeze(dataclasses.asdict(obj), max_depth - 1, max_items)
    except Exception:
        pass

    # Fallback
    return repr(obj)


def _snapshot_args_kwargs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """Create a snapshot of function arguments."""
    try:
        args_copy = deepcopy(args)
        kwargs_copy = deepcopy(kwargs)
    except Exception:
        args_copy, kwargs_copy = args, kwargs

    # If there's exactly one arg and no kwargs, return just that arg
    if len(args_copy) == 1 and not kwargs_copy:
        return _freeze(args_copy[0])

    # If there are kwargs but no args, return just the kwargs
    if not args_copy and kwargs_copy:
        return _freeze(kwargs_copy)

    # If there are multiple args or both args and kwargs, return a dict
    if args_copy and kwargs_copy:
        return {"args": _freeze(args_copy), "kwargs": _freeze(kwargs_copy)}
    elif len(args_copy) > 1:
        return _freeze(args_copy)

    # Fallback
    return _freeze(args_copy)


def _snapshot_output(result: Any) -> Any:
    """Create a snapshot of function output."""
    return _freeze(result)


def _ensure_meta(rec: NodeExecution) -> dict:
    """Ensure the NodeExecution has a metadata dict."""
    md = getattr(rec, "metadata", None)
    if not isinstance(md, dict):
        md = {}
        try:
            rec.metadata = md
        except Exception:
            pass
    return md


def _create_span(span_name: str, inputs: Any) -> NodeExecution:
    """Create a new span execution record."""
    return NodeExecution(
        node_id=str(uuid.uuid4()),
        node_name=span_name,
        framework="custom",
        start_time=_utc(),
        inputs=inputs,
    )


def _trace_base(
    name: Optional[str] = None,
    *,
    span_type: Optional[SpanType] = None,
) -> Callable[[F], F]:
    """
    Base decorator to trace a function as a span in the agent execution.

    Args:
        name: Optional custom name for the span. If not provided, uses function name.
        span_type: Type of span (LLM, TOOL, or RETRIEVER). If None, will auto-detect LLM via network.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        # Handle async generator functions (functions with `yield` that are async)
        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                tracker = get_tracker()
                if not tracker:
                    # No tracker available, just call the function
                    async for chunk in func(*args, **kwargs):
                        yield chunk
                    return

                # Capture network activity
                interceptor = get_network_interceptor()
                try:
                    network_token = interceptor.snapshot_token()
                except Exception:
                    network_token = 0

                # Create span and start tracking
                inputs_snapshot = _snapshot_args_kwargs(args, kwargs)
                span = _create_span(span_name, inputs_snapshot)

                # Mark span type
                if span_type == SpanType.LLM:
                    _ensure_meta(span)["is_llm_call"] = True
                elif span_type == SpanType.TOOL:
                    _ensure_meta(span)["is_tool_call"] = True
                elif span_type == SpanType.RETRIEVER:
                    _ensure_meta(span)["is_retriever_call"] = True

                tracker.on_node_start(span)

                collected: list[str] = []
                try:
                    # Iterate the original generator, collecting content
                    async for chunk in func(*args, **kwargs):
                        # Convert chunk to string for collection
                        if isinstance(chunk, bytes):
                            chunk_str = chunk.decode("utf-8", errors="replace")
                        elif isinstance(chunk, dict):
                            chunk_str = json.dumps(chunk)
                        elif chunk is None:
                            # Skip None values
                            continue
                        else:
                            chunk_str = str(chunk)

                        collected.append(chunk_str)
                        yield chunk

                    # Check for network activity (LLM calls) - only if not already marked
                    if span_type is None:
                        try:
                            if interceptor.hits_since(network_token) > 0:
                                _ensure_meta(span)["is_llm_call"] = True
                        except Exception:
                            pass

                    # Stream complete - finalize span with collected content
                    tracker.on_node_end(span, {"content": "".join(collected)})

                except Exception as e:
                    tracker.on_node_error(span, e)
                    raise

            return async_gen_wrapper  # type: ignore

        # Handle regular async functions (coroutines)
        elif inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracker = get_tracker()
                if not tracker:
                    # No tracker available, just call the function
                    return await func(*args, **kwargs)

                # Capture network activity
                interceptor = get_network_interceptor()
                try:
                    network_token = interceptor.snapshot_token()
                except Exception:
                    network_token = 0

                # Create span and start tracking
                inputs_snapshot = _snapshot_args_kwargs(args, kwargs)
                span = _create_span(span_name, inputs_snapshot)

                # Mark span type
                if span_type == SpanType.LLM:
                    _ensure_meta(span)["is_llm_call"] = True
                elif span_type == SpanType.TOOL:
                    _ensure_meta(span)["is_tool_call"] = True
                elif span_type == SpanType.RETRIEVER:
                    _ensure_meta(span)["is_retriever_call"] = True

                tracker.on_node_start(span)

                try:
                    result = await func(*args, **kwargs)

                    # Check for network activity (LLM calls) - only if not already marked
                    if span_type is None:
                        try:
                            if interceptor.hits_since(network_token) > 0:
                                _ensure_meta(span)["is_llm_call"] = True
                        except Exception:
                            pass

                    # If the result is an async generator, wrap it so we can collect output
                    # without double-iterating. We delay on_node_end until the stream is consumed.
                    if result is not None and (
                        hasattr(result, "__aiter__") or inspect.isasyncgen(result)
                    ):

                        async def _streaming_wrapper(gen):
                            collected: list[str] = []
                            try:
                                async for chunk in gen:
                                    # Convert chunk to string for collection
                                    if isinstance(chunk, bytes):
                                        chunk_str = chunk.decode(
                                            "utf-8", errors="replace"
                                        )
                                    elif isinstance(chunk, dict):
                                        chunk_str = json.dumps(chunk)
                                    elif chunk is None:
                                        # Skip None values
                                        continue
                                    else:
                                        chunk_str = str(chunk)

                                    collected.append(chunk_str)
                                    yield chunk

                                # Stream complete - finalize span
                                tracker.on_node_end(
                                    span, {"content": "".join(collected)}
                                )
                            except Exception as e:
                                tracker.on_node_error(span, e)
                                raise

                        return _streaming_wrapper(result)

                    # Non-streaming path
                    output = _snapshot_output(result)
                    tracker.on_node_end(span, output)
                    return result

                except Exception as e:
                    tracker.on_node_error(span, e)
                    raise

            return async_wrapper  # type: ignore

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracker = get_tracker()
                if not tracker:
                    # No tracker available, just call the function
                    return func(*args, **kwargs)

                # Capture network activity
                interceptor = get_network_interceptor()
                try:
                    network_token = interceptor.snapshot_token()
                except Exception:
                    network_token = 0

                # Create span and start tracking
                inputs_snapshot = _snapshot_args_kwargs(args, kwargs)
                span = _create_span(span_name, inputs_snapshot)

                # Mark span type
                if span_type == SpanType.LLM:
                    _ensure_meta(span)["is_llm_call"] = True
                elif span_type == SpanType.TOOL:
                    _ensure_meta(span)["is_tool_call"] = True
                elif span_type == SpanType.RETRIEVER:
                    _ensure_meta(span)["is_retriever_call"] = True

                tracker.on_node_start(span)

                try:
                    result = func(*args, **kwargs)

                    # Check for network activity (LLM calls) - only if not already marked
                    if span_type is None:
                        try:
                            if interceptor.hits_since(network_token) > 0:
                                _ensure_meta(span)["is_llm_call"] = True
                        except Exception:
                            pass

                    # Check if result is an async generator - pass directly without snapshotting
                    if result is not None and (
                        hasattr(result, "__aiter__") or inspect.isasyncgen(result)
                    ):
                        output = result
                    else:
                        output = _snapshot_output(result)
                    tracker.on_node_end(span, output)
                    return result

                except Exception as e:
                    tracker.on_node_error(span, e)
                    raise

            return sync_wrapper  # type: ignore

    return decorator


def trace_llm(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to trace a function as an LLM call span.

    Args:
        name: Optional custom name for the span. If not provided, uses function name.

    Example:
        @trace_llm("openai_call")
        async def call_openai(prompt: str) -> str:
            response = await openai.chat.completions.create(...)
            return response.choices[0].message.content
    """
    return _trace_base(name, span_type=SpanType.LLM)


def trace_retriever(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to trace a function as a retriever call span.

    Args:
        name: Optional custom name for the span. If not provided, uses function name.

    Example:
        @trace_retriever("vector_search")
        async def search_vectors(query: str) -> list:
            results = await vector_db.search(query)
            return results
    """
    return _trace_base(name, span_type=SpanType.RETRIEVER)


def trace_tool(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to trace a function as a tool call span.

    Args:
        name: Optional custom name for the span. If not provided, uses function name.

    Example:
        @trace_tool("search_database")
        async def search(query: str) -> list:
            results = await db.search(query)
            return results
    """
    return _trace_base(name, span_type=SpanType.TOOL)
