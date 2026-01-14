"""Tracing decorators for manual span tracking.

These decorators allow developers to instrument their custom agent functions
with the same kind of tracing automatically provided for some other frameworks.

Example usage:
    from gradient_adk import entrypoint, trace_llm, trace_tool, trace_retriever, RequestContext

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
    async def my_agent(input: dict, context: RequestContext):
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
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from .runtime.interfaces import NodeExecution
from .runtime.helpers import get_tracker
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

                    # Check for network activity and capture LLM payloads
                    try:
                        has_network_hits = interceptor.hits_since(network_token) > 0
                        # For explicitly marked LLM spans OR auto-detected network activity
                        if has_network_hits or span_type == SpanType.LLM:
                            meta = _ensure_meta(span)
                            if span_type is None and has_network_hits:
                                meta["is_llm_call"] = True
                            # Get captured request/response payloads for LLM metadata extraction
                            captured = interceptor.get_captured_requests_since(
                                network_token
                            )
                            if captured:
                                call = captured[0]
                                if call.request_payload:
                                    meta["llm_request_payload"] = call.request_payload
                                if call.response_payload:
                                    meta["llm_response_payload"] = call.response_payload
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

                    # Check for network activity and capture LLM payloads
                    try:
                        has_network_hits = interceptor.hits_since(network_token) > 0
                        # For explicitly marked LLM spans OR auto-detected network activity
                        if has_network_hits or span_type == SpanType.LLM:
                            meta = _ensure_meta(span)
                            if span_type is None and has_network_hits:
                                meta["is_llm_call"] = True
                            # Get captured request/response payloads for LLM metadata extraction
                            captured = interceptor.get_captured_requests_since(
                                network_token
                            )
                            if captured:
                                call = captured[0]
                                if call.request_payload:
                                    meta["llm_request_payload"] = call.request_payload
                                if call.response_payload:
                                    meta["llm_response_payload"] = call.response_payload
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

                    # Check for network activity and capture LLM payloads
                    try:
                        has_network_hits = interceptor.hits_since(network_token) > 0
                        # For explicitly marked LLM spans OR auto-detected network activity
                        if has_network_hits or span_type == SpanType.LLM:
                            meta = _ensure_meta(span)
                            if span_type is None and has_network_hits:
                                meta["is_llm_call"] = True
                            # Get captured request/response payloads for LLM metadata extraction
                            captured = interceptor.get_captured_requests_since(
                                network_token
                            )
                            if captured:
                                call = captured[0]
                                if call.request_payload:
                                    meta["llm_request_payload"] = call.request_payload
                                if call.response_payload:
                                    meta["llm_response_payload"] = call.response_payload
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


# =============================================================================
# Programmatic Span Functions
# =============================================================================


def add_llm_span(
    name: str,
    input: Any,
    output: Any,
    *,
    model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    num_input_tokens: Optional[int] = None,
    num_output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    time_to_first_token_ns: Optional[int] = None,
    duration_ns: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    status_code: Optional[int] = None,
) -> None:
    """
    Add an LLM span to the current trace.

    Args:
        name: Name for the span (e.g., "call_gpt", "embedding_request")
        input: The input to the LLM call (e.g., messages, prompt)
        output: The output from the LLM call (e.g., response, completion)
        model: Model name (e.g., "gpt-4", "claude-3")
        tools: Tool definitions passed to the model
        num_input_tokens: Number of input/prompt tokens
        num_output_tokens: Number of output/completion tokens
        total_tokens: Total tokens used
        temperature: Temperature setting used
        time_to_first_token_ns: Time to first token in nanoseconds (for streaming)
        duration_ns: Duration of the call in nanoseconds
        metadata: Additional custom metadata
        tags: Tags for the span
        status_code: HTTP status code if applicable

    Example:
        add_llm_span(
            name="call_gpt",
            input={"messages": [{"role": "user", "content": "Hello"}]},
            output={"response": "Hi there!"},
            model="gpt-4",
            num_input_tokens=10,
            num_output_tokens=5,
        )
    """
    tracker = get_tracker()
    if not tracker:
        return

    span = _create_span(name, _freeze(input))
    meta = _ensure_meta(span)
    meta["is_llm_call"] = True
    meta["is_programmatic"] = (
        True  # Mark as programmatic to skip auto-duration calculation
    )

    if model is not None:
        meta["model_name"] = model
    if tools is not None:
        meta["llm_request_payload"] = {"tools": tools}
    if temperature is not None:
        if "llm_request_payload" not in meta:
            meta["llm_request_payload"] = {}
        meta["llm_request_payload"]["temperature"] = temperature
    if time_to_first_token_ns is not None:
        meta["time_to_first_token_ns"] = time_to_first_token_ns
    if (
        num_input_tokens is not None
        or num_output_tokens is not None
        or total_tokens is not None
    ):
        if "llm_response_payload" not in meta:
            meta["llm_response_payload"] = {}
        meta["llm_response_payload"]["usage"] = {
            "prompt_tokens": num_input_tokens,
            "completion_tokens": num_output_tokens,
            "total_tokens": total_tokens,
        }
    if tags is not None:
        meta["tags"] = tags
    if status_code is not None:
        meta["status_code"] = status_code
    if metadata is not None:
        meta["custom_metadata"] = metadata
    if duration_ns is not None:
        meta["duration_ns"] = duration_ns

    tracker.on_node_start(span)
    tracker.on_node_end(span, _freeze(output))


def add_tool_span(
    name: str,
    input: Any,
    output: Any,
    *,
    tool_call_id: Optional[str] = None,
    duration_ns: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    status_code: Optional[int] = None,
) -> None:
    """
    Add a tool span to the current trace.

    Args:
        name: Name for the span (e.g., "calculator", "web_search")
        input: The input to the tool (e.g., function arguments)
        output: The output from the tool (e.g., result)
        tool_call_id: Tool call identifier (from LLM tool calling)
        duration_ns: Duration of the call in nanoseconds
        metadata: Additional custom metadata
        tags: Tags for the span
        status_code: HTTP status code if applicable

    Example:
        add_tool_span(
            name="calculator",
            input={"operation": "add", "x": 5, "y": 3},
            output={"result": 8},
            tool_call_id="call_abc123",
        )
    """
    tracker = get_tracker()
    if not tracker:
        return

    span = _create_span(name, _freeze(input))
    meta = _ensure_meta(span)
    meta["is_tool_call"] = True
    meta["is_programmatic"] = (
        True  # Mark as programmatic to skip auto-duration calculation
    )

    if tool_call_id is not None:
        meta["tool_call_id"] = tool_call_id
    if tags is not None:
        meta["tags"] = tags
    if status_code is not None:
        meta["status_code"] = status_code
    if metadata is not None:
        meta["custom_metadata"] = metadata
    if duration_ns is not None:
        meta["duration_ns"] = duration_ns

    tracker.on_node_start(span)
    tracker.on_node_end(span, _freeze(output))


def add_agent_span(
    name: str,
    input: Any,
    output: Any,
    *,
    duration_ns: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    status_code: Optional[int] = None,
) -> None:
    """
    Add an agent span to the current trace.

    Args:
        name: Name for the span (e.g., "research_agent", "planning_agent")
        input: The input to the agent (e.g., query, task)
        output: The output from the agent (e.g., response, result)
        duration_ns: Duration of the agent execution in nanoseconds
        metadata: Additional custom metadata
        tags: Tags for the span
        status_code: HTTP status code if applicable

    Example:
        add_agent_span(
            name="research_agent",
            input={"query": "What is machine learning?"},
            output={"answer": "Machine learning is..."},
            metadata={"model": "gpt-4"},
        )
    """
    tracker = get_tracker()
    if not tracker:
        return

    span = _create_span(name, _freeze(input))
    meta = _ensure_meta(span)
    meta["is_agent_call"] = True
    meta["is_programmatic"] = (
        True  # Mark as programmatic to skip auto-duration calculation
    )

    if tags is not None:
        meta["tags"] = tags
    if status_code is not None:
        meta["status_code"] = status_code
    if metadata is not None:
        meta["custom_metadata"] = metadata
    if duration_ns is not None:
        meta["duration_ns"] = duration_ns

    tracker.on_node_start(span)
    tracker.on_node_end(span, _freeze(output))
