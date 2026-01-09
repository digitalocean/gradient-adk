from __future__ import annotations

import functools
import inspect
import json
import os
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence, Tuple, Dict

from langgraph.graph import StateGraph

from ..interfaces import NodeExecution
from ..digitalocean_tracker import DigitalOceanTracesTracker
from ..network_interceptor import (
    get_network_interceptor,
    is_inference_url,
    is_kbaas_url,
)


WRAPPED_FLAG = "__do_wrapped__"


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _mk_exec(name: str, inputs: Any) -> NodeExecution:
    return NodeExecution(
        node_id=str(uuid.uuid4()),
        node_name=name,
        framework="langgraph",
        start_time=_utc(),
        inputs=inputs,
    )


def _ensure_meta(rec: NodeExecution) -> dict:
    md = getattr(rec, "metadata", None)
    if not isinstance(md, dict):
        md = {}
        try:
            rec.metadata = md
        except Exception:
            pass
    return md


_MAX_DEPTH = 3
_MAX_ITEMS = 100  # keep payloads bounded


def _freeze(obj: Any, depth: int = _MAX_DEPTH) -> Any:
    """Mutation-safe, JSON-ish snapshot for arbitrary Python objects."""
    # if depth < 0:
    #     return "<max-depth>"
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # dict-like
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= _MAX_ITEMS:
                out["<truncated>"] = True
                break
            out[str(k)] = _freeze(v, depth - 1)
        return out

    # sequences
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        out = []
        for i, v in enumerate(seq):
            if i >= _MAX_ITEMS:
                out.append("<truncated>")
                break
            out.append(_freeze(v, depth - 1))
        return out

    # pydantic
    try:
        from pydantic import BaseModel  # type: ignore

        if isinstance(obj, BaseModel):
            return _freeze(obj.model_dump(), depth - 1)
    except Exception:
        pass

    # dataclass
    try:
        import dataclasses

        if dataclasses.is_dataclass(obj):
            return _freeze(dataclasses.asdict(obj), depth - 1)
    except Exception:
        pass

    # fallback
    return repr(obj)


def _snapshot_args_kwargs(a: Tuple[Any, ...], kw: Dict[str, Any]) -> Any:
    """Deepcopy then freeze to avoid mutation surprises."""
    try:
        a_copy = deepcopy(a)
        kw_copy = deepcopy(kw)
    except Exception:
        a_copy, kw_copy = a, kw  # best-effort

    # If there's exactly one arg and no kwargs, return just that arg
    if len(a_copy) == 1 and not kw_copy:
        return _freeze(a_copy[0])

    # If there are kwargs but no args, return just the kwargs
    if not a_copy and kw_copy:
        return _freeze(kw_copy)

    # If there are multiple args or both args and kwargs, return a dict
    if a_copy and kw_copy:
        return {"args": _freeze(a_copy), "kwargs": _freeze(kw_copy)}
    elif len(a_copy) > 1:
        return _freeze(a_copy)

    # Fallback
    return _freeze(a_copy)


def _diff(a: Any, b: Any, depth: int = 2) -> Any:
    """Small, generic diff for dicts/lists/tuples; returns None if identical."""
    # if depth < 0:
    #     return "<max-depth>"

    # dict diff
    if isinstance(a, dict) and isinstance(b, dict):
        keys = list(set(a.keys()) | set(b.keys()))
        keys.sort(key=str)
        out: Dict[str, Any] = {}
        count = 0
        for k in keys:
            if count >= _MAX_ITEMS:
                out["<truncated_keys>"] = True
                break
            av = a.get(k, "<missing>")
            bv = b.get(k, "<missing>")
            if av == bv:
                continue
            if isinstance(av, (dict, list, tuple)) and isinstance(
                bv, (dict, list, tuple)
            ):
                sub = _diff(av, bv, depth - 1)
                out[k] = sub if sub is not None else {"before": av, "after": bv}
            else:
                out[k] = {"before": av, "after": bv}
            count += 1
        return out or None

    # list/tuple diff
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        length = max(len(a), len(b))
        changed = False
        out_list = []
        for i in range(min(length, _MAX_ITEMS)):
            av = a[i] if i < len(a) else "<missing>"
            bv = b[i] if i < len(b) else "<missing>"
            if av == bv:
                out_list.append("<same>")
            else:
                if isinstance(av, (dict, list, tuple)) and isinstance(
                    bv, (dict, list, tuple)
                ):
                    sub = _diff(av, bv, depth - 1)
                    out_list.append(
                        sub if sub is not None else {"before": av, "after": bv}
                    )
                else:
                    out_list.append({"before": av, "after": bv})
                changed = True
        if length > _MAX_ITEMS:
            out_list.append("<truncated>")
        return out_list if changed else None

    return None if a == b else {"before": a, "after": b}


def _first_arg_after(a: Tuple[Any, ...]) -> Optional[Any]:
    return a[0] if (a and isinstance(a[0], dict)) else None


def _first_arg_before(before_inputs: dict) -> Optional[Any]:
    try:
        args = before_inputs.get("args")
        if isinstance(args, list) and args:
            return args[0]
    except Exception:
        pass
    return None


def _canonical_output(
    before_inputs: dict, a: Tuple[Any, ...], kw: Dict[str, Any], ret: Any
) -> Any:
    """
    Choose a single, compact output:
      1) If ret is a mapping -> return snapshot(ret)
      2) Else if first arg is a dict and appears changed -> snapshot(first arg)
      3) Else -> snapshot(ret)
    """
    if isinstance(ret, Mapping):
        return _freeze(ret)

    arg0_before = _first_arg_before(before_inputs)
    arg0_after = _first_arg_after(a)
    if isinstance(arg0_after, dict):
        arg0_after_frozen = _freeze(arg0_after)
        if not isinstance(arg0_before, dict) or arg0_before != arg0_after_frozen:
            return arg0_after_frozen

    return _freeze(ret)


def _snap():
    intr = get_network_interceptor()
    try:
        tok = intr.snapshot_token()
    except Exception:
        tok = 0
    return intr, tok


def _had_hits_since(intr, token) -> bool:
    try:
        return intr.hits_since(token) > 0
    except Exception:
        return False


def _get_captured_payloads_with_type(intr, token) -> tuple:
    """Get captured API request/response payloads and classify the call type.

    When using httpx, both request() and send() are intercepted, which can result
    in two CapturedRequest records for a single HTTP call. The request() interception
    captures the request payload but not the response (since it delegates to send()),
    while the send() interception captures both request and response payloads.

    This function searches for a captured request that has a response payload,
    preferring the last one (which is typically from send()).

    Returns:
        (request_payload, response_payload, is_llm, is_retriever)
    """
    try:
        captured = intr.get_captured_requests_since(token)
        if captured:
            # Search in reverse order to find a captured request with a response.
            # This handles the case where both request() and send() are intercepted:
            # - captured[0] from request(): has request_payload, no response_payload
            # - captured[1] from send(): has both request_payload and response_payload
            for call in reversed(captured):
                if call.response_payload is not None:
                    url = call.url
                    is_llm = is_inference_url(url)
                    is_retriever = is_kbaas_url(url)
                    return call.request_payload, call.response_payload, is_llm, is_retriever

            # Fallback to the first captured request if none have a response
            # (e.g., if the HTTP request failed or response wasn't captured)
            call = captured[0]
            url = call.url
            is_llm = is_inference_url(url)
            is_retriever = is_kbaas_url(url)
            return call.request_payload, call.response_payload, is_llm, is_retriever
    except Exception:
        pass
    return None, None, False, False


def _transform_kbaas_response(response: Optional[Dict[str, Any]]) -> Optional[list]:
    """Transform KBaaS response to standard retriever format.

    Extracts results and maps content fields to 'page_content'.
    
    For hierarchical KB (parent retrieval):
      - Uses 'parent_chunk_text' as 'page_content' (the context users typically want)
      - Preserves 'text_content' as 'embedded_content' for reference
    
    For standard KB:
      - Uses 'text_content' as 'page_content'
    
    Returns a list of dicts as expected for retriever spans.
    """
    if not isinstance(response, dict):
        return response

    results = response.get("results", [])
    if not isinstance(results, list):
        return response

    transformed_results = []
    for item in results:
        if isinstance(item, dict):
            new_item = dict(item)
            
            # For hierarchical KB: prefer parent_chunk_text as page_content
            if "parent_chunk_text" in new_item:
                new_item["page_content"] = new_item.pop("parent_chunk_text")
                # Preserve embedded text as embedded_content for reference
                if "text_content" in new_item:
                    new_item["embedded_content"] = new_item.pop("text_content")
            elif "text_content" in new_item:
                # Standard KB: use text_content as page_content
                new_item["page_content"] = new_item.pop("text_content")
            
            transformed_results.append(new_item)
        else:
            transformed_results.append(item)

    # Return just the array of results
    return transformed_results


class LangGraphInstrumentor:
    """Wraps LangGraph nodes with tracing."""

    def __init__(self) -> None:
        self._installed = False
        self._tracker: Optional[DigitalOceanTracesTracker] = None

    def install(self, tracker: DigitalOceanTracesTracker) -> None:
        if self._installed:
            return
        self._tracker = tracker

        original_add_node = StateGraph.add_node
        t = tracker  # close over

        def _start(node_name: str, a: Tuple[Any, ...], kw: Dict[str, Any]):
            inputs_snapshot = _snapshot_args_kwargs(a, kw)
            rec = _mk_exec(node_name, inputs_snapshot)
            intr, tok = _snap()
            t.on_node_start(rec)
            return rec, inputs_snapshot, intr, tok

        def _finish_ok(
            rec: NodeExecution,
            inputs_snapshot: dict,
            a: Tuple[Any, ...],
            kw: Dict[str, Any],
            ret: Any,
            intr,
            tok,
            time_to_first_token_ns: Optional[int] = None,
        ):
            # NOTE: Async generators should be handled by the wrapper functions
            # (_wrap_async_func, _wrap_sync_func, etc.) BEFORE calling _finish_ok.
            # The wrappers collect streamed content and pass {"content": "..."} here.

            # Check if this node made any tracked API calls (e.g., LLM inference or KBaaS retrieval)
            if _had_hits_since(intr, tok):
                # Get captured payloads and classify the call type
                api_request, api_response, is_llm, is_retriever = (
                    _get_captured_payloads_with_type(intr, tok)
                )

                # Set metadata based on call type
                meta = _ensure_meta(rec)
                if is_llm:
                    meta["is_llm_call"] = True
                    # Store raw API payloads for LLM field extraction in tracker
                    if api_request:
                        meta["llm_request_payload"] = api_request
                    if api_response:
                        meta["llm_response_payload"] = api_response
                    # Store time-to-first-token if this was a streaming call
                    if time_to_first_token_ns is not None:
                        meta["time_to_first_token_ns"] = time_to_first_token_ns
                elif is_retriever:
                    meta["is_retriever_call"] = True
                else:
                    # Fallback: assume LLM call for backward compatibility
                    meta["is_llm_call"] = True
                    if api_request:
                        meta["llm_request_payload"] = api_request
                    if api_response:
                        meta["llm_response_payload"] = api_response

                if api_request or api_response:
                    # Use actual API payloads instead of function args
                    if api_request:
                        rec.inputs = _freeze(api_request)

                    # Use actual API response as output
                    if api_response:
                        # Transform KBaaS response to standard retriever format
                        if is_retriever:
                            api_response = _transform_kbaas_response(api_response)
                        out_payload = _freeze(api_response)
                    else:
                        out_payload = _canonical_output(inputs_snapshot, a, kw, ret)
                else:
                    out_payload = _canonical_output(inputs_snapshot, a, kw, ret)
            else:
                out_payload = _canonical_output(inputs_snapshot, a, kw, ret)

            t.on_node_end(rec, out_payload)

        def _finish_err(rec: NodeExecution, intr, tok, e: BaseException):
            if _had_hits_since(intr, tok):
                # Get captured payloads and classify the call type
                api_request, _, is_llm, is_retriever = _get_captured_payloads_with_type(
                    intr, tok
                )

                # Set metadata based on call type
                meta = _ensure_meta(rec)
                if is_llm:
                    meta["is_llm_call"] = True
                elif is_retriever:
                    meta["is_retriever_call"] = True
                else:
                    # Fallback: assume LLM call for backward compatibility
                    meta["is_llm_call"] = True

                if api_request:
                    rec.inputs = _freeze(api_request)

            t.on_node_error(rec, e)

        def _wrap_async_func(node_name: str, func):
            @functools.wraps(func)
            async def _wrapped(*a, **kw):
                rec, snap, intr, tok = _start(node_name, a, kw)
                # Record start time for TTFT calculation
                start_time_ns = time.perf_counter_ns()
                try:
                    ret = await func(*a, **kw)

                    # If ret is an async generator, we need to wrap it to collect
                    # content and defer _finish_ok until the stream is consumed
                    if ret is not None and (
                        hasattr(ret, "__aiter__") or inspect.isasyncgen(ret)
                    ):

                        async def _streaming_wrapper(gen):
                            collected: list[str] = []
                            ttft_ns: Optional[int] = None
                            first_chunk_received = False
                            try:
                                async for chunk in gen:
                                    # Record time to first token
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        ttft_ns = time.perf_counter_ns() - start_time_ns

                                    # Convert chunk to string for collection
                                    if isinstance(chunk, bytes):
                                        chunk_str = chunk.decode(
                                            "utf-8", errors="replace"
                                        )
                                    elif isinstance(chunk, dict):
                                        chunk_str = json.dumps(chunk)
                                    elif chunk is None:
                                        continue
                                    else:
                                        chunk_str = str(chunk)

                                    collected.append(chunk_str)
                                    yield chunk

                                # Stream complete - finalize with collected content and TTFT
                                _finish_ok(
                                    rec,
                                    snap,
                                    a,
                                    kw,
                                    {"content": "".join(collected)},
                                    intr,
                                    tok,
                                    time_to_first_token_ns=ttft_ns,
                                )
                            except BaseException as e:
                                _finish_err(rec, intr, tok, e)
                                raise

                        return _streaming_wrapper(ret)

                    # Non-streaming: finalize immediately
                    _finish_ok(rec, snap, a, kw, ret, intr, tok)
                    return ret
                except BaseException as e:
                    _finish_err(rec, intr, tok, e)
                    raise

            setattr(_wrapped, WRAPPED_FLAG, True)
            return _wrapped

        def _wrap_sync_func(node_name: str, func):
            @functools.wraps(func)
            def _wrapped(*a, **kw):
                rec, snap, intr, tok = _start(node_name, a, kw)
                # Record start time for TTFT calculation
                start_time_ns = time.perf_counter_ns()
                try:
                    ret = func(*a, **kw)

                    # If ret is an async generator, we need to wrap it to collect
                    # content and defer _finish_ok until the stream is consumed
                    if ret is not None and (
                        hasattr(ret, "__aiter__") or inspect.isasyncgen(ret)
                    ):

                        async def _streaming_wrapper(gen):
                            collected: list[str] = []
                            ttft_ns: Optional[int] = None
                            first_chunk_received = False
                            try:
                                async for chunk in gen:
                                    # Record time to first token
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        ttft_ns = time.perf_counter_ns() - start_time_ns

                                    # Convert chunk to string for collection
                                    if isinstance(chunk, bytes):
                                        chunk_str = chunk.decode(
                                            "utf-8", errors="replace"
                                        )
                                    elif isinstance(chunk, dict):
                                        chunk_str = json.dumps(chunk)
                                    elif chunk is None:
                                        continue
                                    else:
                                        chunk_str = str(chunk)

                                    collected.append(chunk_str)
                                    yield chunk

                                # Stream complete - finalize with collected content and TTFT
                                _finish_ok(
                                    rec,
                                    snap,
                                    a,
                                    kw,
                                    {"content": "".join(collected)},
                                    intr,
                                    tok,
                                    time_to_first_token_ns=ttft_ns,
                                )
                            except BaseException as e:
                                _finish_err(rec, intr, tok, e)
                                raise

                        return _streaming_wrapper(ret)

                    # Non-streaming: finalize immediately
                    _finish_ok(rec, snap, a, kw, ret, intr, tok)
                    return ret
                except BaseException as e:
                    _finish_err(rec, intr, tok, e)
                    raise

            setattr(_wrapped, WRAPPED_FLAG, True)
            return _wrapped

        def _wrap_async_gen(node_name: str, func):
            @functools.wraps(func)
            async def _wrapped(*a, **kw):
                rec, snap, intr, tok = _start(node_name, a, kw)
                # Record start time for TTFT calculation
                start_time_ns = time.perf_counter_ns()
                ttft_ns: Optional[int] = None
                first_chunk_received = False
                try:
                    # Accumulate a compact, canonical final payload
                    # (string: concatenate; list: extend; else: last write wins)
                    acc: Dict[str, Any] = {}

                    async for chunk in func(*a, **kw):
                        # Record time to first token
                        if not first_chunk_received:
                            first_chunk_received = True
                            ttft_ns = time.perf_counter_ns() - start_time_ns

                        # Merge into acc for the final on_node_end payload
                        for k, v in chunk.items():
                            if isinstance(v, str):
                                acc[k] = acc.get(k, "") + v
                            elif isinstance(v, bytes):
                                acc[k] = acc.get(k, b"") + v
                            elif isinstance(v, list):
                                acc.setdefault(k, []).extend(v)
                            else:
                                acc[k] = v

                        # Pass the live chunk downstream unchanged
                        yield chunk

                    # Finish the span with the aggregated mapping and TTFT
                    _finish_ok(rec, snap, a, kw, acc, intr, tok, time_to_first_token_ns=ttft_ns)
                except BaseException as e:
                    _finish_err(rec, intr, tok, e)
                    raise

            setattr(_wrapped, WRAPPED_FLAG, True)
            return _wrapped

        def _wrap_runnable_ainvoke(node_name: str, runnable):
            async def _wrapped(*a, **kw):
                rec, snap, intr, tok = _start(node_name, a, kw)
                # Record start time for TTFT calculation
                start_time_ns = time.perf_counter_ns()
                try:
                    ret = await runnable.ainvoke(*a, **kw)

                    # If ret is an async generator, wrap it to collect content
                    if ret is not None and (
                        hasattr(ret, "__aiter__") or inspect.isasyncgen(ret)
                    ):

                        async def _streaming_wrapper(gen):
                            collected: list[str] = []
                            ttft_ns: Optional[int] = None
                            first_chunk_received = False
                            try:
                                async for chunk in gen:
                                    # Record time to first token
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        ttft_ns = time.perf_counter_ns() - start_time_ns

                                    if isinstance(chunk, bytes):
                                        chunk_str = chunk.decode(
                                            "utf-8", errors="replace"
                                        )
                                    elif isinstance(chunk, dict):
                                        chunk_str = json.dumps(chunk)
                                    elif chunk is None:
                                        continue
                                    else:
                                        chunk_str = str(chunk)

                                    collected.append(chunk_str)
                                    yield chunk

                                _finish_ok(
                                    rec,
                                    snap,
                                    a,
                                    kw,
                                    {"content": "".join(collected)},
                                    intr,
                                    tok,
                                    time_to_first_token_ns=ttft_ns,
                                )
                            except BaseException as e:
                                _finish_err(rec, intr, tok, e)
                                raise

                        return _streaming_wrapper(ret)

                    _finish_ok(rec, snap, a, kw, ret, intr, tok)
                    return ret
                except BaseException as e:
                    _finish_err(rec, intr, tok, e)
                    raise

            setattr(_wrapped, WRAPPED_FLAG, True)
            return _wrapped

        def _wrap_runnable_invoke(node_name: str, runnable):
            def _wrapped(*a, **kw):
                rec, snap, intr, tok = _start(node_name, a, kw)
                # Record start time for TTFT calculation
                start_time_ns = time.perf_counter_ns()
                try:
                    ret = runnable.invoke(*a, **kw)

                    # If ret is an async generator, wrap it to collect content
                    if ret is not None and (
                        hasattr(ret, "__aiter__") or inspect.isasyncgen(ret)
                    ):

                        async def _streaming_wrapper(gen):
                            collected: list[str] = []
                            ttft_ns: Optional[int] = None
                            first_chunk_received = False
                            try:
                                async for chunk in gen:
                                    # Record time to first token
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        ttft_ns = time.perf_counter_ns() - start_time_ns

                                    if isinstance(chunk, bytes):
                                        chunk_str = chunk.decode(
                                            "utf-8", errors="replace"
                                        )
                                    elif isinstance(chunk, dict):
                                        chunk_str = json.dumps(chunk)
                                    elif chunk is None:
                                        continue
                                    else:
                                        chunk_str = str(chunk)

                                    collected.append(chunk_str)
                                    yield chunk

                                _finish_ok(
                                    rec,
                                    snap,
                                    a,
                                    kw,
                                    {"content": "".join(collected)},
                                    intr,
                                    tok,
                                    time_to_first_token_ns=ttft_ns,
                                )
                            except BaseException as e:
                                _finish_err(rec, intr, tok, e)
                                raise

                        return _streaming_wrapper(ret)

                    _finish_ok(rec, snap, a, kw, ret, intr, tok)
                    return ret
                except BaseException as e:
                    _finish_err(rec, intr, tok, e)
                    raise

            setattr(_wrapped, WRAPPED_FLAG, True)
            return _wrapped

        def wrap_callable(node_name: str, func: Any):
            if getattr(func, WRAPPED_FLAG, False):
                return func

            # Runnable-like objects
            if hasattr(func, "ainvoke"):
                return _wrap_runnable_ainvoke(node_name, func)
            if hasattr(func, "invoke"):
                return _wrap_runnable_invoke(node_name, func)

            # Functions
            if inspect.isasyncgenfunction(func):
                return _wrap_async_gen(node_name, func)
            if inspect.iscoroutinefunction(func):
                return _wrap_async_func(node_name, func)
            if inspect.isfunction(func):
                return _wrap_sync_func(node_name, func)

            # Unknown type -> leave untouched
            return func

        def wrapped_add_node(graph_self, *args, **kwargs):
            # Handle both call signatures:
            # 1. add_node(func) - single arg
            # 2. add_node(name, func) - two args

            if len(args) == 1:
                func = args[0]
                # Infer name from function
                name = getattr(func, "__name__", str(func))
                wrapped_func = wrap_callable(name, func)
                return original_add_node(graph_self, wrapped_func, **kwargs)

            elif len(args) >= 2:
                name = args[0]
                func = args[1]
                wrapped_func = wrap_callable(name, func)
                return original_add_node(
                    graph_self, name, wrapped_func, *args[2:], **kwargs
                )

            # Fallback for edge cases
            return original_add_node(graph_self, *args, **kwargs)

        StateGraph.add_node = wrapped_add_node
        self._installed = True