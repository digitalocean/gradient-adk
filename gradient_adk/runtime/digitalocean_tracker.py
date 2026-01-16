from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Mapping
from dataclasses import dataclass, field
import inspect
import json
from typing import Any, Callable, Dict, List, Optional
import uuid

from gradient_adk.digital_ocean_api import (
    AsyncDigitalOceanGenAI,
    CreateTracesInput,
    Trace,
    Span,
    TraceSpanType,
    SpanCommon,
    LLMSpanDetails,
    ToolSpanDetails,
    RetrieverSpanDetails,
    WorkflowSpanDetails,
)
from .interfaces import NodeExecution
from .network_interceptor import (
    set_request_captured_list,
    reset_request_captured_list,
)

from datetime import datetime, timezone
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse


def _utc(dt: datetime | None = None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# Import CapturedRequest for type annotation
from .network_interceptor import CapturedRequest


@dataclass
class RequestState:
    """Per-request state for tracking spans and trace data.
    
    This class holds all the mutable state needed during a single request's
    lifecycle. Using this with contextvars ensures proper isolation between
    concurrent requests.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    req: Dict[str, Any] = field(default_factory=dict)
    live: Dict[str, NodeExecution] = field(default_factory=dict)
    done: List[NodeExecution] = field(default_factory=list)
    is_evaluation: bool = False
    session_id: Optional[str] = None
    captured_requests: List[CapturedRequest] = field(default_factory=list)  # Per-request HTTP calls
    _captured_list_token: Optional[contextvars.Token] = field(default=None, repr=False)


# Context variable for request-scoped state
# Each async task/request gets its own copy of the RequestState
_request_state: contextvars.ContextVar[Optional[RequestState]] = contextvars.ContextVar(
    "request_state", default=None
)


def get_current_request_state() -> Optional[RequestState]:
    """Get the current request state from context."""
    return _request_state.get()


def set_current_request_state(state: Optional[RequestState]) -> contextvars.Token:
    """Set the current request state in context. Returns a token for resetting."""
    return _request_state.set(state)


def reset_request_state(token: contextvars.Token) -> None:
    """Reset the request state to its previous value."""
    _request_state.reset(token)


class DigitalOceanTracesTracker:
    """Collect executions and submit a single trace on request end.
    
    This tracker uses contextvars to maintain per-request state, ensuring
    proper isolation when multiple requests are processed concurrently.
    Each request gets its own RequestState that tracks:
    - The request metadata (inputs, outputs, errors)
    - In-progress span executions (live)
    - Completed span executions (done)
    - Evaluation mode flag
    - Session ID
    """

    def __init__(
        self,
        *,
        client: AsyncDigitalOceanGenAI,
        agent_workspace_name: str,
        agent_deployment_name: str,
    ) -> None:
        self._client = client
        self._ws = agent_workspace_name
        self._dep = agent_deployment_name

        # Global state that's safe to share across requests
        self._inflight: set[asyncio.Task] = set()
        self._collection_tasks: set[asyncio.Task] = (
            set()
        )  # Tasks collecting async generator outputs

        # Legacy instance variables for backward compatibility with tests
        # These are now only used when no request context is active
        self._req: Dict[str, Any] = {}
        self._live: dict[str, NodeExecution] = {}
        self._done: List[NodeExecution] = []
        self._is_evaluation: bool = False
        self._session_id: Optional[str] = None

    def _get_state(self) -> RequestState:
        """Get the current request state from context, or fall back to legacy instance state."""
        state = get_current_request_state()
        if state is not None:
            return state
        
        # Fallback: wrap legacy instance state (for tests/simple usage)
        return RequestState(
            request_id="legacy",
            req=self._req,
            live=self._live,
            done=self._done,
            is_evaluation=self._is_evaluation,
            session_id=self._session_id,
        )

    def on_request_start(
        self,
        entrypoint: str,
        inputs: Dict[str, Any],
        is_evaluation: bool = False,
        session_id: Optional[str] = None,
    ) -> contextvars.Token:
        """Start tracking a new request.
        
        Creates a new RequestState in the context for this request.
        Returns a token that can be used to reset the context when the request ends.
        
        IMPORTANT: The caller should store the returned token and call
        reset_request_state(token) when the request completes to properly
        clean up the context.
        """
        # Create fresh state for this request (isolated from other concurrent requests)
        state = RequestState(
            is_evaluation=is_evaluation,
            session_id=session_id,
        )
        state.req = {"entrypoint": entrypoint, "inputs": inputs}
        
        # Set up per-request captured list for network interceptor
        # This ensures concurrent requests see only their own HTTP calls
        captured_list_token = set_request_captured_list(state.captured_requests)
        state._captured_list_token = captured_list_token
        
        # Set in context and return token for cleanup
        token = set_current_request_state(state)
        
        # Also update legacy instance variables for backward compatibility
        self._live = state.live
        self._done = state.done
        self._is_evaluation = is_evaluation
        self._session_id = session_id
        self._req = state.req
        
        return token

    def _as_async_iterable_and_setter(
        self, resp
    ) -> Optional[tuple[object, Callable[[object], None]]]:
        """
        If `resp` is a streaming response (FastAPIStreamingResponse or async iterator),
        return (orig_iterable, setter) so we can wrap it for tracking. Else None.

        Note: The decorator handles tracking internally for streaming, so this is mainly
        for edge cases or legacy usage.
        """
        # Handle FastAPIStreamingResponse (from decorator or direct usage)
        if isinstance(resp, FastAPIStreamingResponse):
            # FastAPI/Starlette StreamingResponse stores the iterator in various ways
            # Try common attribute names
            content = (
                getattr(resp, "body_iterator", None)
                or getattr(resp, "iterator", None)
                or getattr(resp, "content", None)
            )

            if content is None:
                return None

            # Check if it's an async iterator/generator
            if hasattr(content, "__aiter__") or inspect.isasyncgen(content):

                def _setter(new_iterable):
                    # Try to set the iterator in the response object
                    # Note: This may not work for all FastAPI versions, but we try
                    if hasattr(resp, "body_iterator"):
                        resp.body_iterator = new_iterable
                    elif hasattr(resp, "iterator"):
                        resp.iterator = new_iterable
                    elif hasattr(resp, "content"):
                        resp.content = new_iterable

                return content, _setter

        # Handle raw async iterators/generators (direct usage - less common)
        if hasattr(resp, "__aiter__") or inspect.isasyncgen(resp):
            # For raw iterators, we can't replace them in place, but we can still
            # wrap them if needed. This case is rare since most go through FastAPIStreamingResponse.
            def _setter(new_iterable):
                # Can't replace the iterator itself
                pass

            return resp, _setter

        return None

    def on_request_end(self, outputs: Any | None, error: Optional[str]) -> None:
        """End tracking for the current request.
        
        This method uses the request state from context to ensure proper
        isolation between concurrent requests.
        """
        state = self._get_state()
        
        # Common fields
        state.req["error"] = error
        # Also update legacy instance variable for backward compatibility
        self._req["error"] = error

        # Streaming path
        wrapped = self._as_async_iterable_and_setter(outputs)
        if wrapped is not None:
            orig_iterable, set_iterable = wrapped
            state.req["outputs"] = None  # will be filled after streaming finishes
            self._req["outputs"] = None

            # Capture state reference for use in the async generator
            captured_state = state

            async def collecting_iter():
                collected: list[str] = []
                try:
                    async for chunk in orig_iterable:
                        # Convert chunk to string for collection
                        if isinstance(chunk, bytes):
                            chunk_str = chunk.decode("utf-8", errors="replace")
                        elif isinstance(chunk, dict):
                            # For dict chunks, try to extract meaningful content
                            content = (
                                chunk.get("content")
                                or chunk.get("data")
                                or chunk.get("delta")
                                or json.dumps(chunk)
                            )
                            chunk_str = str(content)
                        elif chunk is None:
                            # Skip None values
                            continue
                        else:
                            chunk_str = str(chunk)

                        collected.append(chunk_str)
                        yield chunk

                    # Stream complete - submit collected content
                    captured_state.req["outputs"] = "".join(collected)
                    self._req["outputs"] = "".join(collected)
                    await self._submit_state(captured_state)
                except Exception as e:
                    # Error during streaming
                    captured_state.req["error"] = str(e)
                    captured_state.req["outputs"] = "".join(collected) if collected else None
                    self._req["error"] = str(e)
                    self._req["outputs"] = "".join(collected) if collected else None
                    await self._submit_state(captured_state)
                    raise

            set_iterable(collecting_iter())
            return  # important: don't submit yet

        # Non-streaming - always fire-and-forget
        # For evaluation mode, decorator will call submit_and_get_trace_id() directly
        state.req["outputs"] = outputs
        self._req["outputs"] = outputs

        if not state.is_evaluation:
            # Regular fire-and-forget for non-evaluation requests
            # Capture state for the async task
            captured_state = state
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self._submit_state(captured_state))
                self._inflight.add(task)

                def _done_cb(t: asyncio.Task) -> None:
                    self._inflight.discard(t)
                    try:
                        t.result()
                    except Exception:
                        pass

                task.add_done_callback(_done_cb)
            except RuntimeError:
                asyncio.run(self._submit_state(captured_state))

    async def submit_and_get_trace_id(self) -> Optional[str]:
        """
        Submit the trace and return the trace_id.
        Only call this for evaluation requests after on_request_end has been called.
        """
        state = self._get_state()
        return await self._submit_state(state)

    def on_node_start(self, node: NodeExecution) -> None:
        """Start tracking a node execution.
        
        Uses the current request state from context for proper isolation.
        """
        state = self._get_state()
        state.live[node.node_id] = node
        # Also update legacy instance variable for backward compatibility
        self._live[node.node_id] = node

    def on_node_end(self, node: NodeExecution, outputs: Any | None) -> None:
        """End tracking for a node execution.
        
        Uses the current request state from context for proper isolation.
        """
        state = self._get_state()
        
        # Check if we're using context-based state or legacy instance state
        # If state.done is the same object as self._done, we're in legacy mode
        # and should not double-add
        using_legacy = state.done is self._done
        
        # Try to get from state first, fall back to legacy
        live = state.live.pop(node.node_id, None)
        if live is None:
            live = self._live.pop(node.node_id, node)
        else:
            # Also remove from legacy for consistency (only if different objects)
            if not using_legacy:
                self._live.pop(node.node_id, None)
            
        live.end_time = _utc()

        # Handle async generators that may have slipped through
        # NOTE: The instrumentors should now wrap async generators and collect content
        # before calling on_node_end. This is a fallback for any edge cases.
        if outputs is not None:
            is_async_gen = hasattr(outputs, "__aiter__") or inspect.isasyncgen(outputs)
            is_stringified_gen = (
                isinstance(outputs, str)
                and "<async_generator" in outputs
                and "at 0x" in outputs
            )
            if is_async_gen:
                # This shouldn't happen if instrumentors are working correctly
                live.outputs = {
                    "streaming": True,
                    "_debug": "Async generator reached on_node_end without collection. Check instrumentor.",
                }
                state.done.append(live)
                return
            if is_stringified_gen:
                # This happens when _freeze() is called on an async generator
                live.outputs = {
                    "streaming": True,
                    "_debug": "Async generator was converted to string. Instrumentor should not freeze async generators.",
                }
                state.done.append(live)
                return

        live.outputs = outputs
        state.done.append(live)

    async def _collect_async_generator_outputs(self, node: NodeExecution, gen) -> None:
        """Collect content from an async generator and update node outputs.

        This handles cases where a node function returns an async generator
        (e.g., streaming LLM responses) rather than being an async generator function itself.
        """
        collected: list[str] = []
        try:
            # If gen is awaitable (coroutine), await it first
            if hasattr(gen, "__await__"):
                gen = await gen

            async for chunk in gen:
                # Convert chunk to string for collection
                if isinstance(chunk, bytes):
                    chunk_str = chunk.decode("utf-8", errors="replace")
                elif isinstance(chunk, dict):
                    # For dict chunks, try to extract meaningful content
                    # Common patterns: LLM delta chunks, status updates, etc.
                    content = (
                        chunk.get("content")
                        or chunk.get("data")
                        or chunk.get("delta")
                        or chunk.get("text")
                        or json.dumps(chunk)
                    )
                    chunk_str = str(content)
                elif chunk is None:
                    continue
                else:
                    chunk_str = str(chunk)

                collected.append(chunk_str)

            # Update the node's outputs with collected content
            # Wrap in dict to match expected format
            if collected:
                node.outputs = {"content": "".join(collected)}
            else:
                node.outputs = {"content": ""}
        except Exception as e:
            # On error, store error message
            node.outputs = {"error": f"Error collecting stream: {str(e)}"}

    def on_node_error(self, node: NodeExecution, error: BaseException) -> None:
        """Record an error for a node execution.
        
        Uses the current request state from context for proper isolation.
        """
        state = self._get_state()
        
        # Check if we're using context-based state or legacy instance state
        # If state.done is the same object as self._done, we're in legacy mode
        # and should not double-add
        using_legacy = state.done is self._done
        
        # Try to get from state first, fall back to legacy
        live = state.live.pop(node.node_id, None)
        if live is None:
            live = self._live.pop(node.node_id, node)
        else:
            # Also remove from legacy for consistency (only if different objects)
            if not using_legacy:
                self._live.pop(node.node_id, None)
            
        live.end_time = _utc()
        live.error = str(error)
        state.done.append(live)

    async def aclose(self) -> None:
        # Wait for both collection tasks and submission tasks
        all_tasks = list(self._inflight) + list(self._collection_tasks)
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
            self._inflight.clear()
            self._collection_tasks.clear()
        await self._client.aclose()

    async def _submit(self) -> Optional[str]:
        """Submit the trace using legacy instance state (backward compatibility)."""
        try:
            # Wait for any pending async generator collections to complete
            # This ensures node outputs are fully collected before building the trace
            if self._collection_tasks:
                await asyncio.gather(
                    *list(self._collection_tasks), return_exceptions=True
                )

            trace = self._build_trace()
            req = CreateTracesInput(
                agent_workspace_name=self._ws,
                agent_deployment_name=self._dep,
                traces=[trace],
                session_id=getattr(self, "_session_id", None),
            )
            result = await self._client.create_traces(req)
            # Return first trace_uuid if available
            if result.trace_uuids:
                return result.trace_uuids[0]
            return None
        except Exception:
            # never break user code on export errors
            return None

    async def _submit_state(self, state: RequestState) -> Optional[str]:
        """Submit a trace using the provided request state.
        
        This is the concurrent-safe submission method that uses explicit state
        rather than relying on instance variables.
        
        Args:
            state: The RequestState containing the spans and request metadata
                   for this specific request.
        
        Returns:
            The trace UUID if submission succeeded, None otherwise.
        """
        try:
            # Wait for any pending async generator collections to complete
            # This ensures node outputs are fully collected before building the trace
            if self._collection_tasks:
                await asyncio.gather(
                    *list(self._collection_tasks), return_exceptions=True
                )

            trace = self._build_trace_from_state(state)
            req = CreateTracesInput(
                agent_workspace_name=self._ws,
                agent_deployment_name=self._dep,
                traces=[trace],
                session_id=state.session_id,
            )
            result = await self._client.create_traces(req)
            # Return first trace_uuid if available
            if result.trace_uuids:
                return result.trace_uuids[0]
            return None
        except Exception:
            # never break user code on export errors
            return None

    def _build_trace_from_state(self, state: RequestState) -> Trace:
        """Build a Trace from the provided request state.
        
        This is the concurrent-safe trace building method that uses explicit state
        rather than relying on instance variables.
        
        Args:
            state: The RequestState containing the spans and request metadata
                   for this specific request.
        
        Returns:
            A Trace object ready for submission.
        """
        spans = [self._to_span(ex) for ex in state.done]
        created_at = min((s.created_at for s in spans), default=_utc())
        name = str(state.req.get("entrypoint", "request"))

        inputs = self._coerce_top(state.req.get("inputs"), "input")
        outputs = self._coerce_top(state.req.get("outputs"), "output")

        # If there was a request-level error, include it in the top-level output
        if state.req.get("error") is not None:
            outputs = dict(outputs)
            outputs["error"] = state.req["error"]

        trace = Trace(
            created_at=created_at,
            name=name,
            input=inputs,
            output=outputs,
            spans=spans,
        )
        return trace

    def _to_span(self, ex: NodeExecution) -> Span:
        # Base payloads - keep dicts as-is, wrap everything else
        if isinstance(ex.inputs, dict):
            inp = ex.inputs
        else:
            inp = {"input": ex.inputs}

        if isinstance(ex.outputs, dict):
            out = ex.outputs
        else:
            out = {"output": ex.outputs}

        # include error (if any) and matched endpoints (if present)
        if ex.error is not None:
            out = dict(out)
            out["error"] = ex.error
        if ex.metadata and ex.metadata.get("llm_endpoints"):
            out = dict(out)
            out["_llm_endpoints"] = list(ex.metadata["llm_endpoints"])

        # classify span type via metadata set by the instrumentor
        metadata = ex.metadata or {}

        # Check if this is a workflow span
        if metadata.get("is_workflow"):
            span_type = TraceSpanType.TRACE_SPAN_TYPE_WORKFLOW

            # Build sub-spans from the workflow's collected spans
            sub_spans_list = metadata.get("sub_spans", [])
            sub_spans = [self._to_span(sub) for sub in sub_spans_list]

            # Calculate duration from start to end
            duration_ns = None
            if ex.start_time and ex.end_time:
                duration_ns = int(
                    (ex.end_time - ex.start_time).total_seconds() * 1_000_000_000
                )

            # Build common fields
            common = SpanCommon(
                duration_ns=duration_ns,
                metadata={"agent_name": metadata.get("agent_name")},
                status_code=200 if ex.error is None else 500,
            )

            # Build workflow details with nested sub-spans
            workflow_details = WorkflowSpanDetails(spans=sub_spans)

            return Span(
                created_at=_utc(ex.start_time),
                name=ex.node_name,
                input=inp,
                output=out,
                type=span_type,
                common=common,
                workflow=workflow_details,
            )
        elif metadata.get("is_llm_call"):
            span_type = TraceSpanType.TRACE_SPAN_TYPE_LLM

            # For programmatic API, only use user-provided duration_ns
            # For decorators/automatic instrumentation, auto-calculate from start/end times
            duration_ns = metadata.get("duration_ns")
            if duration_ns is None and not metadata.get("is_programmatic"):
                if ex.start_time and ex.end_time:
                    duration_ns = int(
                        (ex.end_time - ex.start_time).total_seconds() * 1_000_000_000
                    )

            # Build LLM-specific details
            llm_common = SpanCommon(
                duration_ns=duration_ns,
                metadata=metadata.get("custom_metadata"),
                tags=metadata.get("tags"),
                status_code=metadata.get("status_code", 200 if ex.error is None else 500),
            )

            # Extract LLM-specific fields from captured API payloads
            llm_request = metadata.get("llm_request_payload", {}) or {}
            llm_response = metadata.get("llm_response_payload", {}) or {}

            # For LLM spans, use just the messages as input (not the full request payload)
            # Must be a dict (not array) because protobuf Struct requires key-value pairs
            if isinstance(llm_request, dict) and "messages" in llm_request:
                inp = {"messages": llm_request.get("messages")}

            # For LLM spans, use just the choices as output (not the full response payload)
            # Must be a dict (not array) because protobuf Struct requires key-value pairs
            if isinstance(llm_response, dict) and "choices" in llm_response:
                out = {"choices": llm_response.get("choices")}

            # Extract model from request payload, fallback to metadata or node name
            model = (
                llm_request.get("model")
                or metadata.get("model_name")
                or ex.node_name.replace("llm:", "")
            )

            # Extract tools from request payload
            tools = llm_request.get("tools") if isinstance(llm_request, dict) else None

            # Extract temperature from request payload
            temperature = llm_request.get("temperature") if isinstance(llm_request, dict) else None

            # Extract token counts from response payload
            num_input_tokens = None
            num_output_tokens = None
            total_tokens = None
            if isinstance(llm_response, dict):
                usage = llm_response.get("usage", {})
                if isinstance(usage, dict):
                    num_input_tokens = usage.get("prompt_tokens")
                    num_output_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")

            # Get time-to-first-token for streaming calls
            time_to_first_token_ns = metadata.get("time_to_first_token_ns")

            llm_details = LLMSpanDetails(
                common=llm_common,
                model=model,
                tools=tools,
                temperature=temperature,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                total_tokens=total_tokens,
                time_to_first_token_ns=time_to_first_token_ns,
            )

            return Span(
                created_at=_utc(ex.start_time),
                name=ex.node_name,
                input=inp,
                output=out,
                type=span_type,
                llm=llm_details,
            )
        elif metadata.get("is_retriever_call"):
            span_type = TraceSpanType.TRACE_SPAN_TYPE_RETRIEVER

            # Calculate duration
            duration_ns = None
            if ex.start_time and ex.end_time:
                duration_ns = int(
                    (ex.end_time - ex.start_time).total_seconds() * 1_000_000_000
                )

            # Build retriever-specific details
            retriever_common = SpanCommon(
                duration_ns=duration_ns,
                status_code=200 if ex.error is None else 500,
            )

            retriever_details = RetrieverSpanDetails(common=retriever_common)

            return Span(
                created_at=_utc(ex.start_time),
                name=ex.node_name,
                input=inp,
                output=out,
                type=span_type,
                retriever=retriever_details,
            )
        elif metadata.get("is_agent_call"):
            span_type = TraceSpanType.TRACE_SPAN_TYPE_AGENT

            # For programmatic API, only use user-provided duration_ns
            # For decorators/automatic instrumentation, auto-calculate from start/end times
            duration_ns = metadata.get("duration_ns")
            if duration_ns is None and not metadata.get("is_programmatic"):
                if ex.start_time and ex.end_time:
                    duration_ns = int(
                        (ex.end_time - ex.start_time).total_seconds() * 1_000_000_000
                    )

            # Build agent-specific details
            agent_common = SpanCommon(
                duration_ns=duration_ns,
                metadata=metadata.get("custom_metadata"),
                tags=metadata.get("tags"),
                status_code=metadata.get("status_code", 200 if ex.error is None else 500),
            )

            return Span(
                created_at=_utc(ex.start_time),
                name=ex.node_name,
                input=inp,
                output=out,
                type=span_type,
                common=agent_common,
            )
        elif metadata.get("is_tool_call"):
            span_type = TraceSpanType.TRACE_SPAN_TYPE_TOOL

            # For programmatic API, only use user-provided duration_ns
            # For decorators/automatic instrumentation, auto-calculate from start/end times
            duration_ns = metadata.get("duration_ns")
            if duration_ns is None and not metadata.get("is_programmatic"):
                if ex.start_time and ex.end_time:
                    duration_ns = int(
                        (ex.end_time - ex.start_time).total_seconds() * 1_000_000_000
                    )

            # Build tool-specific details
            tool_common = SpanCommon(
                duration_ns=duration_ns,
                metadata=metadata.get("custom_metadata"),
                tags=metadata.get("tags"),
                status_code=metadata.get("status_code", 200 if ex.error is None else 500),
            )

            tool_details = ToolSpanDetails(
                common=tool_common,
                tool_call_id=metadata.get("tool_call_id"),
            )

            return Span(
                created_at=_utc(ex.start_time),
                name=ex.node_name,
                input=inp,
                output=out,
                type=span_type,
                tool=tool_details,
            )
        else:
            # Default to tool span (for backward compatibility)
            span_type = TraceSpanType.TRACE_SPAN_TYPE_TOOL

            # Calculate duration
            duration_ns = None
            if ex.start_time and ex.end_time:
                duration_ns = int(
                    (ex.end_time - ex.start_time).total_seconds() * 1_000_000_000
                )

            # Build tool-specific details
            tool_common = SpanCommon(
                duration_ns=duration_ns,
                status_code=200 if ex.error is None else 500,
            )

            tool_details = ToolSpanDetails(common=tool_common)

            return Span(
                created_at=_utc(ex.start_time),
                name=ex.node_name,
                input=inp,
                output=out,
                type=span_type,
                tool=tool_details,
            )

    def _coerce_top(self, val: Any, kind: str) -> Dict[str, Any]:
        """
        Normalize top-level trace input/output to a dict:
        - if already a Mapping -> copy to dict
        - if None -> {}
        - else -> {"input": val} or {"result": val} depending on kind
        """
        if val is None:
            return {}
        if isinstance(val, Mapping):
            return dict(val)
        return {"input": val} if kind == "input" else {"result": val}

    def _build_trace(self) -> Trace:
        spans = [self._to_span(ex) for ex in self._done]
        created_at = min((s.created_at for s in spans), default=_utc())
        name = str(self._req.get("entrypoint", "request"))

        inputs = self._coerce_top(self._req.get("inputs"), "input")
        outputs = self._coerce_top(self._req.get("outputs"), "output")

        # If there was a request-level error, include it in the top-level output
        if self._req.get("error") is not None:
            outputs = dict(outputs)
            outputs["error"] = self._req["error"]

        trace = Trace(
            created_at=created_at,
            name=name,
            input=inputs,
            output=outputs,
            spans=spans,
        )
        return trace