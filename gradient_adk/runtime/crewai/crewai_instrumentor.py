from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Dict, List

from ..interfaces import NodeExecution
from ..digitalocean_tracker import DigitalOceanTracesTracker
from ..network_interceptor import (
    get_network_interceptor,
    get_request_captured_list,
    is_inference_url,
    is_kbaas_url,
)


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _mk_exec(name: str, inputs: Any, framework: str = "crewai") -> NodeExecution:
    return NodeExecution(
        node_id=str(uuid.uuid4()),
        node_name=name,
        framework=framework,
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
        from pydantic import BaseModel

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

    # CrewAI specific types
    try:
        # Handle TaskOutput
        if hasattr(obj, "raw") and hasattr(obj, "pydantic"):
            return _freeze({"raw": obj.raw}, depth - 1)
        # Handle CrewOutput
        if hasattr(obj, "raw") and hasattr(obj, "tasks_output"):
            return _freeze({"raw": obj.raw}, depth - 1)
    except Exception:
        pass

    # fallback
    return repr(obj)


def _snap():
    """Snapshot the current state for tracking HTTP calls during a span.

    Returns:
        (interceptor, snapshot_index) where snapshot_index is the current length
        of the per-request captured list (or 0 if not in a request context).
    """
    intr = get_network_interceptor()
    # Use per-request captured list length as the snapshot token
    request_list = get_request_captured_list()
    if request_list is not None:
        tok = len(request_list)
    else:
        # Fallback to global token (for tests/simple usage without request context)
        try:
            tok = intr.snapshot_token()
        except Exception:
            tok = 0
    return intr, tok


def _had_hits_since(intr, token) -> bool:
    """Check if any tracked HTTP calls happened since the snapshot.

    Uses per-request captured list for proper isolation in concurrent scenarios.
    """
    request_list = get_request_captured_list()
    if request_list is not None:
        return len(request_list) > token
    # Fallback to global interceptor
    try:
        return intr.hits_since(token) > 0
    except Exception:
        return False


def _get_captured_payloads_with_type(intr, token) -> tuple:
    """Get captured API request/response payloads and classify the call type.

    Uses per-request captured list for proper isolation in concurrent scenarios.

    Returns:
        (request_payload, response_payload, is_llm, is_retriever)
    """
    try:
        # Use per-request captured list for concurrent isolation
        request_list = get_request_captured_list()
        if request_list is not None:
            # Get requests captured since the snapshot token
            captured = request_list[token:] if token < len(request_list) else []
        else:
            # Fallback to global interceptor (for tests/simple usage)
            captured = intr.get_captured_requests_since(token)

        if captured:
            # Search in reverse order to find a captured request with a response
            for call in reversed(captured):
                if call.response_payload is not None:
                    url = call.url
                    is_llm = is_inference_url(url)
                    is_retriever = is_kbaas_url(url)
                    return call.request_payload, call.response_payload, is_llm, is_retriever

            # Fallback to the first captured request if none have a response
            call = captured[0]
            url = call.url
            is_llm = is_inference_url(url)
            is_retriever = is_kbaas_url(url)
            return call.request_payload, call.response_payload, is_llm, is_retriever
    except Exception:
        pass
    return None, None, False, False


def _transform_kbaas_response(response: Optional[Dict[str, Any]]) -> Optional[list]:
    """Transform KBaaS response to standard retriever format."""
    if not isinstance(response, dict):
        return response

    results = response.get("results", [])
    if not isinstance(results, list):
        return response

    transformed_results = []
    for item in results:
        if isinstance(item, dict):
            new_item = dict(item)

            if "parent_chunk_text" in new_item:
                new_item["page_content"] = new_item.pop("parent_chunk_text")
                if "text_content" in new_item:
                    new_item["embedded_content"] = new_item.pop("text_content")
            elif "text_content" in new_item:
                new_item["page_content"] = new_item.pop("text_content")

            transformed_results.append(new_item)
        else:
            transformed_results.append(item)

    return transformed_results


# ---- Agent Context Management ----
# NOTE: Using a global list instead of ContextVars because CrewAI's event bus
# callbacks may run in the same thread context, and we need to share state
# between agent start/end events and LLM/tool events.

import threading


@dataclass
class AgentContext:
    """Context for tracking an agent execution and its sub-spans."""

    node: NodeExecution
    sub_spans: List[NodeExecution] = field(default_factory=list)
    agent_role: str = ""


# Thread-safe global stack for agent contexts
_agent_stack_lock = threading.Lock()
_agent_stack: List[AgentContext] = []


def _get_current_agent() -> Optional[AgentContext]:
    """Get the current agent context, if any."""
    with _agent_stack_lock:
        return _agent_stack[-1] if _agent_stack else None


def _push_agent(ctx: AgentContext) -> None:
    """Push an agent context onto the stack."""
    with _agent_stack_lock:
        _agent_stack.append(ctx)


def _pop_agent() -> Optional[AgentContext]:
    """Pop an agent context from the stack."""
    with _agent_stack_lock:
        if _agent_stack:
            return _agent_stack.pop()
        return None


class CrewAIInstrumentor:
    """Wraps CrewAI agents with tracing using the CrewAI event bus."""

    def __init__(self) -> None:
        self._installed = False
        self._tracker: Optional[DigitalOceanTracesTracker] = None
        self._handler_ids: List[Any] = []

    def install(self, tracker: DigitalOceanTracesTracker) -> None:
        if self._installed:
            return
        self._tracker = tracker

        try:
            from crewai.events import crewai_event_bus
            from crewai.events import (
                AgentExecutionStartedEvent,
                AgentExecutionCompletedEvent,
                AgentExecutionErrorEvent,
                LLMCallStartedEvent,
                LLMCallCompletedEvent,
                LLMCallFailedEvent,
                ToolUsageStartedEvent,
                ToolUsageFinishedEvent,
                ToolUsageErrorEvent,
            )
        except ImportError:
            # CrewAI not installed or doesn't have events module
            return

        t = tracker  # close over

        def _start_sub_span(node_name: str, inputs: Any):
            """Start a sub-span that will be nested inside the current agent."""
            inputs_snapshot = _freeze(inputs)
            rec = _mk_exec(node_name, inputs_snapshot)
            intr, tok = _snap()

            # Check if we're inside an agent context
            agent_ctx = _get_current_agent()
            if agent_ctx is None:
                # No agent context - fall back to flat spans
                t.on_node_start(rec)

            return rec, inputs_snapshot, intr, tok

        def _finish_sub_span_ok(
            rec: NodeExecution,
            inputs_snapshot: Any,
            ret: Any,
            intr,
            tok,
            time_to_first_token_ns: Optional[int] = None,
        ):
            """Finish a sub-span successfully.
            
            Uses network interceptor to classify span type (LLM vs retriever),
            following the same pattern as LangGraph instrumentor.
            
            Important: Spans that were explicitly started as tools (via ToolUsageStartedEvent)
            preserve their tool classification. This is because during tool execution,
            CrewAI may make LLM calls internally, and we don't want to reclassify the
            tool span as an LLM span.
            """
            out_payload = _freeze(ret)
            meta = _ensure_meta(rec)
            
            # Check if this span was explicitly started as a tool
            # If so, preserve that classification (don't let LLM HTTP calls override it)
            is_explicit_tool = meta.get("is_tool_call") is True
            
            # Check if this node made any tracked API calls
            if _had_hits_since(intr, tok):
                # Get captured payloads and classify the call type based on URL
                api_request, api_response, is_llm, is_retriever = (
                    _get_captured_payloads_with_type(intr, tok)
                )
                
                # Network interceptor classification is authoritative for LLM/retriever
                # EXCEPT for spans explicitly started as tools
                if is_llm and not is_explicit_tool:
                    meta["is_llm_call"] = True
                    # Store raw API payloads for LLM field extraction in tracker
                    if api_request:
                        meta["llm_request_payload"] = api_request
                    if api_response:
                        meta["llm_response_payload"] = api_response
                    # Store time-to-first-token if this was a streaming call
                    if time_to_first_token_ns is not None:
                        meta["time_to_first_token_ns"] = time_to_first_token_ns
                elif is_retriever and not is_explicit_tool:
                    meta["is_retriever_call"] = True
                # If neither is_llm nor is_retriever, or if it's an explicit tool,
                # keep existing classification

                # Use API payloads for LLM and retriever spans (not tools)
                if (is_llm or is_retriever) and not is_explicit_tool:
                    if api_request:
                        rec.inputs = _freeze(api_request)

                    if api_response:
                        if is_retriever:
                            api_response = _transform_kbaas_response(api_response)
                        out_payload = _freeze(api_response)

            rec.end_time = _utc()
            rec.outputs = out_payload

            # Check if we're inside an agent context
            agent_ctx = _get_current_agent()
            if agent_ctx is not None:
                # Add to agent's sub-spans
                agent_ctx.sub_spans.append(rec)
            else:
                # No agent context - call tracker directly
                t.on_node_end(rec, out_payload)

        def _finish_sub_span_err(rec: NodeExecution, intr, tok, e: BaseException):
            """Finish a sub-span with an error.
            
            Uses network interceptor to classify span type (LLM vs retriever),
            following the same pattern as LangGraph instrumentor.
            
            Important: Spans that were explicitly started as tools preserve their
            tool classification.
            """
            meta = _ensure_meta(rec)
            
            # Check if this span was explicitly started as a tool
            is_explicit_tool = meta.get("is_tool_call") is True
            
            if _had_hits_since(intr, tok):
                # Get captured payloads and classify the call type based on URL
                api_request, api_response, is_llm, is_retriever = (
                    _get_captured_payloads_with_type(intr, tok)
                )
                
                # Network interceptor classification is authoritative for LLM/retriever
                # EXCEPT for spans explicitly started as tools
                if is_llm and not is_explicit_tool:
                    meta["is_llm_call"] = True
                    if api_request:
                        meta["llm_request_payload"] = api_request
                    if api_response:
                        meta["llm_response_payload"] = api_response
                elif is_retriever and not is_explicit_tool:
                    meta["is_retriever_call"] = True
                # If neither is_llm nor is_retriever, or if it's an explicit tool,
                # keep existing classification

                # Use API payloads for LLM and retriever spans (not tools)
                if (is_llm or is_retriever) and not is_explicit_tool and api_request:
                    rec.inputs = _freeze(api_request)

            rec.end_time = _utc()
            rec.error = str(e)

            # Check if we're inside an agent context
            agent_ctx = _get_current_agent()
            if agent_ctx is not None:
                # Add to agent's sub-spans
                agent_ctx.sub_spans.append(rec)
            else:
                # No agent context - call tracker directly
                t.on_node_error(rec, e)

        # Track active spans by event timestamp for correlation
        _active_llm_spans: Dict[str, tuple] = {}
        _active_tool_spans: Dict[str, tuple] = {}

        # ---- Agent Event Handlers ----

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event):
            """Handle agent execution start."""
            try:
                agent = getattr(event, "agent", None)
                agent_role = getattr(agent, "role", "Agent") if agent else "Agent"

                # Get task info if available
                task = getattr(event, "task", None)
                task_description = getattr(task, "description", "") if task else ""

                node_name = f"agent:{agent_role}"
                inputs_snapshot = _freeze({"task": task_description}) if task_description else {}

                agent_node = _mk_exec(node_name, inputs_snapshot)
                meta = _ensure_meta(agent_node)
                meta["is_workflow"] = True  # Use workflow span to enable sub-span rendering
                meta["agent_role"] = agent_role

                # Create agent context and push onto stack
                agent_ctx = AgentContext(
                    node=agent_node,
                    agent_role=agent_role,
                )
                _push_agent(agent_ctx)
            except Exception:
                pass

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event):
            """Handle agent execution completion."""
            try:
                agent_ctx = _pop_agent()
                if agent_ctx is None:
                    return

                agent_node = agent_ctx.node
                agent_node.end_time = _utc()

                # Extract output from event
                output = getattr(event, "output", None)
                if output is not None:
                    agent_node.outputs = {"output": _freeze(output)}
                else:
                    agent_node.outputs = {}

                # Store sub-spans in metadata
                meta = _ensure_meta(agent_node)
                meta["sub_spans"] = agent_ctx.sub_spans

                # Report the agent span to the tracker
                t.on_node_start(agent_node)
                t.on_node_end(agent_node, agent_node.outputs)
            except Exception:
                pass

        @crewai_event_bus.on(AgentExecutionErrorEvent)
        def on_agent_error(source, event):
            """Handle agent execution error."""
            try:
                agent_ctx = _pop_agent()
                if agent_ctx is None:
                    return

                agent_node = agent_ctx.node
                agent_node.end_time = _utc()

                # Extract error from event
                error = getattr(event, "error", None)
                if error is not None:
                    agent_node.error = str(error)
                else:
                    agent_node.error = "Unknown error"

                # Store sub-spans in metadata
                meta = _ensure_meta(agent_node)
                meta["sub_spans"] = agent_ctx.sub_spans

                # Report the agent span to the tracker
                t.on_node_start(agent_node)
                t.on_node_error(agent_node, Exception(agent_node.error))
            except Exception:
                pass

        # ---- LLM Event Handlers ----

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_started(source, event):
            """Handle LLM call start."""
            try:
                # Extract model name
                model = getattr(event, "model", None) or getattr(event, "llm", None)
                model_name = str(model) if model else "unknown"

                # Extract messages/prompt
                messages = getattr(event, "messages", None)
                prompt = getattr(event, "prompt", None)
                inputs = messages if messages else prompt

                node_name = f"llm:{model_name}"
                rec, snap, intr, tok = _start_sub_span(node_name, inputs)

                # Store metadata
                meta = _ensure_meta(rec)
                meta["is_llm_call"] = True
                if model_name != "unknown":
                    meta["model_name"] = model_name

                # Store for correlation with completion event
                event_id = str(id(event))
                _active_llm_spans[event_id] = (rec, snap, intr, tok)
            except Exception:
                pass

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_completed(source, event):
            """Handle LLM call completion."""
            try:
                # Try to find the matching start span
                event_id = str(id(event))
                
                # If we can't find by exact event id, use the most recent one
                if event_id not in _active_llm_spans and _active_llm_spans:
                    event_id = list(_active_llm_spans.keys())[-1]
                
                if event_id not in _active_llm_spans:
                    return

                rec, snap, intr, tok = _active_llm_spans.pop(event_id)

                # Extract response
                response = getattr(event, "response", None)
                output = getattr(event, "output", None)
                result = response if response is not None else output

                _finish_sub_span_ok(rec, snap, result, intr, tok)
            except Exception:
                pass

        @crewai_event_bus.on(LLMCallFailedEvent)
        def on_llm_failed(source, event):
            """Handle LLM call failure."""
            try:
                event_id = str(id(event))
                
                if event_id not in _active_llm_spans and _active_llm_spans:
                    event_id = list(_active_llm_spans.keys())[-1]
                
                if event_id not in _active_llm_spans:
                    return

                rec, _, intr, tok = _active_llm_spans.pop(event_id)

                # Extract error
                error = getattr(event, "error", None) or getattr(event, "exception", None)
                if error is None:
                    error = Exception("LLM call failed")

                _finish_sub_span_err(rec, intr, tok, error if isinstance(error, BaseException) else Exception(str(error)))
            except Exception:
                pass

        # ---- Tool Event Handlers ----

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(source, event):
            """Handle tool usage start."""
            try:
                # Extract tool name
                tool_name = getattr(event, "tool_name", None) or getattr(event, "name", None)
                if tool_name is None:
                    tool = getattr(event, "tool", None)
                    tool_name = getattr(tool, "name", "unknown_tool") if tool else "unknown_tool"

                # Extract tool inputs - try multiple attribute names
                tool_inputs = (
                    getattr(event, "tool_input", None) or 
                    getattr(event, "input", None) or 
                    getattr(event, "args", None) or
                    getattr(event, "arguments", None) or
                    getattr(event, "kwargs", None) or
                    getattr(event, "tool_args", None)
                )

                node_name = str(tool_name)
                rec, snap, intr, tok = _start_sub_span(node_name, tool_inputs)

                # Store metadata
                meta = _ensure_meta(rec)
                meta["is_tool_call"] = True
                meta["tool_name"] = str(tool_name)
                # Also store the tool inputs in metadata for later use
                if tool_inputs:
                    meta["tool_inputs"] = tool_inputs

                # Store for correlation with completion event
                event_id = str(id(event))
                _active_tool_spans[event_id] = (rec, snap, intr, tok)
            except Exception:
                pass

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_finished(source, event):
            """Handle tool usage completion."""
            try:
                event_id = str(id(event))
                
                if event_id not in _active_tool_spans and _active_tool_spans:
                    event_id = list(_active_tool_spans.keys())[-1]
                
                if event_id not in _active_tool_spans:
                    return

                rec, snap, intr, tok = _active_tool_spans.pop(event_id)

                # Try to extract tool input from finished event if not already set
                if rec.inputs is None or rec.inputs == {}:
                    tool_input = (
                        getattr(event, "tool_input", None) or
                        getattr(event, "input", None) or
                        getattr(event, "args", None) or
                        getattr(event, "arguments", None)
                    )
                    if tool_input:
                        rec.inputs = _freeze(tool_input)

                # Extract result
                result = getattr(event, "output", None) or getattr(event, "result", None)

                _finish_sub_span_ok(rec, snap, result, intr, tok)
            except Exception:
                pass

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_error(source, event):
            """Handle tool usage error."""
            try:
                event_id = str(id(event))
                
                if event_id not in _active_tool_spans and _active_tool_spans:
                    event_id = list(_active_tool_spans.keys())[-1]
                
                if event_id not in _active_tool_spans:
                    return

                rec, _, intr, tok = _active_tool_spans.pop(event_id)

                # Extract error
                error = getattr(event, "error", None) or getattr(event, "exception", None)
                if error is None:
                    error = Exception("Tool execution failed")

                _finish_sub_span_err(rec, intr, tok, error if isinstance(error, BaseException) else Exception(str(error)))
            except Exception:
                pass

        self._installed = True

    def uninstall(self) -> None:
        """Remove instrumentation hooks."""
        if not self._installed:
            return

        # Note: CrewAI's event bus doesn't provide a clean way to unregister handlers
        # The handlers will remain registered but will be no-ops since we clear the tracker
        self._tracker = None
        self._installed = False

    def is_installed(self) -> bool:
        """Check if instrumentation is currently installed."""
        return self._installed
