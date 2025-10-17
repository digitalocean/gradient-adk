"""
DigitalOcean traces integration (async) for execution tracking.

Uses AsyncDigitalOceanGenAI + typed Pydantic models to submit traces
to DigitalOcean's GenAI Traces API without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import dataclasses
import threading
from dataclasses import asdict
from collections.abc import Mapping, Sequence, Generator
from datetime import datetime, date, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Union
from uuid import UUID
from enum import Enum

from gradient_adk.digital_ocean_api import (
    AsyncDigitalOceanGenAI,
    CreateTracesInput,
    Trace,
    Span,
    TraceSpanType,
    DOAPIError,
)
from gradient_adk.logging import get_logger

from .interfaces import NodeExecution
from .context import get_current_context
from .tracker import DefaultExecutionTracker

logger = get_logger(__name__)


def _utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class DigitalOceanTracesTracker(DefaultExecutionTracker):
    """
    ExecutionTracker that submits traces to DigitalOcean using an async, typed client.

    Improvements:
      - Safe scheduling across threads/loops (no dead loops).
      - Heavy serialization offloaded with asyncio.to_thread.
      - Robust shutdown: waits for both Tasks and cross-loop Futures.
      - Defensive logging to avoid event-loop stalls from huge payloads.
    """

    def __init__(
        self,
        *,
        agent_workspace_name: str,
        agent_deployment_name: str,
        client: AsyncDigitalOceanGenAI,
        enable_auto_submit: bool = True,
        json_safe_max_depth: int = 6,
        max_logged_chars: int = 2_000,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        super().__init__()

        self._client = client
        self.agent_workspace_name = agent_workspace_name
        self.agent_deployment_name = agent_deployment_name
        self.enable_auto_submit = enable_auto_submit

        self._json_safe_max_depth = json_safe_max_depth
        self._max_logged_chars = max_logged_chars

        self._submitted_traces: Set[str] = set()
        self._submitting_traces: Set[str] = set()  # prevent duplicate scheduling

        self._loop: Optional[asyncio.AbstractEventLoop] = loop
        self._scheduler_lock = threading.Lock()
        self._inflight_tasks: Set[asyncio.Task] = set()
        self._inflight_futures: Set[concurrent.futures.Future] = set()

    def bind_event_loop(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """
        Bind a running event loop to this tracker.
        Call once from async context during initialization (recommended).
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop

    def _schedule_submit(self, rid: str) -> None:
        """
        Schedule _submit_current_trace safely:
          - If called from the bound loop thread: create_task.
          - If called from another thread: run_coroutine_threadsafe.
          - As a last resort: run in a one-off background thread using asyncio.run.
        """
        coro = self._submit_current_trace()

        # Try "current running loop" path first (we're likely on the main loop).
        try:
            running = asyncio.get_running_loop()
            task = running.create_task(coro)
            self._inflight_tasks.add(task)

            def _on_task_done(t: asyncio.Task) -> None:
                self._inflight_tasks.discard(t)
                self._submitting_traces.discard(rid)
                try:
                    success = t.result()
                    if success:
                        logger.debug("Trace submitted to DigitalOcean successfully")
                    else:
                        logger.warning("Failed to submit trace to DigitalOcean")
                except Exception as e:
                    logger.debug(
                        "Error during trace submission", error=str(e), exc_info=True
                    )

            task.add_done_callback(_on_task_done)
            return
        except RuntimeError:
            # Not in any running loop (probably a worker thread).
            pass

        # Cross-thread path using a bound loop.
        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            self._inflight_futures.add(fut)

            def _on_future_done(f: concurrent.futures.Future) -> None:
                self._inflight_futures.discard(f)
                self._submitting_traces.discard(rid)
                try:
                    success = f.result()
                    if success:
                        logger.debug("Trace submitted to DigitalOcean successfully")
                    else:
                        logger.warning("Failed to submit trace to DigitalOcean")
                except Exception as e:
                    logger.debug(
                        "Error during trace submission", error=str(e), exc_info=True
                    )

            fut.add_done_callback(_on_future_done)
            return

        # Last resort: one-off thread that runs the coroutine to completion.
        def _runner() -> None:
            try:
                success = asyncio.run(coro)
                if success:
                    logger.debug("Trace submitted to DigitalOcean successfully")
                else:
                    logger.warning("Failed to submit trace to DigitalOcean")
            except Exception as e:
                logger.warning(
                    "Error during trace submission",
                    error=str(e),
                    exc_info=True,
                )
            finally:
                self._submitting_traces.discard(rid)

        threading.Thread(target=_runner, daemon=True).start()

    def _map_node_to_span_type(
        self, node_name: str, framework: str, execution: Optional[NodeExecution] = None
    ) -> TraceSpanType:
        """Determine the appropriate span type based on node characteristics."""
        if execution and self._is_llm_node(execution):
            if logger.isEnabledFor(10):
                logger.debug("Detected LLM call in node", node_name=node_name)
            return TraceSpanType.TRACE_SPAN_TYPE_LLM
        return TraceSpanType.TRACE_SPAN_TYPE_TOOL

    def _is_llm_node(self, execution: NodeExecution) -> bool:
        """Check if a node execution involves calls to DigitalOcean inference endpoints."""
        if execution.metadata and execution.metadata.get("is_llm_call", False):
            return True
        do_inference_patterns = ["inference.do-ai.run", "inference.do-ai-test.run"]
        all_data = f"{execution.inputs}{execution.outputs}{execution.metadata}"
        return any(p in all_data for p in do_inference_patterns)

    def _span_from_execution(self, execution: NodeExecution) -> Span:
        """
        Convert NodeExecution → Span (protobuf-compatible model).
        Uses JSON-safe conversion; errors embedded into output.
        """
        span_type = self._map_node_to_span_type(
            execution.node_name, execution.framework, execution
        )

        raw_outputs = execution.outputs if execution.outputs is not None else {}
        output = self._unwrap_result_maybe(raw_outputs)

        out_map = self._to_mapping_or_none(output)
        if out_map is not None:
            output = out_map

        output = self._merge_error_into_output(output, execution.error)
        output = self._json_safe(output)

        raw_inputs = execution.inputs if execution.inputs is not None else {}
        first_arg = self._first_arg_or_none(raw_inputs)
        if first_arg is not None:
            input_payload = first_arg
            if logger.isEnabledFor(10):
                logger.debug("Using first arg as input", node_name=execution.node_name)
        else:
            in_map = self._to_mapping_or_none(raw_inputs)
            if in_map is not None:
                input_payload = in_map
            else:
                if isinstance(raw_inputs, Sequence) and not isinstance(
                    raw_inputs, (str, bytes, bytearray)
                ):
                    if self._is_kv_pairs(raw_inputs):
                        input_payload = {str(k): v for k, v in raw_inputs}
                    else:
                        input_payload = {"args": list(raw_inputs)}
                else:
                    input_payload = {"input": raw_inputs}

        input_payload = self._ensure_mapping(self._json_safe(input_payload))

        start = getattr(execution, "start_time", None)
        created_at = _utc(start) if start else _utc(datetime.utcnow())

        return Span(
            created_at=created_at,
            name=execution.node_name or "node",
            input=input_payload,
            output=output,
            type=span_type,
        )

    def _is_kv_pairs(self, seq) -> bool:
        if isinstance(seq, (str, bytes, bytearray)) or not isinstance(seq, Sequence):
            return False
        for item in seq:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes, bytearray))
                or len(item) != 2
            ):
                return False
        return True

    def _to_mapping_or_none(self, obj):
        """Return a dict-like mapping for obj if possible, else None."""
        if isinstance(obj, Mapping):
            return dict(obj)
        if dataclasses.is_dataclass(obj):
            return asdict(obj)

        md = getattr(obj, "model_dump", None)
        if callable(md):
            try:
                return obj.model_dump()
            except Exception:
                pass
        dd = getattr(obj, "dict", None)
        if callable(dd):
            try:
                return obj.dict()
            except Exception:
                pass

        if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__"), dict):
            return dict(vars(obj))

        if isinstance(obj, Sequence) and self._is_kv_pairs(obj):
            try:
                return {k: v for k, v in obj}
            except Exception:
                pass

        return None

    def _json_safe(self, obj, _depth=0) -> Any:
        """Recursively convert obj into JSON-serializable types."""
        if _depth > self._json_safe_max_depth:
            return str(obj)

        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, (UUID, Path)):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, (bytes, bytearray)):
            return {"__bytes__": base64.b64encode(bytes(obj)).decode("ascii")}

        if isinstance(obj, Mapping):
            return {
                str(self._json_safe(k, _depth + 1)): self._json_safe(v, _depth + 1)
                for k, v in obj.items()
            }

        if isinstance(obj, (list, tuple, set, frozenset)) or isinstance(obj, Generator):
            return [self._json_safe(x, _depth + 1) for x in list(obj)]

        if dataclasses.is_dataclass(obj):
            return self._json_safe(asdict(obj), _depth + 1)

        md = getattr(obj, "model_dump", None)
        if callable(md):
            try:
                return self._json_safe(obj.model_dump(), _depth + 1)
            except Exception:
                pass
        dd = getattr(obj, "dict", None)
        if callable(dd):
            try:
                return self._json_safe(obj.dict(), _depth + 1)
            except Exception:
                pass

        if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__"), dict):
            return self._json_safe(vars(obj), _depth + 1)

        return str(obj)

    def _unwrap_result_maybe(self, raw_outputs):
        """If outputs look like {'result': X} and that's the only key, return X."""
        if (
            isinstance(raw_outputs, Mapping)
            and len(raw_outputs) == 1
            and "result" in raw_outputs
        ):
            if logger.isEnabledFor(10):
                logger.debug("Unwrapped single-key 'result' in outputs")
            return raw_outputs["result"]
        return raw_outputs

    def _first_arg_or_none(self, raw_inputs):
        """Return first positional arg if shape is {'args':[...]} with len==1; else None."""
        if isinstance(raw_inputs, Mapping) and "args" in raw_inputs:
            args = raw_inputs.get("args")
            if (
                isinstance(args, Sequence)
                and not isinstance(args, (str, bytes, bytearray))
                and len(args) == 1
            ):
                return args[0]
        return None

    def _ensure_mapping(self, payload):
        """Ensure payload is a dict; if not, wrap under {'input': payload}."""
        return payload if isinstance(payload, Mapping) else {"input": payload}

    def _merge_error_into_output(self, output, error):
        if error is None:
            return output
        if isinstance(output, Mapping):
            merged = dict(output)
            merged["error"] = str(error)
            return merged
        return {"result": output, "error": str(error)}

    def _build_trace(self, trace_name: str, executions: List[NodeExecution]) -> Trace:
        """Build a Trace from a list of executions (CPU-heavy)."""
        spans = [self._span_from_execution(e) for e in executions]

        ctx = get_current_context()
        top_input: Dict[str, Any] = (ctx.inputs or {}) if ctx else {}

        if ctx and ctx.outputs is not None:
            top_output = (
                ctx.outputs
                if isinstance(ctx.outputs, dict)
                else {"result": ctx.outputs}
            )
        else:
            top_output = {}

        if logger.isEnabledFor(10):
            # Avoid giant logs
            def _truncate(v):
                s = str(v)
                return (
                    (s[: self._max_logged_chars] + "…")
                    if len(s) > self._max_logged_chars
                    else s
                )

            logger.debug(
                "Building trace",
                trace_name=trace_name,
                top_input=_truncate(top_input),
                top_output=_truncate(top_output),
                span_count=len(spans),
            )

        created_at = (
            min(_utc(e.start_time) for e in executions)
            if executions
            else _utc(datetime.utcnow())
        )

        return Trace(
            created_at=created_at,
            name=trace_name,
            input=top_input,
            output=top_output,
            spans=spans,
        )

    # ---------------------------- Submission ----------------------------

    async def _submit_trace(self, trace: Trace) -> bool:
        """Submit a single Trace inside CreateTracesInput."""
        req = CreateTracesInput(
            agent_workspace_name=self.agent_workspace_name,
            agent_deployment_name=self.agent_deployment_name,
            traces=[trace],
        )

        try:
            if logger.isEnabledFor(10):
                logger.debug(
                    "Submitting trace to DigitalOcean",
                    workspace=self.agent_workspace_name,
                    deployment=self.agent_deployment_name,
                    trace_name=trace.name,
                    span_count=len(trace.spans),
                )

            await self._client.create_traces(req)
            return True
        except DOAPIError as e:
            logger.debug(
                f"API error submitting trace '{trace.name}'",
                error=str(e),
                status_code=getattr(e, "status_code", None),
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.debug(
                f"Unexpected error submitting trace '{trace.name}'",
                error=str(e),
                exc_info=True,
            )
            return False

    async def _submit_current_trace(self) -> bool:
        """Build and submit the current context's trace."""
        ctx = get_current_context()
        if not ctx:
            logger.debug("No active context for trace submission")
            return False

        if ctx.request_id in self._submitted_traces:
            return True

        executions = [
            ex for ex in self.get_executions() if not self._is_internal_node(ex)
        ]
        if not executions:
            logger.debug("No user executions to submit")
            return True

        trace_name = f"{ctx.entrypoint_name} - {ctx.request_id[:8]}"

        # Offload heavy building to a thread to keep the loop snappy.
        trace = await asyncio.to_thread(self._build_trace, trace_name, executions)

        ok = await self._submit_trace(trace)
        if ok:
            self._submitted_traces.add(ctx.request_id)
        return ok

    def end_node_execution(
        self,
        node_execution: NodeExecution,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        logger.debug(
            "DigitalOceanTracker.end_node_execution", node_name=node_execution.node_name
        )
        super().end_node_execution(node_execution, outputs, error)

        if not self.enable_auto_submit:
            logger.debug("Auto-submit disabled")
            return

        ctx = get_current_context()
        logger.debug("Context status", status=ctx.status if ctx else "None")
        if not ctx:
            return

        # Only auto-submit on synthetic completion node
        meta = node_execution.metadata or {}
        is_request_complete = (
            node_execution.node_name == "RequestComplete"
            or meta.get("internal_request_complete") is True
        )
        if not is_request_complete:
            return

        rid = ctx.request_id
        with self._scheduler_lock:
            if rid in self._submitted_traces or rid in self._submitting_traces:
                return
            self._submitting_traces.add(rid)

        self._schedule_submit(rid)

    async def submit_trace_manually(self, trace_name: Optional[str] = None) -> bool:
        """Manually trigger submission for the current context (awaitable)."""
        return await self._submit_current_trace()

    async def aclose(self) -> None:
        """
        Flush inflight submissions and close the underlying HTTP client.
        Waits for:
          - asyncio.Tasks scheduled on the current loop
          - concurrent.futures.Futures from run_coroutine_threadsafe (cross-loop)
        """
        # Await any tasks that are on our current loop
        if self._inflight_tasks:
            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
            self._inflight_tasks.clear()

        # For cross-loop futures, block them *off* the event loop
        if self._inflight_futures:
            futs = list(self._inflight_futures)
            self._inflight_futures.clear()
            # Wait in a thread to avoid blocking the loop.
            await asyncio.to_thread(concurrent.futures.wait, futs, timeout=None)

        await self._client.aclose()

    def print_summary(self) -> None:
        """Print a summary and submission status."""
        super().print_summary()
        ctx = get_current_context()
        if ctx and ctx.request_id in self._submitted_traces:
            logger.debug("Trace submitted to DigitalOcean")
        elif not self.enable_auto_submit:
            logger.warning("⚠ Auto-submit disabled - call submit_trace_manually()")
