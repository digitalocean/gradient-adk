"""
DigitalOcean traces integration (async) for execution tracking.

Uses AsyncDigitalOceanGenAI + typed Pydantic models to submit traces
to DigitalOcean's GenAI Traces API without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set

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

import base64
import dataclasses
from dataclasses import asdict
from collections.abc import Mapping, Sequence, Generator
from datetime import datetime, date
from pathlib import Path
from uuid import UUID
from enum import Enum


def _utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class DigitalOceanTracesTracker(DefaultExecutionTracker):
    """
    ExecutionTracker that submits traces to DigitalOcean using an async, typed client.

    Constructor supports either passing a ready AsyncDigitalOceanGenAI client or an api_token.
    Prefers agent_workspace_name per the latest schema; optional session_id is supported.
    """

    def __init__(
        self,
        *,
        agent_workspace_name: str,
        agent_deployment_name: str,
        client: AsyncDigitalOceanGenAI,
        enable_auto_submit: bool = True,
    ) -> None:
        super().__init__()

        self._client = client
        self.agent_workspace_name = agent_workspace_name
        self.agent_deployment_name = agent_deployment_name
        self.enable_auto_submit = enable_auto_submit

        self._submitted_traces: Set[str] = set()
        self._inflight_tasks: Set[asyncio.Task] = set()
        self._submitting_traces: Set[str] = set()  # prevent duplicate scheduling

    def _map_node_to_span_type(
        self, node_name: str, framework: str, execution: Optional[NodeExecution] = None
    ) -> TraceSpanType:
        """Determine the appropriate span type based on node characteristics."""

        # Check if this node made calls to DigitalOcean inference endpoints
        if execution and self._is_llm_node(execution):
            logger.debug("Detected LLM call in node", node_name=node_name)
            return TraceSpanType.TRACE_SPAN_TYPE_LLM

        return TraceSpanType.TRACE_SPAN_TYPE_TOOL  # Default to TOOL

    def _is_llm_node(self, execution: NodeExecution) -> bool:
        """Check if a node execution involves calls to DigitalOcean inference endpoints."""

        # First check if the instrumentor detected LLM calls via network interception
        if execution.metadata and execution.metadata.get("is_llm_call", False):
            return True

        # Fallback: check inputs, outputs, and metadata for inference endpoint URLs
        # (in case network interception wasn't available or failed)
        do_inference_patterns = ["inference.do-ai.run", "inference.do-ai-test.run"]
        all_data = (
            str(execution.inputs) + str(execution.outputs) + str(execution.metadata)
        )
        return any(pattern in all_data for pattern in do_inference_patterns)

    def _span_from_execution(self, execution: NodeExecution) -> Span:
        """
        Convert NodeExecution → Span (protobuf-compatible model).
        - `created_at` uses node start time (fallback: now UTC).
        - `error` is embedded into output["error"] (schema has no explicit error field).
        - Inputs/outputs are converted to JSON-safe shapes.
        """
        span_type = self._map_node_to_span_type(
            execution.node_name, execution.framework, execution
        )

        # ------- OUTPUTS -------
        raw_outputs = execution.outputs if execution.outputs is not None else {}
        output = self._unwrap_result_maybe(raw_outputs)

        # If output isn't a mapping, try to coerce; otherwise leave as-is
        out_map = self._to_mapping_or_none(output)
        if out_map is not None:
            output = out_map

        # Merge error safely regardless of output's original shape
        output = self._merge_error_into_output(output, execution.error)
        output = self._json_safe(output)

        # ------- INPUTS --------
        raw_inputs = execution.inputs if execution.inputs is not None else {}

        # Prefer the single positional arg if present (typical tool/LLM wrappers)
        first_arg = self._first_arg_or_none(raw_inputs)
        if first_arg is not None:
            input_payload = first_arg
            logger.debug("Using first arg as input", node_name=execution.node_name)
        else:
            in_map = self._to_mapping_or_none(raw_inputs)
            if in_map is not None:
                input_payload = in_map
            else:
                # If it's a sequence but not KV pairs, treat as args
                if isinstance(raw_inputs, Sequence) and not isinstance(
                    raw_inputs, (str, bytes, bytearray)
                ):
                    if self._is_kv_pairs(raw_inputs):
                        input_payload = {str(k): v for k, v in raw_inputs}
                    else:
                        input_payload = {"args": list(raw_inputs)}
                else:
                    # Anything else: wrap as scalar input
                    input_payload = {"input": raw_inputs}

        # Ensure mapping and attach node metadata
        input_payload = self._ensure_mapping(self._json_safe(input_payload))
        input_payload["_node_meta"] = {
            "framework": execution.framework,
            "node_id": getattr(execution, "node_id", None),
        }

        # ------- CREATED_AT ----
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
        if isinstance(seq, (str, bytes, bytearray)):
            return False
        if not isinstance(seq, Sequence):
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
        # True mappings
        if isinstance(obj, Mapping):
            return dict(obj)

        # Dataclasses
        if dataclasses.is_dataclass(obj):
            return asdict(obj)

        # Pydantic v2
        md = getattr(obj, "model_dump", None)
        if callable(md):
            try:
                return obj.model_dump()
            except Exception:
                pass

        # Pydantic v1 / typical .dict()
        dd = getattr(obj, "dict", None)
        if callable(dd):
            try:
                return obj.dict()
            except Exception:
                pass

        # Generic __dict__
        if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__"), dict):
            return dict(vars(obj))

        # Iterable of KV pairs
        if isinstance(obj, Sequence) and self._is_kv_pairs(obj):
            try:
                return {k: v for k, v in obj}
            except Exception:
                pass

        return None

    def _json_safe(self, obj, _depth=0, _max_depth=6):
        """Recursively convert obj into JSON-serializable types."""
        if _depth > _max_depth:
            return str(obj)

        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Common primitives
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, (UUID, Path)):
            return str(obj)
        if isinstance(obj, Enum):
            # choose name for compactness; use value if you prefer
            return obj.name
        if isinstance(obj, (bytes, bytearray)):
            # Avoid dumping binary directly
            return {"__bytes__": base64.b64encode(bytes(obj)).decode("ascii")}

        # Mappings
        if isinstance(obj, Mapping):
            return {
                str(self._json_safe(k, _depth + 1, _max_depth)): self._json_safe(
                    v, _depth + 1, _max_depth
                )
                for k, v in obj.items()
            }

        # Sequences & generators
        if isinstance(obj, (list, tuple, set, frozenset)) or isinstance(obj, Generator):
            return [self._json_safe(x, _depth + 1, _max_depth) for x in list(obj)]

        # Dataclass
        if dataclasses.is_dataclass(obj):
            return self._json_safe(asdict(obj), _depth + 1, _max_depth)

        # Pydantic models
        md = getattr(obj, "model_dump", None)
        if callable(md):
            try:
                return self._json_safe(obj.model_dump(), _depth + 1, _max_depth)
            except Exception:
                pass
        dd = getattr(obj, "dict", None)
        if callable(dd):
            try:
                return self._json_safe(obj.dict(), _depth + 1, _max_depth)
            except Exception:
                pass

        # Generic object
        if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__"), dict):
            return self._json_safe(vars(obj), _depth + 1, _max_depth)

        # Last resort
        return str(obj)

    def _unwrap_result_maybe(self, raw_outputs):
        """If outputs look like {'result': X} and that's the only key, return X."""
        if (
            isinstance(raw_outputs, Mapping)
            and len(raw_outputs) == 1
            and "result" in raw_outputs
        ):
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
        # embed error string under 'error' key, preserving existing output if mapping
        if isinstance(output, Mapping):
            merged = dict(output)
            merged["error"] = str(error)
            return merged
        return {"result": output, "error": str(error)}

    def _build_trace(self, trace_name: str, executions: List[NodeExecution]) -> Trace:
        """
        Build a Trace from a list of executions.
        - `created_at` = earliest node start time (or now).
        - top-level input/output are taken from the request context, if present.
        """
        spans = [self._span_from_execution(e) for e in executions]

        ctx = get_current_context()
        top_input: Dict[str, Any] = (ctx.inputs or {}) if ctx else {}

        # Ensure output is always a dictionary for the Trace model
        if ctx and ctx.outputs is not None:
            if isinstance(ctx.outputs, dict):
                top_output = ctx.outputs
            else:
                # Wrap non-dict outputs in a result field
                top_output = {"result": ctx.outputs}
        else:
            top_output = {}

        # Debug logging for trace inputs/outputs
        logger.debug(
            "Building trace",
            trace_name=trace_name,
            top_input=top_input,
            top_output=top_output,
            span_count=len(spans),
        )

        for i, span in enumerate(spans):
            logger.debug(
                "Span details",
                span_number=i + 1,
                span_name=span.name,
                span_input=span.input,
                span_output=span.output,
            )

        if executions:
            created_at = min(_utc(e.start_time) for e in executions)
        else:
            created_at = _utc(datetime.utcnow())

        return Trace(
            created_at=created_at,
            name=trace_name,
            input=top_input,
            output=top_output,
            spans=spans,
        )

    async def _submit_trace(self, trace: Trace) -> bool:
        """
        Submit a single Trace inside CreateTracesInput.
        Returns True on success; logs and returns False on failure.
        """
        req = CreateTracesInput(
            agent_workspace_name=self.agent_workspace_name,
            agent_deployment_name=self.agent_deployment_name,
            traces=[trace],
        )

        try:
            logger.debug(
                "Submitting trace to DigitalOcean",
                workspace=self.agent_workspace_name,
                deployment=self.agent_deployment_name,
                trace_name=trace.name,
                span_count=len(trace.spans),
            )

            for i, span in enumerate(trace.spans):
                logger.debug(
                    "Trace span",
                    span_number=i + 1,
                    span_name=span.name,
                    span_type=span.type,
                )

            await self._client.create_traces(req)
            logger.debug(f"Successfully submitted trace '{trace.name}' to DigitalOcean")
            return True
        except DOAPIError as e:
            logger.error(
                f"API error submitting trace '{trace.name}'",
                error=str(e),
                status_code=getattr(e, "status_code", None),
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error submitting trace '{trace.name}'",
                error=str(e),
                exc_info=True,
            )
            return False

    async def _submit_current_trace(self) -> bool:
        """
        Build and submit the current context's trace.
        """
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
        trace = self._build_trace(trace_name, executions)

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

        # NEW: Only allow auto-submit on the synthetic completion node
        meta = node_execution.metadata or {}
        is_request_complete = (
            node_execution.node_name == "RequestComplete"
            or meta.get("internal_request_complete") is True
        )
        if not is_request_complete:
            return

        rid = ctx.request_id
        # NEW: De-dupe scheduling while a submission is in-flight or already done
        if rid in self._submitted_traces or rid in self._submitting_traces:
            return

        self._submitting_traces.add(rid)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        task = loop.create_task(self._submit_current_trace())
        self._inflight_tasks.add(task)

        def _log_submission_result(t: asyncio.Task):
            self._inflight_tasks.discard(t)
            self._submitting_traces.discard(rid)
            try:
                success = t.result()
                if success:
                    logger.debug("Trace submitted to DigitalOcean successfully")
                else:
                    logger.error("Failed to submit trace to DigitalOcean")
            except Exception as e:
                logger.error(
                    "Error during trace submission", error=str(e), exc_info=True
                )

        task.add_done_callback(_log_submission_result)

    async def submit_trace_manually(self, trace_name: Optional[str] = None) -> bool:
        """
        Manually trigger submission for the current context (awaitable).
        `trace_name` is currently derived from context; override could be added if needed.
        """
        return await self._submit_current_trace()

    async def aclose(self) -> None:
        """
        Flush inflight submissions and close the underlying HTTP client.
        """
        if self._inflight_tasks:
            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
            self._inflight_tasks.clear()
        await self._client.aclose()

    def print_summary(self) -> None:
        """Print a summary and submission status."""
        super().print_summary()
        ctx = get_current_context()
        if ctx and ctx.request_id in self._submitted_traces:
            logger.info("✓ Trace submitted to DigitalOcean")
        elif not self.enable_auto_submit:
            logger.warning("⚠ Auto-submit disabled - call submit_trace_manually()")
        # Note: For auto-submit enabled cases without successful submission,
        # the result will be logged by the background task callback
