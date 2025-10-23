from __future__ import annotations

import asyncio
from collections.abc import Mapping
import inspect
from typing import Any, Callable, Dict, List, Optional

from gradient_adk.digital_ocean_api import (
    AsyncDigitalOceanGenAI,
    CreateTracesInput,
    Trace,
    Span,
    TraceSpanType,
)
from .interfaces import NodeExecution

from datetime import datetime, timezone
from gradient_adk.streaming import StreamingResponse, ServerSentEventsResponse


def _utc(dt: datetime | None = None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class DigitalOceanTracesTracker:
    """Collect executions and submit a single trace on request end."""

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

        self._req: Dict[str, Any] = {}
        self._live: dict[str, NodeExecution] = {}
        self._done: List[NodeExecution] = []
        self._inflight: set[asyncio.Task] = set()

    def on_request_start(self, entrypoint: str, inputs: Dict[str, Any]) -> None:
        # NEW: reset buffers per request
        self._live.clear()
        self._done.clear()
        self._req = {"entrypoint": entrypoint, "inputs": inputs}

    def _as_async_iterable_and_setter(
        self, resp
    ) -> Optional[tuple[object, Callable[[object], None]]]:
        """
        If `resp` looks like a streaming response that iterates over `resp.content`,
        return (orig_iterable, setter) so we can replace it. Else None.
        """
        content = getattr(resp, "content", None)
        if content is None:
            return None
        # async iterator / async generator objects
        if hasattr(content, "__aiter__") or inspect.isasyncgen(content):

            def _setter(new_iterable):
                resp.content = new_iterable

            return content, _setter
        return None

    def on_request_end(self, outputs: Any | None, error: Optional[str]) -> None:
        # Common fields
        self._req["error"] = error

        # Streaming path
        wrapped = self._as_async_iterable_and_setter(outputs)
        if wrapped is not None:
            orig_iterable, set_iterable = wrapped
            self._req["outputs"] = None  # will be filled after streaming finishes

            async def collecting_iter():
                collected: list[str] = []
                async for chunk in orig_iterable:
                    # collect safely (bytes/str/other)
                    if isinstance(chunk, (bytes, bytearray)):
                        collected.append(chunk.decode("utf-8", errors="replace"))
                    elif isinstance(chunk, str):
                        collected.append(chunk)
                    else:
                        collected.append(str(chunk))
                    yield chunk
                # when the server finishes sending
                self._req["outputs"] = "".join(collected)
                await self._submit()

            set_iterable(collecting_iter())
            return  # important: don't submit yet

        # Non-streaming
        self._req["outputs"] = outputs
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._submit())
            self._inflight.add(task)

            def _done_cb(t: asyncio.Task) -> None:
                self._inflight.discard(t)
                try:
                    t.result()
                except Exception:
                    pass

            task.add_done_callback(_done_cb)
        except RuntimeError:
            asyncio.run(self._submit())

    def on_node_start(self, node: NodeExecution) -> None:
        self._live[node.node_id] = node

    def on_node_end(self, node: NodeExecution, outputs: Any | None) -> None:
        live = self._live.pop(node.node_id, node)
        live.end_time = _utc()
        live.outputs = outputs
        self._done.append(live)

    def on_node_error(self, node: NodeExecution, error: BaseException) -> None:
        live = self._live.pop(node.node_id, node)
        live.end_time = _utc()
        live.error = str(error)
        self._done.append(live)

    async def aclose(self) -> None:
        if self._inflight:
            await asyncio.gather(*list(self._inflight), return_exceptions=True)
            self._inflight.clear()
        await self._client.aclose()

    async def _submit(self) -> None:
        try:
            trace = self._build_trace()
            req = CreateTracesInput(
                agent_workspace_name=self._ws,
                agent_deployment_name=self._dep,
                traces=[trace],
            )
            await self._client.create_traces(req)
        except Exception as e:
            # never break user code on export errors
            pass

    def _to_span(self, ex: NodeExecution) -> Span:
        # Base payloads
        inp = ex.inputs if isinstance(ex.inputs, dict) else {"input": ex.inputs}
        out = ex.outputs if isinstance(ex.outputs, dict) else {"output": ex.outputs}

        # include error (if any) and matched endpoints (if present)
        if ex.error is not None:
            out = dict(out)
            out["error"] = ex.error
        if ex.metadata and ex.metadata.get("llm_endpoints"):
            out = dict(out)
            out["_llm_endpoints"] = list(ex.metadata["llm_endpoints"])

        # classify LLM/tool via metadata set by the instrumentor
        span_type = (
            TraceSpanType.TRACE_SPAN_TYPE_LLM
            if (ex.metadata or {}).get("is_llm_call")
            else TraceSpanType.TRACE_SPAN_TYPE_TOOL
        )

        return Span(
            created_at=_utc(ex.start_time),
            name=ex.node_name,
            input=inp,
            output=out,
            type=span_type,
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
