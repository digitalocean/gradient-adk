from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from gradient_adk.digital_ocean_api import (
    AsyncDigitalOceanGenAI,
    CreateTracesInput,
    Trace,
    Span,
    TraceSpanType,
)
from .interfaces import NodeExecution

from datetime import datetime, timezone
from gradient_adk.streaming import StreamingResponse


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

    # ---- request ----
    def on_request_start(self, entrypoint: str, inputs: Dict[str, Any]) -> None:
        # NEW: reset buffers per request
        self._live.clear()
        self._done.clear()
        self._req = {"entrypoint": entrypoint, "inputs": inputs}

    def on_request_end(self, outputs: Any | None, error: Optional[str]) -> None:
        if isinstance(outputs, StreamingResponse):
            self._req["error"] = error
            self._req["outputs"] = None

            original_content = outputs.content
            collected: list[str] = []

            # Patch the __aiter__ method so the collector runs *after* the server has sent all chunks
            async def new_aiter():
                async for chunk in original_content:
                    # collect text safely
                    if isinstance(chunk, (bytes, bytearray)):
                        collected.append(chunk.decode("utf-8"))
                    elif isinstance(chunk, str):
                        collected.append(chunk)
                    else:
                        collected.append(str(chunk))
                    yield chunk
                # when Starlette finishes sending
                self._req["outputs"] = "".join(collected)
                await self._submit()

            # Replace the async iterator interface
            outputs.content.__aiter__ = new_aiter
            return

        # --- normal non-streaming path ---
        self._req["outputs"] = outputs
        self._req["error"] = error

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

    # ---- nodes ----
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

    def on_node_chunk(self, node: NodeExecution, chunk: Any) -> None:
        # Optional: add as events/child spans later. No-op for now.
        pass

    async def aclose(self) -> None:
        if self._inflight:
            await asyncio.gather(*list(self._inflight), return_exceptions=True)
            self._inflight.clear()
        await self._client.aclose()

    # ---- submit ----
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
            print(e)

    def _to_span(self, ex: NodeExecution) -> Span:
        # Base payloads
        inp = ex.inputs if isinstance(ex.inputs, dict) else {"input": ex.inputs}
        out = ex.outputs if isinstance(ex.outputs, dict) else {"output": ex.outputs}

        # NEW: include error (if any) and matched endpoints (if present)
        if ex.error is not None:
            out = dict(out)
            out["error"] = ex.error
        if ex.metadata and ex.metadata.get("llm_endpoints"):
            out = dict(out)
            out["_llm_endpoints"] = list(ex.metadata["llm_endpoints"])

        # NEW: classify LLM/tool via metadata set by the instrumentor
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

        # NEW: coerce to dicts so pydantic is happy even if agent returns a string/number/etc.
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
        # optional debug
        # print(trace)
        return trace
