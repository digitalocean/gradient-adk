from __future__ import annotations

import asyncio
from collections.abc import Mapping
import inspect
import json
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
        self._is_evaluation: bool = False

    def on_request_start(
        self, entrypoint: str, inputs: Dict[str, Any], is_evaluation: bool = False
    ) -> None:
        # NEW: reset buffers per request
        self._live.clear()
        self._done.clear()
        self._is_evaluation = is_evaluation
        self._req = {"entrypoint": entrypoint, "inputs": inputs}

    def _as_async_iterable_and_setter(
        self, resp
    ) -> Optional[tuple[object, Callable[[object], None]]]:
        """
        If `resp` looks like a streaming response that iterates over `resp.content`,
        return (orig_iterable, setter) so we can replace it. Else None.
        """
        # For ServerSentEventsResponse and JSONStreamingResponse, we need to access
        # the ORIGINAL content before SSE/JSON formatting wraps it
        if isinstance(resp, (ServerSentEventsResponse, StreamingResponse)):
            # Get the original user generator before formatting
            # The content is already the formatted generator, but we can get the original
            # by checking if it was set in __init__
            # Actually, we need to intercept at a different level

            # Check if we can get to the original content
            # The formatted_content was created from the user's content
            # We need to store the original and intercept there
            content = getattr(resp, "content", None)
            if content is None:
                return None

            # The content is already the SSE/JSON formatted generator
            # We need to wrap it but also parse back the content
            if hasattr(content, "__aiter__") or inspect.isasyncgen(content):

                def _setter(new_iterable):
                    resp.content = new_iterable

                return content, _setter

        # Fallback for plain StreamingResponse
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
                    # Parse SSE formatted strings back to extract content
                    if isinstance(chunk, str) and chunk.startswith("data: "):
                        # SSE format: "data: {json}\n\n"
                        try:
                            json_str = chunk[
                                6:
                            ].strip()  # Remove "data: " prefix and whitespace
                            data = json.loads(json_str) if json_str else {}
                            if isinstance(data, dict):
                                content = (
                                    data.get("content") or data.get("data") or str(data)
                                )
                                collected.append(str(content))
                            else:
                                collected.append(str(data))
                        except Exception:
                            # Fallback: just collect the raw chunk
                            collected.append(chunk)
                    # For ServerSentEventsResponse/JSONStreamingResponse, extract content from dicts
                    elif isinstance(chunk, dict):
                        # Try common keys for content
                        content = (
                            chunk.get("content") or chunk.get("data") or str(chunk)
                        )
                        if isinstance(content, (bytes, bytearray)):
                            collected.append(content.decode("utf-8", errors="replace"))
                        else:
                            collected.append(str(content))
                    # collect safely (bytes/str/other)
                    elif isinstance(chunk, (bytes, bytearray)):
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

        # Non-streaming - always fire-and-forget
        # For evaluation mode, decorator will call submit_and_get_trace_id() directly
        self._req["outputs"] = outputs

        if not self._is_evaluation:
            # Regular fire-and-forget for non-evaluation requests
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

    async def submit_and_get_trace_id(self) -> Optional[str]:
        """
        Submit the trace and return the trace_id.
        Only call this for evaluation requests after on_request_end has been called.
        """
        return await self._submit()

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

    async def _submit(self) -> Optional[str]:
        try:
            trace = self._build_trace()
            req = CreateTracesInput(
                agent_workspace_name=self._ws,
                agent_deployment_name=self._dep,
                traces=[trace],
            )
            result = await self._client.create_traces(req)
            # Return first trace_uuid if available
            if result.trace_uuids:
                return result.trace_uuids[0]
            return None
        except Exception as e:
            # never break user code on export errors
            return None

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

        # classify LLM/tool/retriever via metadata set by the instrumentor
        metadata = ex.metadata or {}
        if metadata.get("is_llm_call"):
            span_type = TraceSpanType.TRACE_SPAN_TYPE_LLM
        elif metadata.get("is_retriever_call"):
            span_type = TraceSpanType.TRACE_SPAN_TYPE_RETRIEVER
        else:
            span_type = TraceSpanType.TRACE_SPAN_TYPE_TOOL

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
