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
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse


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
        self._collection_tasks: set[asyncio.Task] = (
            set()
        )  # Tasks collecting async generator outputs
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
        # Common fields
        self._req["error"] = error

        # Streaming path
        wrapped = self._as_async_iterable_and_setter(outputs)
        if wrapped is not None:
            orig_iterable, set_iterable = wrapped
            self._req["outputs"] = None  # will be filled after streaming finishes

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
                    self._req["outputs"] = "".join(collected)
                    await self._submit()
                except Exception as e:
                    # Error during streaming
                    self._req["error"] = str(e)
                    self._req["outputs"] = "".join(collected) if collected else None
                    await self._submit()
                    raise

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
                self._done.append(live)
                return
            if is_stringified_gen:
                # This happens when _freeze() is called on an async generator
                live.outputs = {
                    "streaming": True,
                    "_debug": "Async generator was converted to string. Instrumentor should not freeze async generators.",
                }
                self._done.append(live)
                return

        live.outputs = outputs
        self._done.append(live)

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
        live = self._live.pop(node.node_id, node)
        live.end_time = _utc()
        live.error = str(error)
        self._done.append(live)

    async def aclose(self) -> None:
        # Wait for both collection tasks and submission tasks
        all_tasks = list(self._inflight) + list(self._collection_tasks)
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
            self._inflight.clear()
            self._collection_tasks.clear()
        await self._client.aclose()

    async def _submit(self) -> Optional[str]:
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
