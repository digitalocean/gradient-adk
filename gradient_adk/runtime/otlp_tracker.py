from __future__ import annotations

import contextvars
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from gradient_adk.digital_ocean_api import AsyncDigitalOceanGenAI

from .interfaces import NodeExecution
from .otel_setup import (
    DEFAULT_MAX_EVENT_BYTES,
    create_tracer_provider,
)


@dataclass
class LiveSpan:
    node: NodeExecution
    span: Any
    token: object


@dataclass
class RequestState:
    req: Dict[str, Any] = field(default_factory=dict)
    live: Dict[str, LiveSpan] = field(default_factory=dict)
    is_evaluation: bool = False
    session_id: Optional[str] = None
    evaluation_run_uuid: Optional[str] = None
    root_span: Any | None = None
    root_token: object | None = None
    trace_id: Optional[str] = None
    logs_enabled: bool = True


_request_state: contextvars.ContextVar[Optional[RequestState]] = contextvars.ContextVar(
    "otlp_request_state", default=None
)


def _utc(dt: datetime | None = None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _truthy_env(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes"}


def _format_trace_id(trace_id: int) -> str:
    return f"{trace_id:032x}"


def _derive_gen_ai_system(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    lowered = url.lower()
    if "openai.com" in lowered:
        return "openai"
    if "anthropic.com" in lowered:
        return "anthropic"
    if "generativelanguage.googleapis.com" in lowered or "googleapis.com" in lowered:
        return "google"
    if "cohere." in lowered:
        return "cohere"
    if "x.ai" in lowered:
        return "xai"
    if "inference.do-ai.run" in lowered or "inference.do-ai-test.run" in lowered:
        return "digitalocean"
    return None


def _extract_query(inputs: Any) -> Optional[str]:
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, dict):
        for key in ("query", "input", "text", "prompt"):
            value = inputs.get(key)
            if isinstance(value, str):
                return value
    return None


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return str(value)


def _truncate_text(text: str, max_bytes: int = DEFAULT_MAX_EVENT_BYTES) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[: max_bytes - len(b"...<truncated>")]
    return truncated.decode("utf-8", errors="ignore") + "...<truncated>"


class OTLPTracesTracker:
    """OpenTelemetry-backed trace tracker for gradient-adk.

    Tenant attribution (`team_id` / `org_id`) is sourced from env vars
    only. The AAPP-1160 collector resource processor is the canonical
    source of truth in production; the SDK does not perform any blocking
    DigitalOcean API lookups at construction time.
    """

    def __init__(
        self,
        *,
        client: Optional[AsyncDigitalOceanGenAI],
        agent_workspace_name: str,
        agent_deployment_name: str,
    ) -> None:
        self._client = client
        self._ws = agent_workspace_name
        self._dep = agent_deployment_name
        self._closed = False
        self._last_trace_id: Optional[str] = None

        team_id = os.environ.get("DIGITALOCEAN_TEAM_ID")
        org_id = os.environ.get("DIGITALOCEAN_ORG_ID")

        self._provider = create_tracer_provider(
            agent_workspace_name=agent_workspace_name,
            agent_deployment_name=agent_deployment_name,
            team_id=team_id,
            org_id=org_id,
        )
        self._tracer = self._provider.get_tracer("gradient_adk.runtime.otlp_tracker")
        self._legacy_live: Dict[str, LiveSpan] = {}

    def _get_state(self) -> Optional[RequestState]:
        return _request_state.get()

    def _logs_enabled(self) -> bool:
        state = self._get_state()
        if state is not None:
            return state.logs_enabled
        return _truthy_env("DIGITALOCEAN_CONVERSATION_LOGS_ENABLED", default=True)

    def _set_payload_event(self, span: Any, event_name: str, value: Any) -> None:
        if value is None:
            return
        span.add_event(event_name, {"content": _truncate_text(_json_text(value))})

    def _mark_redacted(self, span: Any, *, has_input: bool, has_output: bool) -> None:
        if has_input:
            span.set_attribute("gen_ai.input.redacted", True)
        if has_output:
            span.set_attribute("gen_ai.output.redacted", True)

    def _set_common_status(
        self,
        *,
        span: Any,
        metadata: Dict[str, Any],
        error: Optional[str],
        exception: Optional[BaseException] = None,
    ) -> None:
        if metadata.get("status_code") is not None:
            span.set_attribute("http.status_code", int(metadata["status_code"]))
        if exception is not None:
            span.set_status(Status(StatusCode.ERROR, str(exception)))
            span.record_exception(exception)
        elif error:
            span.set_status(Status(StatusCode.ERROR, error))
            span.record_exception(RuntimeError(error))
        else:
            span.set_status(Status(StatusCode.OK))

    def _populate_llm_span(
        self,
        *,
        span: Any,
        node: NodeExecution,
        metadata: Dict[str, Any],
    ) -> None:
        span.set_attribute("gen_ai.operation.name", "chat")

        llm_url = metadata.get("llm_url")
        gen_ai_system = _derive_gen_ai_system(llm_url)
        if gen_ai_system:
            span.set_attribute("gen_ai.system", gen_ai_system)

        llm_request = metadata.get("llm_request_payload", {}) or {}
        llm_response = metadata.get("llm_response_payload", {}) or {}

        model = (
            llm_request.get("model")
            if isinstance(llm_request, dict)
            else metadata.get("model_name")
        ) or metadata.get("model_name")
        if model:
            span.set_attribute("gen_ai.request.model", model)

        if isinstance(llm_request, dict):
            temperature = llm_request.get("temperature")
            if temperature is not None:
                span.set_attribute("gen_ai.request.temperature", float(temperature))

        if isinstance(llm_response, dict):
            usage = llm_response.get("usage", {})
            if isinstance(usage, dict):
                if usage.get("prompt_tokens") is not None:
                    span.set_attribute(
                        "gen_ai.usage.input_tokens", int(usage["prompt_tokens"])
                    )
                if usage.get("completion_tokens") is not None:
                    span.set_attribute(
                        "gen_ai.usage.output_tokens",
                        int(usage["completion_tokens"]),
                    )
                if usage.get("total_tokens") is not None:
                    span.set_attribute(
                        "gen_ai.usage.total_tokens", int(usage["total_tokens"])
                    )

        ttft_ns = metadata.get("time_to_first_token_ns")
        if ttft_ns is not None:
            span.set_attribute(
                "gen_ai.server.time_to_first_token", float(ttft_ns) / 1_000_000_000
            )

        logs_enabled = self._logs_enabled()
        if logs_enabled:
            input_payload = (
                llm_request.get("messages") if isinstance(llm_request, dict) else None
            ) or node.inputs
            output_payload = (
                llm_response.get("choices") if isinstance(llm_response, dict) else None
            ) or node.outputs
            self._set_payload_event(span, "gen_ai.input.messages", input_payload)
            self._set_payload_event(span, "gen_ai.output.messages", output_payload)
            if isinstance(llm_request, dict) and llm_request.get("tools") is not None:
                self._set_payload_event(
                    span, "gen_ai.request.tools", llm_request.get("tools")
                )
        else:
            self._mark_redacted(
                span,
                has_input=node.inputs is not None or bool(llm_request),
                has_output=node.outputs is not None or bool(llm_response),
            )

    def _populate_tool_span(
        self,
        *,
        span: Any,
        node: NodeExecution,
        metadata: Dict[str, Any],
    ) -> None:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", node.node_name)
        tool_call_id = metadata.get("tool_call_id")
        if tool_call_id:
            span.set_attribute("gen_ai.tool.call.id", tool_call_id)
        if self._logs_enabled():
            self._set_payload_event(span, "gen_ai.input.messages", node.inputs)
            self._set_payload_event(span, "gen_ai.output.messages", node.outputs)
        else:
            self._mark_redacted(
                span,
                has_input=node.inputs is not None,
                has_output=node.outputs is not None,
            )

    def _populate_retriever_span(
        self,
        *,
        span: Any,
        node: NodeExecution,
    ) -> None:
        span.set_attribute("gen_ai.operation.name", "retrieval")
        query = _extract_query(node.inputs)
        if query is not None:
            span.set_attribute("gen_ai.retrieval.query", query)
        if self._logs_enabled():
            self._set_payload_event(span, "gen_ai.input.messages", node.inputs)
            self._set_payload_event(span, "gen_ai.output.messages", node.outputs)
        else:
            self._mark_redacted(
                span,
                has_input=node.inputs is not None,
                has_output=node.outputs is not None,
            )

    def _populate_agent_span(
        self,
        *,
        span: Any,
        node: NodeExecution,
    ) -> None:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        if self._logs_enabled():
            self._set_payload_event(span, "gen_ai.input.messages", node.inputs)
            self._set_payload_event(span, "gen_ai.output.messages", node.outputs)
        else:
            self._mark_redacted(
                span,
                has_input=node.inputs is not None,
                has_output=node.outputs is not None,
            )

    def _populate_span_attributes(
        self,
        *,
        span: Any,
        node: NodeExecution,
        metadata: Dict[str, Any],
    ) -> None:
        if metadata.get("is_llm_call"):
            self._populate_llm_span(span=span, node=node, metadata=metadata)
            return
        if metadata.get("is_tool_call"):
            self._populate_tool_span(span=span, node=node, metadata=metadata)
            return
        if metadata.get("is_retriever_call"):
            self._populate_retriever_span(span=span, node=node)
            return
        if metadata.get("is_agent_call") or metadata.get("is_workflow"):
            self._populate_agent_span(span=span, node=node)
            return

        self._populate_tool_span(span=span, node=node, metadata=metadata)

    def on_request_start(
        self,
        entrypoint: str,
        inputs: Dict[str, Any],
        is_evaluation: bool = False,
        session_id: Optional[str] = None,
        parent_context: Any = None,
        evaluation_run_uuid: Optional[str] = None,
    ) -> None:
        state = RequestState(
            req={"entrypoint": entrypoint, "inputs": inputs},
            is_evaluation=is_evaluation,
            session_id=session_id,
            evaluation_run_uuid=evaluation_run_uuid,
            logs_enabled=_truthy_env(
                "DIGITALOCEAN_CONVERSATION_LOGS_ENABLED", default=True
            ),
        )
        parent = parent_context if parent_context is not None else otel_context.get_current()
        root_span = self._tracer.start_span(
            entrypoint,
            context=parent,
            kind=SpanKind.SERVER,
            start_time=int(_utc().timestamp() * 1_000_000_000),
        )
        root_span.set_attribute("gen_ai.operation.name", "invoke_agent")
        if session_id:
            root_span.set_attribute("gen_ai.conversation.id", session_id)
        if evaluation_run_uuid:
            root_span.set_attribute("gen_ai.evaluation.run_uuid", evaluation_run_uuid)

        state.root_span = root_span
        state.trace_id = _format_trace_id(root_span.get_span_context().trace_id)
        state.root_token = otel_context.attach(trace.set_span_in_context(root_span, parent))
        _request_state.set(state)

    def on_request_end(self, outputs: Any | None, error: Optional[str]) -> None:
        state = self._get_state()
        if state is None:
            return

        state.req["outputs"] = outputs
        state.req["error"] = error

        if state.root_span is not None:
            logs_enabled = self._logs_enabled()
            if logs_enabled:
                self._set_payload_event(
                    state.root_span, "gen_ai.input.messages", state.req.get("inputs")
                )
                self._set_payload_event(
                    state.root_span, "gen_ai.output.messages", state.req.get("outputs")
                )
            else:
                self._mark_redacted(
                    state.root_span,
                    has_input=state.req.get("inputs") is not None,
                    has_output=state.req.get("outputs") is not None,
                )

            self._set_common_status(
                span=state.root_span,
                metadata={},
                error=error,
            )
            state.root_span.end()

        if state.root_token is not None:
            otel_context.detach(state.root_token)

        self._last_trace_id = state.trace_id
        _request_state.set(None)

    async def submit_and_get_trace_id(self) -> Optional[str]:
        return self._last_trace_id

    def _span_kind_for_metadata(self, metadata: Dict[str, Any]) -> SpanKind:
        if metadata.get("is_llm_call") or metadata.get("is_retriever_call"):
            return SpanKind.CLIENT
        if metadata.get("is_agent_call") or metadata.get("is_workflow"):
            return SpanKind.INTERNAL
        return SpanKind.INTERNAL

    def on_node_start(self, node: NodeExecution) -> None:
        state = self._get_state()
        metadata = node.metadata or {}
        span = self._tracer.start_span(
            node.node_name,
            context=otel_context.get_current(),
            kind=self._span_kind_for_metadata(metadata),
            start_time=int(_utc(node.start_time).timestamp() * 1_000_000_000),
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        if state is not None:
            state.live[node.node_id] = LiveSpan(node=node, span=span, token=token)
        else:
            self._legacy_live[node.node_id] = LiveSpan(node=node, span=span, token=token)

    def _finish_live_span(
        self,
        node: NodeExecution,
        outputs: Any | None = None,
        error: Optional[str] = None,
        exception: Optional[BaseException] = None,
    ) -> None:
        state = self._get_state()
        live = state.live.pop(node.node_id, None) if state is not None else None
        if live is None:
            live = self._legacy_live.pop(node.node_id, None)
        if live is None:
            return

        live.node.end_time = _utc()
        live.node.outputs = outputs
        if error:
            live.node.error = error

        metadata = live.node.metadata or {}
        self._populate_span_attributes(span=live.span, node=live.node, metadata=metadata)
        self._set_common_status(
            span=live.span,
            metadata=metadata,
            error=error,
            exception=exception,
        )

        if metadata.get("duration_ns") is not None:
            live.span.set_attribute("do.span.duration_ns", int(metadata["duration_ns"]))
        elif live.node.start_time and live.node.end_time:
            live.span.set_attribute(
                "do.span.duration_ns",
                int(
                    (live.node.end_time - live.node.start_time).total_seconds()
                    * 1_000_000_000
                ),
            )

        live.span.end(end_time=int(_utc(live.node.end_time).timestamp() * 1_000_000_000))
        otel_context.detach(live.token)

    def on_node_end(self, node: NodeExecution, outputs: Any | None) -> None:
        self._finish_live_span(node, outputs=outputs)

    def on_node_error(self, node: NodeExecution, error: BaseException) -> None:
        self._finish_live_span(
            node, outputs=None, error=str(error), exception=error
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        try:
            self._provider.force_flush()
            self._provider.shutdown()
        finally:
            self._closed = True
            if self._client is not None:
                await self._client.aclose()
