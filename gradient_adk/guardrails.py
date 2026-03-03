"""Guardrails client for evaluating content against safety rails.

Provides a simple async client to call the DigitalOcean Guardrails service.
When used inside an ``@entrypoint``-decorated function, guardrail evaluations
are automatically captured as spans in the ADK trace.

Example usage::

    from gradient_adk import Guardrails

    guardrails = Guardrails()

    async def check_input(prompt: str):
        result = await guardrails.check(
            rail_type="jailbreak",
            messages=[{"role": "user", "content": prompt}],
        )
        if not result["allowed"]:
            raise ValueError(f"Blocked: {result['violations'][0]['message']}")
        return result
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .runtime.helpers import get_tracker, _is_tracing_disabled
from .runtime.interfaces import NodeExecution

_GUARDRAILS_ENDPOINT = "https://guardrails.do-ai.run/v2/rail"
_DEFAULT_TIMEOUT = 30.0


class Guardrails:
    """Client for the DigitalOcean Guardrails service.

    Evaluates content against safety rails (jailbreak, content_moderation,
    sensitive_data). When used inside an ``@entrypoint`` function, guardrail
    evaluations are automatically captured as tool spans in the ADK trace.
    """

    def __init__(self) -> None:
        self._endpoint = _GUARDRAILS_ENDPOINT
        self._timeout = _DEFAULT_TIMEOUT

    def _resolve_token(self) -> str:
        token = os.environ.get("DIGITALOCEAN_API_TOKEN")
        if not token:
            raise RuntimeError(
                "DIGITALOCEAN_API_TOKEN environment variable is not set."
            )
        return token

    async def check(
        self,
        rail_type: str,
        messages: List[Dict[str, str]],
        *,
        evaluation_type: str = "input",
    ) -> Dict[str, Any]:
        """Evaluate content against a guardrail.

        Args:
            rail_type: Type of guardrail — ``"jailbreak"``,
                ``"content_moderation"``, or ``"sensitive_data"``.
            messages: Messages to evaluate, each with ``role`` and ``content``.
            evaluation_type: ``"input"`` (default) to evaluate user messages
                before LLM processing, or ``"output"`` to evaluate AI responses.

        Returns:
            A dict with ``allowed`` (bool), ``team_id`` (int),
            ``violations`` (list of dicts with ``message`` and ``rule_name``),
            and ``token_usage`` (dict with ``input_tokens``, ``output_tokens``,
            ``total_tokens``).

        Raises:
            RuntimeError: If ``DIGITALOCEAN_API_TOKEN`` is not set.
            httpx.HTTPStatusError: On non-200 responses from the service.

        Example::

            result = await guardrails.check(
                rail_type="jailbreak",
                messages=[{"role": "user", "content": "Hello!"}],
            )
            if result["allowed"]:
                print("Content is safe")
            else:
                for v in result["violations"]:
                    print(f"Violation: {v['message']} ({v['rule_name']})")
        """
        token = self._resolve_token()
        payload = {
            "rail_type": rail_type,
            "messages": messages,
            "evaluation_type": evaluation_type,
        }

        span = _start_guardrail_span(rail_type, payload)
        start_ns = time.monotonic_ns()

        try:
            result = await self._call(token, self._endpoint, payload)
            duration_ns = time.monotonic_ns() - start_ns
            _end_guardrail_span(span, result, duration_ns)
            return result
        except Exception as exc:
            duration_ns = time.monotonic_ns() - start_ns
            _error_guardrail_span(span, exc, duration_ns)
            raise

    async def _call(
        self, token: str, url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )

        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Tracing integration
# ---------------------------------------------------------------------------

def _start_guardrail_span(
    rail_type: str, payload: Dict[str, Any]
) -> Optional[NodeExecution]:
    if _is_tracing_disabled():
        return None
    tracker = get_tracker()
    if not tracker:
        return None

    span = NodeExecution(
        node_id=str(uuid.uuid4()),
        node_name=f"guardrail:{rail_type}",
        framework="guardrails",
        start_time=datetime.now(timezone.utc),
        inputs=payload,
        metadata={
            "is_tool_call": True,
            "is_programmatic": True,
            "rail_type": rail_type,
        },
    )
    tracker.on_node_start(span)
    return span


def _end_guardrail_span(
    span: Optional[NodeExecution],
    result: Dict[str, Any],
    duration_ns: int,
) -> None:
    if span is None:
        return
    tracker = get_tracker()
    if not tracker:
        return

    meta = span.metadata or {}
    meta["duration_ns"] = duration_ns
    meta["guardrail_allowed"] = result.get("allowed")
    meta["guardrail_violations"] = len(result.get("violations", []))
    token_usage = result.get("token_usage", {})
    meta["guardrail_total_tokens"] = token_usage.get("total_tokens", 0)
    span.metadata = meta

    tracker.on_node_end(span, result)


def _error_guardrail_span(
    span: Optional[NodeExecution],
    exc: Exception,
    duration_ns: int,
) -> None:
    if span is None:
        return
    tracker = get_tracker()
    if not tracker:
        return

    meta = span.metadata or {}
    meta["duration_ns"] = duration_ns
    span.metadata = meta

    tracker.on_node_error(span, exc)
