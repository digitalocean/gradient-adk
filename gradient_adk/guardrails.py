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
        if not result.allowed:
            raise ValueError(f"Blocked: {result.violations[0].message}")
        return result
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .runtime.helpers import get_tracker, _is_tracing_disabled
from .runtime.interfaces import NodeExecution

_DEFAULT_TIMEOUT = 30.0


@dataclass
class GuardrailViolation:
    """A single guardrail violation."""

    message: str
    rule_name: str


@dataclass
class TokenUsage:
    """Token consumption for a guardrail evaluation."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GuardrailResult:
    """Result of a guardrail evaluation."""

    allowed: bool
    team_id: int
    violations: List[GuardrailViolation] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)


class GuardrailsError(Exception):
    """Raised when a guardrails evaluation fails."""

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class Guardrails:
    """Client for the DigitalOcean Guardrails service.

    Evaluates content against safety rails (jailbreak, content_moderation,
    sensitive_data). Authentication and service configuration are handled
    automatically via environment variables. Guardrail evaluations are
    captured as tool spans in the ADK trace.
    """

    def __init__(self) -> None:
        self._base_url = os.environ.get("GUARDRAILS_URL", "")
        self._timeout = _DEFAULT_TIMEOUT

    def _resolve_token(self) -> str:
        token = os.environ.get("DIGITALOCEAN_API_TOKEN")
        if not token:
            raise GuardrailsError(
                "DIGITALOCEAN_API_TOKEN environment variable is not set."
            )
        return token

    def _resolve_url(self) -> str:
        if not self._base_url:
            raise GuardrailsError(
                "GUARDRAILS_URL environment variable is not set."
            )
        return self._base_url.rstrip("/")

    async def check(
        self,
        rail_type: str,
        messages: List[Dict[str, str]],
        *,
        evaluation_type: str = "input",
    ) -> GuardrailResult:
        """Evaluate content against a guardrail.

        Args:
            rail_type: Type of guardrail — ``"jailbreak"``,
                ``"content_moderation"``, or ``"sensitive_data"``.
            messages: Messages to evaluate, each with ``role`` and ``content``.
            evaluation_type: ``"input"`` (default) to evaluate user messages
                before LLM processing, or ``"output"`` to evaluate AI responses.

        Returns:
            :class:`GuardrailResult` with ``allowed``, ``violations``,
            ``team_id``, and ``token_usage``.

        Raises:
            GuardrailsError: On authentication failure, invalid rail type,
                or service unavailability.

        Example::

            result = await guardrails.check(
                rail_type="jailbreak",
                messages=[{"role": "user", "content": "Hello!"}],
            )
            if result.allowed:
                print("Content is safe")
            else:
                for v in result.violations:
                    print(f"Violation: {v.message} ({v.rule_name})")
        """
        token = self._resolve_token()
        url = self._resolve_url()
        payload = {
            "rail_type": rail_type,
            "messages": messages,
            "evaluation_type": evaluation_type,
        }

        span = _start_guardrail_span(rail_type, payload)
        start_ns = time.monotonic_ns()

        try:
            result = await self._call(token, url, payload)
            duration_ns = time.monotonic_ns() - start_ns
            _end_guardrail_span(span, result, duration_ns)
            return result
        except Exception as exc:
            duration_ns = time.monotonic_ns() - start_ns
            _error_guardrail_span(span, exc, duration_ns)
            raise

    async def _call(
        self, token: str, url: str, payload: Dict[str, Any]
    ) -> GuardrailResult:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )

        if resp.status_code == 401:
            body = resp.json()
            raise GuardrailsError(
                body.get("description", "Authentication failed"),
                status_code=401,
            )

        if resp.status_code != 200:
            try:
                body = resp.json()
                detail = body.get("detail", body.get("message", resp.text))
            except Exception:
                detail = resp.text
            raise GuardrailsError(
                f"Guardrails service error ({resp.status_code}): {detail}",
                status_code=resp.status_code,
            )

        body = resp.json()
        violations = [
            GuardrailViolation(message=v["message"], rule_name=v["rule_name"])
            for v in body.get("violations", [])
        ]
        usage = body.get("token_usage", {})
        return GuardrailResult(
            allowed=body["allowed"],
            team_id=body["team_id"],
            violations=violations,
            token_usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )


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
    result: GuardrailResult,
    duration_ns: int,
) -> None:
    if span is None:
        return
    tracker = get_tracker()
    if not tracker:
        return

    output = {
        "allowed": result.allowed,
        "team_id": result.team_id,
        "violations": [
            {"message": v.message, "rule_name": v.rule_name}
            for v in result.violations
        ],
        "token_usage": {
            "input_tokens": result.token_usage.input_tokens,
            "output_tokens": result.token_usage.output_tokens,
            "total_tokens": result.token_usage.total_tokens,
        },
    }

    meta = span.metadata or {}
    meta["duration_ns"] = duration_ns
    meta["guardrail_allowed"] = result.allowed
    meta["guardrail_violations"] = len(result.violations)
    span.metadata = meta

    tracker.on_node_end(span, output)


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
