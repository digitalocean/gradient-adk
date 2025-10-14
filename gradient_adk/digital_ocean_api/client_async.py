from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence
import httpx
from pydantic import BaseModel, ValidationError

from gradient_adk.logging import get_logger
from .models import (
    CreateTracesInput,
    EmptyResponse,
    GetDefaultProjectResponse,
    TracingServiceJWTOutput,
)
from .errors import (
    DOAPIAuthError,
    DOAPIRateLimitError,
    DOAPIClientError,
    DOAPIServerError,
    DOAPINetworkError,
    DOAPIValidationError,
    DOAPIError,
)
from .utils.utils import async_backoff_sleep, DEFAULT_RETRY_STATUSES

logger = get_logger(__name__)


class AsyncDigitalOceanGenAI:
    """
    Non-blocking DigitalOcean GenAI client (httpx.AsyncClient) with:
      - Pydantic validation
      - Exponential backoff + Retry-After support
      - Typed exceptions
    """

    def __init__(
        self,
        api_token: str,
        base_url: str = "https://api.digitalocean.com/v2",
        *,
        timeout_sec: float = 15.0,
        max_retries: int = 5,
        retry_statuses: Optional[Sequence[int]] = None,
        transport: httpx.AsyncBaseTransport | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_sec
        self.max_retries = max_retries
        self.retry_statuses = set(retry_statuses or DEFAULT_RETRY_STATUSES)
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        # A single shared async client; call `aclose()` when done
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_traces(self, req: CreateTracesInput) -> EmptyResponse:
        body = self._model_dump(req)
        logger.debug("Creating traces", request_body=body)
        data = await self._post_json("/gen-ai/traces", body)
        # Always return empty response until create_traces starts returning data
        if not data:
            return EmptyResponse()
        return EmptyResponse()

    async def get_default_project(self) -> GetDefaultProjectResponse:
        """Get the default project for the authenticated user."""
        logger.debug("Getting default project")
        data = await self._get_json("/projects/default")
        return GetDefaultProjectResponse(**data)

    async def get_tracing_token(
        self, agent_workspace_uuid: str, agent_deployment_name: str
    ) -> TracingServiceJWTOutput:
        """Get tracing token for the specified agent workspace and deployment."""
        logger.debug(
            "Getting tracing token",
            agent_workspace_uuid=agent_workspace_uuid,
            agent_deployment_name=agent_deployment_name,
        )
        path = f"/genai/tracing_tokens/{agent_workspace_uuid}/{agent_deployment_name}"
        data = await self._get_json(path)
        return TracingServiceJWTOutput(**data)

    @staticmethod
    def _model_dump(model: BaseModel) -> dict:
        try:
            # Use mode="json" to properly serialize datetime objects to ISO strings
            return model.model_dump(by_alias=True, exclude_none=True, mode="json")
        except ValidationError as e:
            raise DOAPIValidationError(f"Invalid request payload: {e}") from e

    async def _get_json(self, path: str) -> Optional[dict]:
        attempt = 0
        last_exc: Exception | None = None

        while True:
            attempt += 1
            try:
                resp = await self._client.get(path)
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.TimeoutException,
            ) as e:
                last_exc = e
                if attempt > self.max_retries:
                    raise DOAPINetworkError(f"Network error on GET {path}: {e}") from e
                await async_backoff_sleep(attempt)
                continue

            status = resp.status_code
            text = resp.text or ""
            payload = None
            try:
                if text.strip():
                    payload = resp.json()
            except json.JSONDecodeError:
                payload = None

            if 200 <= status < 300:
                return payload

            # Retryable?
            if status in self.retry_statuses and attempt < self.max_retries:
                retry_after_s = None
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after_s = float(ra)
                    except ValueError:
                        retry_after_s = None
                await async_backoff_sleep(attempt, retry_after=retry_after_s)
                continue

            # Non-retryable or out of retries, raise typed errors
            message = self._extract_error_message(payload) or f"HTTP {status}"
            if status in (401, 403):
                raise DOAPIAuthError(message, status_code=status, payload=payload)
            if status == 429:
                raise DOAPIRateLimitError(message, status_code=status, payload=payload)
            if 400 <= status < 500:
                raise DOAPIClientError(message, status_code=status, payload=payload)
            if 500 <= status < 600:
                raise DOAPIServerError(message, status_code=status, payload=payload)

            raise DOAPIError(message, status_code=status, payload=payload)

    async def _post_json(self, path: str, body: Dict[str, Any]) -> Optional[dict]:
        attempt = 0
        last_exc: Exception | None = None

        while True:
            attempt += 1
            try:
                resp = await self._client.post(path, json=body)
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.TimeoutException,
            ) as e:
                last_exc = e
                if attempt > self.max_retries:
                    raise DOAPINetworkError(f"Network error on POST {path}: {e}") from e
                await async_backoff_sleep(attempt)
                continue

            status = resp.status_code
            text = resp.text or ""
            payload = None
            try:
                if text.strip():
                    payload = resp.json()
            except json.JSONDecodeError:
                payload = None

            if 200 <= status < 300:
                return payload

            # Retryable?
            if status in self.retry_statuses and attempt < self.max_retries:
                retry_after_s = None
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after_s = float(ra)
                    except ValueError:
                        retry_after_s = None
                await async_backoff_sleep(attempt, retry_after=retry_after_s)
                continue

            # Non-retryable or out of retries, raise typed errors
            message = self._extract_error_message(payload) or f"HTTP {status}"
            if status in (401, 403):
                raise DOAPIAuthError(message, status_code=status, payload=payload)
            if status == 429:
                raise DOAPIRateLimitError(message, status_code=status, payload=payload)
            if 400 <= status < 500:
                raise DOAPIClientError(message, status_code=status, payload=payload)
            if 500 <= status < 600:
                raise DOAPIServerError(message, status_code=status, payload=payload)

            raise DOAPIError(message, status_code=status, payload=payload)

    @staticmethod
    def _extract_error_message(payload: Optional[dict]) -> Optional[str]:
        if not payload or not isinstance(payload, dict):
            return None
        if isinstance(payload.get("message"), str):
            return payload["message"]
        err = payload.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        return None
