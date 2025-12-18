from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence
import httpx
from pydantic import BaseModel, ValidationError

from gradient_adk.logging import get_logger
from .models import (
    CreateTracesInput,
    CreateTracesOutput,
    EmptyResponse,
    GetDefaultProjectResponse,
    TracingServiceJWTOutput,
    GetAgentWorkspaceDeploymentOutput,
    GetAgentWorkspaceOutput,
    ListAgentWorkspacesOutput,
    CreateAgentDeploymentFileUploadPresignedURLInput,
    CreateAgentDeploymentFileUploadPresignedURLOutput,
    CreateAgentWorkspaceDeploymentInput,
    CreateAgentWorkspaceDeploymentOutput,
    CreateAgentDeploymentReleaseInput,
    CreateAgentDeploymentReleaseOutput,
    GetAgentDeploymentReleaseOutput,
    CreateAgentWorkspaceInput,
    CreateAgentWorkspaceOutput,
    GetAgentWorkspaceDeploymentRuntimeLogsOutput,
    ListEvaluationTestCasesByWorkspaceInput,
    ListEvaluationTestCasesByWorkspaceOutput,
    CreateEvaluationTestCaseInput,
    CreateEvaluationTestCaseOutput,
    UpdateEvaluationTestCaseInput,
    UpdateEvaluationTestCaseOutput,
    RunEvaluationTestCaseInput,
    RunEvaluationTestCaseOutput,
    CreateEvaluationDatasetInput,
    CreateEvaluationDatasetOutput,
    CreateEvaluationDatasetFileUploadPresignedUrlsInput,
    CreateEvaluationDatasetFileUploadPresignedUrlsOutput,
    GetEvaluationRunOutput,
    ListEvaluationMetricsOutput,
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

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()
        return False

    async def create_traces(self, req: CreateTracesInput) -> CreateTracesOutput:
        # can not have more than 1,000 spans
        for trace in req.traces:
            if len(trace.spans) > 1000:
                raise DOAPIValidationError(
                    "A single trace can not have more than 1,000 spans"
                )
        body = self._model_dump(req)
        logger.debug("Creating traces", request_body=body)
        data = await self._post_json("/gen-ai/traces", body)
        return CreateTracesOutput(**data)

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
        path = f"/gen-ai/tracing_tokens/{agent_workspace_uuid}/{agent_deployment_name}"
        data = await self._get_json(path)
        return TracingServiceJWTOutput(**data)

    async def get_agent_workspace_deployment(
        self, agent_workspace_name: str, agent_deployment_name: str
    ) -> GetAgentWorkspaceDeploymentOutput:
        """Get agent workspace deployment details.

        Args:
            agent_workspace_name: The name of the agent workspace
            agent_deployment_name: The name of the agent deployment

        Returns:
            GetAgentWorkspaceDeploymentOutput containing the agent workspace deployment details
        """
        logger.debug(
            "Getting agent workspace deployment",
            agent_workspace_name=agent_workspace_name,
            agent_deployment_name=agent_deployment_name,
        )
        path = f"/gen-ai/agent-workspaces/{agent_workspace_name}/agent-deployments/{agent_deployment_name}"
        data = await self._get_json(path)
        return GetAgentWorkspaceDeploymentOutput(**data)

    async def get_agent_workspace(
        self, agent_workspace_name: str
    ) -> GetAgentWorkspaceOutput:
        """Get an agent workspace by name.

        Args:
            agent_workspace_name: The name of the agent workspace

        Returns:
            GetAgentWorkspaceOutput containing the agent workspace
        """
        logger.debug(
            "Getting agent workspace",
            agent_workspace_name=agent_workspace_name,
        )
        path = f"/gen-ai/agent-workspaces/{agent_workspace_name}"
        data = await self._get_json(path)
        return GetAgentWorkspaceOutput(**data)

    async def list_agent_workspaces(self) -> ListAgentWorkspacesOutput:
        """List all agent workspaces.

        Returns:
            ListAgentWorkspacesOutput containing the list of agent workspaces
        """
        logger.debug("Listing agent workspaces")
        path = "/gen-ai/agent-workspaces"
        data = await self._get_json(path)
        return ListAgentWorkspacesOutput(**data)

    async def create_agent_workspace(
        self, input_data: CreateAgentWorkspaceInput
    ) -> CreateAgentWorkspaceOutput:
        """Create an agent workspace.

        Args:
            input_data: The workspace configuration including workspace name, deployment name, code artifact, and project id

        Returns:
            CreateAgentWorkspaceOutput containing the created agent workspace
        """
        logger.debug(
            "Creating agent workspace",
            agent_workspace_name=input_data.agent_workspace_name,
            agent_deployment_name=input_data.agent_deployment_name,
            project_id=input_data.project_id,
        )
        path = "/gen-ai/agent-workspaces"
        body = self._model_dump(input_data)
        data = await self._post_json(path, body)
        return CreateAgentWorkspaceOutput(**data)

    async def create_agent_deployment_file_upload_presigned_url(
        self, input_data: CreateAgentDeploymentFileUploadPresignedURLInput
    ) -> CreateAgentDeploymentFileUploadPresignedURLOutput:
        """Create a presigned URL for uploading agent deployment files.

        Args:
            input_data: The file metadata for which to generate a presigned URL

        Returns:
            CreateAgentDeploymentFileUploadPresignedURLOutput containing the presigned URL and request ID
        """
        logger.debug(
            "Creating agent deployment file upload presigned URL",
            file_name=input_data.file.file_name,
            file_size=input_data.file.file_size,
        )
        path = "/gen-ai/agent-workspace-deployments/file_upload_presigned_url"
        body = self._model_dump(input_data)
        data = await self._post_json(path, body)
        return CreateAgentDeploymentFileUploadPresignedURLOutput(**data)

    async def create_agent_workspace_deployment(
        self, input_data: CreateAgentWorkspaceDeploymentInput
    ) -> CreateAgentWorkspaceDeploymentOutput:
        """Create an agent workspace deployment.

        Args:
            input_data: The deployment configuration including workspace name, deployment name, and code artifact

        Returns:
            CreateAgentWorkspaceDeploymentOutput containing the created agent workspace deployment
        """
        logger.debug(
            "Creating agent workspace deployment",
            agent_workspace_name=input_data.agent_workspace_name,
            agent_deployment_name=input_data.agent_deployment_name,
            agent_code_file_path=input_data.agent_deployment_code_artifact.agent_code_file_path,
        )
        path = f"/gen-ai/agent-workspaces/{input_data.agent_workspace_name}/agent-deployments"
        body = self._model_dump(input_data)
        data = await self._post_json(path, body)
        return CreateAgentWorkspaceDeploymentOutput(**data)

    async def create_agent_deployment_release(
        self, input_data: CreateAgentDeploymentReleaseInput
    ) -> CreateAgentDeploymentReleaseOutput:
        """Create an agent deployment release.

        Args:
            input_data: The release configuration including workspace name, deployment name, and code artifact

        Returns:
            CreateAgentDeploymentReleaseOutput containing the created agent deployment release
        """
        logger.debug(
            "Creating agent deployment release",
            agent_workspace_name=input_data.agent_workspace_name,
            agent_deployment_name=input_data.agent_deployment_name,
            agent_code_file_path=input_data.agent_deployment_code_artifact.agent_code_file_path,
        )
        path = f"/gen-ai/agent-workspaces/{input_data.agent_workspace_name}/agent-deployments/{input_data.agent_deployment_name}/releases"
        body = self._model_dump(input_data)
        data = await self._post_json(path, body)
        return CreateAgentDeploymentReleaseOutput(**data)

    async def get_agent_deployment_release(
        self, uuid: str
    ) -> GetAgentDeploymentReleaseOutput:
        """Get an agent deployment release by UUID.

        Args:
            uuid: The unique agent deployment release id

        Returns:
            GetAgentDeploymentReleaseOutput containing the agent deployment release details
        """
        logger.debug(
            "Getting agent deployment release",
            uuid=uuid,
        )
        path = f"/gen-ai/agent-workspace-deployment-releases/{uuid}"
        data = await self._get_json(path)
        return GetAgentDeploymentReleaseOutput(**data)

    async def get_agent_workspace_deployment_runtime_logs(
        self, agent_workspace_name: str, agent_deployment_name: str
    ) -> GetAgentWorkspaceDeploymentRuntimeLogsOutput:
        """Get runtime logs URL for an agent workspace deployment.

        Args:
            agent_workspace_name: The name of the agent workspace
            agent_deployment_name: The name of the agent deployment

        Returns:
            GetAgentWorkspaceDeploymentRuntimeLogsOutput containing the live logs URL
        """
        logger.debug(
            "Getting agent workspace deployment runtime logs",
            agent_workspace_name=agent_workspace_name,
            agent_deployment_name=agent_deployment_name,
        )
        path = f"/gen-ai/agent-workspaces/{agent_workspace_name}/agent-deployments/{agent_deployment_name}/logs?tail_lines=50"
        data = await self._get_json(path)
        return GetAgentWorkspaceDeploymentRuntimeLogsOutput(**data)

    async def list_evaluation_test_cases_by_workspace(
        self,
        workspace_uuid: Optional[str] = None,
        agent_workspace_name: Optional[str] = None,
    ) -> ListEvaluationTestCasesByWorkspaceOutput:
        """List evaluation test cases for a workspace.

        Args:
            workspace_uuid: Workspace UUID (optional if agent_workspace_name is provided)
            agent_workspace_name: Workspace name (optional if workspace_uuid is provided)

        Returns:
            ListEvaluationTestCasesByWorkspaceOutput containing the list of evaluation test cases
        """
        if not workspace_uuid and not agent_workspace_name:
            raise ValueError(
                "Either workspace_uuid or agent_workspace_name must be provided"
            )

        logger.debug(
            "Listing evaluation test cases by workspace",
            workspace_uuid=workspace_uuid,
            agent_workspace_name=agent_workspace_name,
        )

        # Build query parameters
        params = {}
        if workspace_uuid:
            params["workspace_uuid"] = workspace_uuid
        if agent_workspace_name:
            params["agent_workspace_name"] = agent_workspace_name

        path = "/gen-ai/evaluation_test_cases"
        if params:
            # Construct query string
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            path = f"{path}?{query_string}"

        data = await self._get_json(path)
        return ListEvaluationTestCasesByWorkspaceOutput(**data)

    async def create_evaluation_test_case(
        self, input_data: CreateEvaluationTestCaseInput
    ) -> CreateEvaluationTestCaseOutput:
        """Create an evaluation test case.

        Args:
            input_data: The test case configuration including name, description, dataset, metrics, and workspace

        Returns:
            CreateEvaluationTestCaseOutput containing the created test case UUID
        """
        if not input_data.workspace_uuid and not input_data.agent_workspace_name:
            raise ValueError(
                "Either workspace_uuid or agent_workspace_name must be provided"
            )

        logger.debug(
            "Creating evaluation test case",
            name=input_data.name,
            dataset_uuid=input_data.dataset_uuid,
            workspace_uuid=input_data.workspace_uuid,
            agent_workspace_name=input_data.agent_workspace_name,
        )

        path = "/gen-ai/evaluation_test_cases"
        body = self._model_dump(input_data)
        data = await self._post_json(path, body)
        return CreateEvaluationTestCaseOutput(**data)

    async def update_evaluation_test_case(
        self, input_data: UpdateEvaluationTestCaseInput
    ) -> UpdateEvaluationTestCaseOutput:
        """Update an evaluation test case.

        Args:
            input_data: The test case updates including optional name, description, dataset, metrics, and star_metric

        Returns:
            UpdateEvaluationTestCaseOutput containing the test case UUID and new version
        """
        logger.info(
            "Updating evaluation test case",
            test_case_uuid=input_data.test_case_uuid,
            name=input_data.name,
            dataset_uuid=input_data.dataset_uuid,
        )

        path = f"/gen-ai/evaluation_test_cases/{input_data.test_case_uuid}"
        # Exclude test_case_uuid from body since it's in the URL path
        body = input_data.model_dump(
            by_alias=True, exclude_none=True, mode="json", exclude={"test_case_uuid"}
        )
        data = await self._put_json(path, body)
        return UpdateEvaluationTestCaseOutput(**data)

    async def run_evaluation_test_case(
        self, input_data: RunEvaluationTestCaseInput
    ) -> RunEvaluationTestCaseOutput:
        """Run an evaluation test case.

        Args:
            input_data: The run configuration including test case UUID, agent UUIDs/deployment names, and run name

        Returns:
            RunEvaluationTestCaseOutput containing the evaluation run UUIDs
        """
        logger.debug(
            "Running evaluation test case",
            test_case_uuid=input_data.test_case_uuid,
            run_name=input_data.run_name,
            agent_uuids=input_data.agent_uuids,
            agent_deployment_names=input_data.agent_deployment_names,
        )

        path = "/gen-ai/evaluation_runs"
        body = self._model_dump(input_data)
        data = await self._post_json(path, body)
        return RunEvaluationTestCaseOutput(**data)

    async def create_evaluation_dataset(
        self, input_data: CreateEvaluationDatasetInput
    ) -> CreateEvaluationDatasetOutput:
        """Create an evaluation dataset.

        Args:
            input_data: The dataset configuration including name and file upload data source

        Returns:
            CreateEvaluationDatasetOutput containing the dataset UUID
        """
        logger.debug(
            "Creating evaluation dataset",
            name=input_data.name,
            file_size=input_data.file_upload_dataset.size_in_bytes,
        )

        path = "/gen-ai/evaluation_datasets"
        body = self._model_dump(input_data)
        data = await self._post_json(path, body)
        return CreateEvaluationDatasetOutput(**data)

    async def create_evaluation_dataset_file_upload_presigned_urls(
        self, input_data: CreateEvaluationDatasetFileUploadPresignedUrlsInput
    ) -> CreateEvaluationDatasetFileUploadPresignedUrlsOutput:
        """Create presigned URLs for uploading evaluation dataset files.

        Args:
            input_data: The file metadata for which to generate presigned URLs

        Returns:
            CreateEvaluationDatasetFileUploadPresignedUrlsOutput containing the presigned URLs and request ID
        """
        logger.debug(
            "Creating evaluation dataset file upload presigned URLs",
            file_count=len(input_data.files),
        )

        path = "/gen-ai/evaluation_datasets/file_upload_presigned_urls"
        body = self._model_dump(input_data)
        data = await self._post_json(path, body)
        return CreateEvaluationDatasetFileUploadPresignedUrlsOutput(**data)

    async def get_evaluation_run(
        self, evaluation_run_uuid: str
    ) -> GetEvaluationRunOutput:
        """Get evaluation run details by UUID.

        Args:
            evaluation_run_uuid: The evaluation run UUID

        Returns:
            GetEvaluationRunOutput containing the evaluation run details
        """
        logger.debug(
            "Getting evaluation run",
            evaluation_run_uuid=evaluation_run_uuid,
        )

        path = f"/gen-ai/evaluation_runs/{evaluation_run_uuid}"
        data = await self._get_json(path)
        return GetEvaluationRunOutput(**data)

    async def list_evaluation_metrics(self) -> ListEvaluationMetricsOutput:
        """List all available evaluation metrics.

        Returns:
            ListEvaluationMetricsOutput containing the list of evaluation metrics
        """
        logger.debug("Listing evaluation metrics")

        path = "/gen-ai/evaluation_metrics"
        data = await self._get_json(path)
        return ListEvaluationMetricsOutput(**data)

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

    async def _put_json(self, path: str, body: Dict[str, Any]) -> Optional[dict]:
        attempt = 0
        last_exc: Exception | None = None

        while True:
            attempt += 1
            try:
                resp = await self._client.put(path, json=body)
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.TimeoutException,
            ) as e:
                last_exc = e
                if attempt > self.max_retries:
                    raise DOAPINetworkError(f"Network error on PUT {path}: {e}") from e
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
