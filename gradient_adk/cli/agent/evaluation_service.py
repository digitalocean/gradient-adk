from __future__ import annotations
import os
import asyncio
from pathlib import Path
from typing import Optional
import httpx

from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI
from gradient_adk.digital_ocean_api.models import (
    PresignedUrlFile,
    CreateEvaluationDatasetFileUploadPresignedUrlsInput,
    CreateEvaluationDatasetInput,
    FileUploadDataSource,
    StarMetric,
    UpdateEvaluationTestCaseInput,
    RunEvaluationTestCaseInput,
    EvaluationRun,
    EvaluationRunStatus,
)
from gradient_adk.logging import get_logger

logger = get_logger(__name__)


class EvaluationService:
    """Service for managing agent evaluations."""

    def __init__(self, client: AsyncDigitalOceanGenAI):
        self.client = client

    async def run_evaluation(
        self,
        agent_workspace_name: str,
        agent_deployment_name: str,
        test_case_name: str,
        dataset_file_path: Path,
    ) -> str:
        """Run an evaluation test case for an agent.

        Args:
            agent_workspace_name: The name of the agent workspace
            agent_deployment_name: The name of the agent deployment
            test_case_name: The name of the evaluation test case
            dataset_file_path: Path to the CSV dataset file

        Returns:
            The evaluation run UUID

        Raises:
            ValueError: If dataset file is not a CSV or doesn't exist
        """
        # Validate dataset file
        if not dataset_file_path.exists():
            raise ValueError(f"Dataset file not found: {dataset_file_path}")
        if dataset_file_path.suffix.lower() != ".csv":
            raise ValueError(f"Dataset file must be a CSV file: {dataset_file_path}")

        logger.info(
            "Starting evaluation",
            agent_workspace_name=agent_workspace_name,
            agent_deployment_name=agent_deployment_name,
            test_case_name=test_case_name,
            dataset_file=str(dataset_file_path),
        )

        # Step 1: List evaluation test cases by workspace
        logger.debug("Listing evaluation test cases", workspace=agent_workspace_name)
        test_cases_response = await self.client.list_evaluation_test_cases_by_workspace(
            agent_workspace_name=agent_workspace_name
        )

        # Step 2: Find existing test case or create new one
        test_case_uuid = None
        existing_test_case = None
        for test_case in test_cases_response.evaluation_test_cases:
            if test_case.name == test_case_name:
                existing_test_case = test_case
                test_case_uuid = test_case.test_case_uuid
                logger.info("Found existing test case", test_case_uuid=test_case_uuid)
                break

        # Upload dataset file
        logger.debug("Uploading dataset file")
        dataset_uuid = await self._upload_dataset(
            dataset_file_path=dataset_file_path,
            dataset_name=f"{test_case_name}_dataset",
        )
        logger.info("Dataset uploaded", dataset_uuid=dataset_uuid)

        # Hardcoded metric configuration
        # metric_uuids = ["11f04651-dfae-39b3-bf8f-4e013e2ddde4"]
        metric_uuids = ["11f04584-4b03-e119-bf8f-4e013e2ddde4"]
        star_metric = StarMetric(
            # metric_uuid="11f04651-dfae-39b3-bf8f-4e013e2ddde4",
            metric_uuid="11f04584-4b03-e119-bf8f-4e013e2ddde4",
            name="Correctness",
            success_threshold=80.0,
        )

        if existing_test_case:
            # Step 3a: Update existing test case with new metrics and dataset
            logger.debug(
                "Updating test case",
                test_case_uuid=test_case_uuid,
                dataset_uuid=dataset_uuid,
            )
            update_input = UpdateEvaluationTestCaseInput(
                test_case_uuid=test_case_uuid,
                dataset_uuid=dataset_uuid,
                metrics=metric_uuids,
                star_metric=star_metric,
            )
            await self.client.update_evaluation_test_case(update_input)
            logger.info("Test case updated", test_case_uuid=test_case_uuid)
        else:
            # Step 3b: Create new test case
            logger.debug("Creating new test case", name=test_case_name)
            from gradient_adk.digital_ocean_api.models import (
                CreateEvaluationTestCaseInput,
            )

            create_input = CreateEvaluationTestCaseInput(
                name=test_case_name,
                description=f"Evaluation test case for {agent_workspace_name}",
                dataset_uuid=dataset_uuid,
                metrics=metric_uuids,
                star_metric=star_metric,
                agent_workspace_name=agent_workspace_name,
            )
            create_response = await self.client.create_evaluation_test_case(
                create_input
            )
            test_case_uuid = create_response.test_case_uuid
            logger.info("Test case created", test_case_uuid=test_case_uuid)

        # Step 4: Run evaluation test case
        logger.debug(
            "Running evaluation",
            test_case_uuid=test_case_uuid,
            deployment_name=agent_deployment_name,
        )
        run_input = RunEvaluationTestCaseInput(
            test_case_uuid=test_case_uuid,
            agent_deployment_names=[agent_deployment_name],
            run_name=f"{test_case_name}_run",
        )
        run_response = await self.client.run_evaluation_test_case(run_input)

        if run_response.evaluation_run_uuids:
            evaluation_run_uuid = run_response.evaluation_run_uuids[0]
            logger.info("Evaluation started", evaluation_run_uuid=evaluation_run_uuid)
            return evaluation_run_uuid
        else:
            raise RuntimeError("Failed to start evaluation - no run UUID returned")

    async def _upload_dataset(self, dataset_file_path: Path, dataset_name: str) -> str:
        """Upload a dataset file and return the dataset UUID.

        Args:
            dataset_file_path: Path to the dataset file
            dataset_name: Name for the dataset

        Returns:
            The dataset UUID
        """
        file_size = dataset_file_path.stat().st_size
        file_name = dataset_file_path.name

        # Step 1: Get presigned URL for upload
        logger.debug("Getting presigned URL", file_name=file_name, size=file_size)
        presigned_input = CreateEvaluationDatasetFileUploadPresignedUrlsInput(
            files=[PresignedUrlFile(file_name=file_name, file_size=file_size)]
        )
        presigned_response = (
            await self.client.create_evaluation_dataset_file_upload_presigned_urls(
                presigned_input
            )
        )

        if not presigned_response.uploads:
            raise RuntimeError("Failed to get presigned URL for dataset upload")

        upload_info = presigned_response.uploads[0]
        presigned_url = upload_info.presigned_url
        object_key = upload_info.object_key

        # Step 2: Upload file to presigned URL
        logger.debug("Uploading file to presigned URL", url=presigned_url)
        async with httpx.AsyncClient() as upload_client:
            with open(dataset_file_path, "rb") as f:
                file_content = f.read()
            response = await upload_client.put(
                presigned_url,
                content=file_content,
                headers={"Content-Type": "text/csv"},
            )
            response.raise_for_status()

        logger.debug("File uploaded successfully", object_key=object_key)

        # Step 3: Create dataset record
        logger.debug("Creating dataset record", name=dataset_name)
        dataset_input = CreateEvaluationDatasetInput(
            name=dataset_name,
            file_upload_dataset=FileUploadDataSource(
                original_file_name=file_name,
                stored_object_key=object_key,
                size_in_bytes=file_size,
            ),
        )
        dataset_response = await self.client.create_evaluation_dataset(dataset_input)

        return dataset_response.evaluation_dataset_uuid

    async def poll_evaluation_run(
        self,
        evaluation_run_uuid: str,
        poll_interval_seconds: float = 5.0,
        max_wait_seconds: float = 600.0,
    ) -> EvaluationRun:
        """Poll an evaluation run until completion.

        Args:
            evaluation_run_uuid: The evaluation run UUID to poll
            poll_interval_seconds: Time to wait between polls (default: 5 seconds)
            max_wait_seconds: Maximum time to wait before timing out (default: 600 seconds)

        Returns:
            The completed EvaluationRun

        Raises:
            TimeoutError: If the evaluation doesn't complete within max_wait_seconds
            RuntimeError: If the evaluation fails
        """
        logger.info(
            "Polling evaluation run",
            evaluation_run_uuid=evaluation_run_uuid,
            poll_interval=poll_interval_seconds,
        )

        start_time = asyncio.get_event_loop().time()
        terminal_statuses = {
            EvaluationRunStatus.EVALUATION_RUN_STATUS_COMPLETED,
            EvaluationRunStatus.EVALUATION_RUN_STATUS_FAILED,
            EvaluationRunStatus.EVALUATION_RUN_STATUS_CANCELLED,
        }

        while True:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait_seconds:
                raise TimeoutError(
                    f"Evaluation run {evaluation_run_uuid} did not complete within {max_wait_seconds} seconds"
                )

            # Get current status
            response = await self.client.get_evaluation_run(evaluation_run_uuid)
            evaluation_run = response.evaluation_run

            logger.debug(
                "Evaluation run status",
                evaluation_run_uuid=evaluation_run_uuid,
                status=evaluation_run.status,
            )

            # Check if terminal status
            if evaluation_run.status in terminal_statuses:
                if (
                    evaluation_run.status
                    == EvaluationRunStatus.EVALUATION_RUN_STATUS_FAILED
                ):
                    error_msg = evaluation_run.error_description or "Unknown error"
                    raise RuntimeError(
                        f"Evaluation run {evaluation_run_uuid} failed: {error_msg}"
                    )
                elif (
                    evaluation_run.status
                    == EvaluationRunStatus.EVALUATION_RUN_STATUS_CANCELLED
                ):
                    raise RuntimeError(
                        f"Evaluation run {evaluation_run_uuid} was cancelled"
                    )
                else:
                    # Completed successfully
                    logger.info(
                        "Evaluation run completed",
                        evaluation_run_uuid=evaluation_run_uuid,
                    )
                    return evaluation_run

            # Wait before next poll
            await asyncio.sleep(poll_interval_seconds)
