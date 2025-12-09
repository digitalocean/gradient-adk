from __future__ import annotations
import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict
import httpx

from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI
from gradient_adk.digital_ocean_api.models import (
    PresignedUrlFile,
    CreateEvaluationDatasetFileUploadPresignedUrlsInput,
    CreateEvaluationDatasetInput,
    FileUploadDataSource,
    StarMetric,
    UpdateEvaluationTestCaseInput,
    UpdateEvaluationTestCaseMetrics,
    RunEvaluationTestCaseInput,
    EvaluationRun,
    EvaluationRunStatus,
    EvaluationMetric,
    EvaluationMetricCategory,
    EvaluationMetricValueType,
)
from gradient_adk.logging import get_logger

logger = get_logger(__name__)


class EvaluationService:
    """Service for managing agent evaluations."""

    def __init__(self, client: AsyncDigitalOceanGenAI):
        self.client = client
        self._metrics_cache: Optional[List[EvaluationMetric]] = None

    async def get_available_metrics(self) -> List[EvaluationMetric]:
        """Get all available evaluation metrics from the API.

        Returns:
            List of EvaluationMetric objects
        """
        if self._metrics_cache is None:
            logger.debug("Fetching evaluation metrics from API")
            response = await self.client.list_evaluation_metrics()
            self._metrics_cache = response.metrics
            logger.debug(f"Loaded {len(self._metrics_cache)} evaluation metrics")
        return self._metrics_cache

    async def get_metrics_by_category(self) -> Dict[str, List[EvaluationMetric]]:
        """Organize metrics by category.

        Returns:
            Dictionary mapping category names to lists of metrics
        """
        metrics = await self.get_available_metrics()
        by_category: Dict[str, List[EvaluationMetric]] = {}

        for metric in metrics:
            if metric.category:
                # Convert enum to readable string (e.g., METRIC_CATEGORY_CORRECTNESS -> correctness)
                category_name = metric.category.value.replace(
                    "METRIC_CATEGORY_", ""
                ).lower()
                if category_name not in by_category:
                    by_category[category_name] = []
                by_category[category_name].append(metric)

        # Sort metrics within each category by rank
        for metrics_list in by_category.values():
            metrics_list.sort(key=lambda m: m.metric_rank or 999)

        return by_category

    async def find_metric_by_name(self, metric_name: str) -> Optional[EvaluationMetric]:
        """Find a metric by its name.

        Args:
            metric_name: The name of the metric to find

        Returns:
            The matching EvaluationMetric or None if not found
        """
        metrics = await self.get_available_metrics()
        for metric in metrics:
            if metric.metric_name == metric_name:
                return metric
        return None

    async def run_evaluation(
        self,
        agent_workspace_name: str,
        agent_deployment_name: str,
        test_case_name: str,
        dataset_file_path: Path,
        metric_categories: List[str],
        star_metric_name: Optional[str] = None,
        success_threshold: Optional[float] = None,
    ) -> str:
        """Run an evaluation test case for an agent.

        Args:
            agent_workspace_name: The name of the agent workspace
            agent_deployment_name: The name of the agent deployment
            test_case_name: The name of the evaluation test case
            dataset_file_path: Path to the CSV dataset file
            metric_categories: List of metric categories to use (e.g., ["correctness", "safety_and_security"])
            star_metric_name: Name of the star metric (optional, will use first metric if not specified)
            success_threshold: Success threshold for star metric (optional, only for number/percentage metrics)

        Returns:
            The evaluation run UUID

        Raises:
            ValueError: If dataset file is not a CSV or doesn't exist, or invalid categories
        """
        # Validate dataset file
        if not dataset_file_path.exists():
            raise ValueError(f"Dataset file not found: {dataset_file_path}")
        if dataset_file_path.suffix.lower() != ".csv":
            raise ValueError(f"Dataset file must be a CSV file: {dataset_file_path}")

        # Get metrics organized by category
        metrics_by_category = await self.get_metrics_by_category()
        available_categories = set(metrics_by_category.keys())

        # Validate categories and collect metrics
        selected_metrics: List[EvaluationMetric] = []
        for category in metric_categories:
            if category not in available_categories:
                valid_categories = ", ".join(sorted(available_categories))
                raise ValueError(
                    f"Invalid metric category: {category}. Valid categories: {valid_categories}"
                )
            selected_metrics.extend(metrics_by_category[category])

        if not selected_metrics:
            raise ValueError(
                "No metrics selected. Please specify at least one category."
            )

        logger.info(
            "Starting evaluation",
            agent_workspace_name=agent_workspace_name,
            agent_deployment_name=agent_deployment_name,
            test_case_name=test_case_name,
            dataset_file=str(dataset_file_path),
            metric_categories=metric_categories,
            total_metrics=len(selected_metrics),
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

        # Find star metric
        star_metric_obj: Optional[EvaluationMetric] = None
        if star_metric_name:
            star_metric_obj = await self.find_metric_by_name(star_metric_name)
            if not star_metric_obj:
                raise ValueError(f"Star metric not found: {star_metric_name}")
        else:
            # Use first selected metric as star metric
            star_metric_obj = selected_metrics[0]
            logger.info(
                "No star metric specified, using first metric",
                metric_name=star_metric_obj.metric_name,
            )

        # Validate threshold for star metric
        if success_threshold is not None:
            if star_metric_obj.metric_value_type not in [
                EvaluationMetricValueType.METRIC_VALUE_TYPE_NUMBER,
                EvaluationMetricValueType.METRIC_VALUE_TYPE_PERCENTAGE,
            ]:
                raise ValueError(
                    f"Success threshold can only be set for number or percentage metrics. "
                    f"Star metric '{star_metric_obj.metric_name}' is type: {star_metric_obj.metric_value_type}"
                )
            # Validate threshold range
            if (
                star_metric_obj.range_min is not None
                and success_threshold < star_metric_obj.range_min
            ):
                raise ValueError(
                    f"Success threshold {success_threshold} is below minimum {star_metric_obj.range_min} "
                    f"for metric '{star_metric_obj.metric_name}'"
                )
            if (
                star_metric_obj.range_max is not None
                and success_threshold > star_metric_obj.range_max
            ):
                raise ValueError(
                    f"Success threshold {success_threshold} is above maximum {star_metric_obj.range_max} "
                    f"for metric '{star_metric_obj.metric_name}'"
                )

        # Build metric UUIDs list from selected metrics
        metric_uuid_strings = [m.metric_uuid for m in selected_metrics]

        star_metric = StarMetric(
            metric_uuid=star_metric_obj.metric_uuid,
            name=star_metric_obj.metric_name,
            success_threshold=success_threshold,
        )

        if existing_test_case:
            # Step 3a: Update existing test case with new metrics and dataset
            logger.info(
                "Updating test case",
                test_case_uuid=test_case_uuid,
                dataset_uuid=dataset_uuid,
            )
            update_input = UpdateEvaluationTestCaseInput(
                test_case_uuid=test_case_uuid,
                dataset_uuid=dataset_uuid,
                metrics=UpdateEvaluationTestCaseMetrics(
                    metric_uuids=metric_uuid_strings
                ),
                star_metric=star_metric,
            )
            # Debug: log what we're sending
            import json

            payload = update_input.model_dump(
                by_alias=True, exclude_none=True, mode="json"
            )
            logger.debug("Update payload", payload=json.dumps(payload, indent=2))
            await self.client.update_evaluation_test_case(update_input)
            logger.debug("Test case updated", test_case_uuid=test_case_uuid)
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
                metrics=metric_uuid_strings,
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
            "Waiting for evaluation run to complete. This can take several minutes...",
            evaluation_run_uuid=evaluation_run_uuid,
        )

        start_time = asyncio.get_event_loop().time()
        terminal_statuses = {
            EvaluationRunStatus.EVALUATION_RUN_SUCCESSFUL,
            EvaluationRunStatus.EVALUATION_RUN_FAILED,
            EvaluationRunStatus.EVALUATION_RUN_CANCELLED,
            EvaluationRunStatus.EVALUATION_RUN_PARTIALLY_SUCCESSFUL,
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
                if evaluation_run.status == EvaluationRunStatus.EVALUATION_RUN_FAILED:
                    error_msg = evaluation_run.error_description or "Unknown error"
                    raise RuntimeError(
                        f"Evaluation run {evaluation_run_uuid} failed: {error_msg}"
                    )
                elif (
                    evaluation_run.status
                    == EvaluationRunStatus.EVALUATION_RUN_CANCELLED
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
