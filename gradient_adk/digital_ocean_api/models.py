from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TraceSpanType(str, Enum):
    TRACE_SPAN_TYPE_UNKNOWN = "TRACE_SPAN_TYPE_UNKNOWN"
    TRACE_SPAN_TYPE_LLM = "TRACE_SPAN_TYPE_LLM"
    TRACE_SPAN_TYPE_RETRIEVER = "TRACE_SPAN_TYPE_RETRIEVER"
    TRACE_SPAN_TYPE_TOOL = "TRACE_SPAN_TYPE_TOOL"


class Span(BaseModel):
    """
    Represents a span within a trace (e.g., LLM call, retriever, tool).
    - created_at: RFC3339 timestamp (protobuf Timestamp)
    - input/output: json
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    created_at: datetime = Field(..., description="RFC3339 timestamp")
    input: Dict[str, Any]
    name: str
    output: Dict[str, Any]
    type: TraceSpanType = Field(default=TraceSpanType.TRACE_SPAN_TYPE_UNKNOWN)


class Trace(BaseModel):
    """
    Represents a complete trace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    created_at: datetime = Field(..., description="RFC3339 timestamp")
    input: Dict[str, Any]
    name: str
    output: Dict[str, Any]
    spans: List[Span] = Field(default_factory=list)


class CreateTracesInput(BaseModel):
    """
    Input for creating traces.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_deployment_name: str
    session_id: Optional[str] = None
    traces: List[Trace]
    agent_workspace_name: str


class CreateTracesOutput(BaseModel):
    """
    Response for creating traces.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    trace_uuids: List[str] = Field(
        default_factory=list, description="Trace UUIDs created"
    )
    session_id: Optional[str] = Field(None, description="Session ID")


class Project(BaseModel):
    """
    Represents a DigitalOcean project.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    owner_uuid: str
    owner_id: int
    name: str
    description: str
    purpose: str
    environment: str
    is_default: bool
    created_at: datetime = Field(..., description="RFC3339 timestamp")
    updated_at: datetime = Field(..., description="RFC3339 timestamp")


class GetDefaultProjectResponse(BaseModel):
    """
    Response for getting the default project.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    project: Project


class TracingServiceJWTOutput(BaseModel):
    """
    Response for getting tracing token.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    access_token: str = Field(
        ..., description="Access token for the clickout to the tracing service"
    )
    expires_at: str = Field(..., description="Expiry time of the access token")
    base_url: str = Field(..., description="Base URL for the tracing service instance")


class EmptyResponse(BaseModel):
    pass


class ReleaseStatus(str, Enum):
    RELEASE_STATUS_UNKNOWN = "RELEASE_STATUS_UNKNOWN"
    RELEASE_STATUS_BUILDING = "RELEASE_STATUS_BUILDING"
    RELEASE_STATUS_WAITING_FOR_DEPLOYMENT = "RELEASE_STATUS_WAITING_FOR_DEPLOYMENT"
    RELEASE_STATUS_DEPLOYING = "RELEASE_STATUS_DEPLOYING"
    RELEASE_STATUS_RUNNING = "RELEASE_STATUS_RUNNING"
    RELEASE_STATUS_FAILED = "RELEASE_STATUS_FAILED"
    RELEASE_STATUS_WAITING_FOR_UNDEPLOYMENT = "RELEASE_STATUS_WAITING_FOR_UNDEPLOYMENT"
    RELEASE_STATUS_UNDEPLOYING = "RELEASE_STATUS_UNDEPLOYING"
    RELEASE_STATUS_UNDEPLOYMENT_FAILED = "RELEASE_STATUS_UNDEPLOYMENT_FAILED"
    RELEASE_STATUS_DELETED = "RELEASE_STATUS_DELETED"


class AgentDeploymentRelease(BaseModel):
    """
    Represents an Agent Deployment Release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    uuid: str = Field(..., description="Unique release id")
    status: Optional[ReleaseStatus] = Field(
        None, description="The status of the release"
    )
    url: Optional[str] = Field(
        None, description="The URL to access the agent workspace deployment"
    )
    error_msg: Optional[str] = Field(
        None,
        description="Error message providing a hint which part of the system experienced an error",
    )
    created_at: datetime = Field(
        ..., description="Creation date/time (RFC3339 timestamp)"
    )
    updated_at: datetime = Field(
        ..., description="Last modified date/time (RFC3339 timestamp)"
    )
    created_by_user_id: int = Field(
        ..., description="ID of user that created the agent deployment release"
    )
    created_by_user_email: Optional[str] = Field(
        None, description="Email of user that created the agent deployment release"
    )


class AgentLoggingConfig(BaseModel):
    """
    Represents Agent Logging Config Details.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    galileo_project_id: str = Field(..., description="Galileo project identifier")
    galileo_project_name: str = Field(..., description="Name of the Galileo project")
    log_stream_id: str = Field(..., description="Identifier for the log stream")
    log_stream_name: str = Field(..., description="Name of the log stream")
    insights_enabled_at: Optional[datetime] = Field(
        None, description="Timestamp when insights were enabled (RFC3339 timestamp)"
    )
    insights_enabled: Optional[bool] = Field(
        None, description="Whether insights are enabled"
    )


class AgentWorkspaceDeployment(BaseModel):
    """
    Represents an Agent Workspace Deployment.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    uuid: str = Field(..., description="Unique agent id")
    name: str = Field(..., description="Agent name")
    created_at: datetime = Field(
        ..., description="Creation date/time (RFC3339 timestamp)"
    )
    updated_at: datetime = Field(
        ..., description="Last modified date/time (RFC3339 timestamp)"
    )
    created_by_user_id: int = Field(
        ..., description="ID of user that created the agent workspace"
    )
    created_by_user_email: Optional[str] = Field(
        None, description="Email of user that created the agent workspace"
    )
    latest_release: Optional[AgentDeploymentRelease] = Field(
        None, description="The latest release"
    )
    active_release: Optional[AgentDeploymentRelease] = Field(
        None, description="The active release"
    )
    logging_config: Optional[AgentLoggingConfig] = Field(
        None, description="Agent Logging Config Details"
    )


class GetAgentWorkspaceDeploymentOutput(BaseModel):
    """
    Response for getting an agent workspace deployment.
    """

    model_config = ConfigDict(populate_by_name=True)

    agent_workspace_deployment: AgentWorkspaceDeployment = Field(
        ..., description="The agent workspace deployment"
    )


class AgentWorkspace(BaseModel):
    """
    Represents an Agent Workspace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    uuid: str = Field(..., description="Unique agent id")
    name: str = Field(..., description="Agent name")
    created_at: datetime = Field(
        ..., description="Creation date/time (RFC3339 timestamp)"
    )
    updated_at: datetime = Field(
        ..., description="Last modified date/time (RFC3339 timestamp)"
    )
    created_by_user_id: int = Field(
        ..., description="ID of user that created the agent workspace"
    )
    created_by_user_email: Optional[str] = Field(
        None, description="Email of user that created the agent workspace"
    )
    team_id: int = Field(..., description="Team ID the agent workspace belongs to")
    project_id: Optional[str] = Field(
        None, description="The project ID the agent workspace belongs to"
    )
    deployments: list[AgentWorkspaceDeployment] = Field(
        default_factory=list, description="The deployments the agent workspace has"
    )


class ListAgentWorkspacesOutput(BaseModel):
    """
    Response for listing agent workspaces.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_workspaces: list[AgentWorkspace] = Field(
        default_factory=list, description="List of agent workspaces"
    )


class GetAgentWorkspaceOutput(BaseModel):
    """
    Response for getting a single agent workspace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_workspace: AgentWorkspace = Field(..., description="The agent workspace")


class PresignedUrlFile(BaseModel):
    """
    A single file's metadata in the request.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    file_name: str = Field(..., description="Local filename")
    file_size: int = Field(..., description="The size of the file in bytes")


class CreateAgentDeploymentFileUploadPresignedURLInput(BaseModel):
    """
    Input for creating agent deployment file upload presigned URL.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    file: PresignedUrlFile = Field(
        ..., description="The file to generate presigned URL for"
    )


class FilePresignedUrlResponse(BaseModel):
    """
    Detailed info about each presigned URL returned to the client.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    object_key: str = Field(
        ..., description="The unique object key to store the file as"
    )
    original_file_name: Optional[str] = Field(
        None, description="The original file name"
    )
    presigned_url: str = Field(
        ...,
        description="The actual presigned URL the client can use to upload the file directly",
    )
    expires_at: datetime = Field(
        ..., description="The time the url expires at (RFC3339 timestamp)"
    )


class CreateAgentDeploymentFileUploadPresignedURLOutput(BaseModel):
    """
    Response for creating agent deployment file upload presigned URL.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    request_id: str = Field(
        ..., description="The ID generated for the request for Presigned URLs"
    )
    upload: FilePresignedUrlResponse = Field(
        ..., description="The generated presigned URL and object key"
    )


class AgentDeploymentCodeArtifact(BaseModel):
    """
    File to upload for agent deployment.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_code_file_path: str = Field(..., description="The agent code file path")
    stored_object_key: str = Field(
        ..., description="The object key the file was stored as"
    )
    size_in_bytes: int = Field(..., description="The size of the file in bytes")


class CreateAgentWorkspaceDeploymentInput(BaseModel):
    """
    Input for creating an agent workspace deployment.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_workspace_name: str = Field(..., description="The name of agent workspace")
    agent_deployment_name: str = Field(..., description="The deployment name")
    agent_deployment_code_artifact: AgentDeploymentCodeArtifact = Field(
        ..., description="The agent deployment code artifact"
    )


class CreateAgentWorkspaceDeploymentOutput(BaseModel):
    """
    Response for creating an agent workspace deployment.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_workspace_deployment: AgentWorkspaceDeployment = Field(
        ..., description="The agent workspace deployment"
    )


class CreateAgentDeploymentReleaseInput(BaseModel):
    """
    Input for creating an agent deployment release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_workspace_name: str = Field(..., description="The name of agent workspace")
    agent_deployment_name: str = Field(..., description="The deployment name")
    agent_deployment_code_artifact: AgentDeploymentCodeArtifact = Field(
        ..., description="The agent deployment code artifact"
    )


class CreateAgentDeploymentReleaseOutput(BaseModel):
    """
    Response for creating an agent deployment release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_deployment_release: AgentDeploymentRelease = Field(
        ..., description="The agent deployment release"
    )


class GetAgentDeploymentReleaseInput(BaseModel):
    """
    Input for getting an agent deployment release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    uuid: str = Field(..., description="Unique agent deployment release id")


class GetAgentDeploymentReleaseOutput(BaseModel):
    """
    Response for getting an agent deployment release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_deployment_release: AgentDeploymentRelease = Field(
        ..., description="The agent deployment release"
    )


class CreateAgentWorkspaceInput(BaseModel):
    """
    Input for creating an agent workspace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_workspace_name: str = Field(..., description="The name of agent workspace")
    agent_deployment_name: str = Field(..., description="The deployment name")
    agent_deployment_code_artifact: AgentDeploymentCodeArtifact = Field(
        ..., description="The agent deployment code artifact"
    )
    project_id: str = Field(..., description="The project id")


class CreateAgentWorkspaceOutput(BaseModel):
    """
    Response for creating an agent workspace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    agent_workspace: AgentWorkspace = Field(..., description="The agent workspace")


class GetAgentWorkspaceDeploymentRuntimeLogsOutput(BaseModel):
    """
    Response for getting agent workspace deployment runtime logs.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    live_url: str = Field(..., description="URL for live logs")


class StarMetric(BaseModel):
    """
    Represents a star metric for evaluations.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    metric_uuid: str = Field(..., description="UUID of the metric")
    name: str = Field(..., description="Name of the metric")
    success_threshold_pct: Optional[int] = Field(
        None,
        description="The success threshold percentage (0-100) - deprecated",
        deprecated=True,
    )
    success_threshold: Optional[float] = Field(
        None,
        description="The success threshold value the metric must reach",
    )


class EvaluationMetric(BaseModel):
    """
    Represents an evaluation metric.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    metric_uuid: str = Field(..., description="UUID of the metric")
    metric_name: str = Field(..., description="Name of the metric")
    description: Optional[str] = Field(None, description="Description of the metric")
    metric_type: Optional[EvaluationMetricType] = Field(
        None, description="Type of the metric"
    )
    metric_value_type: Optional[EvaluationMetricValueType] = Field(
        None, description="Value type of the metric"
    )
    range_min: Optional[float] = Field(None, description="Minimum range value")
    range_max: Optional[float] = Field(None, description="Maximum range value")
    inverted: Optional[bool] = Field(
        None, description="If true, lower values are better"
    )
    category: Optional[EvaluationMetricCategory] = Field(
        None, description="Category of the metric"
    )
    is_metric_goal: Optional[bool] = Field(
        None, description="Whether this is a goal metric"
    )
    metric_rank: Optional[int] = Field(None, description="Rank of the metric")


class EvaluationDataset(BaseModel):
    """
    Represents an evaluation dataset.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    dataset_uuid: str = Field(..., description="UUID of the dataset")
    dataset_name: str = Field(..., description="Name of the dataset")
    row_count: Optional[int] = Field(None, description="Number of rows in the dataset")
    has_ground_truth: Optional[bool] = Field(
        None, description="Does the dataset have a ground truth column?"
    )
    file_size: Optional[int] = Field(
        None, description="The size of the dataset uploaded file in bytes"
    )
    created_at: datetime = Field(..., description="Time created at (RFC3339 timestamp)")


class EvaluationTestCase(BaseModel):
    """
    Represents an evaluation test case.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    test_case_uuid: str = Field(..., description="UUID of the test case")
    name: str = Field(..., description="Name of the test case")
    description: Optional[str] = Field(None, description="Description of the test case")
    version: int = Field(..., description="Version number")
    metrics: List[EvaluationMetric] = Field(
        default_factory=list, description="Evaluation metrics"
    )
    star_metric: Optional[StarMetric] = Field(
        None, description="Star metric for the evaluation"
    )
    total_runs: Optional[int] = Field(None, description="Total number of runs")
    latest_version_number_of_runs: Optional[int] = Field(
        None, description="Number of runs for the latest version"
    )
    updated_by_user_id: int = Field(..., description="ID of user who last updated")
    updated_by_user_email: Optional[str] = Field(
        None, description="Email of user who last updated"
    )
    created_by_user_email: Optional[str] = Field(
        None, description="Email of user who created"
    )
    created_by_user_id: int = Field(..., description="ID of user who created")
    created_at: datetime = Field(
        ..., description="Creation date/time (RFC3339 timestamp)"
    )
    updated_at: datetime = Field(
        ..., description="Last modified date/time (RFC3339 timestamp)"
    )
    archived_at: Optional[datetime] = Field(
        None, description="Archived date/time (RFC3339 timestamp)"
    )
    dataset: Optional[EvaluationDataset] = Field(
        None, description="The evaluation dataset information"
    )


class ListEvaluationTestCasesByWorkspaceInput(BaseModel):
    """
    Input for listing evaluation test cases by workspace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    workspace_uuid: Optional[str] = Field(
        None,
        description="Workspace UUID (optional if agent_workspace_name is provided)",
    )
    agent_workspace_name: Optional[str] = Field(
        None, description="Workspace name (optional if workspace_uuid is provided)"
    )


class ListEvaluationTestCasesByWorkspaceOutput(BaseModel):
    """
    Response for listing evaluation test cases by workspace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    evaluation_test_cases: List[EvaluationTestCase] = Field(
        default_factory=list, description="List of evaluation test cases"
    )


class CreateEvaluationTestCaseInput(BaseModel):
    """
    Input for creating an evaluation test case.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str = Field(..., description="Name of the test case")
    description: str = Field(..., description="Description of the test case")
    dataset_uuid: str = Field(
        ..., description="Dataset against which the test case is executed"
    )
    metrics: List[str] = Field(
        default_factory=list, description="Full metric list to use for evaluation"
    )
    star_metric: StarMetric = Field(..., description="Star metric for test case")
    workspace_uuid: Optional[str] = Field(
        None,
        description="The workspace UUID (optional if agent_workspace_name provided)",
    )
    agent_workspace_name: Optional[str] = Field(
        None, description="The workspace name (optional if workspace_uuid provided)"
    )


class CreateEvaluationTestCaseOutput(BaseModel):
    """
    Response for creating an evaluation test case.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    test_case_uuid: str = Field(..., description="Test case UUID")


class UpdateEvaluationTestCaseMetrics(BaseModel):
    """
    Metrics update structure for evaluation test cases.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    metric_uuids: List[str] = Field(
        default_factory=list, description="List of metric UUIDs"
    )


class UpdateEvaluationTestCaseInput(BaseModel):
    """
    Input for updating an evaluation test case.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    test_case_uuid: str = Field(..., description="Test case UUID to update")
    name: Optional[str] = Field(None, description="Name of the test case")
    description: Optional[str] = Field(None, description="Description of the test case")
    dataset_uuid: Optional[str] = Field(
        None, description="Dataset against which the test case is executed"
    )
    metrics: Optional[UpdateEvaluationTestCaseMetrics] = Field(
        None, description="Metrics to use for evaluation"
    )
    star_metric: Optional[StarMetric] = Field(
        None, description="Optional star metric to update"
    )


class UpdateEvaluationTestCaseOutput(BaseModel):
    """
    Response for updating an evaluation test case.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    test_case_uuid: str = Field(..., description="Test case UUID")
    version: int = Field(..., description="The new version of the test case")


class RunEvaluationTestCaseInput(BaseModel):
    """
    Input for running an evaluation test case.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    test_case_uuid: str = Field(..., description="Test case UUID to run")
    agent_uuids: List[str] = Field(
        default_factory=list,
        description="Agent UUIDs to run the test case against (legacy agents)",
    )
    agent_deployment_names: List[str] = Field(
        default_factory=list,
        description="Agent deployment names to run the test case against (ADK agent workspaces)",
    )
    run_name: str = Field(..., description="The name of the run")


class RunEvaluationTestCaseOutput(BaseModel):
    """
    Response for running an evaluation test case.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    evaluation_run_uuids: List[str] = Field(
        default_factory=list, description="Evaluation run UUIDs"
    )


class FileUploadDataSource(BaseModel):
    """
    Represents a file upload data source.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    original_file_name: str = Field(..., description="The original file name")
    stored_object_key: str = Field(
        ..., description="The object key the file was stored as"
    )
    size_in_bytes: int = Field(..., description="The size of the file in bytes")


class CreateEvaluationDatasetInput(BaseModel):
    """
    Input for creating an evaluation dataset.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str = Field(..., description="The name of the agent evaluation dataset")
    file_upload_dataset: FileUploadDataSource = Field(
        ..., description="File to upload as the agent evaluation dataset"
    )


class CreateEvaluationDatasetOutput(BaseModel):
    """
    Response for creating an evaluation dataset.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    evaluation_dataset_uuid: str = Field(
        ..., description="Dataset UUID", alias="dataset_uuid"
    )


class CreateEvaluationDatasetFileUploadPresignedUrlsInput(BaseModel):
    """
    Input for creating evaluation dataset file upload presigned URLs.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    files: List[PresignedUrlFile] = Field(
        ..., description="A list of files to generate presigned URLs for"
    )


class CreateEvaluationDatasetFileUploadPresignedUrlsOutput(BaseModel):
    """
    Response for creating evaluation dataset file upload presigned URLs.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    request_id: str = Field(
        ..., description="The ID generated for the request for Presigned URLs"
    )
    uploads: List[FilePresignedUrlResponse] = Field(
        ...,
        description="A list of generated presigned URLs and object keys, one per file",
    )


class EvaluationMetricType(str, Enum):
    """Metric type enumeration."""

    METRIC_TYPE_UNSPECIFIED = "METRIC_TYPE_UNSPECIFIED"
    METRIC_TYPE_GENERAL_QUALITY = "METRIC_TYPE_GENERAL_QUALITY"
    METRIC_TYPE_RAG_AND_TOOL = "METRIC_TYPE_RAG_AND_TOOL"


class EvaluationMetricValueType(str, Enum):
    """Metric value type enumeration."""

    METRIC_VALUE_TYPE_UNSPECIFIED = "METRIC_VALUE_TYPE_UNSPECIFIED"
    METRIC_VALUE_TYPE_NUMBER = "METRIC_VALUE_TYPE_NUMBER"
    METRIC_VALUE_TYPE_STRING = "METRIC_VALUE_TYPE_STRING"
    METRIC_VALUE_TYPE_PERCENTAGE = "METRIC_VALUE_TYPE_PERCENTAGE"


class EvaluationMetricCategory(str, Enum):
    """Metric category enumeration."""

    METRIC_CATEGORY_UNSPECIFIED = "METRIC_CATEGORY_UNSPECIFIED"
    METRIC_CATEGORY_CORRECTNESS = "METRIC_CATEGORY_CORRECTNESS"
    METRIC_CATEGORY_USER_OUTCOMES = "METRIC_CATEGORY_USER_OUTCOMES"
    METRIC_CATEGORY_SAFETY_AND_SECURITY = "METRIC_CATEGORY_SAFETY_AND_SECURITY"
    METRIC_CATEGORY_CONTEXT_QUALITY = "METRIC_CATEGORY_CONTEXT_QUALITY"
    METRIC_CATEGORY_MODEL_FIT = "METRIC_CATEGORY_MODEL_FIT"


class EvaluationMetricResult(BaseModel):
    """
    Result of an evaluation metric.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    metric_name: str = Field(..., description="Metric name")
    number_value: Optional[float] = Field(
        None, description="The value of the metric as a number"
    )
    string_value: Optional[str] = Field(
        None, description="The value of the metric as a string"
    )
    reasoning: Optional[str] = Field(None, description="Reasoning of the metric result")
    error_description: Optional[str] = Field(
        None, description="Error description if the metric could not be calculated"
    )
    metric_value_type: Optional[EvaluationMetricValueType] = Field(
        None, description="The type of the metric"
    )


class EvaluationRunStatus(str, Enum):
    """Evaluation run status enumeration."""

    EVALUATION_RUN_STATUS_UNKNOWN = "EVALUATION_RUN_STATUS_UNKNOWN"
    EVALUATION_RUN_QUEUED = "EVALUATION_RUN_QUEUED"
    EVALUATION_RUN_RUNNING = "EVALUATION_RUN_RUNNING"
    EVALUATION_RUN_COMPLETED = "EVALUATION_RUN_COMPLETED"
    EVALUATION_RUN_FAILED = "EVALUATION_RUN_FAILED"
    EVALUATION_RUN_CANCELLED = "EVALUATION_RUN_CANCELLED"
    EVALUATION_RUN_RUNNING_DATASET = "EVALUATION_RUN_RUNNING_DATASET"
    EVALUATION_RUN_EVALUATING_RESULTS = "EVALUATION_RUN_EVALUATING_RESULTS"
    EVALUATION_RUN_PARTIALLY_SUCCESSFUL = "EVALUATION_RUN_PARTIALLY_SUCCESSFUL"
    EVALUATION_RUN_SUCCESSFUL = "EVALUATION_RUN_SUCCESSFUL"


class EvaluationRun(BaseModel):
    """
    Represents an evaluation run.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    evaluation_run_uuid: str = Field(..., description="Evaluation run UUID")
    test_case_uuid: str = Field(..., description="Test-case UUID")
    test_case_version: int = Field(..., description="Test-case-version")
    test_case_name: str = Field(..., description="Test case name")
    test_case_description: Optional[str] = Field(
        None, description="Test case description"
    )
    agent_uuid: str = Field(..., description="Agent UUID")
    agent_version_hash: Optional[str] = Field(None, description="version hash")
    run_name: str = Field(..., description="Run name")
    status: EvaluationRunStatus = Field(..., description="Run status")
    started_at: Optional[datetime] = Field(
        None, description="Run start time (RFC3339 timestamp)"
    )
    finished_at: Optional[datetime] = Field(
        None, description="Run end time (RFC3339 timestamp)"
    )
    pass_status: Optional[bool] = Field(
        None,
        description="The pass status of the evaluation run based on the star metric",
    )
    star_metric_result: Optional[EvaluationMetricResult] = Field(
        None, description="The result of the star metric"
    )
    run_level_metric_results: List[EvaluationMetricResult] = Field(
        default_factory=list, description="Run level metric results"
    )
    agent_name: Optional[str] = Field(None, description="agent name")
    agent_workspace_uuid: Optional[str] = Field(
        None, description="agent workspace uuid"
    )
    evaluation_test_case_workspace_uuid: Optional[str] = Field(
        None, description="evaluation test case workspace uuid"
    )
    agent_deleted: Optional[bool] = Field(None, description="whether agent is deleted")
    error_description: Optional[str] = Field(None, description="The error description")
    created_by_user_email: Optional[str] = Field(
        None, description="Email of user who created"
    )
    created_by_user_id: Optional[int] = Field(
        None, description="ID of user who created"
    )
    queued_at: Optional[datetime] = Field(
        None, description="Run queued time (RFC3339 timestamp)"
    )
    agent_deployment_name: Optional[str] = Field(
        None, description="The agent deployment name"
    )


class GetEvaluationRunOutput(BaseModel):
    """
    Response for getting an evaluation run.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    evaluation_run: EvaluationRun = Field(..., description="The evaluation run")


class ListEvaluationMetricsOutput(BaseModel):
    """
    Response for listing evaluation metrics.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    metrics: List[EvaluationMetric] = Field(
        default_factory=list, description="List of evaluation metrics"
    )
