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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    created_at: datetime = Field(..., description="RFC3339 timestamp")
    input: Dict[str, Any]
    name: str
    output: Dict[str, Any]
    type: TraceSpanType = Field(default=TraceSpanType.TRACE_SPAN_TYPE_UNKNOWN)


class Trace(BaseModel):
    """
    Represents a complete trace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    created_at: datetime = Field(..., description="RFC3339 timestamp")
    input: Dict[str, Any]
    name: str
    output: Dict[str, Any]
    spans: List[Span] = Field(default_factory=list)


class CreateTracesInput(BaseModel):
    """
    Input for creating traces.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_deployment_name: str
    session_id: Optional[str] = None
    traces: List[Trace]
    agent_workspace_name: str


class Project(BaseModel):
    """
    Represents a DigitalOcean project.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    project: Project


class TracingServiceJWTOutput(BaseModel):
    """
    Response for getting tracing token.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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
    logging_config: Optional[AgentLoggingConfig] = Field(
        None, description="Agent Logging Config Details"
    )


class GetAgentWorkspaceDeploymentOutput(BaseModel):
    """
    Response for getting an agent workspace deployment.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_workspace_deployment: AgentWorkspaceDeployment = Field(
        ..., description="The agent workspace deployment"
    )


class AgentWorkspace(BaseModel):
    """
    Represents an Agent Workspace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_workspaces: list[AgentWorkspace] = Field(
        default_factory=list, description="List of agent workspaces"
    )


class PresignedUrlFile(BaseModel):
    """
    A single file's metadata in the request.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    file_name: str = Field(..., description="Local filename")
    file_size: int = Field(..., description="The size of the file in bytes")


class CreateAgentDeploymentFileUploadPresignedURLInput(BaseModel):
    """
    Input for creating agent deployment file upload presigned URL.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    file: PresignedUrlFile = Field(
        ..., description="The file to generate presigned URL for"
    )


class FilePresignedUrlResponse(BaseModel):
    """
    Detailed info about each presigned URL returned to the client.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    object_key: str = Field(
        ..., description="The unique object key to store the file as"
    )
    original_file_name: str = Field(..., description="The original file name")
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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_code_file_path: str = Field(..., description="The agent code file path")
    stored_object_key: str = Field(
        ..., description="The object key the file was stored as"
    )
    size_in_bytes: int = Field(..., description="The size of the file in bytes")


class CreateAgentWorkspaceDeploymentInput(BaseModel):
    """
    Input for creating an agent workspace deployment.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_workspace_name: str = Field(..., description="The name of agent workspace")
    agent_deployment_name: str = Field(..., description="The deployment name")
    agent_deployment_code_artifact: AgentDeploymentCodeArtifact = Field(
        ..., description="The agent deployment code artifact"
    )


class CreateAgentWorkspaceDeploymentOutput(BaseModel):
    """
    Response for creating an agent workspace deployment.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_workspace_deployment: AgentWorkspaceDeployment = Field(
        ..., description="The agent workspace deployment"
    )


class CreateAgentDeploymentReleaseInput(BaseModel):
    """
    Input for creating an agent deployment release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_workspace_name: str = Field(..., description="The name of agent workspace")
    agent_deployment_name: str = Field(..., description="The deployment name")
    agent_deployment_code_artifact: AgentDeploymentCodeArtifact = Field(
        ..., description="The agent deployment code artifact"
    )


class CreateAgentDeploymentReleaseOutput(BaseModel):
    """
    Response for creating an agent deployment release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_deployment_release: AgentDeploymentRelease = Field(
        ..., description="The agent deployment release"
    )


class GetAgentDeploymentReleaseInput(BaseModel):
    """
    Input for getting an agent deployment release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    uuid: str = Field(..., description="Unique agent deployment release id")


class GetAgentDeploymentReleaseOutput(BaseModel):
    """
    Response for getting an agent deployment release.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_deployment_release: AgentDeploymentRelease = Field(
        ..., description="The agent deployment release"
    )


class CreateAgentWorkspaceInput(BaseModel):
    """
    Input for creating an agent workspace.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent_workspace: AgentWorkspace = Field(..., description="The agent workspace")


class GetAgentWorkspaceDeploymentRuntimeLogsOutput(BaseModel):
    """
    Response for getting agent workspace deployment runtime logs.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    live_url: str = Field(..., description="URL for live logs")
