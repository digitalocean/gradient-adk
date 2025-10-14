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
