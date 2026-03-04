from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


class DeploymentTarget(str, Enum):
    """Supported deployment targets for ADK agents."""
    GENAI_API = "genai_api"
    DOCC = "docc"


class DOCCConfig(BaseModel):
    """DOCC-specific deployment configuration."""
    context: str = Field(..., description="DOCC cluster context (e.g. 'puff', 'wolf')")
    namespace: str = Field(default="gen-ai", description="DOCC namespace")
    service_id: Optional[str] = Field(None, description="Service catalog UUID for the DOCC manifest")
    scale: int = Field(default=2, description="Number of replicas")
    image_registry: str = Field(
        default="docker.internal.digitalocean.com/gen-ai/adk-agents",
        description="Docker image registry base path",
    )
    maintainer: str = Field(
        default="gen-ai-engineering@digitalocean.com",
        description="Maintainer email for the DOCC manifest",
    )
    github_acl: Optional[Dict[str, list]] = Field(
        None, description="GitHub ACL for DOCC access control"
    )
    mtls_acl: Optional[Dict[str, list]] = Field(
        None, description="mTLS ACL for DOCC access control"
    )
    resources_request_memory: str = Field(default="256", description="Memory request in MB")
    resources_request_cpu: str = Field(default="0.5", description="CPU request")
    resources_limit_memory: str = Field(default="1024", description="Memory limit in MB")
    resources_limit_cpu: str = Field(default="2", description="CPU limit")


class AgentConfigManager:
    """Interface for reading and writing agent configuration."""

    def load_config(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def get_agent_name(self) -> Optional[str]:
        raise NotImplementedError

    def get_agent_environment(self) -> Optional[str]:
        raise NotImplementedError

    def get_entrypoint_file(self) -> Optional[str]:
        raise NotImplementedError

    def get_description(self) -> Optional[str]:
        raise NotImplementedError

    def get_deployment_target(self) -> DeploymentTarget:
        raise NotImplementedError

    def get_docc_config(self) -> Optional[DOCCConfig]:
        raise NotImplementedError

    def configure(
        self,
        agent_name: Optional[str] = None,
        agent_environment: Optional[str] = None,
        entrypoint_file: Optional[str] = None,
        description: Optional[str] = None,
        deployment_target: Optional[DeploymentTarget] = None,
        docc_config: Optional[DOCCConfig] = None,
        interactive: bool = True,
    ) -> None:
        raise NotImplementedError
