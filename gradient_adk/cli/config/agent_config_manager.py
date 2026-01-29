from __future__ import annotations
from typing import Dict, Any, Optional, List


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

    def configure(
        self,
        agent_name: Optional[str] = None,
        agent_environment: Optional[str] = None,
        entrypoint_file: Optional[str] = None,
        description: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        raise NotImplementedError

    # Multi-deployment support methods

    def get_deployment_names(self) -> List[str]:
        """Get list of deployment names from the deployments section.

        Returns empty list if no deployments section exists (old format).
        """
        raise NotImplementedError

    def get_config_for_deployment(self, deployment_name: str) -> Optional[Dict[str, Any]]:
        """Get merged config for a specific deployment.

        The deployment name becomes the agent_environment.
        Returns None if deployment not found.
        """
        raise NotImplementedError

    def add_deployment(
        self,
        deployment_name: str,
        entrypoint_file: str,
        description: Optional[str] = None,
    ) -> None:
        """Add a deployment to the deployments section."""
        raise NotImplementedError
