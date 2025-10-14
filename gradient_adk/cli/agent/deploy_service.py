"""Agent deployment service interface."""

from __future__ import annotations
from abc import ABC, abstractmethod


class DeployService(ABC):
    """Abstract interface for agent deployment operations."""

    @abstractmethod
    def deploy_agent(self) -> None:
        """Deploy the agent to the configured environment."""
        pass
