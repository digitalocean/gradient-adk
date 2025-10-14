"""Agent configuration reader interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class ConfigReader(ABC):
    """Abstract interface for reading agent configuration."""

    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """Load and return the agent configuration."""
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """Get the agent name from configuration."""
        pass

    @abstractmethod
    def get_agent_environment(self) -> str:
        """Get the agent environment from configuration."""
        pass

    @abstractmethod
    def get_entrypoint_file(self) -> str:
        """Get the entrypoint file from configuration."""
        pass
