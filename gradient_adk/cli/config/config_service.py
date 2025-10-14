"""Agent configuration service interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class AgentConfigService(ABC):
    """Abstract interface for agent configuration operations."""

    @abstractmethod
    def configure(
        self,
        agent_name: Optional[str] = None,
        agent_environment: Optional[str] = None,
        entrypoint_file: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        """Configure agent settings and save to configuration file."""
        pass
