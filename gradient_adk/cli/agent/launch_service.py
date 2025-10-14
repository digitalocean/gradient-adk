"""Agent launch service interface."""

from __future__ import annotations
from abc import ABC, abstractmethod


class LaunchService(ABC):
    """Abstract interface for agent launch operations."""

    @abstractmethod
    def launch_locally(
        self, dev_mode: bool = False, host: str = "0.0.0.0", port: int = 8080
    ) -> None:
        """Launch the agent locally."""
        pass
