from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List
from pathlib import Path


class AuthService(ABC):
    """Abstract interface for authentication operations."""

    @abstractmethod
    def init(
        self,
        context: Optional[str] = None,
        token: Optional[str] = None,
        api_url: Optional[str] = None,
        interactive: bool = True,
        output: Optional[str] = None,
        verbose: bool = False,
        trace: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """Initialize authentication with the given parameters."""
        pass

    @abstractmethod
    def list(
        self,
        output: Optional[str] = None,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """List authentication contexts."""
        pass

    @abstractmethod
    def remove(
        self,
        context: str,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """Remove an authentication context."""
        pass

    @abstractmethod
    def switch(
        self,
        context: str,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """Switch to an authentication context."""
        pass


class ToolResolver(ABC):
    """Abstract interface for resolving external tool paths."""

    @abstractmethod
    def resolve_tool_path(self, tool_name: str) -> Path:
        """Resolve the path to an external tool binary."""
        pass

    @abstractmethod
    def get_config_path(self) -> str:
        """Get the configuration file path."""
        pass


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
        """Configure agent settings and save to YAML file."""
        pass


class LaunchService(ABC):
    """Abstract interface for agent launch operations."""

    @abstractmethod
    def launch_locally(self) -> None:
        """Launch the agent locally using Docker."""
        pass
