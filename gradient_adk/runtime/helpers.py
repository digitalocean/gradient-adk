"""
Centralized instrumentation helpers for Gradient ADK.

Provides a registry pattern for optional framework instrumentors (PydanticAI, LangGraph, etc.)
that handles:
- Tracker creation and lifecycle management
- Environment variable disable checks
- Framework availability checks
- Automatic installation at import time
"""

from __future__ import annotations
import os
from typing import Optional, Callable, Dict, Any, Protocol
from gradient_adk.cli.config.yaml_agent_config_manager import YamlAgentConfigManager
from gradient_adk.runtime.digitalocean_tracker import DigitalOceanTracesTracker
from gradient_adk.digital_ocean_api import AsyncDigitalOceanGenAI
from gradient_adk.runtime.network_interceptor import setup_digitalocean_interception


class InstrumentorProtocol(Protocol):
    """Protocol for instrumentor classes."""

    def install(self, tracker: DigitalOceanTracesTracker) -> None:
        """Install the instrumentor with the given tracker."""
        ...

    def uninstall(self) -> None:
        """Uninstall the instrumentor and restore original behavior."""
        ...

    def is_installed(self) -> bool:
        """Check if the instrumentor is currently installed."""
        ...


class InstrumentorRegistry:
    """
    Registry for optional framework instrumentors.

    Provides centralized management for:
    - Tracker creation (single shared tracker for all instrumentors)
    - Framework availability checks
    - Environment variable disable flags
    - Installation lifecycle

    Usage:
        # In instrumentor module:
        registry.register(
            name="pydanticai",
            env_disable_var="GRADIENT_DISABLE_PYDANTICAI_INSTRUMENTOR",
            availability_check=lambda: _has_pydantic_ai(),
            instrumentor_factory=lambda: PydanticAIInstrumentor()
        )

        # In decorator or main module:
        registry.install_all()
        tracker = registry.get_tracker()
    """

    def __init__(self):
        self._tracker: Optional[DigitalOceanTracesTracker] = None
        self._instrumentors: Dict[str, InstrumentorProtocol] = {}
        self._registrations: Dict[str, Dict[str, Any]] = {}
        self._config_reader = YamlAgentConfigManager()
        self._tracker_initialized = False

    def _is_env_disabled(self, env_var: str) -> bool:
        """Check if instrumentation is disabled via environment variable."""
        val = os.environ.get(env_var, "").lower()
        return val in ("true", "1", "yes")

    def _ensure_tracker(self) -> Optional[DigitalOceanTracesTracker]:
        """
        Create the shared tracker if not already created.

        Returns None if:
        - No API token is available
        - Tracker already failed to initialize
        """
        if self._tracker is not None:
            return self._tracker

        if self._tracker_initialized:
            # Already tried and failed
            return None

        self._tracker_initialized = True

        try:
            api_token = os.environ.get("DIGITALOCEAN_API_TOKEN")
            if not api_token:
                return None

            ws = self._config_reader.get_agent_name()
            dep = self._config_reader.get_agent_environment()

            self._tracker = DigitalOceanTracesTracker(
                client=AsyncDigitalOceanGenAI(api_token=api_token),
                agent_workspace_name=ws,
                agent_deployment_name=dep,
            )
            setup_digitalocean_interception()
            return self._tracker

        except Exception:
            return None

    def register(
        self,
        name: str,
        env_disable_var: str,
        availability_check: Callable[[], bool],
        instrumentor_factory: Callable[[], InstrumentorProtocol],
    ) -> None:
        """
        Register an instrumentor for later installation.

        Args:
            name: Unique name for this instrumentor (e.g., "pydanticai", "langgraph")
            env_disable_var: Environment variable name to disable this instrumentor
            availability_check: Callable that returns True if the framework is available
            instrumentor_factory: Callable that creates the instrumentor instance
        """
        self._registrations[name] = {
            "env_disable_var": env_disable_var,
            "availability_check": availability_check,
            "instrumentor_factory": instrumentor_factory,
        }

    def install(self, name: str) -> Optional[DigitalOceanTracesTracker]:
        """
        Install a specific registered instrumentor.

        Returns the tracker if installation succeeded, None otherwise.
        """
        if name in self._instrumentors:
            # Already installed
            return self._tracker

        if name not in self._registrations:
            return None

        reg = self._registrations[name]

        # Check if disabled via env var
        if self._is_env_disabled(reg["env_disable_var"]):
            return None

        # Check if framework is available
        if not reg["availability_check"]():
            return None

        # Ensure we have a tracker
        tracker = self._ensure_tracker()
        if tracker is None:
            return None

        # Create and install instrumentor
        try:
            instrumentor = reg["instrumentor_factory"]()
            instrumentor.install(tracker)
            self._instrumentors[name] = instrumentor
            return tracker
        except Exception:
            return None

    def install_all(self) -> Optional[DigitalOceanTracesTracker]:
        """
        Install all registered instrumentors that are available.

        Returns the tracker if at least one instrumentor was installed.
        """
        for name in self._registrations:
            self.install(name)
        return self._tracker

    def uninstall(self, name: str) -> None:
        """Uninstall a specific instrumentor."""
        if name in self._instrumentors:
            try:
                self._instrumentors[name].uninstall()
            except Exception:
                pass
            del self._instrumentors[name]

    def uninstall_all(self) -> None:
        """Uninstall all instrumentors."""
        for name in list(self._instrumentors.keys()):
            self.uninstall(name)

    def get_tracker(self) -> Optional[DigitalOceanTracesTracker]:
        """Get the shared tracker instance."""
        return self._tracker

    def is_installed(self, name: str) -> bool:
        """Check if a specific instrumentor is installed."""
        return name in self._instrumentors

    def get_installed_names(self) -> list[str]:
        """Get list of installed instrumentor names."""
        return list(self._instrumentors.keys())


# Global registry instance
registry = InstrumentorRegistry()


def get_tracker() -> Optional[DigitalOceanTracesTracker]:
    """Get the shared tracker from the global registry."""
    return registry.get_tracker()


# ---- Auto-registration functions for known instrumentors ----
# These are called to register instrumentors without importing their heavy dependencies


def _register_langgraph() -> None:
    """Register LangGraph instrumentor if available."""

    def is_available() -> bool:
        try:
            from langgraph.graph import StateGraph

            return True
        except ImportError:
            return False

    def factory():
        from gradient_adk.runtime.langgraph.langgraph_instrumentor import (
            LangGraphInstrumentor,
        )

        return LangGraphInstrumentor()

    registry.register(
        name="langgraph",
        env_disable_var="GRADIENT_DISABLE_LANGGRAPH_INSTRUMENTOR",
        availability_check=is_available,
        instrumentor_factory=factory,
    )


def _register_pydanticai() -> None:
    """Register PydanticAI instrumentor if available."""

    def is_available() -> bool:
        try:
            from pydantic_ai import Agent

            return True
        except ImportError:
            return False

    def factory():
        from gradient_adk.runtime.pydanticai.pydanticai_instrumentor import (
            PydanticAIInstrumentor,
        )

        return PydanticAIInstrumentor()

    registry.register(
        name="pydanticai",
        env_disable_var="GRADIENT_DISABLE_PYDANTICAI_INSTRUMENTOR",
        availability_check=is_available,
        instrumentor_factory=factory,
    )


def register_all_instrumentors() -> None:
    """Register all known instrumentors with the registry."""
    _register_langgraph()
    _register_pydanticai()


def capture_all() -> Optional[DigitalOceanTracesTracker]:
    """
    Register and install all available instrumentors.

    This is the main entry point for the decorator module.
    Call this once at startup to automatically instrument all available frameworks.

    Returns:
        The shared tracker if at least one instrumentor was installed, None otherwise.
    """
    register_all_instrumentors()
    return registry.install_all()
