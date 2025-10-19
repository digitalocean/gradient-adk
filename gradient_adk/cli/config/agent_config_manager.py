from __future__ import annotations
from typing import Dict, Any, Optional


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

    def configure(
        self,
        agent_name: Optional[str] = None,
        agent_environment: Optional[str] = None,
        entrypoint_file: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        raise NotImplementedError
