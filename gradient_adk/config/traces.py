"""
Configuration utilities for DigitalOcean traces integration.

This module provides helper functions to configure gradient-agent with
DigitalOcean traces support.
"""

import os
from typing import Optional

from gradient_adk.runtime import configure_digitalocean_traces
from gradient_adk.logging import get_logger

logger = get_logger(__name__)


def configure_traces_from_env(
    api_token_env: str = "DIGITALOCEAN_API_TOKEN",
    auto_submit: bool = True,
) -> bool:
    """Configure DigitalOcean traces from API token and agent config.

    Args:
        api_token_env: Environment variable name for API token
        auto_submit: Whether to automatically submit traces

    Returns:
        True if configuration was successful, False otherwise
    """
    api_token = os.getenv(api_token_env)

    if not api_token:
        logger.warning(
            f"{api_token_env} not set - traces disabled", env_var=api_token_env
        )
        return False

    try:
        manager = configure_digitalocean_traces(
            api_token=api_token,
            enable_auto_submit=auto_submit,
        )

        tracker = manager.get_tracker()
        logger.info(
            "✓ DigitalOcean traces configured successfully",
            workspace=tracker.agent_workspace_name,
            deployment=tracker.agent_deployment_name,
            auto_submit="enabled" if auto_submit else "disabled",
        )
        return True

    except Exception as e:
        logger.error(
            "Error configuring DigitalOcean traces", error=str(e), exc_info=True
        )
        return False


def configure_traces_with_token(
    api_token: str,
    agent_workspace_name: Optional[str] = None,
    agent_deployment_name: Optional[str] = None,
    auto_submit: bool = True,
) -> bool:
    """Configure DigitalOcean traces with explicit parameters.

    Args:
        api_token: DigitalOcean API token
        agent_workspace_name: Name of the agent workspace (uses agent config if None)
        agent_deployment_name: Name of the agent deployment (uses agent config if None)
        auto_submit: Whether to automatically submit traces

    Returns:
        True if configuration was successful, False otherwise
    """
    if not api_token:
        logger.error("api_token is required")
        return False

    try:
        manager = configure_digitalocean_traces(
            api_token=api_token,
            agent_workspace_name=agent_workspace_name,
            agent_deployment_name=agent_deployment_name,
            enable_auto_submit=auto_submit,
        )

        tracker = manager.get_tracker()
        logger.info(
            "✓ DigitalOcean traces configured successfully",
            workspace=tracker.agent_workspace_name,
            deployment=tracker.agent_deployment_name,
            auto_submit="enabled" if auto_submit else "disabled",
        )
        return True

    except Exception as e:
        logger.error(
            "Error configuring DigitalOcean traces", error=str(e), exc_info=True
        )
        return False
