"""FastAPI server setup using A2A SDK."""

import os
from typing import Any

from fastapi import FastAPI

from a2a.server.apps.jsonrpc import A2AFastAPIApplication
from a2a.types import AgentCard, AgentCapabilities

from gradient_adk.a2a.infrastructure.composition import compose_request_handler
from gradient_adk.a2a.domain.constants import A2A_PROTOCOL_VERSION


def _validate_path(path: str, param_name: str) -> str:
    """
    Validate URL path format.

    Args:
        path: The path to validate
        param_name: Name of the parameter (for error messages)

    Returns:
        The validated path

    Raises:
        ValueError: If path format is invalid
    """
    if not path.startswith("/"):
        raise ValueError(f"{param_name} must start with '/'")
    if "//" in path:
        raise ValueError(f"{param_name} cannot contain double slashes")
    return path


def create_a2a_server(
    agent_func: Any,
    rpc_url: str = "/",
    agent_card_url: str = "/.well-known/agent-card.json",
    base_url: str | None = None,
    agent_name: str = "Gradient A2A Agent",
    input_key: str = "prompt",
    output_keys: list[str] | None = None,
) -> FastAPI:
    """
    Create A2A server.

    Args:
        agent_func: Gradient agent function
        rpc_url: RPC endpoint URL path
        agent_card_url: Agent card endpoint URL path
        base_url: Base URL for agent (used in AgentCard for discovery).
                  Reads from A2A_BASE_URL env var if not provided.
                  Falls back to "http://localhost:8000" for local development.
        agent_name: Agent name for AgentCard
        input_key: Key for agent input
        output_keys: Keys to try for output

    Returns:
        FastAPI application

    Raises:
        ValueError: If rpc_url or agent_card_url have invalid format
    """
    if base_url is None:
        base_url = os.getenv("A2A_BASE_URL", "http://localhost:8000")

    rpc_url = _validate_path(rpc_url, "rpc_url")
    agent_card_url = _validate_path(agent_card_url, "agent_card_url")

    agent_card = AgentCard(
        name=agent_name,
        version=A2A_PROTOCOL_VERSION,
        description="A2A-enabled Gradient agent",
        url=base_url,
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
            state_transition_history=True,
        ),
        supports_authenticated_extended_card=False,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[],
    )

    request_handler = compose_request_handler(
        agent_func,
        input_key=input_key,
        output_keys=output_keys,
    )

    app_builder = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    return app_builder.build(
        rpc_url=rpc_url,
        agent_card_url=agent_card_url
    )
