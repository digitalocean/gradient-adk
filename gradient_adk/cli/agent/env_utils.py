"""Environment variable utilities for CLI commands."""

import os
import typer
from typing import Optional


class EnvironmentError(Exception):
    """Raised when required environment variables are missing or invalid."""

    pass


def get_do_api_token(
    token_override: Optional[str] = None,
    env_var_name: str = "DIGITALOCEAN_API_TOKEN",
    required: bool = True,
) -> Optional[str]:
    """
    Get DigitalOcean API token from environment or override.

    Args:
        token_override: Optional token provided via CLI argument
        env_var_name: Name of environment variable to check (default: DO_API_TOKEN)
        required: Whether the token is required (raises error if missing)

    Returns:
        The API token if found

    Raises:
        EnvironmentError: If required=True and no token is found
    """
    # Priority: CLI argument > Environment variable
    token = token_override or os.getenv(env_var_name)

    if required and not token:
        raise EnvironmentError(
            f"DigitalOcean API token is required. Please either:\n"
            f"  1. Set the {env_var_name} environment variable, or\n"
            f"  2. Pass the token via --api-token argument"
        )

    if token and not token.strip():
        raise EnvironmentError(f"DigitalOcean API token cannot be empty")

    return token.strip() if token else None


def get_agent_uuid():
    """
    Get the agent UUID from the environment variable.
    """
    uuid = os.getenv("AGENT_UUID")
    if not uuid:
        raise EnvironmentError("AGENT_UUID environment variable is not set.")
    return uuid


def validate_api_token(token: str) -> str:
    """
    Validate that an API token looks reasonable.

    Args:
        token: The token to validate

    Returns:
        The token if valid

    Raises:
        typer.BadParameter: If token format is invalid
    """
    if not token or not token.strip():
        raise typer.BadParameter("API token cannot be empty")

    token = token.strip()

    # Basic validation - DigitalOcean tokens are typically 64 characters
    if len(token) < 10:
        raise typer.BadParameter("API token appears to be too short")

    return token


def prompt_for_missing_token(env_var_name: str = "DO_API_TOKEN") -> str:
    """
    Interactively prompt for API token if missing.

    Args:
        env_var_name: Name of the environment variable

    Returns:
        The provided token
    """
    typer.echo(
        f"No {env_var_name} found in environment.\n"
        f"You can set it permanently with: export {env_var_name}=your_token_here"
    )

    token = typer.prompt(
        "Enter your DigitalOcean API token",
        hide_input=True,  # Hide the token input for security
        show_default=False,
    )

    return validate_api_token(token)


def get_optional_env_var(
    env_var_name: str, default: Optional[str] = None, description: Optional[str] = None
) -> Optional[str]:
    """
    Get an optional environment variable with helpful error context.

    Args:
        env_var_name: Name of environment variable
        default: Default value if not set
        description: Human-readable description for error messages

    Returns:
        The environment variable value or default
    """
    value = os.getenv(env_var_name, default)

    if value is None and description:
        typer.echo(
            f"Tip: Set {env_var_name} environment variable for {description}", err=True
        )

    return value
