from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import typer
import yaml

from .config_reader import ConfigReader


class YamlConfigReader(ConfigReader):
    """YAML-based implementation of configuration reader."""

    def __init__(self):
        self.config_dir = Path.cwd() / ".gradient"
        self.config_file = self.config_dir / "agent.yml"

    def load_config(self) -> Dict[str, Any]:
        """Load and return the agent configuration."""
        if not self.config_file.exists():
            typer.echo("Error: No agent configuration found.", err=True)
            typer.echo(
                "Please run 'gradient agent init' first to set up your agent.", err=True
            )
            raise typer.Exit(1)

        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
                if not config:
                    raise ValueError("Configuration file is empty")
                return config
        except Exception as e:
            typer.echo(f"Error reading agent configuration: {e}", err=True)
            raise typer.Exit(1)

    def get_agent_name(self) -> str:
        """Get the agent name from configuration."""
        config = self.load_config()
        agent_name = config.get("agent_name")
        if not agent_name:
            typer.echo("Error: No agent name found in configuration.", err=True)
            raise typer.Exit(1)
        return agent_name

    def get_agent_environment(self) -> str:
        """Get the agent environment from configuration."""
        config = self.load_config()
        agent_environment = config.get("agent_environment")
        if not agent_environment:
            typer.echo("Error: No agent environment found in configuration.", err=True)
            raise typer.Exit(1)
        return agent_environment

    def get_entrypoint_file(self) -> str:
        """Get the entrypoint file from configuration."""
        config = self.load_config()
        entrypoint_file = config.get("entrypoint_file")
        if not entrypoint_file:
            typer.echo("Error: No entrypoint file found in configuration.", err=True)
            raise typer.Exit(1)
        return entrypoint_file
