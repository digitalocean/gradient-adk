from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import typer
import yaml

from .interfaces import AgentConfigService, LaunchService


class YamlAgentConfigService(AgentConfigService):
    """YAML-based implementation of agent configuration service."""

    def __init__(self):
        self.config_dir = Path.cwd() / ".gradient"
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "agent.yml"

    def configure(
        self,
        agent_name: Optional[str] = None,
        agent_environment: Optional[str] = None,
        entrypoint_file: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        if interactive:
            if agent_name is None:
                agent_name = typer.prompt("Agent name")
            if agent_environment is None:
                agent_environment = typer.prompt("Agent environment name")
            if entrypoint_file is None:
                entrypoint_file = typer.prompt(
                    "Entrypoint file (e.g., main.py, agent.py)", default="main.py"
                )
        else:
            if (
                agent_name is None
                or agent_environment is None
                or entrypoint_file is None
            ):
                typer.echo(
                    "Error: --agent-name, --agent-environment, and --entrypoint-file are required in non-interactive mode.",
                    err=True,
                )
                raise typer.Exit(2)

        entrypoint_path = Path.cwd() / entrypoint_file
        if not entrypoint_path.exists():
            typer.echo(
                f"Error: Entrypoint file '{entrypoint_file}' does not exist.",
                err=True,
            )
            typer.echo(
                "Please create this file with your @entrypoint decorated function before configuring the agent."
            )
            raise typer.Exit(1)

        try:
            with open(entrypoint_path, "r") as f:
                file_content = f.read()
            import re

            decorator_pattern = r"^\s*@entrypoint\s*$"
            if not re.search(decorator_pattern, file_content, re.MULTILINE):
                typer.echo(
                    f"Error: No @entrypoint decorator found in '{entrypoint_file}'.",
                    err=True,
                )
                typer.echo(
                    "Please add the @entrypoint decorator to a function in this file."
                )
                typer.echo("Example:")
                typer.echo("         from gradient_agents import entrypoint")
                typer.echo("         @entrypoint")
                typer.echo("         def my_agent(data, context):")
                typer.echo("             return {'result': data}")
                typer.echo(
                    "Note: Entrypoint functions must accept exactly 2 parameters (data, context)"
                )
                raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception as e:
            typer.echo(
                f"Error: Could not read entrypoint file '{entrypoint_file}': {e}",
                err=True,
            )
            raise typer.Exit(1)

        config = {
            "agent_name": agent_name,
            "agent_environment": agent_environment,
            "entrypoint_file": entrypoint_file,
        }

        try:
            with open(self.config_file, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            typer.echo(f"Configuration saved to {self.config_file}")
            typer.echo(f"Agent name: {agent_name}")
            typer.echo(f"Agent environment: {agent_environment}")
            typer.echo(f"Entrypoint file: {entrypoint_file}")
        except Exception as e:
            typer.echo(f"Error writing configuration file: {e}", err=True)
            raise typer.Exit(1)


class DirectLaunchService(LaunchService):
    """Direct FastAPI implementation of launch service."""

    def __init__(self):
        self.config_dir = Path.cwd() / ".gradient"
        self.config_file = self.config_dir / "agent.yml"

    def launch_locally(self) -> None:
        if not self.config_file.exists():
            typer.echo("Error: No agent configuration found.", err=True)
            typer.echo(
                "Please run 'gradient agent init' first to set up your agent.", err=True
            )
            raise typer.Exit(1)

        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
            entrypoint_file = config.get("entrypoint_file")
            agent_name = config.get("agent_name", "gradient-agent")
        except Exception as e:
            typer.echo(f"Error reading agent configuration: {e}", err=True)
            raise typer.Exit(1)

        if not entrypoint_file:
            typer.echo(
                "Error: No entrypoint file specified in configuration.", err=True
            )
            raise typer.Exit(1)

        entrypoint_path = Path.cwd() / entrypoint_file
        if not entrypoint_path.exists():
            typer.echo(
                f"Error: Entrypoint file '{entrypoint_file}' does not exist.",
                err=True,
            )
            raise typer.Exit(1)

        try:
            import sys

            current_dir = str(Path.cwd())
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            module_name = (
                entrypoint_file.replace(".py", "").replace("/", ".").replace("\\", ".")
            )
            import importlib

            typer.echo(f"Importing module: {module_name}")
            importlib.import_module(module_name)

            typer.echo(f"Starting {agent_name} server...")
            typer.echo("Server will be accessible at http://localhost:8080")
            typer.echo("Press Ctrl+C to stop the server")

            try:
                from gradient_agents import run_server

                run_server(host="0.0.0.0", port=8080)
            except ImportError:
                typer.echo(
                    "Error: gradient_agents package not found.",
                    err=True,
                )
                typer.echo(
                    "Please install it with: pip install gradient-agent",
                    err=True,
                )
                raise typer.Exit(1)

        except ImportError as e:
            error_msg = str(e)
            typer.echo(
                f"Error: Error importing entrypoint module '{module_name}': {error_msg}",
                err=True,
            )
            typer.echo(
                "Please install the gradient-agent package and ensure imports are correct:",
                err=True,
            )
            typer.echo("  pip install gradient-agent", err=True)
            typer.echo("  from gradient_agents import entrypoint", err=True)
            raise typer.Exit(1)
