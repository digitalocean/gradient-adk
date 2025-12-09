from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, Optional
import typer
import yaml

from gradient_adk.cli.config.agent_config_manager import AgentConfigManager


class YamlAgentConfigManager(AgentConfigManager):
    """YAML-based implementation of agent configuration manager."""

    def __init__(self):
        self.config_dir = Path.cwd() / ".gradient"
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "agent.yml"

    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load and return the agent configuration."""
        if not self.config_file.exists():
            return None

        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
                return config if config else None
        except Exception as e:
            typer.echo(f"Error reading agent configuration: {e}", err=True)
            raise typer.Exit(1)

    def get_agent_name(self) -> Optional[str]:
        config = self.load_config()
        return config.get("agent_name") if config else None

    def get_agent_environment(self) -> Optional[str]:
        config = self.load_config()
        return config.get("agent_environment") if config else None

    def get_entrypoint_file(self) -> Optional[str]:
        config = self.load_config()
        return config.get("entrypoint_file") if config else None

    def configure(
        self,
        agent_name: Optional[str] = None,
        agent_environment: Optional[str] = None,
        entrypoint_file: Optional[str] = None,
        interactive: bool = True,
    ) -> None:
        """Configure agent settings and save to YAML file."""
        if interactive:
            if agent_name is None:
                agent_name = self._prompt_with_validation(
                    "Agent workspace name",
                    self._validate_name,
                    "Agent workspace name can only contain alphanumeric characters, hyphens, and underscores",
                )
            if agent_environment is None:
                agent_environment = self._prompt_with_validation(
                    "Agent deployment name",
                    self._validate_name,
                    "Deployment name can only contain alphanumeric characters, hyphens, and underscores",
                    default="main",
                )
            if entrypoint_file is None:
                entrypoint_file = typer.prompt(
                    "Entrypoint file (e.g., main.py, agent.py)", default="main.py"
                )
        else:
            if not all([agent_name, agent_environment, entrypoint_file]):
                typer.echo(
                    "Error: --agent-workspace-name, --agent-deployment-name, and --entrypoint-file are required in non-interactive mode.",
                    err=True,
                )
                raise typer.Exit(2)

            # Validate names in non-interactive mode
            if not self._validate_name(agent_name):
                typer.echo(
                    f"Error: Agent workspace name '{agent_name}' is invalid. "
                    "It can only contain alphanumeric characters, hyphens, and underscores.",
                    err=True,
                )
                raise typer.Exit(1)

            if not self._validate_name(agent_environment):
                typer.echo(
                    f"Error: Agent deployment name '{agent_environment}' is invalid. "
                    "It can only contain alphanumeric characters, hyphens, and underscores.",
                    err=True,
                )
                raise typer.Exit(1)

        self._validate_entrypoint_file(entrypoint_file)
        self._save_config(agent_name, agent_environment, entrypoint_file)

    def _validate_name(self, name: str) -> bool:
        """Validate that a name only contains alphanumeric characters, hyphens, and underscores."""
        if not name:
            return False
        return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))

    def _prompt_with_validation(
        self,
        prompt_text: str,
        validator: callable,
        error_message: str,
        default: Optional[str] = None,
    ) -> str:
        """Prompt user for input with validation, re-prompting on invalid input."""
        while True:
            if default:
                value = typer.prompt(prompt_text, default=default)
            else:
                value = typer.prompt(prompt_text)

            if validator(value):
                return value
            else:
                typer.echo(f"❌ {error_message}", err=True)
                typer.echo(f"   You entered: '{value}'", err=True)
                typer.echo("   Please try again.\n", err=True)

    def _validate_entrypoint_file(self, entrypoint_file: str) -> None:
        """Validate that the entrypoint file exists and contains @entrypoint decorator."""
        entrypoint_path = Path.cwd() / entrypoint_file
        if not entrypoint_path.exists():
            typer.echo(
                f"Error: Entrypoint file '{entrypoint_file}' does not exist.", err=True
            )
            typer.echo(
                "Please create this file with your @entrypoint decorated function before configuring the agent."
            )
            raise typer.Exit(1)

        try:
            content = entrypoint_path.read_text()
            if not re.search(r"^\s*@entrypoint\s*$", content, re.MULTILINE):
                typer.echo(
                    f"Error: No @entrypoint decorator found in '{entrypoint_file}'.",
                    err=True,
                )
                self._show_entrypoint_example()
                raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Error reading '{entrypoint_file}': {e}", err=True)
            raise typer.Exit(1)

    def _show_entrypoint_example(self) -> None:
        """Show example of correct @entrypoint usage."""
        typer.echo("\nExample of correct @entrypoint usage:")
        typer.echo("  from gradient_adk import entrypoint")
        typer.echo("  @entrypoint")
        typer.echo("  async def my_agent(data, context):")
        typer.echo("      return {'result': data}\n")
        typer.echo(
            "Note: Entrypoint functions must accept exactly 2 parameters (data, context)."
        )

    def _save_config(
        self, agent_name: str, agent_environment: str, entrypoint_file: str
    ) -> None:
        """Save configuration to YAML file."""
        config = {
            "agent_name": agent_name,
            "agent_environment": agent_environment,
            "entrypoint_file": entrypoint_file,
        }

        try:
            with open(self.config_file, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            typer.echo(f"✅ Configuration saved to {self.config_file}")
            typer.echo(f"  Agent workspace name: {agent_name}")
            typer.echo(f"  Agent deployment name: {agent_environment}")
            typer.echo(f"  Entrypoint: {entrypoint_file}")
        except Exception as e:
            typer.echo(f"Error writing configuration file: {e}", err=True)
            raise typer.Exit(1)
