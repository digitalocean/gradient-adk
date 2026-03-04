from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, Optional
import typer
import yaml

from gradient_adk.cli.config.agent_config_manager import (
    AgentConfigManager,
    DeploymentTarget,
    DOCCConfig,
)


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

    def get_description(self) -> Optional[str]:
        config = self.load_config()
        return config.get("description") if config else None

    def get_deployment_target(self) -> DeploymentTarget:
        config = self.load_config()
        if not config:
            return DeploymentTarget.GENAI_API
        raw = config.get("deployment_target", "genai_api")
        try:
            return DeploymentTarget(raw)
        except ValueError:
            return DeploymentTarget.GENAI_API

    def get_docc_config(self) -> Optional[DOCCConfig]:
        config = self.load_config()
        if not config or "docc" not in config:
            return None
        try:
            return DOCCConfig(**config["docc"])
        except Exception:
            return None

    def configure(
        self,
        agent_name: Optional[str] = None,
        agent_environment: Optional[str] = None,
        entrypoint_file: Optional[str] = None,
        description: Optional[str] = None,
        deployment_target: Optional[DeploymentTarget] = None,
        docc_config: Optional[DOCCConfig] = None,
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
            if deployment_target is None:
                target_choice = typer.prompt(
                    "Deployment target (genai_api or docc)",
                    default="genai_api",
                )
                try:
                    deployment_target = DeploymentTarget(target_choice)
                except ValueError:
                    typer.echo(
                        f"Invalid deployment target '{target_choice}'. Using 'genai_api'.",
                        err=True,
                    )
                    deployment_target = DeploymentTarget.GENAI_API

            if deployment_target == DeploymentTarget.DOCC and docc_config is None:
                docc_config = self._prompt_docc_config()
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

            if deployment_target is None:
                deployment_target = DeploymentTarget.GENAI_API

        # Validate description length if provided
        if description is not None and len(description) > 1000:
            typer.echo(
                f"Error: Description exceeds maximum length of 1000 characters (got {len(description)}).",
                err=True,
            )
            raise typer.Exit(1)

        self._validate_entrypoint_file(entrypoint_file)
        self._save_config(
            agent_name,
            agent_environment,
            entrypoint_file,
            description,
            deployment_target,
            docc_config,
        )

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

    def _prompt_docc_config(self) -> DOCCConfig:
        """Interactively prompt for DOCC-specific configuration."""
        context = typer.prompt("DOCC cluster context (e.g. puff, wolf, flux)")
        namespace = typer.prompt("DOCC namespace", default="gen-ai")
        service_id = typer.prompt(
            "Service catalog UUID (or press Enter to auto-generate)",
            default="",
        )
        scale = typer.prompt("Number of replicas", default=2, type=int)
        image_registry = typer.prompt(
            "Docker image registry base path",
            default="docker.internal.digitalocean.com/gen-ai/adk-agents",
        )
        maintainer = typer.prompt(
            "Maintainer email",
            default="gen-ai-engineering@digitalocean.com",
        )

        return DOCCConfig(
            context=context,
            namespace=namespace,
            service_id=service_id if service_id else None,
            scale=scale,
            image_registry=image_registry,
            maintainer=maintainer,
        )

    def _save_config(
        self,
        agent_name: str,
        agent_environment: str,
        entrypoint_file: str,
        description: Optional[str] = None,
        deployment_target: Optional[DeploymentTarget] = None,
        docc_config: Optional[DOCCConfig] = None,
    ) -> None:
        """Save configuration to YAML file."""
        config = {
            "agent_name": agent_name,
            "agent_environment": agent_environment,
            "entrypoint_file": entrypoint_file,
        }

        if description is not None:
            config["description"] = description

        if deployment_target is not None:
            config["deployment_target"] = deployment_target.value

        if docc_config is not None:
            config["docc"] = docc_config.model_dump(exclude_none=True)

        try:
            with open(self.config_file, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            typer.echo(f"✅ Configuration saved to {self.config_file}")
            typer.echo(f"  Agent workspace name: {agent_name}")
            typer.echo(f"  Agent deployment name: {agent_environment}")
            typer.echo(f"  Entrypoint: {entrypoint_file}")
            if description:
                typer.echo(f"  Description: {description[:50]}{'...' if len(description) > 50 else ''}")
            if deployment_target:
                typer.echo(f"  Deployment target: {deployment_target.value}")
            if docc_config:
                typer.echo(f"  DOCC context: {docc_config.context}")
                typer.echo(f"  DOCC namespace: {docc_config.namespace}")
        except Exception as e:
            typer.echo(f"Error writing configuration file: {e}", err=True)
            raise typer.Exit(1)
