from __future__ import annotations
from typing import Optional
import typer

from .interfaces import AgentConfigService, LaunchService
from .services import (
    YamlAgentConfigService,
    DirectLaunchService,
)

_agent_config_service = YamlAgentConfigService()
_launch_service = DirectLaunchService()

app = typer.Typer(no_args_is_help=True, add_completion=False, help="gradient CLI")

agent_app = typer.Typer(
    no_args_is_help=True,
    help="Agent configuration and management",
)
app.add_typer(agent_app, name="agent")


def get_agent_config_service() -> AgentConfigService:
    return _agent_config_service


def get_launch_service() -> LaunchService:
    return _launch_service


@agent_app.command("init")
def agent_init(
    agent_name: Optional[str] = typer.Option(
        None, "--agent-name", help="Name of the agent"
    ),
    agent_environment: Optional[str] = typer.Option(
        None, "--agent-environment", help="Agent environment name"
    ),
    entrypoint_file: Optional[str] = typer.Option(
        None,
        "--entrypoint-file",
        help="Python file containing @entrypoint decorated function",
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive prompt mode"
    ),
):
    agent_config_service = get_agent_config_service()
    agent_config_service.configure(
        agent_name=agent_name,
        agent_environment=agent_environment,
        entrypoint_file=entrypoint_file,
        interactive=interactive,
    )


@agent_app.command("run")
def agent_run():
    launch_service = get_launch_service()
    launch_service.launch_locally()


def run():
    app()
