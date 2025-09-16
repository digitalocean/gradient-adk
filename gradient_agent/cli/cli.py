from __future__ import annotations
from typing import Optional
import typer

from .interfaces import AuthService, AgentConfigService, LaunchService
from .services import (
    DoctlAuthService,
    DoctlResolver,
    YamlAgentConfigService,
    DirectLaunchService,
)

_resolver = DoctlResolver()
_auth_service = DoctlAuthService(_resolver)
_agent_config_service = YamlAgentConfigService()
_launch_service = DirectLaunchService()

app = typer.Typer(no_args_is_help=True, add_completion=False, help="gradient CLI")

auth_app = typer.Typer(
    no_args_is_help=True,
    help="Authenticate and manage credentials",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
app.add_typer(auth_app, name="auth")

agent_app = typer.Typer(
    no_args_is_help=True,
    help="Agent configuration and management",
)
app.add_typer(agent_app, name="agent")


def get_auth_service() -> AuthService:
    return _auth_service


def get_agent_config_service() -> AgentConfigService:
    return _agent_config_service


def get_launch_service() -> LaunchService:
    return _launch_service


@auth_app.command("init")
def auth_init(
    ctx: typer.Context,
    context: Optional[str] = typer.Option(None, "--context"),
    token: Optional[str] = typer.Option(
        None, "--token", help="If provided, runs non-interactively."
    ),
    api_url: Optional[str] = typer.Option(None, "--api-url"),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive prompt mode"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    trace: bool = typer.Option(False, "--trace"),
):
    auth_service = get_auth_service()
    auth_service.init(
        context=context,
        token=token,
        api_url=api_url,
        interactive=interactive,
        output=output,
        verbose=verbose,
        trace=trace,
        extra_args=ctx.args,
    )


@auth_app.command("list")
def auth_list(
    ctx: typer.Context,
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    auth_service = get_auth_service()
    auth_service.list(
        output=output,
        verbose=verbose,
        extra_args=ctx.args,
    )


@auth_app.command("remove")
def auth_remove(
    ctx: typer.Context,
    context: str = typer.Option(..., "--context"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    auth_service = get_auth_service()
    auth_service.remove(
        context=context,
        verbose=verbose,
        extra_args=ctx.args,
    )


@auth_app.command("switch")
def auth_switch(
    ctx: typer.Context,
    context: str = typer.Option(..., "--context"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    auth_service = get_auth_service()
    auth_service.switch(
        context=context,
        verbose=verbose,
        extra_args=ctx.args,
    )


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
