from __future__ import annotations
import os
from typing import Optional
import typer
import importlib.metadata

from gradient_adk.cli.config.yaml_agent_config_manager import YamlAgentConfigManager
from gradient_adk.cli.agent.deployment.deploy_service import AgentDeployService
from gradient_adk.cli.agent.direct_launch_service import DirectLaunchService
from gradient_adk.cli.agent.traces_service import GalileoTracesService
from gradient_adk.cli.agent.env_utils import get_do_api_token, EnvironmentError


def get_version() -> str:
    """Get the version from package metadata."""
    try:
        return importlib.metadata.version("gradient-adk")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


_agent_config_manager = YamlAgentConfigManager()
_launch_service = DirectLaunchService()


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        version = get_version()
        typer.echo(f"gradient-adk version {version}")
        raise typer.Exit()


app = typer.Typer(no_args_is_help=True, add_completion=False, help="gradient CLI")


# Add version option to main app
@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    )
):
    """Gradient ADK CLI"""
    pass


agent_app = typer.Typer(
    no_args_is_help=True,
    help="Agent configuration and management",
)
app.add_typer(agent_app, name="agent")


def _configure_agent(
    agent_name: Optional[str] = None,
    deployment_name: Optional[str] = None,
    entrypoint_file: Optional[str] = None,
    interactive: bool = True,
    skip_entrypoint_prompt: bool = False,  # New parameter for init
) -> None:
    """Configure agent settings and save to YAML file."""
    # If we're skipping entrypoint prompt (init case), we need to handle interactive mode specially
    if skip_entrypoint_prompt and interactive:
        # Handle the prompts manually for init case
        if agent_name is None:
            agent_name = typer.prompt("Agent workspace name")
        if deployment_name is None:
            deployment_name = typer.prompt("Agent deployment name", default="main")
        # entrypoint_file is already set and we don't prompt for it

        # Now call configure in non-interactive mode since we have all values
        _agent_config_manager.configure(
            agent_name=agent_name,
            agent_environment=deployment_name,
            entrypoint_file=entrypoint_file,
            interactive=False,
        )
    else:
        # Normal configure case - let the manager handle prompts
        _agent_config_manager.configure(
            agent_name=agent_name,
            agent_environment=deployment_name,
            entrypoint_file=entrypoint_file,
            interactive=interactive,
        )


def _create_project_structure() -> None:
    """Create the project structure with folders and template files."""
    import pathlib

    # Define folders to create
    folders_to_create = ["agents", "tools"]

    for folder in folders_to_create:
        folder_path = pathlib.Path(folder)
        if not folder_path.exists():
            folder_path.mkdir(exist_ok=True)

    # Create main.py if it doesn't exist
    main_py_path = pathlib.Path("main.py")
    if not main_py_path.exists():
        # Read the template file
        template_path = pathlib.Path(__file__).parent / "templates" / "main.py.template"
        if template_path.exists():
            main_py_content = template_path.read_text()
            main_py_path.write_text(main_py_content)

    # Create .gitignore if it doesn't exist
    gitignore_path = pathlib.Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/

# Environments
.env
"""
        gitignore_path.write_text(gitignore_content)

    # Create requirements.txt if it doesn't exist
    requirements_path = pathlib.Path("requirements.txt")
    if not requirements_path.exists():
        requirements_content = """gradient-adk
langgraph
langchain-core
gradient
"""
        requirements_path.write_text(requirements_content)

    # Create a .env file with placeholder variables if it doesn't exist
    env_path = pathlib.Path(".env")
    if not env_path.exists():
        env_content = ""
        env_path.write_text(env_content)


@agent_app.command("init")
def agent_init(
    agent_name: Optional[str] = typer.Option(
        None, "--agent-workspace-name", help="Name of the agent workspace"
    ),
    deployment_name: Optional[str] = typer.Option(
        None, "--deployment-name", help="Deployment name"
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive prompt mode"
    ),
):
    """Initializes a new agent with configuration and project structure."""
    # Create project structure first (including template files)
    _create_project_structure()

    # For init, always use main.py as the entrypoint (it was just created)
    entrypoint_file = "main.py"

    # Configure the agent (main.py is guaranteed to exist now)
    _configure_agent(
        agent_name=agent_name,
        deployment_name=deployment_name,
        entrypoint_file=entrypoint_file,
        interactive=interactive,
        skip_entrypoint_prompt=True,  # Don't prompt for entrypoint in init
    )

    typer.echo("\n🚀 Next steps:")
    typer.echo("   1. Edit main.py to implement your agent logic")
    typer.echo(
        "   2. Update your .env file with your GRADIENT_MODEL_ACCESS_KEY (https://cloud.digitalocean.com/gen-ai/model-access-keys)"
    )
    typer.echo("   3. Run 'gradient agent run' to test locally")
    typer.echo("   4. Use 'gradient agent deploy' when ready to deploy")


@agent_app.command("configure")
def agent_configure(
    agent_name: Optional[str] = typer.Option(
        None, "--agent-workspace-name", help="Name of the agent workspace"
    ),
    deployment_name: Optional[str] = typer.Option(
        None, "--deployment-name", help="Deployment name"
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
    """Configure agent settings in config.yaml for an existing project."""
    _configure_agent(
        agent_name=agent_name,
        deployment_name=deployment_name,
        entrypoint_file=entrypoint_file,
        interactive=interactive,
    )

    typer.echo("\n🚀 Configuration complete! Next steps:")
    typer.echo("   • Run 'gradient agent run' to test locally")
    typer.echo("   • Use 'gradient agent deploy' when ready to deploy")


@agent_app.command("run")
def agent_run(
    dev: bool = typer.Option(
        True, "--dev/--no-dev", help="Run in development mode with auto-reload"
    ),
    port: int = typer.Option(8080, "--port", help="Port to run the server on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind the server to"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging for debugging traces"
    ),
):
    """Runs the agent locally."""
    # Set verbose mode globally if requested
    if verbose:
        import os

        os.environ["GRADIENT_VERBOSE"] = "1"
        # Configure logging with verbose mode
        from gradient_adk.logging import configure_logging

        configure_logging(force_verbose=True)
        typer.echo("🔍 Verbose mode enabled - detailed trace logging will be shown")
    else:
        # Configure normal logging
        from gradient_adk.logging import configure_logging

        configure_logging()

    _launch_service.launch_locally(dev_mode=dev, host=host, port=port)


@agent_app.command("deploy")
def agent_deploy(
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="DigitalOcean API token (overrides DIGITALOCEAN_API_TOKEN env var)",
        envvar="DIGITALOCEAN_API_TOKEN",
        hide_input=True,
    )
):
    """Deploy the agent to DigitalOcean."""
    import asyncio
    from pathlib import Path
    from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI

    try:
        # Get configuration
        agent_workspace_name = _agent_config_manager.get_agent_name()
        agent_deployment_name = _agent_config_manager.get_agent_environment()

        # Get API token
        if not api_token:
            api_token = get_do_api_token()

        typer.echo(f"🚀 Deploying {agent_workspace_name}/{agent_deployment_name}...")
        typer.echo()

        # Get project ID from default project
        async def deploy():
            async with AsyncDigitalOceanGenAI(api_token=api_token) as client:
                # Get default project
                project_response = await client.get_default_project()
                project_id = project_response.project.id

                # Create deploy service with injected client
                deploy_service = AgentDeployService(client=client)

                # Deploy from current directory
                workspace_uuid = await deploy_service.deploy_agent(
                    agent_workspace_name=agent_workspace_name,
                    agent_deployment_name=agent_deployment_name,
                    source_dir=Path.cwd(),
                    project_id=project_id,
                    api_token=api_token,
                )

                typer.echo(
                    f"Agent deployed successfully! ({agent_workspace_name}/{agent_deployment_name})"
                )
                invoke_url = f"https://agents.do-ai.run/{workspace_uuid}/{agent_deployment_name}/run"

                typer.echo(
                    f"To invoke your deployed agent, send a POST request to {invoke_url} with your properly formatted payload."
                )
                example_cmd = f"""Example:
  curl -X POST {invoke_url} -d '{{"prompt": "hello"}}' """
                typer.echo(example_cmd)

        asyncio.run(deploy())

    except EnvironmentError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("\nTo set your token:", err=True)
        typer.echo("  export DIGITALOCEAN_API_TOKEN=your_token_here", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Deployment failed: {e}", err=True)
        typer.echo(
            f"Ensure that your agent can start up successfully with the correct environment variables prior to deploying.",
            err=True,
        )
        raise typer.Exit(1)


@agent_app.command("traces")
def agent_traces(
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="DigitalOcean API token (overrides DIGITALOCEAN_API_TOKEN env var)",
        envvar="DIGITALOCEAN_API_TOKEN",
        hide_input=True,
    )
):
    """Open the Galileo traces UI for monitoring agent execution."""
    import asyncio
    from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI

    try:
        # Get configuration
        agent_workspace_name = _agent_config_manager.get_agent_name()
        agent_deployment_name = _agent_config_manager.get_agent_environment()

        # Get API token
        if not api_token:
            api_token = get_do_api_token()

        typer.echo(
            f"🔍 Opening Galileo Traces UI for {agent_workspace_name}/{agent_deployment_name}..."
        )
        typer.echo()

        # Create async function to use context manager
        async def open_traces():
            async with AsyncDigitalOceanGenAI(api_token=api_token) as client:
                traces_service = GalileoTracesService(client=client)
                await traces_service.open_traces_console(
                    agent_workspace_name=agent_workspace_name,
                    agent_deployment_name=agent_deployment_name,
                )

        asyncio.run(open_traces())

        typer.echo("✅ DigitalOcean Traces UI opened in your browser")

    except EnvironmentError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("\nTo set your token permanently:", err=True)
        typer.echo("  export DIGITALOCEAN_API_TOKEN=your_token_here", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to open traces UI: {e}", err=True)
        raise typer.Exit(1)


@agent_app.command("logs")
def agent_logs(
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="DigitalOcean API token (overrides DIGITALOCEAN_API_TOKEN env var)",
        envvar="DIGITALOCEAN_API_TOKEN",
        hide_input=True,
    )
):
    """View runtime logs for the deployed agent."""
    import asyncio
    from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI

    try:
        # Get configuration
        agent_workspace_name = _agent_config_manager.get_agent_name()
        agent_deployment_name = _agent_config_manager.get_agent_environment()

        # Get API token
        if not api_token:
            api_token = get_do_api_token()

        # Create async function to use context manager
        async def fetch_logs():
            async with AsyncDigitalOceanGenAI(api_token=api_token) as client:
                traces_service = GalileoTracesService(client=client)
                logs = await traces_service.get_runtime_logs(
                    agent_workspace_name=agent_workspace_name,
                    agent_deployment_name=agent_deployment_name,
                )
                return logs

        logs = asyncio.run(fetch_logs())
        typer.echo(logs)

    except EnvironmentError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("\nTo set your token:", err=True)
        typer.echo("  export DIGITALOCEAN_API_TOKEN=your_token_here", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to fetch logs: {e}", err=True)
        raise typer.Exit(1)


def run():
    app()
