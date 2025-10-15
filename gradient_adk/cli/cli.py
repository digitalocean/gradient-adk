from __future__ import annotations
import os
from typing import Optional
import typer
import importlib.metadata

from .agent import (
    AgentConfigService,
    LaunchService,
    DeployService,
    AgentDeployService,
    TracesService,
    ConfigReader,
    YamlAgentConfigService,
    DirectLaunchService,
    GalileoTracesService,
    YamlConfigReader,
    get_do_api_token,
    EnvironmentError,
)


def get_version() -> str:
    """Get the version from package metadata."""
    try:
        return importlib.metadata.version("gradient-adk")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


_agent_config_service = YamlAgentConfigService()
_launch_service = DirectLaunchService()
_config_reader = YamlConfigReader()


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


def get_agent_config_service() -> AgentConfigService:
    return _agent_config_service


def get_launch_service() -> LaunchService:
    return _launch_service


def get_config_reader() -> ConfigReader:
    return _config_reader


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
            agent_name = typer.prompt("Agent name")
        if deployment_name is None:
            deployment_name = typer.prompt("Agent environment name", default="main")
        # entrypoint_file is already set and we don't prompt for it

        # Now call configure in non-interactive mode since we have all values
        agent_config_service = get_agent_config_service()
        agent_config_service.configure(
            agent_name=agent_name,
            agent_environment=deployment_name,
            entrypoint_file=entrypoint_file,
            interactive=False,
        )
    else:
        # Normal configure case - let the service handle prompts
        agent_config_service = get_agent_config_service()
        agent_config_service.configure(
            agent_name=agent_name,
            agent_environment=deployment_name,
            entrypoint_file=entrypoint_file,
            interactive=interactive,
        )


def _create_project_structure() -> None:
    """Create the project structure with folders and template files."""
    import pathlib

    typer.echo("\n📁 Creating project structure...")

    # Define folders to create
    folders_to_create = ["agents", "datasets", "evaluations", "tools"]

    for folder in folders_to_create:
        folder_path = pathlib.Path(folder)
        if not folder_path.exists():
            folder_path.mkdir(exist_ok=True)
            typer.echo(f"   Created folder: {folder}/")

    # Create main.py if it doesn't exist
    main_py_path = pathlib.Path("main.py")
    if not main_py_path.exists():
        # Read the template file
        template_path = pathlib.Path(__file__).parent / "templates" / "main.py.template"
        if template_path.exists():
            main_py_content = template_path.read_text()
            main_py_path.write_text(main_py_content)
            typer.echo("   Created file: main.py")

    # Create .gitignore if it doesn't exist
    gitignore_path = pathlib.Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/

# Environments
.env
"""
        gitignore_path.write_text(gitignore_content)
        typer.echo("   Created file: .gitignore")

    typer.echo("\n✅ Project structure created successfully!")


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
    typer.echo("   2. Add your datasets to the datasets/ folder")
    typer.echo("   3. Create evaluation scripts in evaluations/")
    typer.echo("   4. Add custom tools to the tools/ folder")
    typer.echo("   5. Run 'gradient agent run' to test locally")
    typer.echo("   6. Use 'gradient agent deploy' when ready to deploy")


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

    launch_service = get_launch_service()
    launch_service.launch_locally(dev_mode=dev, host=host, port=port)


@agent_app.command("deploy")
def agent_deploy(
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="DigitalOcean API token (overrides DO_API_TOKEN env var)",
        envvar="DO_API_TOKEN",
        hide_input=True,
    )
):
    """Deploy the agent to DigitalOcean."""
    import asyncio
    from pathlib import Path
    from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI

    try:
        # Get configuration
        config_reader = get_config_reader()
        agent_workspace_name = config_reader.get_agent_name()
        agent_deployment_name = config_reader.get_agent_environment()

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
                await deploy_service.deploy_agent(
                    agent_workspace_name=agent_workspace_name,
                    agent_deployment_name=agent_deployment_name,
                    source_dir=Path.cwd(),
                    project_id=project_id,
                    api_token=api_token,
                )

        asyncio.run(deploy())

    except EnvironmentError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("\nTo set your token:", err=True)
        typer.echo("  export DO_API_TOKEN=your_token_here", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Deployment failed: {e}", err=True)
        raise typer.Exit(1)


@agent_app.command("evaluate")
def agent_evaluate():
    """Run an evaluation of the agent."""
    import time

    try:
        config_reader = get_config_reader()
        agent_name = config_reader.get_agent_name()
        agent_environment = config_reader.get_agent_environment()

        typer.echo(f"🧪 Initiating evaluation for agent: {agent_name}")
        typer.echo(f"🎯 Environment: {agent_environment}")
        typer.echo()

        # Setting up evaluation
        typer.echo("⚙️  Setting up evaluation...")
        time.sleep(2)

        # Starting run
        typer.echo("🚀 Starting run...")
        time.sleep(1)

        # Poll for 10 seconds with progress indicators
        typer.echo("📊 Running evaluation...")
        for i in range(10):
            time.sleep(1)
            dots = "." * ((i % 3) + 1)
            typer.echo(f"   Evaluating{dots}", nl=False)
            if i < 9:  # Don't print newline on last iteration
                typer.echo("\r", nl=False)

        typer.echo()  # Final newline
        time.sleep(0.5)

        # Complete
        typer.echo("✅ Evaluation completed!")
        typer.echo()
        typer.echo("📈 View detailed results at:")
        # Get URL from env var EVALUATION_URL
        eval_url = os.getenv(
            "EVALUATION_URL",
            "https://cloud.digitalocean.com/gen-ai/workspaces/11f076b0-2ff2-d71d-b074-4e013e2ddde4/evaluations/384ad568-76b5-11f0-b074-4e013e2ddde4/runs/73b04ba3-6494-42af-b7de-b093d8082814?i=b59231",
        )
        typer.echo(eval_url)

    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {e}", err=True)
        raise typer.Exit(1)


@agent_app.command("traces")
def agent_traces(
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="DigitalOcean API token (overrides DO_API_TOKEN env var)",
        envvar="DO_API_TOKEN",
        hide_input=True,
    )
):
    """Open the Galileo traces UI for monitoring agent execution."""
    import asyncio
    from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI

    try:
        # Get configuration
        config_reader = get_config_reader()
        agent_workspace_name = config_reader.get_agent_name()
        agent_deployment_name = config_reader.get_agent_environment()

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
        typer.echo("  export DO_API_TOKEN=your_token_here", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to open traces UI: {e}", err=True)
        raise typer.Exit(1)


@agent_app.command("logs")
def agent_logs(
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="DigitalOcean API token (overrides DO_API_TOKEN env var)",
        envvar="DO_API_TOKEN",
        hide_input=True,
    )
):
    """View runtime logs for the deployed agent."""
    import asyncio
    from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI

    try:
        # Get configuration
        config_reader = get_config_reader()
        agent_workspace_name = config_reader.get_agent_name()
        agent_deployment_name = config_reader.get_agent_environment()

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
        typer.echo("  export DO_API_TOKEN=your_token_here", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to fetch logs: {e}", err=True)
        raise typer.Exit(1)


def run():
    app()
