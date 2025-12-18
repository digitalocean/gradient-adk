from __future__ import annotations
import os
from typing import Optional
import typer
import importlib.metadata

from gradient_adk.cli.config.yaml_agent_config_manager import YamlAgentConfigManager
from gradient_adk.cli.agent.deployment.deploy_service import AgentDeployService
from gradient_adk.cli.agent.direct_launch_service import DirectLaunchService
from gradient_adk.cli.agent.traces_service import GalileoTracesService
from gradient_adk.cli.agent.evaluation_service import EvaluationService
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
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging for debugging deployment"
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip pre-deployment validation (not recommended)",
    ),
):
    """Deploy the agent to DigitalOcean."""
    import asyncio
    from pathlib import Path
    from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI
    from gradient_adk.cli.agent.deployment.validation import (
        validate_agent_entrypoint,
        ValidationError,
    )

    # Set verbose mode globally if requested
    if verbose:
        import os

        os.environ["GRADIENT_VERBOSE"] = "1"
        # Configure logging with verbose mode
        from gradient_adk.logging import configure_logging

        configure_logging(force_verbose=True)
        typer.echo(
            "🔍 Verbose mode enabled - detailed deployment logging will be shown"
        )
        typer.echo()

    try:
        # Get configuration
        agent_workspace_name = _agent_config_manager.get_agent_name()
        agent_deployment_name = _agent_config_manager.get_agent_environment()
        entrypoint_file = _agent_config_manager.get_entrypoint_file()

        # Validate names follow requirements (alphanumeric, hyphens, underscores only)
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", agent_workspace_name):
            typer.echo(
                f"❌ Invalid agent workspace name: '{agent_workspace_name}'", err=True
            )
            typer.echo(
                "Agent workspace name can only contain alphanumeric characters, hyphens, and underscores.",
                err=True,
            )
            raise typer.Exit(1)

        if not re.match(r"^[a-zA-Z0-9_-]+$", agent_deployment_name):
            typer.echo(
                f"❌ Invalid deployment name: '{agent_deployment_name}'", err=True
            )
            typer.echo(
                "Deployment name can only contain alphanumeric characters, hyphens, and underscores.",
                err=True,
            )
            raise typer.Exit(1)

        # Validate agent before deploying (unless skipped)
        if not skip_validation:
            try:
                validate_agent_entrypoint(
                    source_dir=Path.cwd(),
                    entrypoint_file=entrypoint_file,
                    verbose=verbose,
                )
            except ValidationError as e:
                typer.echo(f"❌ Validation failed:\n{e}", err=True)
                typer.echo(
                    "\nFix the issues above and try again, or use --skip-validation to bypass (not recommended).",
                    err=True,
                )
                raise typer.Exit(1)
        else:
            typer.echo(
                "⚠️  Skipping validation - deployment may fail if agent has issues"
            )
            typer.echo()

        # Get API token
        if not api_token:
            api_token = get_do_api_token()

        typer.echo(f"🚀 Deploying {agent_workspace_name}/{agent_deployment_name}...")
        typer.echo()

        # Get project ID from default project
        async def deploy():
            from gradient_adk.digital_ocean_api.errors import (
                DOAPIClientError,
                DOAPIAuthError,
            )

            async with AsyncDigitalOceanGenAI(api_token=api_token) as client:
                # Get default project
                try:
                    project_response = await client.get_default_project()
                    project_id = project_response.project.id
                except (DOAPIClientError, DOAPIAuthError) as e:
                    typer.echo(
                        "❌ Failed to retrieve default project from DigitalOcean API",
                        err=True,
                    )
                    typer.echo(
                        "\nYour API token must have the following scopes:", err=True
                    )
                    typer.echo("  • projects:read (or projects:*)", err=True)
                    typer.echo("  • genai:*", err=True)
                    typer.echo(
                        "\nPlease create a new API token with these scopes at:",
                        err=True,
                    )
                    typer.echo(
                        "  https://cloud.digitalocean.com/account/api/tokens", err=True
                    )
                    raise typer.Exit(1)

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
  curl -X POST {invoke_url} \\
    -H "Authorization: Bearer $DIGITALOCEAN_API_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{{"prompt": "hello"}}'"""
                typer.echo(example_cmd)

        asyncio.run(deploy())

    except EnvironmentError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("\nTo set your token:", err=True)
        typer.echo("  export DIGITALOCEAN_API_TOKEN=your_token_here", err=True)
        raise typer.Exit(1)
    except Exception as e:
        import traceback

        # Get error message with fallback
        error_msg = str(e) if str(e) else repr(e)

        typer.echo(f"❌ Deployment failed: {error_msg}", err=True)

        typer.echo(
            "\nEnsure that your agent can start up successfully with the correct environment variables prior to deploying.",
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
    """Open the DigitalOcean traces UI for monitoring agent execution."""
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
            f"🔍 Opening DigitalOcean Traces UI for {agent_workspace_name}/{agent_deployment_name}..."
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


@agent_app.command("evaluate")
def agent_evaluate(
    test_case_name: Optional[str] = typer.Option(
        None, "--test-case-name", help="Name of the evaluation test case"
    ),
    dataset_file: Optional[str] = typer.Option(
        None, "--dataset-file", help="Path to the CSV dataset file for evaluation"
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        help="Comma-separated list of metric categories (correctness,context_quality,user_outcomes,model_fit,safety_and_security)",
    ),
    star_metric_name: Optional[str] = typer.Option(
        None,
        "--star-metric-name",
        help="Name of the star metric (default: Correctness (general hallucinations))",
    ),
    success_threshold: float = typer.Option(
        80.0, "--success-threshold", help="Success threshold for star metric (0-100)"
    ),
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="DigitalOcean API token (overrides DIGITALOCEAN_API_TOKEN env var)",
        envvar="DIGITALOCEAN_API_TOKEN",
        hide_input=True,
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive prompt mode"
    ),
):
    """Run an evaluation test case for the agent."""
    import asyncio
    from pathlib import Path
    from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI
    from gradient_adk.cli.agent.evaluation_service import EvaluationService
    from gradient_adk.digital_ocean_api.models import EvaluationMetricValueType

    try:
        # Get configuration
        agent_workspace_name = _agent_config_manager.get_agent_name()
        agent_deployment_name = _agent_config_manager.get_agent_environment()

        # Get API token
        if not api_token:
            api_token = get_do_api_token()

        # Create async function to handle interactive prompts with API access
        async def get_interactive_inputs():
            async with AsyncDigitalOceanGenAI(api_token=api_token) as client:
                eval_service = EvaluationService(client=client)
                metrics_by_category = await eval_service.get_metrics_by_category()

                nonlocal test_case_name, dataset_file, categories, star_metric_name, success_threshold

                if test_case_name is None:
                    test_case_name = typer.prompt("Evaluation test case name")
                if dataset_file is None:
                    dataset_file = typer.prompt("Dataset file path")

                if categories is None:
                    typer.echo()
                    typer.echo("Available metric categories:")
                    for cat, metrics in sorted(metrics_by_category.items()):
                        typer.echo(f"  • {cat} ({len(metrics)} metrics)")
                    typer.echo()
                    categories = typer.prompt(
                        "Select metric categories (comma-separated)",
                        default="correctness,context_quality",
                    )

                # Parse categories to get selected metrics
                metric_categories_list = [cat.strip() for cat in categories.split(",")]
                selected_metrics = []
                for cat in metric_categories_list:
                    if cat in metrics_by_category:
                        selected_metrics.extend(metrics_by_category[cat])

                if star_metric_name is None and selected_metrics:
                    typer.echo()
                    typer.echo("Available star metrics from selected categories:")
                    for i, metric in enumerate(selected_metrics[:10], 1):
                        value_type = (
                            metric.metric_value_type.value.replace(
                                "METRIC_VALUE_TYPE_", ""
                            ).lower()
                            if metric.metric_value_type
                            else "unknown"
                        )
                        typer.echo(f"  {i}. {metric.metric_name} ({value_type})")
                    if len(selected_metrics) > 10:
                        typer.echo(f"  ... and {len(selected_metrics) - 10} more")
                    typer.echo()
                    star_metric_name = typer.prompt(
                        "Star metric name",
                        default=(
                            selected_metrics[0].metric_name if selected_metrics else ""
                        ),
                    )

                # Get star metric details for threshold validation
                star_metric_obj = None
                if star_metric_name:
                    star_metric_obj = await eval_service.find_metric_by_name(
                        star_metric_name
                    )

                if success_threshold is None and star_metric_obj:
                    # Only prompt for threshold if it's a number or percentage metric
                    if star_metric_obj.metric_value_type in [
                        EvaluationMetricValueType.METRIC_VALUE_TYPE_NUMBER,
                        EvaluationMetricValueType.METRIC_VALUE_TYPE_PERCENTAGE,
                    ]:
                        typer.echo()
                        range_info = ""
                        if (
                            star_metric_obj.range_min is not None
                            and star_metric_obj.range_max is not None
                        ):
                            range_info = f" (range: {star_metric_obj.range_min}-{star_metric_obj.range_max})"
                        elif star_metric_obj.range_min is not None:
                            range_info = f" (min: {star_metric_obj.range_min})"
                        elif star_metric_obj.range_max is not None:
                            range_info = f" (max: {star_metric_obj.range_max})"

                        default_threshold = 80.0
                        if star_metric_obj.range_max is not None:
                            default_threshold = min(
                                80.0, star_metric_obj.range_max * 0.8
                            )

                        success_threshold = typer.prompt(
                            f"Success threshold for '{star_metric_obj.metric_name}'{range_info}",
                            default=default_threshold,
                            type=float,
                        )

        # Handle interactive prompts
        if interactive:
            asyncio.run(get_interactive_inputs())

        # Validate required inputs
        if not test_case_name:
            typer.echo("❌ Test case name is required", err=True)
            raise typer.Exit(1)
        if not dataset_file:
            typer.echo("❌ Dataset file is required", err=True)
            raise typer.Exit(1)
        if not categories:
            typer.echo("❌ Metric categories are required", err=True)
            raise typer.Exit(1)

        # Parse categories
        metric_categories = [cat.strip() for cat in categories.split(",")]

        # Validate dataset file path
        dataset_path = Path(dataset_file)
        if not dataset_path.exists():
            typer.echo(f"❌ Dataset file not found: {dataset_file}", err=True)
            raise typer.Exit(1)

        if dataset_path.suffix.lower() != ".csv":
            typer.echo(
                f"❌ Dataset file must be a CSV file, got: {dataset_path.suffix}",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(
            f"🧪 Running evaluation '{test_case_name}' for {agent_workspace_name}/{agent_deployment_name}..."
        )
        typer.echo(f"📊 Dataset: {dataset_file}")
        typer.echo(f"📈 Metric categories: {', '.join(metric_categories)}")
        if star_metric_name:
            typer.echo(f"⭐ Star metric: {star_metric_name}")
        if success_threshold is not None:
            typer.echo(f"🎯 Success threshold: {success_threshold}")
        typer.echo()
        evaluation_run_uuid = None

        # Create async function to run evaluation
        async def run_evaluation():
            async with AsyncDigitalOceanGenAI(api_token=api_token) as client:
                evaluation_service = EvaluationService(client=client)

                # Start the evaluation
                evaluation_run_uuid = await evaluation_service.run_evaluation(
                    agent_workspace_name=agent_workspace_name,
                    agent_deployment_name=agent_deployment_name,
                    test_case_name=test_case_name,
                    dataset_file_path=dataset_path,
                    metric_categories=metric_categories,
                    star_metric_name=star_metric_name,
                    success_threshold=success_threshold,
                )

                typer.echo(f"✅ Evaluation started successfully!")
                typer.echo(f"📋 Evaluation run UUID: {evaluation_run_uuid}")
                typer.echo()
                typer.echo("⏳ Waiting for evaluation to complete...")

                # Poll for completion
                evaluation_run = await evaluation_service.poll_evaluation_run(
                    evaluation_run_uuid=evaluation_run_uuid
                )

                return evaluation_run

        evaluation_run = asyncio.run(run_evaluation())

        # Display results
        typer.echo()
        typer.echo("=" * 60)
        typer.echo("✅ Evaluation Completed!")
        typer.echo("=" * 60)
        typer.echo()

        # Display basic info
        typer.echo(f"Test Case: {evaluation_run.test_case_name}")
        typer.echo(
            f"Agent deployment name: {evaluation_run.agent_deployment_name or 'N/A'}"
        )
        typer.echo(f"Status: {evaluation_run.status.value}")

        # Display pass status if available
        if evaluation_run.pass_status is not None:
            status_emoji = "✅" if evaluation_run.pass_status else "❌"
            typer.echo(
                f"Pass Status: {status_emoji} {'PASSED' if evaluation_run.pass_status else 'FAILED'}"
            )

        # Display star metric result if available
        if evaluation_run.star_metric_result:
            typer.echo()
            typer.echo("Star Metric Result:")
            metric = evaluation_run.star_metric_result
            typer.echo(f"  Metric: {metric.metric_name}")

            if metric.number_value is not None:
                typer.echo(f"  Value: {metric.number_value:.2f}")
            elif metric.string_value is not None:
                typer.echo(f"  Value: {metric.string_value}")

            if metric.reasoning:
                typer.echo(f"  Reasoning: {metric.reasoning}")

        # Display run-level metrics if available
        if evaluation_run.run_level_metric_results:
            typer.echo()
            typer.echo("Run-Level Metrics:")
            for metric in evaluation_run.run_level_metric_results:
                value_str = ""
                if metric.number_value is not None:
                    value_str = f"{metric.number_value:.2f}"
                elif metric.string_value is not None:
                    value_str = metric.string_value
                typer.echo(f"  • {metric.metric_name}: {value_str}")

        typer.echo()
        typer.echo("=" * 60)
        typer.echo(f"📊 View full results in the DigitalOcean console:")
        typer.echo(
            f"   https://cloud.digitalocean.com/gen-ai/agent-workspaces/{agent_workspace_name}/evaluations/test-cases/{evaluation_run.test_case_uuid}"
        )
        typer.echo("=" * 60)

    except TimeoutError as e:
        typer.echo()
        typer.echo(f"⏱️  {e}", err=True)
        typer.echo(
            "\nThe evaluation is still running. Check the console for status:", err=True
        )
        typer.echo(
            f"  https://cloud.digitalocean.com/gen-ai/agent-workspaces/{agent_workspace_name}/evaluations/test-cases/{evaluation_run.test_case_uuid}",
            err=True,
        )
        raise typer.Exit(1)
    except EnvironmentError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("\nTo set your token:", err=True)
        typer.echo("  export DIGITALOCEAN_API_TOKEN=your_token_here", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {e}", err=True)
        raise typer.Exit(1)


def run():
    app()
