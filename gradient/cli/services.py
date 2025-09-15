from __future__ import annotations
import platform
import shutil
import subprocess
import os
from pathlib import Path
from typing import Optional, List
from importlib import resources
import typer
import yaml

from .interfaces import AuthService, ToolResolver, AgentConfigService, LaunchService


class DoctlResolver(ToolResolver):
    """Resolves paths for the doctl binary and configuration."""

    def resolve_tool_path(self, tool_name: str = "doctl") -> Path:
        """
        Prefer the bundled doctl (inside gradient._vendor.doctl.<os>_<arch>),
        but if it's not present in this build/layout, fall back to a system
        `doctl` on PATH. Error out only if neither exists.
        """
        if tool_name != "doctl":
            raise ValueError("DoctlResolver only supports 'doctl' tool")

        # 1) Try bundled
        sys_os = platform.system().lower()
        machine = platform.machine().lower()

        if sys_os.startswith("darwin"):
            os_dir = "darwin"
        elif sys_os.startswith("linux"):
            os_dir = "linux"
        elif sys_os.startswith("windows"):
            os_dir = "windows"
        else:
            raise typer.Exit(64)

        if machine in ("arm64", "aarch64"):
            arch = "arm64"
        elif machine in ("x86_64", "amd64"):
            arch = "amd64"
        else:
            raise typer.Exit(64)

        bin_name = "doctl.exe" if os_dir == "windows" else "doctl"
        pkg = f"gradient._vendor.doctl.{os_dir}_{arch}"

        try:
            base = resources.files(pkg)  # py>=3.9
            path = Path(base / bin_name)
            if path.exists():
                # ensure exec bit in dev checkouts on unix
                if os.name != "nt":
                    try:
                        path.chmod(path.stat().st_mode | 0o111)
                    except Exception:
                        pass
                return path
        except Exception:
            # Package (or resource) may not exist in this dev layout; that's fine.
            pass

        # 2) Fallback: system doctl on PATH
        system = shutil.which("doctl")
        if system:
            return Path(system)

        # 3) Nothing found
        typer.echo(
            "Error: doctl not found. Bundle it with gradient or install doctl on your PATH.",
            err=True,
        )
        raise typer.Exit(127)

    def get_config_path(self) -> str:
        """Get the doctl configuration file path."""
        p = Path.home() / ".config" / "gradient" / "doctl" / "config.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)


class DoctlAuthService(AuthService):
    """DoCtl-based implementation of authentication service."""

    def __init__(self, resolver: ToolResolver):
        self.resolver = resolver

    def _build_base_command(self) -> List[str]:
        """Build the base doctl command with config path."""
        return [
            str(self.resolver.resolve_tool_path("doctl")),
            "--config",
            self.resolver.get_config_path(),
        ]

    def _run_command(self, cmd: List[str], input_bytes: Optional[bytes] = None) -> None:
        """Execute a command and handle exit codes."""
        rc = subprocess.run(cmd, input=input_bytes).returncode
        if rc != 0:
            raise typer.Exit(rc)

    def init(
        self,
        context: Optional[str] = None,
        token: Optional[str] = None,
        api_url: Optional[str] = None,
        interactive: bool = True,
        output: Optional[str] = None,
        verbose: bool = False,
        trace: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """Initialize authentication with the given parameters."""
        # Global flags come before the subcommand
        cmd = self._build_base_command()

        if api_url:
            cmd += ["--api-url", api_url]
        if output:
            cmd += ["--output", output]
        if verbose:
            cmd += ["--verbose"]
        if trace:
            cmd += ["--trace"]

        # If token is provided, run fully non-interactively by giving -t
        if token:
            cmd += ["--access-token", token]

        # Subcommand and its flags
        cmd += ["auth", "init"]
        if context:
            cmd += ["--context", context]

        # forward any extra/unknown options straight to doctl
        if extra_args:
            cmd += extra_args

        # Guardrails: if non-interactive but no token, we'd hang → error out.
        if not interactive and not token:
            typer.echo("Error: --no-interactive requires --token.", err=True)
            raise typer.Exit(2)

        # If interactive and no token provided, let doctl prompt directly (no stdin piping)
        self._run_command(cmd)

    def list(
        self,
        output: Optional[str] = None,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """List authentication contexts."""
        cmd = self._build_base_command() + ["auth", "list"]
        if output:
            cmd += ["--output", output]
        if verbose:
            cmd += ["--verbose"]
        if extra_args:
            cmd += extra_args
        self._run_command(cmd)

    def remove(
        self,
        context: str,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """Remove an authentication context."""
        cmd = self._build_base_command() + ["auth", "remove", "--context", context]
        if verbose:
            cmd += ["--verbose"]
        if extra_args:
            cmd += extra_args
        self._run_command(cmd)

    def switch(
        self,
        context: str,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """Switch to an authentication context."""
        cmd = self._build_base_command() + ["auth", "switch", "--context", context]
        if verbose:
            cmd += ["--verbose"]
        if extra_args:
            cmd += extra_args
        self._run_command(cmd)


class YamlAgentConfigService(AgentConfigService):
    """YAML-based implementation of agent configuration service."""

    def __init__(self):
        # Create .gradient directory in current working directory
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
        """Configure agent settings and save to YAML file."""
        # If interactive and values not provided, prompt for them
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
            # Non-interactive mode requires all values
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

        # Validate that the entrypoint file exists
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

        # Validate that the entrypoint file contains @entrypoint decorator
        try:
            with open(entrypoint_path, "r") as f:
                file_content = f.read()

            # Look for @entrypoint decorator pattern (at start of line, possibly with whitespace)
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
                typer.echo("Example: from gradient.sdk import entrypoint")
                typer.echo("         @entrypoint")
                typer.echo("         def my_function(prompt: str) -> str:")
                raise typer.Exit(1)

        except typer.Exit:
            raise  # Re-raise typer.Exit without modification
        except Exception as e:
            typer.echo(
                f"Error: Could not read entrypoint file '{entrypoint_file}': {e}",
                err=True,
            )
            raise typer.Exit(1)

        # Create the configuration dictionary
        config = {
            "agent_name": agent_name,
            "agent_environment": agent_environment,
            "entrypoint_file": entrypoint_file,
        }

        # Write to YAML file (overwrites if exists)
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
        """Launch the agent locally using FastAPI directly."""
        # Check if agent.yml exists
        if not self.config_file.exists():
            typer.echo("Error: No agent configuration found.", err=True)
            typer.echo(
                "Please run 'gradient agent init' first to set up your agent.", err=True
            )
            raise typer.Exit(1)

        # Read agent config
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

        # Validate that the entrypoint file exists
        entrypoint_path = Path.cwd() / entrypoint_file
        if not entrypoint_path.exists():
            typer.echo(
                f"Error: Entrypoint file '{entrypoint_file}' does not exist.",
                err=True,
            )
            raise typer.Exit(1)

        # Import the entrypoint module to trigger the @entrypoint decorator
        try:
            # Add current directory to Python path so we can import the module
            import sys

            current_dir = str(Path.cwd())
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            # Convert file path to module name
            module_name = (
                entrypoint_file.replace(".py", "").replace("/", ".").replace("\\", ".")
            )

            # Import the module
            import importlib

            importlib.import_module(module_name)

            typer.echo(f"Starting {agent_name} server...")
            typer.echo("Server will be accessible at http://localhost:8080")
            typer.echo("Press Ctrl+C to stop the server")

            # Get the FastAPI app from the decorator and start uvicorn
            from gradient.sdk.decorator import get_app
            import uvicorn

            uvicorn.run(get_app(), host="0.0.0.0", port=8080)

        except ImportError as e:
            typer.echo(
                f"Error importing entrypoint module '{module_name}': {e}", err=True
            )
            typer.echo(
                "Make sure all dependencies are installed and the module is valid.",
                err=True,
            )
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Error starting server: {e}", err=True)
            raise typer.Exit(1)
        except KeyboardInterrupt:
            typer.echo("\nServer stopped by user.")
            raise typer.Exit(0)
