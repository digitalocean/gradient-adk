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
        Prefer the bundled doctl (inside gradient_agent._vendor.doctl.<os>_<arch>),
        else fall back to a system `doctl` on PATH. Error out only if neither exists.
        """
        if tool_name != "doctl":
            raise ValueError("DoctlResolver only supports 'doctl' tool")

        # 1) Determine platform
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

        # 2) Try unified vendor path
        pkg = f"gradient_agent._vendor.doctl.{os_dir}_{arch}"
        try:
            base = resources.files(pkg)
            path = Path(base / bin_name)
            if path.exists():
                if os.name != "nt":
                    try:
                        path.chmod(path.stat().st_mode | 0o111)
                    except Exception:
                        pass
                return path
        except Exception:
            pass

        # 3) Fallback: system doctl on PATH
        system = shutil.which("doctl")
        if system:
            return Path(system)

        # 4) Nothing found
        typer.echo(
            "Error: doctl not found. Bundle it or install doctl on your PATH.",
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
        cmd = self._build_base_command()

        if api_url:
            cmd += ["--api-url", api_url]
        if output:
            cmd += ["--output", output]
        if verbose:
            cmd += ["--verbose"]
        if trace:
            cmd += ["--trace"]
        if token:
            cmd += ["--access-token", token]

        cmd += ["auth", "init"]
        if context:
            cmd += ["--context", context]
        if extra_args:
            cmd += extra_args

        if not interactive and not token:
            typer.echo("Error: --no-interactive requires --token.", err=True)
            raise typer.Exit(2)

        self._run_command(cmd)

    def list(
        self,
        output: Optional[str] = None,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> None:
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
        cmd = self._build_base_command() + ["auth", "switch", "--context", context]
        if verbose:
            cmd += ["--verbose"]
        if extra_args:
            cmd += extra_args
        self._run_command(cmd)


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
                typer.echo("Example: from gradient_agent import entrypoint")
                typer.echo("         @entrypoint")
                typer.echo("         def my_function(prompt: str) -> str:")
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
                from gradient_agent import run_server

                run_server(host="0.0.0.0", port=8080)
            except ImportError:
                typer.echo(
                    "Error: gradient_agent package not found.",
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
            typer.echo("  from gradient_agent import entrypoint", err=True)
            raise typer.Exit(1)
