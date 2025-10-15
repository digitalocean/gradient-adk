"""Direct FastAPI launch service implementation.

Reverted to a simpler implementation that directly calls `uvicorn.run` in-process.
Adds clear guidance for users who rely on relative imports.
"""

from __future__ import annotations
import importlib
import sys
from pathlib import Path
import os
import re
import typer
import yaml

from .launch_service import LaunchService


class DirectLaunchService(LaunchService):
    """Direct FastAPI implementation of launch service."""

    def __init__(self):
        self.config_dir = Path.cwd() / ".gradient"
        self.config_file = self.config_dir / "agent.yml"

    def launch_locally(
        self, dev_mode: bool = False, host: str = "0.0.0.0", port: int = 8080
    ) -> None:
        """Launch the agent locally using FastAPI server."""
        config = self._load_config()
        entrypoint_file = config.get("entrypoint_file")
        agent_name = config.get("agent_name", "gradient-agent")

        if not entrypoint_file:
            typer.echo(
                "Error: No entrypoint file specified in configuration.", err=True
            )
            raise typer.Exit(1)

        self._validate_entrypoint_file(entrypoint_file)

        if dev_mode:
            self._start_server(
                agent_name, entrypoint_file, host, port, reload=True, dev_banner=True
            )
        else:
            typer.echo("🚀 Running in production mode...")
            self._import_entrypoint_module(entrypoint_file)
            self._start_server(
                agent_name, entrypoint_file, host, port, reload=False, dev_banner=False
            )

    def _load_config(self) -> dict:
        """Load agent configuration from YAML file."""
        if not self.config_file.exists():
            typer.echo("Error: No agent configuration found.", err=True)
            typer.echo(
                "Please run 'gradient agent init' first to set up your agent.", err=True
            )
            raise typer.Exit(1)

        try:
            with open(self.config_file, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            typer.echo(f"Error reading agent configuration: {e}", err=True)
            raise typer.Exit(1)

    def _validate_entrypoint_file(self, entrypoint_file: str) -> None:
        """Validate that the entrypoint file exists."""
        entrypoint_path = Path.cwd() / entrypoint_file
        if not entrypoint_path.exists():
            typer.echo(
                f"Error: Entrypoint file '{entrypoint_file}' does not exist.",
                err=True,
            )
            raise typer.Exit(1)

    def _import_entrypoint_module(self, entrypoint_file: str) -> None:
        """Validate the module can be imported."""
        try:
            current_dir = str(Path.cwd())
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            module_name = (
                entrypoint_file.replace(".py", "").replace("/", ".").replace("\\", ".")
            )
            module = importlib.import_module(module_name)
            if not hasattr(module, "app"):
                # Soft warning only; user will see clearer error from uvicorn
                typer.echo(
                    "⚠️  Warning: module has no 'app' attribute yet. Ensure @entrypoint is applied."
                )
        except Exception:
            # Suppress to allow uvicorn to attempt import in correct package context
            pass

    def _derive_module_name(self, entrypoint_file: str) -> str:
        """Convert an entrypoint file path to a module name (best effort).

        NOTE: This does not guarantee proper relative import support—users needing
        relative imports should structure code as a package (see guidance).
        """
        return entrypoint_file.replace(".py", "").replace("/", ".").replace("\\", ".")

    def _start_server(
        self,
        agent_name: str,
        entrypoint_file: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        reload: bool = False,
        dev_banner: bool = False,
    ) -> None:
        """Start the FastAPI server (in-process uvicorn).

        This keeps things simple; relative imports require user packaging.
        """
        # Resolve entrypoint path relative to the current working directory
        original_cwd = Path.cwd()
        entry_path = (original_cwd / entrypoint_file).resolve()
        entry_dir = entry_path.parent
        module_name = entry_path.stem  # Use file name without extension

        if dev_banner:
            typer.echo(f"📂 Entrypoint: {entrypoint_file}")
            typer.echo(f"🌐 Server: http://{host}:{port}")
            typer.echo(f"🏷️  Agent: {agent_name}")
            typer.echo(f"🌐 Entrypoint endpoint: http://{host}:{port}/completions")
            typer.echo("Auto-reload enabled - server will restart on file changes")
            typer.echo("Press Ctrl+C to stop the server\n")
        else:
            typer.echo(f"Starting {agent_name} server...")
            typer.echo(f"Server will be accessible at http://{host}:{port}")
            typer.echo("Press Ctrl+C to stop the server")

        app_target = f"{module_name}:app"

        try:
            import uvicorn

            # For reload mode, we need to ensure the working directory is correct
            # when the uvicorn subprocess starts
            if reload:
                # Add the entry directory to Python path so imports work
                current_sys_path = sys.path.copy()
                if str(entry_dir) not in sys.path:
                    sys.path.insert(0, str(entry_dir))

                try:
                    # Change to the entry directory for uvicorn
                    os.chdir(entry_dir)
                    uvicorn.run(
                        app_target,
                        host=host,
                        port=port,
                        reload=reload,
                        reload_dirs=[str(entry_dir)],
                    )
                finally:
                    # Restore original state
                    sys.path[:] = current_sys_path
                    try:
                        os.chdir(original_cwd)
                    except Exception:
                        pass
            else:
                # Non-reload mode - change directory and run
                try:
                    os.chdir(entry_dir)
                    uvicorn.run(
                        app_target,
                        host=host,
                        port=port,
                        reload=False,
                    )
                finally:
                    try:
                        os.chdir(original_cwd)
                    except Exception:
                        pass
        except ImportError as e:
            typer.echo(f"❌ Unable to import module '{module_name}': {e}", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"❌ Server failed to start: {e}", err=True)
            raise typer.Exit(1)

    # _start_dev_server removed; unified into _start_server with flags.

    def _show_import_help(self) -> None:
        """Show help for import errors."""
        typer.echo(
            "Please install the gradient-adk package and ensure imports are correct:",
            err=True,
        )
        typer.echo("  pip install gradient-adk", err=True)
        typer.echo("  from gradient_adk import entrypoint", err=True)

    def _show_entrypoint_example(self) -> None:
        """Show example of correct @entrypoint usage."""
        typer.echo("Please add the @entrypoint decorator to a function in this file:")
        typer.echo("Example:")
        typer.echo("  from gradient_adk import entrypoint")
        typer.echo("  ")
        typer.echo("  @entrypoint")
        typer.echo("  def main(query, context):")
        typer.echo("      return {'result': 'Hello World'}")
        typer.echo("  ")
        typer.echo(
            "  # The decorator automatically creates 'app' - no manual assignment needed!"
        )
        typer.echo("  # You can now run: uvicorn main:app")
        typer.echo("")
        typer.echo("Note: The entrypoint function must accept exactly 2 parameters")
