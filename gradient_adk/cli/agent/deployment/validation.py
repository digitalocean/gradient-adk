"""Pre-deployment validation for agent entrypoints."""

from __future__ import annotations
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import sys


class ValidationError(Exception):
    """Raised when agent validation fails."""

    pass


def validate_agent_entrypoint(
    source_dir: Path, entrypoint_file: str, verbose: bool = False
) -> None:
    """
    Validate that the agent can run successfully in a fresh environment.

    This creates a temporary virtual environment, installs dependencies,
    and attempts to start the agent to verify it works before deployment.

    Args:
        source_dir: Directory containing the agent source code
        entrypoint_file: Relative path to the entrypoint file (e.g., "main.py")
        verbose: Whether to print verbose validation output

    Raises:
        ValidationError: If validation fails
    """
    print(
        f"🔍 Validating agent can run before deployment... (skip this step with --skip-validation)"
    )
    if verbose:
        print(f"🔍 Validating agent before deployment...")
        print(f"   Source: {source_dir}")
        print(f"   Entrypoint: {entrypoint_file}")

    # Check file exists
    entrypoint_path = source_dir / entrypoint_file
    if not entrypoint_path.exists():
        raise ValidationError(
            f"Entrypoint file not found: {entrypoint_file}\n"
            f"Expected at: {entrypoint_path}"
        )

    # Check if requirements.txt exists
    requirements_path = source_dir / "requirements.txt"
    if not requirements_path.exists():
        raise ValidationError(
            f"No requirements.txt found in {source_dir}\n"
            f"A requirements.txt file is required for deployment.\n\n"
            f"Create a requirements.txt with at minimum:\n"
            f"  gradient-adk\n"
        )

    # Check for config file
    config_path = source_dir / ".gradient" / "agent.yml"
    if not config_path.exists():
        raise ValidationError(
            f"No agent configuration found at {config_path}\n"
            f"Run 'gradient agent configure' first to set up your agent."
        )

    # Create temporary directory for validation
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="gradient_validation_"))

        if verbose:
            print(f"\n� Created temporary validation environment: {temp_dir}")

        # Copy source files to temp directory
        if verbose:
            print(f"📋 Copying source files...")

        # Copy all files except common exclusions
        _copy_source_files(source_dir, temp_dir, verbose=verbose)

        # Create virtual environment
        venv_path = temp_dir / ".venv"
        if verbose:
            print(f"\n🔨 Creating virtual environment...")

        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise ValidationError(
                f"Failed to create virtual environment:\n{result.stderr}"
            )

        # Get pip path
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip"
            python_path = venv_path / "Scripts" / "python"
        else:
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"

        # Install requirements
        if verbose:
            print(f"📦 Installing dependencies from requirements.txt...")

        result = subprocess.run(
            [str(pip_path), "install", "-r", "requirements.txt"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for dependency installation
        )

        if result.returncode != 0:
            raise ValidationError(
                f"Failed to install dependencies:\n{result.stderr}\n\n"
                f"Fix your requirements.txt and try again."
            )

        if verbose:
            print(f"✅ Dependencies installed successfully")

        # Try to import the entrypoint and verify decorator
        if verbose:
            print(f"\n🔍 Verifying agent entrypoint has @entrypoint decorator...")

        # Check that the entrypoint file has the @entrypoint decorator
        # Convert file path to module path (e.g., "agents/my_agent.py" -> "agents.my_agent")
        module_path = entrypoint_file.replace(".py", "").replace("/", ".")

        check_script = f"""
import sys
import re
import os
sys.path.insert(0, '.')

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Read the entrypoint file
with open('{entrypoint_file}', 'r') as f:
    content = f.read()

# Check for @entrypoint decorator
if not re.search(r'^\\s*@entrypoint\\s*$', content, re.MULTILINE):
    print("ERROR: No @entrypoint decorator found")
    sys.exit(1)

# Try to import it
try:
    import {module_path}
    print('✅ Entrypoint imported successfully with @entrypoint decorator')
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

        result = subprocess.run(
            [str(python_path), "-c", check_script],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            error_output = result.stdout + result.stderr
            if "No @entrypoint decorator found" in error_output:
                raise ValidationError(
                    f"No @entrypoint decorator found in {entrypoint_file}\n\n"
                    f"Your agent must have a function decorated with @entrypoint.\n"
                    f"Example:\n\n"
                    f"  from gradient_adk import entrypoint\n\n"
                    f"  @entrypoint\n"
                    f"  async def my_agent(data, context):\n"
                    f"      return {{'result': 'hello'}}\n"
                )
            else:
                raise ValidationError(
                    f"Failed to import entrypoint:\n{error_output}\n\n"
                    f"The agent code has errors or missing dependencies."
                )

        if verbose:
            print(result.stdout.strip())

        if verbose:
            print(f"\n✅ Agent validation passed - ready to deploy!")

    except subprocess.TimeoutExpired:
        raise ValidationError(
            "Validation timed out - the agent may have infinite loops or be waiting for input.\n"
            "Ensure your agent can start up quickly without requiring user interaction."
        )
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Validation failed with unexpected error:\n"
            f"  {type(e).__name__}: {e}\n\n"
            f"Try running 'gradient agent run' locally to debug the issue."
        )
    finally:
        # Clean up temp directory
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                if verbose:
                    print(f"🧹 Cleaned up temporary files")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Warning: Failed to clean up {temp_dir}: {e}")


def _copy_source_files(source_dir: Path, dest_dir: Path, verbose: bool = False) -> None:
    """Copy source files to destination, excluding common patterns."""

    # Common exclusions
    exclude_patterns = {
        "__pycache__",
        "*.pyc",
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        "*.egg-info",
        "dist",
        "build",
    }

    def should_exclude(path: Path) -> bool:
        """Check if a path should be excluded."""
        for pattern in exclude_patterns:
            if pattern.startswith("*"):
                if path.name.endswith(pattern[1:]):
                    return True
            elif path.name == pattern:
                return True
        return False

    # Copy files and directories
    for item in source_dir.iterdir():
        if should_exclude(item):
            if verbose:
                print(f"   Skipping: {item.name}")
            continue

        dest_path = dest_dir / item.name

        try:
            if item.is_dir():
                shutil.copytree(
                    item,
                    dest_path,
                    ignore=lambda d, files: [
                        f for f in files if should_exclude(Path(d) / f)
                    ],
                )
            else:
                shutil.copy2(item, dest_path)

            if verbose:
                print(f"   Copied: {item.name}")
        except Exception as e:
            if verbose:
                print(f"   Warning: Failed to copy {item.name}: {e}")
