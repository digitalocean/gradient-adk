"""Tests for agent deployment validation."""

import pytest
from pathlib import Path
import tempfile
import shutil

from gradient_adk.cli.agent.deployment.validation import (
    validate_agent_entrypoint,
    ValidationError,
)

# Mark slow integration tests
slow = pytest.mark.slow


@pytest.fixture
def temp_agent_dir():
    """Create a temporary directory for test agents."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_agent_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_validation_fails_when_entrypoint_missing(temp_agent_dir):
    """Test that validation fails when entrypoint file doesn't exist."""
    # Create minimal structure without entrypoint
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\n")

    with pytest.raises(ValidationError) as exc_info:
        validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)

    assert "Entrypoint file not found" in str(exc_info.value)


def test_validation_fails_when_dependency_file_missing(temp_agent_dir):
    """Test that validation fails when no dependency file exists."""
    # Create minimal structure without requirements.txt or pyproject.toml
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "main.py").write_text("# placeholder\n")

    with pytest.raises(ValidationError) as exc_info:
        validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)

    assert "No requirements.txt or pyproject.toml found" in str(exc_info.value)


def test_validation_fails_when_config_missing(temp_agent_dir):
    """Test that validation fails when agent config is missing."""
    # Create structure without config
    (temp_agent_dir / "main.py").write_text("# placeholder\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\n")

    with pytest.raises(ValidationError) as exc_info:
        validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)

    assert "No agent configuration found" in str(exc_info.value)


def test_validation_fails_when_no_entrypoint_decorator(temp_agent_dir):
    """Test that validation fails when @entrypoint decorator is missing."""
    # Create full structure but without @entrypoint decorator
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\n")
    (temp_agent_dir / "main.py").write_text(
        """
def my_agent(data, context):
    return {"result": "hello"}
"""
    )

    with pytest.raises(ValidationError) as exc_info:
        validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)

    assert "No @entrypoint decorator found" in str(exc_info.value)


@slow
def test_validation_passes_with_valid_agent(temp_agent_dir):
    """Test that validation passes with a properly structured agent."""
    # Create full valid structure
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\n")
    (temp_agent_dir / "main.py").write_text(
        """
from gradient_adk import entrypoint

@entrypoint
async def my_agent(data, context):
    return {"result": "hello"}
"""
    )

    # Should not raise
    validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)


@slow
def test_validation_passes_with_env_file(temp_agent_dir):
    """Test that validation loads .env file during validation."""
    # Create structure with .env file
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\npython-dotenv\n")
    (temp_agent_dir / ".env").write_text("TEST_VAR=test_value\n")
    (temp_agent_dir / "main.py").write_text(
        """
import os
from gradient_adk import entrypoint

# This would fail if .env wasn't loaded
assert os.getenv("TEST_VAR") == "test_value"

@entrypoint
async def my_agent(data, context):
    return {"result": os.getenv("TEST_VAR")}
"""
    )

    # Should not raise - .env should be loaded
    validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)


def test_validation_fails_with_import_error(temp_agent_dir):
    """Test that validation fails when agent has import errors."""
    # Create structure with bad import
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\n")
    (temp_agent_dir / "main.py").write_text(
        """
from gradient_adk import entrypoint
import nonexistent_module  # This will fail

@entrypoint
async def my_agent(data, context):
    return {"result": "hello"}
"""
    )

    with pytest.raises(ValidationError) as exc_info:
        validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)

    assert "Failed to import entrypoint" in str(exc_info.value)


def test_validation_fails_with_syntax_error(temp_agent_dir):
    """Test that validation fails when agent has syntax errors."""
    # Create structure with syntax error
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\n")
    (temp_agent_dir / "main.py").write_text(
        """
from gradient_adk import entrypoint

@entrypoint
async def my_agent(data, context)  # Missing colon
    return {"result": "hello"}
"""
    )

    with pytest.raises(ValidationError) as exc_info:
        validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)

    assert "Failed to import entrypoint" in str(exc_info.value)


@slow
def test_validation_with_subdirectory_entrypoint(temp_agent_dir):
    """Test that validation works with entrypoint in a subdirectory."""
    # Create structure with entrypoint in subdirectory
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\n")
    (temp_agent_dir / "agents").mkdir()
    (temp_agent_dir / "agents" / "__init__.py").write_text("")
    (temp_agent_dir / "agents" / "my_agent.py").write_text(
        """
from gradient_adk import entrypoint

@entrypoint
async def my_agent(data, context):
    return {"result": "hello"}
"""
    )

    # Should validate successfully
    validate_agent_entrypoint(temp_agent_dir, "agents/my_agent.py", verbose=False)


@slow
def test_validation_excludes_common_patterns(temp_agent_dir):
    """Test that validation doesn't copy excluded files but keeps .env."""
    # Create structure with files that should be excluded
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\n")
    (temp_agent_dir / "main.py").write_text(
        """
from gradient_adk import entrypoint

@entrypoint
async def my_agent(data, context):
    return {"result": "hello"}
"""
    )

    # Create files/dirs that should be excluded
    (temp_agent_dir / "__pycache__").mkdir()
    (temp_agent_dir / "__pycache__" / "test.pyc").write_text("compiled")
    (temp_agent_dir / ".git").mkdir()
    (temp_agent_dir / ".git" / "config").write_text("git config")
    (temp_agent_dir / "venv").mkdir()
    (temp_agent_dir / "venv" / "bin").mkdir()
    # .env should NOT be excluded - it's needed for validation
    (temp_agent_dir / ".env").write_text("SECRET=value")

    # Validation should pass and not fail due to excluded files
    validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)


@slow
def test_validation_with_dependencies(temp_agent_dir):
    """Test that validation installs and uses dependencies from requirements.txt."""
    # Create structure that uses an external library
    (temp_agent_dir / ".gradient").mkdir()
    (temp_agent_dir / ".gradient" / "agent.yml").write_text("workspace_name: test\n")
    # Include a real package that can be installed quickly
    (temp_agent_dir / "requirements.txt").write_text("gradient-adk\nsix\n")
    (temp_agent_dir / "main.py").write_text(
        """
from gradient_adk import entrypoint
import six  # External dependency

@entrypoint
async def my_agent(data, context):
    return {"result": six.text_type("hello")}
"""
    )

    # Should install six and validate successfully
    validate_agent_entrypoint(temp_agent_dir, "main.py", verbose=False)