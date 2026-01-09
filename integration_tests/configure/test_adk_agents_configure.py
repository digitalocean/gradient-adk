"""
Integration tests for the `gradient agent configure` CLI command.
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class TestADKAgentsConfigure:
    """Tests for the gradient agent configure command."""

    @pytest.mark.cli
    def test_configure_happy_path(self):
        """
        Test the happy path for "gradient agent configure" command.
        Verifies that:
        - The command exits with code 0
        - Config file is created with correct values
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file (required by configure)
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            workspace_name = "test-agent-workspace"
            deployment_name = "production"
            entrypoint_file = "main.py"

            logger.info(f"Running gradient agent configure in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    workspace_name,
                    "--deployment-name",
                    deployment_name,
                    "--entrypoint-file",
                    entrypoint_file,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Verify config file was created
            config_file = temp_path / ".gradient" / "agent.yml"
            assert config_file.exists(), ".gradient/agent.yml was not created"

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config["agent_name"] == workspace_name, \
                f"Expected agent_name '{workspace_name}', got '{config.get('agent_name')}'"
            assert config["agent_environment"] == deployment_name, \
                f"Expected agent_environment '{deployment_name}', got '{config.get('agent_environment')}'"
            assert config["entrypoint_file"] == entrypoint_file, \
                f"Expected entrypoint_file '{entrypoint_file}', got '{config.get('entrypoint_file')}'"

            logger.info("All assertions passed for happy path test")

    @pytest.mark.cli
    def test_configure_updates_existing_config(self):
        """
        Test that configure updates an existing config file.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create initial config
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()
            config_file = gradient_dir / "agent.yml"

            initial_config = {
                "agent_name": "old-name",
                "agent_environment": "old-env",
                "entrypoint_file": "main.py",
            }
            with open(config_file, "w") as f:
                yaml.safe_dump(initial_config, f)

            # Run configure to update
            new_workspace_name = "new-workspace-name"
            new_deployment_name = "staging"

            logger.info(f"Running gradient agent configure to update config in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    new_workspace_name,
                    "--deployment-name",
                    new_deployment_name,
                    "--entrypoint-file",
                    "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Verify config was updated
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config["agent_name"] == new_workspace_name, \
                f"Expected agent_name '{new_workspace_name}', got '{config.get('agent_name')}'"
            assert config["agent_environment"] == new_deployment_name, \
                f"Expected agent_environment '{new_deployment_name}', got '{config.get('agent_environment')}'"

            logger.info("Config successfully updated")

    @pytest.mark.cli
    def test_configure_missing_entrypoint_file(self):
        """
        Test that configure fails when the entrypoint file doesn't exist.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Don't create the entrypoint file

            logger.info(f"Running gradient agent configure with missing entrypoint in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    "test-agent",
                    "--deployment-name",
                    "main",
                    "--entrypoint-file",
                    "nonexistent.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Command should have failed with missing entrypoint"

            combined_output = result.stdout + result.stderr
            assert any(
                term in combined_output.lower()
                for term in ["entrypoint", "not exist", "not found", "error"]
            ), f"Expected error about missing entrypoint, got: {combined_output}"

            logger.info("Correctly failed with missing entrypoint file")

    @pytest.mark.cli
    def test_configure_entrypoint_without_decorator(self):
        """
        Test that configure fails when entrypoint file doesn't have @entrypoint decorator.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file WITHOUT @entrypoint decorator
            main_py = temp_path / "main.py"
            main_py.write_text("""
def main(query, context):
    return {"result": "test"}
""")

            logger.info(f"Running gradient agent configure with invalid entrypoint in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    "test-agent",
                    "--deployment-name",
                    "main",
                    "--entrypoint-file",
                    "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Command should have failed without @entrypoint decorator"

            combined_output = result.stdout + result.stderr
            assert any(
                term in combined_output.lower()
                for term in ["@entrypoint", "decorator", "error"]
            ), f"Expected error about missing @entrypoint decorator, got: {combined_output}"

            logger.info("Correctly failed without @entrypoint decorator")

    @pytest.mark.cli
    def test_configure_invalid_workspace_name(self):
        """
        Test that configure fails with invalid workspace name (special characters).
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Invalid workspace name with spaces and special characters
            invalid_workspace_name = "bad workspace name!"

            logger.info(f"Running gradient agent configure with invalid workspace name in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    invalid_workspace_name,
                    "--deployment-name",
                    "main",
                    "--entrypoint-file",
                    "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Command should have failed with invalid workspace name"

            combined_output = result.stdout + result.stderr
            assert any(
                term in combined_output.lower()
                for term in ["invalid", "alphanumeric", "error"]
            ), f"Expected error about invalid workspace name, got: {combined_output}"

            logger.info("Correctly failed with invalid workspace name")

    @pytest.mark.cli
    def test_configure_invalid_deployment_name(self):
        """
        Test that configure fails with invalid deployment name (special characters).
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Invalid deployment name with spaces
            invalid_deployment_name = "bad deployment!"

            logger.info(f"Running gradient agent configure with invalid deployment name in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    "valid-workspace",
                    "--deployment-name",
                    invalid_deployment_name,
                    "--entrypoint-file",
                    "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Command should have failed with invalid deployment name"

            combined_output = result.stdout + result.stderr
            assert any(
                term in combined_output.lower()
                for term in ["invalid", "alphanumeric", "error"]
            ), f"Expected error about invalid deployment name, got: {combined_output}"

            logger.info("Correctly failed with invalid deployment name")

    @pytest.mark.cli
    def test_configure_with_underscores_and_hyphens(self):
        """
        Test that workspace and deployment names with underscores and hyphens are accepted.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            workspace_name = "my_test-agent_v2"
            deployment_name = "staging-env_01"

            logger.info(f"Running gradient agent configure with underscores and hyphens in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    workspace_name,
                    "--deployment-name",
                    deployment_name,
                    "--entrypoint-file",
                    "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Verify config file has correct values
            config_file = temp_path / ".gradient" / "agent.yml"
            assert config_file.exists(), ".gradient/agent.yml was not created"

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config["agent_name"] == workspace_name
            assert config["agent_environment"] == deployment_name

            logger.info("All assertions passed for underscores and hyphens test")

    @pytest.mark.cli
    def test_configure_custom_entrypoint_path(self):
        """
        Test that configure works with a custom entrypoint file path.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file in a subdirectory
            agents_dir = temp_path / "agents"
            agents_dir.mkdir()
            entrypoint_file = agents_dir / "my_agent.py"
            entrypoint_file.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def my_agent(query, context):
    return {"result": "custom agent"}
""")

            workspace_name = "test-agent"
            deployment_name = "main"
            entrypoint_path = "agents/my_agent.py"

            logger.info(f"Running gradient agent configure with custom entrypoint path in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    workspace_name,
                    "--deployment-name",
                    deployment_name,
                    "--entrypoint-file",
                    entrypoint_path,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Verify config file has correct entrypoint path
            config_file = temp_path / ".gradient" / "agent.yml"
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config["entrypoint_file"] == entrypoint_path, \
                f"Expected entrypoint_file '{entrypoint_path}', got '{config.get('entrypoint_file')}'"

            logger.info("Custom entrypoint path configured correctly")

    @pytest.mark.cli
    def test_configure_with_description_happy_path(self):
        """
        Test that configure works with a valid description.
        Verifies that:
        - The command exits with code 0
        - Description is saved correctly in config file
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            workspace_name = "test-agent"
            deployment_name = "main"
            entrypoint_file = "main.py"
            description = "This is a test agent that does amazing things."

            logger.info(f"Running gradient agent configure with description in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    workspace_name,
                    "--deployment-name",
                    deployment_name,
                    "--entrypoint-file",
                    entrypoint_file,
                    "--description",
                    description,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Verify config file was created with description
            config_file = temp_path / ".gradient" / "agent.yml"
            assert config_file.exists(), ".gradient/agent.yml was not created"

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config["agent_name"] == workspace_name
            assert config["agent_environment"] == deployment_name
            assert config["entrypoint_file"] == entrypoint_file
            assert config.get("description") == description, \
                f"Expected description '{description}', got '{config.get('description')}'"

            logger.info("Description configured correctly")

    @pytest.mark.cli
    def test_configure_description_too_long(self):
        """
        Test that configure fails when description exceeds 1000 characters.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create a description that exceeds 1000 characters
            long_description = "x" * 1001

            logger.info(f"Running gradient agent configure with long description in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    "test-agent",
                    "--deployment-name",
                    "main",
                    "--entrypoint-file",
                    "main.py",
                    "--description",
                    long_description,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Command should have failed with long description"

            combined_output = result.stdout + result.stderr
            assert any(
                term in combined_output.lower()
                for term in ["description", "1000", "length", "exceeds", "error"]
            ), f"Expected error about description length, got: {combined_output}"

            logger.info("Correctly failed with description too long")

    @pytest.mark.cli
    def test_configure_description_at_max_length(self):
        """
        Test that configure works with description at exactly 1000 characters.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create a description that is exactly 1000 characters
            max_description = "x" * 1000

            logger.info(f"Running gradient agent configure with max length description in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    "test-agent",
                    "--deployment-name",
                    "main",
                    "--entrypoint-file",
                    "main.py",
                    "--description",
                    max_description,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Verify config file has correct description
            config_file = temp_path / ".gradient" / "agent.yml"
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config.get("description") == max_description, \
                f"Expected 1000-char description, got {len(config.get('description', ''))} chars"

            logger.info("Max length description configured correctly")

    @pytest.mark.cli
    def test_configure_updates_existing_config_with_description(self):
        """
        Test that configure can add a description to an existing config.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create initial config without description
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()
            config_file = gradient_dir / "agent.yml"

            initial_config = {
                "agent_name": "test-agent",
                "agent_environment": "main",
                "entrypoint_file": "main.py",
            }
            with open(config_file, "w") as f:
                yaml.safe_dump(initial_config, f)

            # Verify initial config has no description
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            assert "description" not in config, "Initial config should not have description"

            # Run configure to add description
            new_description = "Adding a description to existing config."

            logger.info(f"Running gradient agent configure to add description in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    "test-agent",
                    "--deployment-name",
                    "main",
                    "--entrypoint-file",
                    "main.py",
                    "--description",
                    new_description,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Verify config was updated with description
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config.get("description") == new_description, \
                f"Expected description '{new_description}', got '{config.get('description')}'"

            logger.info("Description successfully added to existing config")

    @pytest.mark.cli
    def test_configure_without_description_does_not_add_description(self):
        """
        Test that configure without --description does not add a description field.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            logger.info(f"Running gradient agent configure without description in {temp_dir}")

            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    "test-agent",
                    "--deployment-name",
                    "main",
                    "--entrypoint-file",
                    "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Verify config file does NOT have description field
            config_file = temp_path / ".gradient" / "agent.yml"
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert "description" not in config, \
                f"Config should not have description field when not provided, but got: {config}"

            logger.info("Config correctly created without description field")

    @pytest.mark.cli
    def test_configure_requires_all_options_in_non_interactive(self):
        """
        Test that configure in non-interactive mode requires all options.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create an entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            logger.info(f"Running gradient agent configure without all required options in {temp_dir}")

            # Missing --entrypoint-file
            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "configure",
                    "--agent-workspace-name",
                    "test-agent",
                    "--deployment-name",
                    "main",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Command should have failed without all required options"

            combined_output = result.stdout + result.stderr
            assert any(
                term in combined_output.lower()
                for term in ["required", "error", "entrypoint"]
            ), f"Expected error about missing required options, got: {combined_output}"

            logger.info("Correctly failed without all required options")


class TestADKAgentsConfigureHelp:
    """Tests for configure command help."""

    @pytest.mark.cli
    def test_configure_help(self):
        """
        Test that 'gradient agent configure --help' shows proper usage.
        """
        logger = logging.getLogger(__name__)

        result = subprocess.run(
            ["gradient", "agent", "configure", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "NO_COLOR": "1"},  # Disable colors
        )

        assert result.returncode == 0, "Help command should succeed"

        # Strip ANSI codes and check for expected options
        combined_output = strip_ansi_codes(result.stdout + result.stderr)
        # Check for expected options in help
        assert "--agent-workspace-name" in combined_output, \
            f"Should show --agent-workspace-name option. Got: {combined_output}"
        assert "--deployment-name" in combined_output, "Should show --deployment-name option"
        assert "--entrypoint-file" in combined_output, "Should show --entrypoint-file option"
        assert "--description" in combined_output, "Should show --description option"
        assert "--interactive" in combined_output or "--no-interactive" in combined_output, \
            "Should show --interactive option"

        logger.info("Help output is correct")
