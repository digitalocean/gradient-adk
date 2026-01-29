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
            # New format uses deployments section
            assert "deployments" in config, "Config should have deployments section"
            assert deployment_name in config["deployments"], \
                f"Expected deployment '{deployment_name}' in deployments"
            assert config["deployments"][deployment_name]["entrypoint_file"] == entrypoint_file, \
                f"Expected entrypoint_file '{entrypoint_file}', got '{config['deployments'][deployment_name].get('entrypoint_file')}'"

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
            # New format uses deployments section
            assert "deployments" in config, "Config should have deployments section"
            assert new_deployment_name in config["deployments"], \
                f"Expected deployment '{new_deployment_name}' in deployments"

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
            # New format uses deployments section
            assert "deployments" in config, "Config should have deployments section"
            assert deployment_name in config["deployments"], \
                f"Expected deployment '{deployment_name}' in deployments"

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

            # New format uses deployments section
            assert "deployments" in config, "Config should have deployments section"
            assert deployment_name in config["deployments"], \
                f"Expected deployment '{deployment_name}' in deployments"
            assert config["deployments"][deployment_name]["entrypoint_file"] == entrypoint_path, \
                f"Expected entrypoint_file '{entrypoint_path}', got '{config['deployments'][deployment_name].get('entrypoint_file')}'"

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
            # New format uses deployments section
            assert "deployments" in config, "Config should have deployments section"
            assert deployment_name in config["deployments"], \
                f"Expected deployment '{deployment_name}' in deployments"
            assert config["deployments"][deployment_name]["entrypoint_file"] == entrypoint_file, \
                f"Expected entrypoint_file '{entrypoint_file}'"
            assert config["deployments"][deployment_name].get("description") == description, \
                f"Expected description '{description}', got '{config['deployments'][deployment_name].get('description')}'"

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

            # New format uses deployments section
            assert "deployments" in config, "Config should have deployments section"
            assert "main" in config["deployments"], "Expected deployment 'main' in deployments"
            assert config["deployments"]["main"].get("description") == max_description, \
                f"Expected 1000-char description, got {len(config['deployments']['main'].get('description', ''))} chars"

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

            # Verify config was updated with description (now in new format)
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # New format uses deployments section
            assert "deployments" in config, "Config should have deployments section"
            assert "main" in config["deployments"], "Expected deployment 'main' in deployments"
            assert config["deployments"]["main"].get("description") == new_description, \
                f"Expected description '{new_description}', got '{config['deployments']['main'].get('description')}'"

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

            # New format uses deployments section
            assert "deployments" in config, "Config should have deployments section"
            assert "main" in config["deployments"], "Expected deployment 'main' in deployments"
            assert "description" not in config["deployments"]["main"], \
                f"Deployment should not have description field when not provided, but got: {config['deployments']['main']}"

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


class TestADKAgentsMultiDeployment:
    """Tests for multi-deployment support."""

    @pytest.mark.cli
    def test_add_deployment_to_existing_config(self):
        """
        Test that --add-deployment adds a new deployment to existing config.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create entrypoint files
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "prod"}
""")
            staging_py = temp_path / "staging.py"
            staging_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def staging(query, context):
    return {"result": "staging"}
""")

            # Create initial config with one deployment
            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--agent-workspace-name", "test-agent",
                    "--deployment-name", "prod",
                    "--entrypoint-file", "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0, f"Initial configure failed: {result.stderr}"

            # Add a second deployment
            logger.info("Adding staging deployment")
            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--add-deployment", "staging",
                    "--entrypoint-file", "staging.py",
                    "--description", "Staging environment",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0, f"Add deployment failed: {result.stderr}"

            # Verify both deployments exist
            config_file = temp_path / ".gradient" / "agent.yml"
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert "deployments" in config, "Config should have deployments section"
            assert "prod" in config["deployments"], "Should have prod deployment"
            assert "staging" in config["deployments"], "Should have staging deployment"
            assert config["deployments"]["staging"]["entrypoint_file"] == "staging.py"
            assert config["deployments"]["staging"]["description"] == "Staging environment"

            logger.info("Successfully added second deployment")

    @pytest.mark.cli
    def test_add_deployment_requires_entrypoint(self):
        """
        Test that --add-deployment requires --entrypoint-file.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create initial config
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--agent-workspace-name", "test-agent",
                    "--deployment-name", "prod",
                    "--entrypoint-file", "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0

            # Try to add deployment without entrypoint
            logger.info("Trying to add deployment without entrypoint")
            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--add-deployment", "staging",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Should fail without --entrypoint-file"
            combined_output = result.stdout + result.stderr
            assert "entrypoint" in combined_output.lower(), \
                f"Should mention entrypoint requirement, got: {combined_output}"

            logger.info("Correctly failed without entrypoint")

    @pytest.mark.cli
    def test_deploy_auto_selects_single_deployment(self):
        """
        Test that deploy auto-selects when only one deployment exists (new format).
        This test verifies the CLI behavior without actually deploying.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create requirements.txt
            (temp_path / "requirements.txt").write_text("gradient-adk\n")

            # Configure with new format (single deployment)
            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--agent-workspace-name", "test-agent",
                    "--deployment-name", "main",
                    "--entrypoint-file", "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0, f"Configure failed: {result.stderr}"

            # Verify config has new format with deployments section
            config_file = temp_path / ".gradient" / "agent.yml"
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            assert "deployments" in config, "Should use new deployments format"

            # Try deploy without --deployment-name (should work with single deployment)
            # Note: This will fail due to missing API token, but should NOT fail
            # due to missing deployment name
            logger.info("Testing deploy without --deployment-name")
            result = subprocess.run(
                [
                    "gradient", "agent", "deploy",
                    "--skip-validation",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            combined_output = result.stdout + result.stderr
            # Should NOT complain about multiple deployments
            assert "multiple deployments" not in combined_output.lower(), \
                f"Should auto-select single deployment, got: {combined_output}"

            logger.info("Deploy correctly auto-selected single deployment")

    @pytest.mark.cli
    def test_deploy_requires_deployment_name_for_multiple(self):
        """
        Test that deploy requires --deployment-name when multiple deployments exist.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create entrypoint files
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "prod"}
""")
            staging_py = temp_path / "staging.py"
            staging_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def staging(query, context):
    return {"result": "staging"}
""")

            # Create requirements.txt
            (temp_path / "requirements.txt").write_text("gradient-adk\n")

            # Configure first deployment
            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--agent-workspace-name", "test-agent",
                    "--deployment-name", "prod",
                    "--entrypoint-file", "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0

            # Add second deployment
            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--add-deployment", "staging",
                    "--entrypoint-file", "staging.py",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0

            # Try deploy without --deployment-name (should fail)
            logger.info("Testing deploy without --deployment-name (multiple deployments)")
            result = subprocess.run(
                [
                    "gradient", "agent", "deploy",
                    "--skip-validation",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Should fail without --deployment-name"
            combined_output = result.stdout + result.stderr
            assert "multiple deployments" in combined_output.lower(), \
                f"Should mention multiple deployments, got: {combined_output}"
            assert "prod" in combined_output, "Should list 'prod' deployment"
            assert "staging" in combined_output, "Should list 'staging' deployment"

            logger.info("Deploy correctly required --deployment-name")

    @pytest.mark.cli
    def test_deploy_with_deployment_name_selects_correct_one(self):
        """
        Test that --deployment-name selects the correct deployment.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create entrypoint files
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "prod"}
""")
            staging_py = temp_path / "staging.py"
            staging_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def staging(query, context):
    return {"result": "staging"}
""")

            # Create requirements.txt
            (temp_path / "requirements.txt").write_text("gradient-adk\n")

            # Configure both deployments
            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--agent-workspace-name", "test-agent",
                    "--deployment-name", "prod",
                    "--entrypoint-file", "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0

            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--add-deployment", "staging",
                    "--entrypoint-file", "staging.py",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0

            # Try deploy with --deployment-name (should work, but fail on API token)
            logger.info("Testing deploy with --deployment-name=staging")
            result = subprocess.run(
                [
                    "gradient", "agent", "deploy",
                    "--deployment-name", "staging",
                    "--skip-validation",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            combined_output = result.stdout + result.stderr
            # Should NOT complain about multiple deployments or missing deployment name
            assert "multiple deployments" not in combined_output.lower(), \
                f"Should not mention multiple deployments when name specified, got: {combined_output}"

            logger.info("Deploy correctly selected specified deployment")

    @pytest.mark.cli
    def test_deploy_with_invalid_deployment_name(self):
        """
        Test that deploy fails gracefully with non-existent deployment name.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create requirements.txt
            (temp_path / "requirements.txt").write_text("gradient-adk\n")

            # Configure with one deployment
            result = subprocess.run(
                [
                    "gradient", "agent", "configure",
                    "--agent-workspace-name", "test-agent",
                    "--deployment-name", "prod",
                    "--entrypoint-file", "main.py",
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )
            assert result.returncode == 0

            # Try deploy with non-existent deployment name
            logger.info("Testing deploy with invalid deployment name")
            result = subprocess.run(
                [
                    "gradient", "agent", "deploy",
                    "--deployment-name", "nonexistent",
                    "--skip-validation",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            assert result.returncode != 0, "Should fail with non-existent deployment"
            combined_output = result.stdout + result.stderr
            assert "not found" in combined_output.lower(), \
                f"Should mention deployment not found, got: {combined_output}"

            logger.info("Deploy correctly failed with invalid deployment name")

    @pytest.mark.cli
    def test_deploy_backwards_compat_old_format(self):
        """
        Test that deploy still works with old config format (no deployments section).
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create entrypoint file
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create requirements.txt
            (temp_path / "requirements.txt").write_text("gradient-adk\n")

            # Create OLD format config manually
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()
            config_file = gradient_dir / "agent.yml"

            old_config = {
                "agent_name": "test-agent",
                "agent_environment": "main",
                "entrypoint_file": "main.py",
            }
            with open(config_file, "w") as f:
                yaml.safe_dump(old_config, f)

            # Try deploy (should work with old format, fail on API token)
            logger.info("Testing deploy with old config format")
            result = subprocess.run(
                [
                    "gradient", "agent", "deploy",
                    "--skip-validation",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            combined_output = result.stdout + result.stderr
            # Should NOT complain about configuration format
            assert "configuration not found" not in combined_output.lower(), \
                f"Should read old format config, got: {combined_output}"
            # Should NOT ask for deployment name
            assert "multiple deployments" not in combined_output.lower(), \
                f"Old format should work without --deployment-name, got: {combined_output}"

            logger.info("Deploy correctly handled old config format")
