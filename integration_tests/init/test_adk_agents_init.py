import logging
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


class TestADKAgentsInit:

    @pytest.mark.cli
    def test_adk_agent_init_happy_path(self):
        """
        Test the happy path for "gradient agent init" command.
        Verifies that:
        - The command exits with code 0
        - Project structure is created (main.py, agents/, tools/, .gradient/agent.yml, etc.)
        - Config file contains correct values
        """
        logger = logging.getLogger(__name__)

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_name = "test-agent-workspace"
            deployment_name = "main"

            logger.info(f"Running gradient agent init in {temp_dir}")

            # Run gradient agent init
            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "init",
                    "--agent-workspace-name",
                    workspace_name,
                    "--deployment-name",
                    deployment_name,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            # Assert command succeeded
            assert (
                result.returncode == 0
            ), f"Command failed with stderr: {result.stderr}"

            # Verify project structure was created
            temp_path = Path(temp_dir)

            # Check main.py exists
            main_py = temp_path / "main.py"
            assert main_py.exists(), "main.py was not created"
            main_content = main_py.read_text()
            assert (
                "@entrypoint" in main_content
            ), "main.py does not contain @entrypoint decorator"

            # Check agents/ folder exists
            agents_dir = temp_path / "agents"
            assert agents_dir.exists(), "agents/ directory was not created"
            assert agents_dir.is_dir(), "agents/ is not a directory"

            # Check tools/ folder exists
            tools_dir = temp_path / "tools"
            assert tools_dir.exists(), "tools/ directory was not created"
            assert tools_dir.is_dir(), "tools/ is not a directory"

            # Check .gradient/agent.yml exists and has correct values
            config_file = temp_path / ".gradient" / "agent.yml"
            assert config_file.exists(), ".gradient/agent.yml was not created"

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert (
                config["agent_name"] == workspace_name
            ), f"Expected agent_name '{workspace_name}', got '{config.get('agent_name')}'"
            assert (
                config["agent_environment"] == deployment_name
            ), f"Expected agent_environment '{deployment_name}', got '{config.get('agent_environment')}'"
            assert (
                config["entrypoint_file"] == "main.py"
            ), f"Expected entrypoint_file 'main.py', got '{config.get('entrypoint_file')}'"

            # Check .gitignore exists
            gitignore = temp_path / ".gitignore"
            assert gitignore.exists(), ".gitignore was not created"

            # Check requirements.txt exists
            requirements = temp_path / "requirements.txt"
            assert requirements.exists(), "requirements.txt was not created"
            req_content = requirements.read_text()
            assert (
                "gradient-adk" in req_content
            ), "requirements.txt does not contain gradient-adk"

            # Check .env exists
            env_file = temp_path / ".env"
            assert env_file.exists(), ".env was not created"

            logger.info("All assertions passed for happy path test")

    @pytest.mark.cli
    def test_adk_agent_init_bad_workspace_name(self):
        """
        Test "gradient agent init" with an invalid workspace name.
        Workspace names can only contain alphanumeric characters, hyphens, and underscores.
        Verifies that:
        - The command exits with a non-zero code
        - Error message mentions invalid workspace name
        """
        logger = logging.getLogger(__name__)

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Invalid workspace name contains spaces and special characters
            invalid_workspace_name = "bad workspace name!"
            deployment_name = "main"

            logger.info(
                f"Running gradient agent init with invalid workspace name in {temp_dir}"
            )

            # Run gradient agent init with invalid workspace name
            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "init",
                    "--agent-workspace-name",
                    invalid_workspace_name,
                    "--deployment-name",
                    deployment_name,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            # Assert command failed
            assert (
                result.returncode != 0
            ), f"Command should have failed but succeeded with stdout: {result.stdout}"

            # Check error message mentions invalid name
            combined_output = result.stdout + result.stderr
            assert (
                "invalid" in combined_output.lower()
                or "error" in combined_output.lower()
            ), f"Expected error message about invalid name, got: {combined_output}"

            logger.info("All assertions passed for bad workspace name test")

    @pytest.mark.cli
    def test_adk_agent_init_bad_deployment_name(self):
        """
        Test "gradient agent init" with an invalid deployment name.
        Deployment names can only contain alphanumeric characters, hyphens, and underscores.
        Verifies that:
        - The command exits with a non-zero code
        - Error message mentions invalid deployment name
        """
        logger = logging.getLogger(__name__)

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_name = "valid-workspace"
            # Invalid deployment name contains spaces
            invalid_deployment_name = "bad deployment"

            logger.info(
                f"Running gradient agent init with invalid deployment name in {temp_dir}"
            )

            # Run gradient agent init with invalid deployment name
            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "init",
                    "--agent-workspace-name",
                    workspace_name,
                    "--deployment-name",
                    invalid_deployment_name,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            # Assert command failed
            assert (
                result.returncode != 0
            ), f"Command should have failed but succeeded with stdout: {result.stdout}"

            # Check error message mentions invalid name
            combined_output = result.stdout + result.stderr
            assert (
                "invalid" in combined_output.lower()
                or "error" in combined_output.lower()
            ), f"Expected error message about invalid name, got: {combined_output}"

            logger.info("All assertions passed for bad deployment name test")

    @pytest.mark.cli
    def test_adk_agent_init_with_underscores_and_hyphens(self):
        """
        Test that workspace and deployment names with underscores and hyphens are accepted.
        """
        logger = logging.getLogger(__name__)

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_name = "my_test-agent_v2"
            deployment_name = "staging-env_01"

            logger.info(
                f"Running gradient agent init with underscores and hyphens in {temp_dir}"
            )

            # Run gradient agent init
            result = subprocess.run(
                [
                    "gradient",
                    "agent",
                    "init",
                    "--agent-workspace-name",
                    workspace_name,
                    "--deployment-name",
                    deployment_name,
                    "--no-interactive",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir,
            )

            # Assert command succeeded
            assert (
                result.returncode == 0
            ), f"Command failed with stderr: {result.stderr}"

            # Verify config file has correct values
            config_file = Path(temp_dir) / ".gradient" / "agent.yml"
            assert config_file.exists(), ".gradient/agent.yml was not created"

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config["agent_name"] == workspace_name
            assert config["agent_environment"] == deployment_name

            logger.info("All assertions passed for underscores and hyphens test")
