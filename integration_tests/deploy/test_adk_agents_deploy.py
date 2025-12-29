"""
Integration tests for the `gradient agent deploy` CLI command.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class TestADKAgentsDeployValidation:
    """Tests for deploy validation that don't require API access."""

    @pytest.fixture
    def echo_agent_dir(self):
        """Get the path to the echo agent directory."""
        return Path(__file__).parent.parent / "example_agents" / "echo_agent"

    @pytest.fixture
    def setup_valid_agent(self, echo_agent_dir):
        """
        Setup a temporary directory with a valid agent structure.
        Yields the temp directory path and cleans up after.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy the echo agent main.py
            shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

            # Create .gradient directory and config
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()

            config = {
                "agent_name": "test-echo-agent",
                "agent_environment": "main",
                "entrypoint_file": "main.py",
            }

            with open(gradient_dir / "agent.yml", "w") as f:
                yaml.safe_dump(config, f)

            # Create requirements.txt
            requirements_path = temp_path / "requirements.txt"
            requirements_path.write_text("gradient-adk\n")

            yield temp_path

    @pytest.mark.cli
    def test_deploy_missing_config(self):
        """
        Test that deploy fails with helpful error when .gradient/agent.yml is missing.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a minimal main.py
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create requirements.txt
            requirements_path = temp_path / "requirements.txt"
            requirements_path.write_text("gradient-adk\n")

            # Don't create .gradient/agent.yml

            logger.info(f"Testing deploy without config in {temp_dir}")

            result = subprocess.run(
                ["gradient", "agent", "deploy", "--skip-validation"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
            )

            assert result.returncode != 0, "Deploy should have failed without config"

            combined_output = result.stdout + result.stderr
            # Should show helpful error about missing configuration
            assert "agent configuration not found" in combined_output.lower() or \
                   ".gradient/agent.yml" in combined_output.lower() or \
                   "gradient agent configure" in combined_output.lower(), \
                f"Expected helpful error about missing configuration, got: {combined_output}"

            logger.info("Correctly failed without configuration with helpful error message")

    @pytest.mark.cli
    def test_deploy_missing_entrypoint_file(self, setup_valid_agent):
        """
        Test that deploy fails when the entrypoint file doesn't exist.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Remove the main.py file
        (temp_path / "main.py").unlink()

        logger.info(f"Testing deploy with missing entrypoint in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Deploy should have failed with missing entrypoint"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["entrypoint", "not found", "not exist", "main.py", "error"]
        ), f"Expected error about missing entrypoint, got: {combined_output}"

        logger.info("Correctly failed with missing entrypoint file")

    @pytest.mark.cli
    def test_deploy_missing_requirements_txt(self, echo_agent_dir):
        """
        Test that deploy fails when requirements.txt is missing.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy the echo agent main.py
            shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

            # Create .gradient directory and config
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()

            config = {
                "agent_name": "test-echo-agent",
                "agent_environment": "main",
                "entrypoint_file": "main.py",
            }

            with open(gradient_dir / "agent.yml", "w") as f:
                yaml.safe_dump(config, f)

            # Don't create requirements.txt

            logger.info(f"Testing deploy without requirements.txt in {temp_dir}")

            result = subprocess.run(
                ["gradient", "agent", "deploy"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
            )

            assert result.returncode != 0, "Deploy should have failed without requirements.txt"

            combined_output = result.stdout + result.stderr
            assert any(
                term in combined_output.lower()
                for term in ["requirements", "requirements.txt", "error"]
            ), f"Expected error about missing requirements.txt, got: {combined_output}"

            logger.info("Correctly failed without requirements.txt")

    @pytest.mark.cli
    def test_deploy_entrypoint_without_decorator(self, setup_valid_agent):
        """
        Test that deploy fails when entrypoint file doesn't have @entrypoint decorator.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Overwrite main.py with code that doesn't have @entrypoint
        main_py = temp_path / "main.py"
        main_py.write_text("""
def main(query, context):
    return {"result": "no decorator"}
""")

        logger.info(f"Testing deploy with invalid entrypoint (no decorator) in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=120,  # Validation can take time
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Deploy should have failed without @entrypoint decorator"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["@entrypoint", "decorator", "validation", "error"]
        ), f"Expected error about missing @entrypoint decorator, got: {combined_output}"

        logger.info("Correctly failed without @entrypoint decorator")

    @pytest.mark.cli
    def test_deploy_invalid_agent_workspace_name(self, setup_valid_agent):
        """
        Test that deploy fails with invalid agent workspace name (special characters).
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Update config with invalid agent name
        config_path = temp_path / ".gradient" / "agent.yml"
        config = {
            "agent_name": "test agent with spaces!",  # Invalid: spaces and special chars
            "agent_environment": "main",
            "entrypoint_file": "main.py",
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        logger.info(f"Testing deploy with invalid agent workspace name in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--skip-validation"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Deploy should have failed with invalid agent name"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["invalid", "agent workspace name", "alphanumeric", "error"]
        ), f"Expected error about invalid agent name, got: {combined_output}"

        logger.info("Correctly failed with invalid agent workspace name")

    @pytest.mark.cli
    def test_deploy_invalid_deployment_name(self, setup_valid_agent):
        """
        Test that deploy fails with invalid deployment name (special characters).
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Update config with invalid deployment name
        config_path = temp_path / ".gradient" / "agent.yml"
        config = {
            "agent_name": "test-agent",
            "agent_environment": "main deployment!",  # Invalid: spaces and special chars
            "entrypoint_file": "main.py",
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        logger.info(f"Testing deploy with invalid deployment name in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--skip-validation"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Deploy should have failed with invalid deployment name"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["invalid", "deployment name", "alphanumeric", "error"]
        ), f"Expected error about invalid deployment name, got: {combined_output}"

        logger.info("Correctly failed with invalid deployment name")

    @pytest.mark.cli
    def test_deploy_missing_api_token(self, setup_valid_agent):
        """
        Test that deploy fails with helpful error when DIGITALOCEAN_API_TOKEN is not set.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        logger.info(f"Testing deploy without API token in {temp_path}")

        # Create a clean environment without the API token
        clean_env = {k: v for k, v in os.environ.items() if k != "DIGITALOCEAN_API_TOKEN"}

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--skip-validation"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env=clean_env,
        )

        assert result.returncode != 0, "Deploy should have failed without API token"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["digitalocean_api_token", "api token", "token", "error"]
        ), f"Expected error about missing API token, got: {combined_output}"

        logger.info("Correctly failed without API token")

    @pytest.mark.cli
    def test_deploy_skip_validation_flag(self, setup_valid_agent):
        """
        Test that --skip-validation flag works and skips the validation step.
        Note: This test will still fail at the API call stage, but should skip validation.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Overwrite main.py with code that doesn't have @entrypoint
        # This would normally fail validation
        main_py = temp_path / "main.py"
        main_py.write_text("""
def main(query, context):
    return {"result": "no decorator"}
""")

        logger.info(f"Testing deploy with --skip-validation in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--skip-validation"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        # The command will fail (likely at API call), but the output should show
        # that validation was skipped
        combined_output = result.stdout + result.stderr
        assert "skipping validation" in combined_output.lower(), \
            f"Expected message about skipping validation, got: {combined_output}"

        logger.info("Correctly skipped validation")

    @pytest.mark.cli
    def test_deploy_validation_import_error(self, setup_valid_agent):
        """
        Test that deploy fails with helpful error when entrypoint has import errors.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Overwrite main.py with code that has import errors
        main_py = temp_path / "main.py"
        main_py.write_text("""
from gradient_adk import entrypoint
from nonexistent_module import something  # This will fail to import

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

        logger.info(f"Testing deploy with import error in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Deploy should have failed with import error"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["import", "nonexistent_module", "module", "error"]
        ), f"Expected error about import failure, got: {combined_output}"

        logger.info("Correctly failed with import error")

    @pytest.mark.cli
    def test_deploy_validation_syntax_error(self, setup_valid_agent):
        """
        Test that deploy fails with helpful error when entrypoint has syntax errors.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Overwrite main.py with code that has syntax errors
        main_py = temp_path / "main.py"
        main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context)  # Missing colon!
    return {"result": "test"}
""")

        logger.info(f"Testing deploy with syntax error in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Deploy should have failed with syntax error"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["syntax", "error", "invalid"]
        ), f"Expected error about syntax error, got: {combined_output}"

        logger.info("Correctly failed with syntax error")


class TestADKAgentsDeployVerbose:
    """Tests for deploy verbose mode."""

    @pytest.fixture
    def echo_agent_dir(self):
        """Get the path to the echo agent directory."""
        return Path(__file__).parent.parent / "example_agents" / "echo_agent"

    @pytest.fixture
    def setup_valid_agent(self, echo_agent_dir):
        """
        Setup a temporary directory with a valid agent structure.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy the echo agent main.py
            shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

            # Create .gradient directory and config
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()

            config = {
                "agent_name": "test-echo-agent",
                "agent_environment": "main",
                "entrypoint_file": "main.py",
            }

            with open(gradient_dir / "agent.yml", "w") as f:
                yaml.safe_dump(config, f)

            # Create requirements.txt
            requirements_path = temp_path / "requirements.txt"
            requirements_path.write_text("gradient-adk\n")

            yield temp_path

    @pytest.mark.cli
    def test_deploy_verbose_flag(self, setup_valid_agent):
        """
        Test that --verbose flag produces more detailed output.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        logger.info(f"Testing deploy with --verbose in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--verbose"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        combined_output = result.stdout + result.stderr
        # Verbose mode should show "Verbose mode enabled"
        assert "verbose mode enabled" in combined_output.lower(), \
            f"Expected verbose mode message, got: {combined_output}"

        logger.info("Verbose flag working correctly")


class TestADKAgentsDeployHelp:
    """Tests for deploy command help and options."""

    @pytest.mark.cli
    def test_deploy_help(self):
        """
        Test that 'gradient agent deploy --help' shows proper usage.
        """
        logger = logging.getLogger(__name__)

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, "Help command should succeed"

        combined_output = strip_ansi_codes(result.stdout + result.stderr)
        # Check for expected options in help
        assert "--api-token" in combined_output, f"Should show --api-token option. Got: {combined_output}"
        assert "--verbose" in combined_output or "-v" in combined_output, "Should show --verbose option"
        assert "--skip-validation" in combined_output, "Should show --skip-validation option"

        logger.info("Help output is correct")