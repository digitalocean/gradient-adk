"""
Integration tests for the `gradient agent traces` CLI command.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


class TestADKAgentsTracesValidation:
    """Tests for traces command validation that don't require a deployed agent."""

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
    def test_traces_missing_config(self):
        """
        Test that traces fails with helpful error when .gradient/agent.yml is missing.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Testing traces without config in {temp_dir}")

            result = subprocess.run(
                ["gradient", "agent", "traces"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
            )

            assert result.returncode != 0, "Traces should have failed without config"

            combined_output = result.stdout + result.stderr
            # Should show helpful error about missing configuration
            assert "agent configuration not found" in combined_output.lower() or \
                   ".gradient/agent.yml" in combined_output.lower() or \
                   "gradient agent configure" in combined_output.lower(), \
                f"Expected helpful error about missing configuration, got: {combined_output}"

            logger.info("Correctly failed without configuration with helpful error message")

    @pytest.mark.cli
    def test_traces_missing_api_token(self, setup_valid_agent):
        """
        Test that traces fails with helpful error when DIGITALOCEAN_API_TOKEN is not set.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        logger.info(f"Testing traces without API token in {temp_path}")

        # Create a clean environment without the API token
        clean_env = {k: v for k, v in os.environ.items() if k != "DIGITALOCEAN_API_TOKEN"}

        result = subprocess.run(
            ["gradient", "agent", "traces"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env=clean_env,
        )

        assert result.returncode != 0, "Traces should have failed without API token"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["digitalocean_api_token", "api token", "token", "error"]
        ), f"Expected error about missing API token, got: {combined_output}"

        logger.info("Correctly failed without API token")

    @pytest.mark.cli
    def test_traces_non_deployed_agent(self, setup_valid_agent):
        """
        Test that traces fails with helpful error when agent is not deployed.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Configure with a non-existent agent name
        config_path = temp_path / ".gradient" / "agent.yml"
        config = {
            "agent_name": "nonexistent-agent-12345",
            "agent_environment": "main",
            "entrypoint_file": "main.py",
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        # Need a real API token to test the 404 response
        api_token = os.getenv("DIGITALOCEAN_API_TOKEN") or os.getenv("TEST_DIGITALOCEAN_API_TOKEN")

        if not api_token:
            pytest.skip("DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN required for this test")

        logger.info(f"Testing traces for non-deployed agent in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "traces"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": api_token},
        )

        assert result.returncode != 0, "Traces should have failed for non-deployed agent"

        combined_output = result.stdout + result.stderr
        # Should indicate agent not found or not deployed
        assert any(
            term in combined_output.lower()
            for term in ["not found", "not deployed", "deploy", "error"]
        ), f"Expected error about agent not found/deployed, got: {combined_output}"

        logger.info("Correctly failed for non-deployed agent with helpful error")


class TestADKAgentsTracesHappyPath:
    """Tests for traces command happy path - requires a deployed agent."""

    @pytest.mark.cli
    @pytest.mark.e2e
    def test_traces_happy_path(self):
        """
        Test the happy path for 'gradient agent traces'.
        Requires:
        - TEST_AGENT_WORKSPACE_NAME env var (name of a deployed agent workspace)
        - TEST_AGENT_DEPLOYMENT_NAME env var (name of the deployment, e.g., 'main')
        - DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN env var
        
        Note: This test will attempt to open a browser. In CI environments,
        it may fail if no display is available, but it should still succeed
        in making the API calls.
        """
        logger = logging.getLogger(__name__)

        # Get required environment variables
        agent_workspace_name = os.getenv("TEST_AGENT_WORKSPACE_NAME")
        agent_deployment_name = os.getenv("TEST_AGENT_DEPLOYMENT_NAME", "main")
        api_token = os.getenv("DIGITALOCEAN_API_TOKEN") or os.getenv("TEST_DIGITALOCEAN_API_TOKEN")

        if not agent_workspace_name:
            pytest.skip("TEST_AGENT_WORKSPACE_NAME required for this test")
        if not api_token:
            pytest.skip("DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN required for this test")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a minimal agent structure pointing to the deployed agent
            main_py = temp_path / "main.py"
            main_py.write_text("""
from gradient_adk import entrypoint

@entrypoint
async def main(query, context):
    return {"result": "test"}
""")

            # Create .gradient directory and config pointing to the deployed agent
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()

            config = {
                "agent_name": agent_workspace_name,
                "agent_environment": agent_deployment_name,
                "entrypoint_file": "main.py",
            }

            with open(gradient_dir / "agent.yml", "w") as f:
                yaml.safe_dump(config, f)

            logger.info(f"Testing traces for deployed agent {agent_workspace_name}/{agent_deployment_name}")

            result = subprocess.run(
                ["gradient", "agent", "traces"],
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": api_token},
            )

            # The command may succeed (return code 0) if it can open the browser
            # or it may fail if no display is available (in CI)
            combined_output = result.stdout + result.stderr
            
            if result.returncode == 0:
                # Should show success message
                assert "opened" in combined_output.lower() or "traces" in combined_output.lower(), \
                    f"Expected success message about traces UI, got: {combined_output}"
                logger.info("Successfully opened traces UI for deployed agent")
            else:
                # If it failed, it should be due to browser issues, not API/config issues
                # (i.e., the command got past validation and API calls)
                assert "opening" in combined_output.lower() or "traces ui" in combined_output.lower(), \
                    f"Command failed before reaching browser step: {combined_output}"
                logger.info("Traces command reached browser step (may have failed to open browser in headless env)")


class TestADKAgentsTracesHelp:
    """Tests for traces command help."""

    @pytest.mark.cli
    def test_traces_help(self):
        """
        Test that 'gradient agent traces --help' shows proper usage.
        """
        logger = logging.getLogger(__name__)

        result = subprocess.run(
            ["gradient", "agent", "traces", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, "Help command should succeed"

        combined_output = result.stdout + result.stderr
        # Check for expected options in help
        assert "--api-token" in combined_output, "Should show --api-token option"
        assert "traces" in combined_output.lower(), "Should describe traces functionality"

        logger.info("Help output is correct")