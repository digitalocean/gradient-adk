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
        assert "--output" in combined_output or "-o" in combined_output, "Should show --output option"

        logger.info("Help output is correct")


class TestADKAgentsDeployJsonOutput:
    """Tests for deploy command JSON output mode."""

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
    def test_deploy_json_output_missing_config(self):
        """
        Test that deploy with --output json returns valid JSON error when config is missing.
        """
        import json
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

            logger.info(f"Testing deploy --output json without config in {temp_dir}")

            result = subprocess.run(
                ["gradient", "agent", "deploy", "--output", "json", "--skip-validation"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
            )

            assert result.returncode != 0, "Deploy should have failed without config"

            # stderr should contain valid JSON error
            try:
                parsed = json.loads(result.stderr)
                assert parsed["status"] == "error", "JSON should have status: error"
                assert "error" in parsed, "JSON should have error field"
                assert "configuration" in parsed["error"].lower(), \
                    f"Error should mention configuration, got: {parsed['error']}"
                logger.info("JSON error output is valid")
            except json.JSONDecodeError as e:
                pytest.fail(f"stderr should be valid JSON, got: {result.stderr}")

    @pytest.mark.cli
    def test_deploy_json_output_invalid_agent_name(self, setup_valid_agent):
        """
        Test that deploy with --output json returns valid JSON error for invalid agent name.
        """
        import json
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Update config with invalid agent name
        config_path = temp_path / ".gradient" / "agent.yml"
        config = {
            "agent_name": "test agent with spaces!",  # Invalid
            "agent_environment": "main",
            "entrypoint_file": "main.py",
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        logger.info(f"Testing deploy --output json with invalid agent name in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--output", "json", "--skip-validation"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Deploy should have failed with invalid agent name"

        # stderr should contain valid JSON error
        try:
            parsed = json.loads(result.stderr)
            assert parsed["status"] == "error", "JSON should have status: error"
            assert "invalid" in parsed["error"].lower(), \
                f"Error should mention invalid, got: {parsed['error']}"
            logger.info("JSON error output is valid for invalid agent name")
        except json.JSONDecodeError as e:
            pytest.fail(f"stderr should be valid JSON, got: {result.stderr}")

    @pytest.mark.cli
    def test_deploy_json_output_missing_api_token(self, setup_valid_agent):
        """
        Test that deploy with --output json returns valid JSON error when API token is missing.
        """
        import json
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        logger.info(f"Testing deploy --output json without API token in {temp_path}")

        # Create a clean environment without the API token
        clean_env = {k: v for k, v in os.environ.items() if k != "DIGITALOCEAN_API_TOKEN"}

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--output", "json", "--skip-validation"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env=clean_env,
        )

        assert result.returncode != 0, "Deploy should have failed without API token"

        # stderr should contain valid JSON error
        try:
            parsed = json.loads(result.stderr)
            assert parsed["status"] == "error", "JSON should have status: error"
            assert "token" in parsed["error"].lower(), \
                f"Error should mention token, got: {parsed['error']}"
            logger.info("JSON error output is valid for missing API token")
        except json.JSONDecodeError as e:
            pytest.fail(f"stderr should be valid JSON, got: {result.stderr}")


class TestADKAgentsDeployPythonEnvironment:
    """Tests for Python environment detection during deploy."""

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
    def test_deploy_missing_dependency_file(self, echo_agent_dir):
        """
        Test that deploy fails when neither requirements.txt nor pyproject.toml exists.
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

            # Don't create requirements.txt or pyproject.toml

            logger.info(f"Testing deploy without dependency file in {temp_dir}")

            result = subprocess.run(
                ["gradient", "agent", "deploy", "--skip-validation"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
            )

            assert result.returncode != 0, "Deploy should have failed without dependency file"

            combined_output = result.stdout + result.stderr
            assert any(
                term in combined_output.lower()
                for term in ["no dependency file", "requirements.txt", "pyproject.toml"]
            ), f"Expected error about missing dependency file, got: {combined_output}"

            logger.info("Correctly failed without dependency file")

    @pytest.mark.cli
    def test_deploy_unsupported_python_version(self, setup_valid_agent):
        """
        Test that deploy fails when Python version is not supported (3.10-3.14).
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Create .python-version with unsupported version
        (temp_path / ".python-version").write_text("3.9\n")

        logger.info(f"Testing deploy with unsupported Python version in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--skip-validation"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Deploy should have failed with unsupported Python version"

        combined_output = result.stdout + result.stderr
        assert any(
            term in combined_output.lower()
            for term in ["not supported", "3.9", "supported versions"]
        ), f"Expected error about unsupported Python version, got: {combined_output}"

        logger.info("Correctly failed with unsupported Python version")

    @pytest.mark.cli
    def test_deploy_both_dependency_files_warning(self, setup_valid_agent, caplog):
        """
        Test that deploy warns when both requirements.txt and pyproject.toml exist.
        This test just verifies the warning appears, not full deployment success.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Both requirements.txt (already exists from fixture) and pyproject.toml
        (temp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')

        logger.info(f"Testing deploy with both dependency files in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--skip-validation", "--verbose"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        combined_output = result.stdout + result.stderr
        # Should show warning about both files (may fail at API call, but warning should appear)
        assert "both requirements.txt and pyproject.toml" in combined_output.lower() or \
               "using requirements.txt" in combined_output.lower(), \
            f"Expected warning about both dependency files, got: {combined_output}"

        logger.info("Correctly warned about both dependency files")

    @pytest.mark.cli
    def test_deploy_with_pyproject_toml_only(self, echo_agent_dir):
        """
        Test that deploy works with only pyproject.toml (no requirements.txt).
        This test verifies the environment detection works, not full deployment.
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

            # Create pyproject.toml instead of requirements.txt
            pyproject_content = '''[project]
name = "test-agent"
requires-python = ">=3.12"
dependencies = ["gradient-adk"]
'''
            (temp_path / "pyproject.toml").write_text(pyproject_content)

            logger.info(f"Testing deploy with pyproject.toml only in {temp_dir}")

            result = subprocess.run(
                ["gradient", "agent", "deploy", "--skip-validation", "--verbose"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
            )

            combined_output = result.stdout + result.stderr
            # Should not fail with "no dependency file" error
            assert "no dependency file found" not in combined_output.lower(), \
                f"Should have detected pyproject.toml, got: {combined_output}"

            logger.info("Correctly detected pyproject.toml as dependency file")

    @pytest.mark.cli
    def test_deploy_with_uv_lock(self, setup_valid_agent):
        """
        Test that deploy detects UV package manager from uv.lock file.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Create uv.lock file
        (temp_path / "uv.lock").write_text("# uv lock file\n")

        logger.info(f"Testing deploy with uv.lock in {temp_path}")

        result = subprocess.run(
            ["gradient", "agent", "deploy", "--skip-validation", "--verbose"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        combined_output = result.stdout + result.stderr
        # Should detect UV (debug message or no pip warning)
        # The test just verifies it doesn't crash and processes uv.lock
        assert "no dependency file found" not in combined_output.lower(), \
            f"Should have processed with uv.lock present, got: {combined_output}"

        logger.info("Correctly processed with uv.lock file")

    @pytest.mark.cli
    def test_deploy_with_tool_uv_config(self, echo_agent_dir):
        """
        Test that deploy detects UV package manager from [tool.uv] in pyproject.toml.
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

            # Create pyproject.toml with [tool.uv] section
            pyproject_content = '''[project]
name = "test-agent"
requires-python = ">=3.12"
dependencies = ["gradient-adk"]

[tool.uv]
dev-dependencies = []
'''
            (temp_path / "pyproject.toml").write_text(pyproject_content)

            logger.info(f"Testing deploy with [tool.uv] in pyproject.toml in {temp_dir}")

            result = subprocess.run(
                ["gradient", "agent", "deploy", "--skip-validation", "--verbose"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
            )

            combined_output = result.stdout + result.stderr
            # Should detect UV from [tool.uv]
            assert "no dependency file found" not in combined_output.lower(), \
                f"Should have processed with [tool.uv] config, got: {combined_output}"

            logger.info("Correctly processed with [tool.uv] config")

    @pytest.mark.cli
    def test_deploy_json_output_python_env_error(self, echo_agent_dir):
        """
        Test that deploy with --output json returns valid JSON for Python env errors.
        """
        import json
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

            # Don't create any dependency file

            logger.info(f"Testing deploy --output json without dependency file in {temp_dir}")

            result = subprocess.run(
                ["gradient", "agent", "deploy", "--output", "json", "--skip-validation"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
            )

            assert result.returncode != 0, "Deploy should have failed"

            # stderr should contain valid JSON error
            try:
                parsed = json.loads(result.stderr)
                assert parsed["status"] == "error", f"JSON should have status: error, got: {parsed}"
                assert "dependency" in parsed["error"].lower() or "requirements" in parsed["error"].lower(), \
                    f"Error should mention dependency file, got: {parsed['error']}"
                logger.info("JSON error output is valid for missing dependency file")
            except json.JSONDecodeError:
                pytest.fail(f"stderr should be valid JSON, got: {result.stderr}")


class TestADKAgentsDeployE2E:
    """End-to-end tests for successful agent deployment."""

    @pytest.fixture
    def echo_agent_dir(self):
        """Get the path to the echo agent directory."""
        return Path(__file__).parent.parent / "example_agents" / "echo_agent"

    @staticmethod
    async def _cleanup_agent_workspace(api_token: str, agent_name: str):
        """Helper to cleanup an agent workspace after test."""
        from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI

        logger = logging.getLogger(__name__)
        try:
            async with AsyncDigitalOceanGenAI(api_token=api_token) as client:
                await client.delete_agent_workspace(agent_name)
                logger.info(f"Cleaned up agent workspace: {agent_name}")
        except Exception as e:
            # Log but don't fail the test on cleanup errors
            logger.warning(f"Failed to cleanup agent workspace {agent_name}: {e}")

    @pytest.mark.cli
    @pytest.mark.e2e
    def test_deploy_success(self, echo_agent_dir):
        """
        Test successful agent deployment with text output.

        Requires:
        - DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN env var

        Note: This test will deploy an agent named 'e2e-test-deploy-{timestamp}'
        to avoid conflicts with other tests.
        """
        import asyncio
        import time
        logger = logging.getLogger(__name__)

        # Get API token
        api_token = os.getenv("DIGITALOCEAN_API_TOKEN") or os.getenv("TEST_DIGITALOCEAN_API_TOKEN")

        if not api_token:
            pytest.skip("DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN required for this test")

        # Use a unique agent name to avoid conflicts
        timestamp = int(time.time())
        agent_name = f"e2e-test-deploy-{timestamp}"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy the echo agent main.py
                shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

                # Create .gradient directory and config
                gradient_dir = temp_path / ".gradient"
                gradient_dir.mkdir()

                config = {
                    "agent_name": agent_name,
                    "agent_environment": "main",
                    "entrypoint_file": "main.py",
                }

                with open(gradient_dir / "agent.yml", "w") as f:
                    yaml.safe_dump(config, f)

                # Create requirements.txt
                requirements_path = temp_path / "requirements.txt"
                requirements_path.write_text("gradient-adk\n")

                logger.info(f"Testing successful deploy for agent {agent_name}")

                result = subprocess.run(
                    ["gradient", "agent", "deploy"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for deployment
                    env={**os.environ, "DIGITALOCEAN_API_TOKEN": api_token},
                )

                combined_output = result.stdout + result.stderr

                assert result.returncode == 0, f"Deploy should have succeeded. Output: {combined_output}"

                # Check for success indicators in output
                assert "deployed successfully" in combined_output.lower() or \
                       "agent deployed" in combined_output.lower(), \
                    f"Expected success message in output, got: {combined_output}"

                # Check that invoke URL is shown
                assert "invoke" in combined_output.lower() or \
                       "agents.do-ai.run" in combined_output, \
                    f"Expected invoke URL in output, got: {combined_output}"

                logger.info(f"Successfully deployed agent {agent_name}")
        finally:
            # Cleanup the deployed agent workspace
            asyncio.run(self._cleanup_agent_workspace(api_token, agent_name))

    @pytest.mark.cli
    @pytest.mark.e2e
    def test_deploy_json_output_success(self, echo_agent_dir):
        """
        Test successful agent deployment with JSON output.

        Requires:
        - DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN env var

        Note: This test will deploy an agent named 'e2e-test-json-{timestamp}'
        to avoid conflicts with other tests.
        """
        import asyncio
        import json
        import time
        logger = logging.getLogger(__name__)

        # Get API token
        api_token = os.getenv("DIGITALOCEAN_API_TOKEN") or os.getenv("TEST_DIGITALOCEAN_API_TOKEN")

        if not api_token:
            pytest.skip("DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN required for this test")

        # Use a unique agent name to avoid conflicts
        timestamp = int(time.time())
        agent_name = f"e2e-test-json-{timestamp}"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy the echo agent main.py
                shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

                # Create .gradient directory and config
                gradient_dir = temp_path / ".gradient"
                gradient_dir.mkdir()

                config = {
                    "agent_name": agent_name,
                    "agent_environment": "main",
                    "entrypoint_file": "main.py",
                }

                with open(gradient_dir / "agent.yml", "w") as f:
                    yaml.safe_dump(config, f)

                # Create requirements.txt
                requirements_path = temp_path / "requirements.txt"
                requirements_path.write_text("gradient-adk\n")

                logger.info(f"Testing successful deploy --output json for agent {agent_name}")

                result = subprocess.run(
                    ["gradient", "agent", "deploy", "--output", "json"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for deployment
                    env={**os.environ, "DIGITALOCEAN_API_TOKEN": api_token},
                )

                assert result.returncode == 0, f"Deploy should have succeeded. stderr: {result.stderr}"

                # stdout should contain valid JSON with deployment info
                try:
                    parsed = json.loads(result.stdout)
                    assert parsed["status"] == "success", f"JSON should have status: success, got: {parsed}"
                    assert "workspace_name" in parsed, f"JSON should have workspace_name, got: {parsed}"
                    assert "deployment_name" in parsed, f"JSON should have deployment_name, got: {parsed}"
                    assert "workspace_uuid" in parsed, f"JSON should have workspace_uuid, got: {parsed}"
                    assert "invoke_url" in parsed, f"JSON should have invoke_url, got: {parsed}"

                    # Verify the values match what we configured
                    assert parsed["workspace_name"] == agent_name, \
                        f"workspace_name should be {agent_name}, got: {parsed['workspace_name']}"
                    assert parsed["deployment_name"] == "main", \
                        f"deployment_name should be 'main', got: {parsed['deployment_name']}"

                    # Verify invoke_url format
                    assert "agents.do-ai.run" in parsed["invoke_url"], \
                        f"invoke_url should contain agents.do-ai.run, got: {parsed['invoke_url']}"
                    assert parsed["workspace_uuid"] in parsed["invoke_url"], \
                        f"invoke_url should contain workspace_uuid, got: {parsed['invoke_url']}"

                    logger.info(f"Successfully deployed agent {agent_name} with JSON output")
                    logger.info(f"  workspace_uuid: {parsed['workspace_uuid']}")
                    logger.info(f"  invoke_url: {parsed['invoke_url']}")

                except json.JSONDecodeError as e:
                    pytest.fail(f"stdout should be valid JSON, got: {result.stdout}")
        finally:
            # Cleanup the deployed agent workspace
            asyncio.run(self._cleanup_agent_workspace(api_token, agent_name))

    @pytest.mark.cli
    @pytest.mark.e2e
    def test_deploy_json_can_pipe_to_jq(self, echo_agent_dir):
        """
        Test that JSON output can be piped to jq to extract fields.

        Requires:
        - DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN env var
        - jq command available in PATH

        Note: This test will deploy an agent named 'e2e-test-jq-{timestamp}'
        to avoid conflicts with other tests.
        """
        import asyncio
        import json
        import time
        logger = logging.getLogger(__name__)

        # Check if jq is available
        jq_check = subprocess.run(["which", "jq"], capture_output=True)
        if jq_check.returncode != 0:
            pytest.skip("jq command not available")

        # Get API token
        api_token = os.getenv("DIGITALOCEAN_API_TOKEN") or os.getenv("TEST_DIGITALOCEAN_API_TOKEN")

        if not api_token:
            pytest.skip("DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN required for this test")

        # Use a unique agent name to avoid conflicts
        timestamp = int(time.time())
        agent_name = f"e2e-test-jq-{timestamp}"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy the echo agent main.py
                shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

                # Create .gradient directory and config
                gradient_dir = temp_path / ".gradient"
                gradient_dir.mkdir()

                config = {
                    "agent_name": agent_name,
                    "agent_environment": "main",
                    "entrypoint_file": "main.py",
                }

                with open(gradient_dir / "agent.yml", "w") as f:
                    yaml.safe_dump(config, f)

                # Create requirements.txt
                requirements_path = temp_path / "requirements.txt"
                requirements_path.write_text("gradient-adk\n")

                logger.info(f"Testing deploy --output json | jq for agent {agent_name}")

                # Deploy and capture the full JSON output first, then pipe to jq
                # This avoids potential issues with broken pipes
                result = subprocess.run(
                    ["gradient", "agent", "deploy", "--output", "json"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for deployment
                    env={**os.environ, "DIGITALOCEAN_API_TOKEN": api_token},
                )

                assert result.returncode == 0, f"Deploy should have succeeded. stderr: {result.stderr}"

                # Verify we can parse the JSON
                parsed = json.loads(result.stdout)
                assert parsed["status"] == "success", f"JSON should have status: success, got: {parsed}"
                assert "workspace_uuid" in parsed, f"JSON should have workspace_uuid, got: {parsed}"
                assert "invoke_url" in parsed, f"JSON should have invoke_url, got: {parsed}"

                # Now test that jq can extract the fields
                jq_result = subprocess.run(
                    ["jq", "-r", ".workspace_uuid"],
                    input=result.stdout,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                assert jq_result.returncode == 0, f"jq should have succeeded. stderr: {jq_result.stderr}"
                workspace_uuid = jq_result.stdout.strip()
                assert workspace_uuid == parsed["workspace_uuid"], \
                    f"jq extracted workspace_uuid should match: {workspace_uuid} vs {parsed['workspace_uuid']}"

                logger.info(f"Successfully extracted workspace_uuid via jq: {workspace_uuid}")

                # Test extracting invoke_url with jq
                jq_result2 = subprocess.run(
                    ["jq", "-r", ".invoke_url"],
                    input=result.stdout,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                assert jq_result2.returncode == 0, f"jq should have succeeded for invoke_url. stderr: {jq_result2.stderr}"
                invoke_url = jq_result2.stdout.strip()
                assert invoke_url == parsed["invoke_url"], \
                    f"jq extracted invoke_url should match: {invoke_url} vs {parsed['invoke_url']}"
                assert "agents.do-ai.run" in invoke_url, \
                    f"invoke_url should contain agents.do-ai.run, got: {invoke_url}"

                logger.info(f"Successfully extracted invoke_url via jq: {invoke_url}")
        finally:
            # Cleanup the deployed agent workspace
            asyncio.run(self._cleanup_agent_workspace(api_token, agent_name))

    @pytest.mark.cli
    @pytest.mark.e2e
    def test_deploy_with_pyproject_toml_e2e(self, echo_agent_dir):
        """
        Test successful agent deployment with pyproject.toml instead of requirements.txt.

        Requires:
        - DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN env var

        Note: This test will deploy an agent named 'e2e-test-pyproject-{timestamp}'
        to avoid conflicts with other tests.
        """
        import asyncio
        import time
        logger = logging.getLogger(__name__)

        # Get API token
        api_token = os.getenv("DIGITALOCEAN_API_TOKEN") or os.getenv("TEST_DIGITALOCEAN_API_TOKEN")

        if not api_token:
            pytest.skip("DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN required for this test")

        # Use a unique agent name to avoid conflicts
        timestamp = int(time.time())
        agent_name = f"e2e-test-pyproject-{timestamp}"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy the echo agent main.py
                shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

                # Create .gradient directory and config
                gradient_dir = temp_path / ".gradient"
                gradient_dir.mkdir()

                config = {
                    "agent_name": agent_name,
                    "agent_environment": "main",
                    "entrypoint_file": "main.py",
                }

                with open(gradient_dir / "agent.yml", "w") as f:
                    yaml.safe_dump(config, f)

                # Create pyproject.toml instead of requirements.txt
                pyproject_content = '''[project]
name = "e2e-test-agent"
requires-python = ">=3.12"
dependencies = ["gradient-adk"]
'''
                (temp_path / "pyproject.toml").write_text(pyproject_content)

                logger.info(f"Testing deploy with pyproject.toml for agent {agent_name}")

                result = subprocess.run(
                    ["gradient", "agent", "deploy"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for deployment
                    env={**os.environ, "DIGITALOCEAN_API_TOKEN": api_token},
                )

                combined_output = result.stdout + result.stderr

                assert result.returncode == 0, f"Deploy should have succeeded with pyproject.toml. Output: {combined_output}"

                # Check for success indicators in output
                assert "deployed successfully" in combined_output.lower() or \
                       "agent deployed" in combined_output.lower(), \
                    f"Expected success message in output, got: {combined_output}"

                logger.info(f"Successfully deployed agent {agent_name} with pyproject.toml")
        finally:
            # Cleanup the deployed agent workspace
            asyncio.run(self._cleanup_agent_workspace(api_token, agent_name))

    @pytest.mark.cli
    @pytest.mark.e2e
    def test_deploy_with_uv_environment_e2e(self, echo_agent_dir):
        """
        Test successful agent deployment with UV package manager configuration.

        Requires:
        - DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN env var

        Note: This test will deploy an agent named 'e2e-test-uv-{timestamp}'
        to avoid conflicts with other tests.
        """
        import asyncio
        import time
        logger = logging.getLogger(__name__)

        # Get API token
        api_token = os.getenv("DIGITALOCEAN_API_TOKEN") or os.getenv("TEST_DIGITALOCEAN_API_TOKEN")

        if not api_token:
            pytest.skip("DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN required for this test")

        # Use a unique agent name to avoid conflicts
        timestamp = int(time.time())
        agent_name = f"e2e-test-uv-{timestamp}"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy the echo agent main.py
                shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

                # Create .gradient directory and config
                gradient_dir = temp_path / ".gradient"
                gradient_dir.mkdir()

                config = {
                    "agent_name": agent_name,
                    "agent_environment": "main",
                    "entrypoint_file": "main.py",
                }

                with open(gradient_dir / "agent.yml", "w") as f:
                    yaml.safe_dump(config, f)

                # Create pyproject.toml with [tool.uv] section
                pyproject_content = '''[project]
name = "e2e-test-uv-agent"
requires-python = ">=3.12"
dependencies = ["gradient-adk"]

[tool.uv]
dev-dependencies = []
'''
                (temp_path / "pyproject.toml").write_text(pyproject_content)

                # Create uv.lock file to indicate UV package manager
                (temp_path / "uv.lock").write_text("# uv lock file placeholder\n")

                logger.info(f"Testing deploy with UV environment for agent {agent_name}")

                result = subprocess.run(
                    ["gradient", "agent", "deploy"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for deployment
                    env={**os.environ, "DIGITALOCEAN_API_TOKEN": api_token},
                )

                combined_output = result.stdout + result.stderr

                assert result.returncode == 0, f"Deploy should have succeeded with UV config. Output: {combined_output}"

                # Check for success indicators in output
                assert "deployed successfully" in combined_output.lower() or \
                       "agent deployed" in combined_output.lower(), \
                    f"Expected success message in output, got: {combined_output}"

                logger.info(f"Successfully deployed agent {agent_name} with UV environment")
        finally:
            # Cleanup the deployed agent workspace
            asyncio.run(self._cleanup_agent_workspace(api_token, agent_name))

    @pytest.mark.cli
    @pytest.mark.e2e
    def test_deploy_with_specific_python_version_e2e(self, echo_agent_dir):
        """
        Test successful agent deployment with specific Python version in .python-version file.

        Requires:
        - DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN env var

        Note: This test will deploy an agent named 'e2e-test-pyver-{timestamp}'
        to avoid conflicts with other tests.
        """
        import asyncio
        import time
        logger = logging.getLogger(__name__)

        # Get API token
        api_token = os.getenv("DIGITALOCEAN_API_TOKEN") or os.getenv("TEST_DIGITALOCEAN_API_TOKEN")

        if not api_token:
            pytest.skip("DIGITALOCEAN_API_TOKEN or TEST_DIGITALOCEAN_API_TOKEN required for this test")

        # Use a unique agent name to avoid conflicts
        timestamp = int(time.time())
        agent_name = f"e2e-test-pyver-{timestamp}"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy the echo agent main.py
                shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

                # Create .gradient directory and config
                gradient_dir = temp_path / ".gradient"
                gradient_dir.mkdir()

                config = {
                    "agent_name": agent_name,
                    "agent_environment": "main",
                    "entrypoint_file": "main.py",
                }

                with open(gradient_dir / "agent.yml", "w") as f:
                    yaml.safe_dump(config, f)

                # Create requirements.txt
                (temp_path / "requirements.txt").write_text("gradient-adk\n")

                # Create .python-version file with specific version
                (temp_path / ".python-version").write_text("3.12\n")

                logger.info(f"Testing deploy with .python-version for agent {agent_name}")

                result = subprocess.run(
                    ["gradient", "agent", "deploy"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout for deployment
                    env={**os.environ, "DIGITALOCEAN_API_TOKEN": api_token},
                )

                combined_output = result.stdout + result.stderr

                assert result.returncode == 0, f"Deploy should have succeeded with .python-version. Output: {combined_output}"

                # Check for success indicators in output
                assert "deployed successfully" in combined_output.lower() or \
                       "agent deployed" in combined_output.lower(), \
                    f"Expected success message in output, got: {combined_output}"

                logger.info(f"Successfully deployed agent {agent_name} with specific Python version")
        finally:
            # Cleanup the deployed agent workspace
            asyncio.run(self._cleanup_agent_workspace(api_token, agent_name))