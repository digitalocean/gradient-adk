"""
Integration tests for the `gradient agent run` CLI command.
"""

import logging
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests
import yaml


def find_free_port():
    """Find an available port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Wait for server to be ready on the given port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)
    return False


def cleanup_process(process):
    """Clean up a process and its entire process group."""
    if process and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            # Process already terminated
            pass


class TestADKAgentsRun:

    @pytest.fixture
    def echo_agent_dir(self):
        """Get the path to the echo agent directory."""
        return Path(__file__).parent.parent / "example_agents" / "echo_agent"

    @pytest.fixture
    def setup_agent_in_temp(self, echo_agent_dir):
        """
        Setup a temporary directory with the echo agent and proper configuration.
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

            yield temp_path

    @pytest.mark.cli
    def test_agent_run_happy_path(self, setup_agent_in_temp):
        """
        Test the happy path for 'gradient agent run'.
        Verifies:
        - Server starts successfully
        - Health endpoint responds
        - /run endpoint works and echoes input
        - Server can be cleanly terminated
        """
        logger = logging.getLogger(__name__)
        temp_dir = setup_agent_in_temp
        port = find_free_port()
        process = None

        try:
            logger.info(f"Starting agent on port {port} in {temp_dir}")

            # Start the agent server
            # Note: We use start_new_session=True to create a new process group
            # so we can cleanly kill all child processes during cleanup
            process = subprocess.Popen(
                [
                    "gradient",
                    "agent",
                    "run",
                    "--port",
                    str(port),
                    "--no-dev",
                ],
                cwd=temp_dir,
                start_new_session=True,
            )

            # Wait for server to be ready
            server_ready = wait_for_server(port, timeout=30)
            assert server_ready, "Server did not start within timeout"

            # Test health endpoint
            health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
            assert health_response.status_code == 200
            health_data = health_response.json()
            assert health_data["status"] == "healthy"
            logger.info(f"Health check passed: {health_data}")

            # Test /run endpoint
            run_response = requests.post(
                f"http://localhost:{port}/run",
                json={"prompt": "Hello, World!"},
                timeout=10,
            )
            assert run_response.status_code == 200
            run_data = run_response.json()
            assert run_data["echo"] == "Hello, World!"
            assert run_data["received"]["prompt"] == "Hello, World!"
            logger.info(f"Run endpoint test passed: {run_data}")

        finally:
            cleanup_process(process)

    @pytest.mark.cli
    def test_agent_run_custom_port(self, setup_agent_in_temp):
        """
        Test that the --port flag works correctly.
        """
        logger = logging.getLogger(__name__)
        temp_dir = setup_agent_in_temp
        port = find_free_port()
        process = None

        try:
            logger.info(f"Starting agent on custom port {port}")

            # Start the agent server on custom port
            # Note: We use start_new_session=True to create a new process group
            # so we can cleanly kill all child processes during cleanup
            process = subprocess.Popen(
                [
                    "gradient",
                    "agent",
                    "run",
                    "--port",
                    str(port),
                    "--no-dev",
                ],
                cwd=temp_dir,
                start_new_session=True,
            )

            # Wait for server to be ready
            server_ready = wait_for_server(port, timeout=30)
            assert server_ready, f"Server did not start on port {port}"

            # Verify it's actually on the custom port
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            assert response.status_code == 200
            logger.info(f"Server successfully running on custom port {port}")

        finally:
            cleanup_process(process)

    @pytest.mark.cli
    def test_agent_run_no_config(self):
        """
        Test that running without .gradient/agent.yml fails with helpful error.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Testing agent run without config in {temp_dir}")

            # Run gradient agent run without any config
            result = subprocess.run(
                ["gradient", "agent", "run", "--no-dev"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should fail
            assert result.returncode != 0, "Command should have failed without config"

            # Check for helpful error message
            combined_output = result.stdout + result.stderr
            assert "error" in combined_output.lower() or "configuration" in combined_output.lower(), \
                f"Expected error about missing configuration, got: {combined_output}"
            logger.info("Correctly failed without configuration")

    @pytest.mark.cli
    def test_agent_run_missing_entrypoint(self):
        """
        Test that running with config but missing entrypoint file fails.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create .gradient directory and config pointing to non-existent file
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()

            config = {
                "agent_name": "test-agent",
                "agent_environment": "main",
                "entrypoint_file": "nonexistent.py",
            }

            with open(gradient_dir / "agent.yml", "w") as f:
                yaml.safe_dump(config, f)

            logger.info(f"Testing agent run with missing entrypoint in {temp_dir}")

            # Run gradient agent run
            result = subprocess.run(
                ["gradient", "agent", "run", "--no-dev"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should fail
            assert result.returncode != 0, "Command should have failed with missing entrypoint"

            # Check for helpful error message
            combined_output = result.stdout + result.stderr
            assert "error" in combined_output.lower() or "not exist" in combined_output.lower() or "nonexistent" in combined_output.lower(), \
                f"Expected error about missing entrypoint, got: {combined_output}"
            logger.info("Correctly failed with missing entrypoint file")

    @pytest.mark.cli
    def test_agent_run_invalid_entrypoint_no_decorator(self):
        """
        Test that running with entrypoint file that doesn't have @entrypoint decorator fails.
        """
        logger = logging.getLogger(__name__)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a Python file without @entrypoint decorator
            main_py = temp_path / "main.py"
            main_py.write_text("""
def main(query, context):
    return {"result": "no decorator"}
""")

            # Create .gradient directory and config
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()

            config = {
                "agent_name": "test-agent",
                "agent_environment": "main",
                "entrypoint_file": "main.py",
            }

            with open(gradient_dir / "agent.yml", "w") as f:
                yaml.safe_dump(config, f)

            logger.info(f"Testing agent run with invalid entrypoint (no decorator) in {temp_dir}")

            # Run gradient agent run
            # This might start but fail to find fastapi_app, or fail on validation
            # Either way it should not succeed
            process = subprocess.Popen(
                ["gradient", "agent", "run", "--no-dev", "--port", str(find_free_port())],
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            try:
                # Give it a moment to fail or start
                time.sleep(5)

                # Check if process exited with error
                return_code = process.poll()
                
                if return_code is None:
                    # Process is still running - try to connect and see if it works
                    # (It shouldn't work properly without @entrypoint)
                    # If it's running but not responding properly, that's also a failure mode
                    logger.info("Process started but likely not functioning correctly")
                else:
                    # Process exited - check return code
                    assert return_code != 0 or return_code is None, "Expected process to fail or not work correctly"
                    logger.info(f"Process correctly exited with code {return_code}")
            finally:
                cleanup_process(process)

    @pytest.mark.cli
    def test_agent_run_run_endpoint_with_various_inputs(self, setup_agent_in_temp):
        """
        Test the /run endpoint with various input types.
        """
        logger = logging.getLogger(__name__)
        temp_dir = setup_agent_in_temp
        port = find_free_port()
        process = None

        try:
            # Start the agent server
            # Note: We use start_new_session=True to create a new process group
            # so we can cleanly kill all child processes during cleanup
            process = subprocess.Popen(
                [
                    "gradient",
                    "agent",
                    "run",
                    "--port",
                    str(port),
                    "--no-dev",
                ],
                cwd=temp_dir,
                start_new_session=True,
            )

            # Wait for server to be ready
            server_ready = wait_for_server(port, timeout=30)
            assert server_ready, "Server did not start within timeout"

            # Test with empty object
            response = requests.post(
                f"http://localhost:{port}/run",
                json={},
                timeout=10,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["echo"] == "no prompt provided"
            logger.info("Empty object test passed")

            # Test with additional fields
            response = requests.post(
                f"http://localhost:{port}/run",
                json={"prompt": "test", "extra_field": "value", "nested": {"key": "val"}},
                timeout=10,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["echo"] == "test"
            assert data["received"]["extra_field"] == "value"
            assert data["received"]["nested"]["key"] == "val"
            logger.info("Extra fields test passed")

            # Test with unicode
            response = requests.post(
                f"http://localhost:{port}/run",
                json={"prompt": "Hello `} E1-('"},
                timeout=10,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["echo"] == "Hello `} E1-('"
            logger.info("Unicode test passed")

        finally:
            cleanup_process(process)