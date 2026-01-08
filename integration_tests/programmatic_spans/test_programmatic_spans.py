"""
Integration tests for the programmatic span functions (add_llm_span, add_tool_span, add_agent_span).
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
            pass


class TestProgrammaticSpans:
    """Integration tests for programmatic span functions."""

    @pytest.fixture
    def programmatic_spans_agent_dir(self):
        """Get the path to the programmatic spans agent directory."""
        return Path(__file__).parent.parent / "example_agents" / "programmatic_spans_agent"

    @pytest.fixture
    def setup_agent_in_temp(self, programmatic_spans_agent_dir):
        """
        Setup a temporary directory with the programmatic spans agent and proper configuration.
        Yields the temp directory path and cleans up after.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy the agent main.py
            shutil.copy(programmatic_spans_agent_dir / "main.py", temp_path / "main.py")

            # Create .gradient directory and config
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()

            config = {
                "agent_name": "test-programmatic-spans-agent",
                "agent_environment": "main",
                "entrypoint_file": "main.py",
            }

            with open(gradient_dir / "agent.yml", "w") as f:
                yaml.safe_dump(config, f)

            yield temp_path

    @pytest.mark.cli
    def test_programmatic_spans_agent_runs_successfully(self, setup_agent_in_temp):
        """
        Test that an agent using add_llm_span, add_tool_span, and add_agent_span
        can start and respond without errors.

        Verifies:
        - Server starts successfully
        - Health endpoint responds
        - /run endpoint works with programmatic span functions
        - Server can be cleanly terminated
        """
        logger = logging.getLogger(__name__)
        temp_dir = setup_agent_in_temp
        port = find_free_port()
        process = None

        try:
            logger.info(f"Starting programmatic spans agent on port {port} in {temp_dir}")

            # Start the agent server
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

            # Test /run endpoint - this will exercise all three programmatic span functions
            run_response = requests.post(
                f"http://localhost:{port}/run",
                json={"prompt": "Test prompt for programmatic spans"},
                timeout=10,
            )
            assert run_response.status_code == 200
            run_data = run_response.json()

            # Verify the response
            assert run_data["success"] is True
            assert run_data["message"] == "All programmatic spans created successfully"
            assert run_data["prompt_received"] == "Test prompt for programmatic spans"
            logger.info(f"Run endpoint test passed: {run_data}")

        finally:
            cleanup_process(process)

    @pytest.mark.cli
    def test_programmatic_spans_with_empty_input(self, setup_agent_in_temp):
        """
        Test programmatic spans with empty input.
        """
        logger = logging.getLogger(__name__)
        temp_dir = setup_agent_in_temp
        port = find_free_port()
        process = None

        try:
            logger.info(f"Starting agent on port {port}")

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

            server_ready = wait_for_server(port, timeout=30)
            assert server_ready, "Server did not start within timeout"

            # Test with empty object
            run_response = requests.post(
                f"http://localhost:{port}/run",
                json={},
                timeout=10,
            )
            assert run_response.status_code == 200
            run_data = run_response.json()

            assert run_data["success"] is True
            assert run_data["prompt_received"] == "no prompt provided"
            logger.info(f"Empty input test passed: {run_data}")

        finally:
            cleanup_process(process)

    @pytest.mark.cli
    def test_programmatic_spans_multiple_requests(self, setup_agent_in_temp):
        """
        Test that multiple requests with programmatic spans work correctly.
        """
        logger = logging.getLogger(__name__)
        temp_dir = setup_agent_in_temp
        port = find_free_port()
        process = None

        try:
            logger.info(f"Starting agent on port {port}")

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

            server_ready = wait_for_server(port, timeout=30)
            assert server_ready, "Server did not start within timeout"

            # Make multiple requests to ensure spans work repeatedly
            for i in range(3):
                run_response = requests.post(
                    f"http://localhost:{port}/run",
                    json={"prompt": f"Request {i + 1}"},
                    timeout=10,
                )
                assert run_response.status_code == 200
                run_data = run_response.json()

                assert run_data["success"] is True
                assert run_data["prompt_received"] == f"Request {i + 1}"
                logger.info(f"Request {i + 1} passed: {run_data}")

            logger.info("Multiple requests test passed")

        finally:
            cleanup_process(process)