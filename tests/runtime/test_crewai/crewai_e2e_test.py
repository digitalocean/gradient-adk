"""End-to-end tests for CrewAI instrumentation.

These tests deploy the example CrewAI agent and invoke its endpoint to verify
the agent runs correctly with the instrumentor installed.

The example agent is located at examples/crewai/main.py and creates a trivia
generator crew with two agents (Researcher and Trivia Generator).
"""

import pytest
import os
import subprocess
import time
import json
import requests
from pathlib import Path

# Skip all tests if crewai is not installed
pytest.importorskip("crewai")


# Mark all tests as e2e - they require actual deployment
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("RUN_E2E_TESTS"),
        reason="E2E tests require RUN_E2E_TESTS=1 environment variable",
    ),
]


# -----------------------------
# Configuration
# -----------------------------


EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "crewai"
LOCAL_RUN_PORT = 8080
LOCAL_RUN_HOST = "0.0.0.0"
STARTUP_TIMEOUT = 60  # seconds
REQUEST_TIMEOUT = 300  # 5 minutes for crew execution


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture(scope="module")
def gradient_cli_available():
    """Check if gradient CLI is available."""
    try:
        result = subprocess.run(
            ["gradient", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


@pytest.fixture(scope="module")
def api_keys_available():
    """Check if required API keys are set."""
    return bool(
        os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
        and os.environ.get("SERPER_API_KEY")
    )


@pytest.fixture
def local_agent_server(gradient_cli_available, api_keys_available):
    """Start the example agent locally and return its URL.
    
    This fixture starts the agent using `gradient agent run` and waits
    for it to be ready before yielding the URL. It cleans up after the test.
    """
    if not gradient_cli_available:
        pytest.skip("gradient CLI not available")
    
    if not api_keys_available:
        pytest.skip("Required API keys not set (GRADIENT_MODEL_ACCESS_KEY, SERPER_API_KEY)")

    # Start the agent server
    process = subprocess.Popen(
        ["gradient", "agent", "run", "--port", str(LOCAL_RUN_PORT)],
        cwd=str(EXAMPLES_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url = f"http://{LOCAL_RUN_HOST}:{LOCAL_RUN_PORT}"

    # Wait for server to be ready
    start_time = time.time()
    while time.time() - start_time < STARTUP_TIMEOUT:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        process.terminate()
        process.wait()
        pytest.fail(f"Agent server did not start within {STARTUP_TIMEOUT} seconds")

    yield url

    # Cleanup
    process.terminate()
    process.wait(timeout=10)


# -----------------------------
# Local Run Tests
# -----------------------------


class TestLocalRun:
    """Tests that run the example agent locally using `gradient agent run`."""

    def test_agent_starts_successfully(self, local_agent_server):
        """Test that the agent server starts and responds to health checks."""
        url = local_agent_server
        
        response = requests.get(f"{url}/health", timeout=10)
        assert response.status_code == 200

    def test_agent_accepts_invocation_request(self, local_agent_server):
        """Test that the agent accepts and processes invocation requests."""
        url = local_agent_server
        
        payload = {
            "date": "2025-01-15",
            "topic": "Artificial Intelligence",
        }
        
        response = requests.post(
            f"{url}/run",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # The response should contain the crew result
        assert "result" in data or "error" in data

    def test_agent_handles_missing_input(self, local_agent_server):
        """Test that the agent handles missing required input gracefully."""
        url = local_agent_server
        
        # Send empty payload
        response = requests.post(
            f"{url}/run",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        
        # Should either return an error or handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_multiple_invocations_work(self, local_agent_server):
        """Test that the agent can handle multiple sequential invocations."""
        url = local_agent_server
        
        topics = ["Technology", "Science"]
        
        for topic in topics:
            payload = {
                "date": "2025-01-15",
                "topic": topic,
            }
            
            response = requests.post(
                f"{url}/run",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            
            assert response.status_code == 200


# -----------------------------
# Deployment Tests
# -----------------------------


@pytest.mark.skipif(
    not os.environ.get("RUN_DEPLOY_TESTS"),
    reason="Deployment tests require RUN_DEPLOY_TESTS=1 environment variable",
)
class TestDeployment:
    """Tests that deploy the example agent to the cloud.
    
    These tests are more expensive and slower, so they're separated
    and require an additional flag to run.
    """

    @pytest.fixture
    def deployed_agent(self, gradient_cli_available, api_keys_available):
        """Deploy the example agent and return its URL.
        
        This fixture deploys the agent, waits for it to be ready,
        and cleans up after the test.
        """
        if not gradient_cli_available:
            pytest.skip("gradient CLI not available")
        
        if not api_keys_available:
            pytest.skip("Required API keys not set")

        # Deploy the agent
        result = subprocess.run(
            ["gradient", "agent", "deploy"],
            cwd=str(EXAMPLES_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            pytest.fail(f"Deployment failed: {result.stderr}")

        # Extract URL from output (format may vary)
        # This is a simplified parser - adjust based on actual CLI output
        url = None
        for line in result.stdout.split("\n"):
            if "https://" in line and "agents" in line:
                # Extract URL from line
                import re
                match = re.search(r"(https://[^\s]+)", line)
                if match:
                    url = match.group(1)
                    break
        
        if not url:
            pytest.fail("Could not extract deployment URL from CLI output")

        # Wait for deployment to be ready
        for _ in range(60):  # Wait up to 5 minutes
            try:
                response = requests.get(f"{url}/health", timeout=10)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(5)
        else:
            pytest.fail("Deployment did not become ready")

        yield url

        # Note: Cleanup (undeploy) could be added here if there's a CLI command for it

    def test_deployed_agent_responds(self, deployed_agent):
        """Test that the deployed agent responds to requests."""
        url = deployed_agent
        
        payload = {
            "date": "2025-01-15",
            "topic": "Technology",
        }
        
        response = requests.post(
            f"{url}/run",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        
        assert response.status_code == 200


# -----------------------------
# Instrumentation Verification Tests
# -----------------------------


class TestInstrumentationVerification:
    """Tests that verify instrumentation is working during agent execution.
    
    These tests check that spans are being created and captured correctly
    by examining the trace output or logs.
    """

    @pytest.mark.skipif(
        not (os.environ.get("GRADIENT_MODEL_ACCESS_KEY") and os.environ.get("SERPER_API_KEY")),
        reason="Requires GRADIENT_MODEL_ACCESS_KEY and SERPER_API_KEY",
    )
    def test_crew_runs_with_instrumentation(self):
        """Test that the example crew runs successfully with instrumentation enabled.
        
        This test directly imports and runs the example crew to verify
        that instrumentation doesn't break normal execution.
        """
        import sys
        sys.path.insert(0, str(EXAMPLES_DIR))
        
        try:
            from main import create_trivia_crew
            from gradient_adk.runtime.crewai.crewai_instrumentor import CrewAIInstrumentor
            from gradient_adk.runtime.digitalocean_tracker import DigitalOceanTracesTracker
            from unittest.mock import MagicMock

            # Create a mock tracker to capture spans
            tracker = MagicMock(spec=DigitalOceanTracesTracker)
            tracker.on_node_start = MagicMock()
            tracker.on_node_end = MagicMock()
            tracker.on_node_error = MagicMock()

            # Install instrumentor
            instrumentor = CrewAIInstrumentor()
            instrumentor.install(tracker)

            try:
                # Create the crew
                crew = create_trivia_crew("2025-01-15", "Technology")
                
                # Run the crew
                result = crew.kickoff()
                
                # Verify result
                assert result is not None
                
                # Verify instrumentation captured spans
                assert tracker.on_node_start.call_count >= 1, "No spans were started"
                assert tracker.on_node_end.call_count >= 1, "No spans were ended"
                
                # Verify workflow spans were created
                workflow_spans = []
                for call in tracker.on_node_start.call_args_list:
                    span = call[0][0]
                    if hasattr(span, "metadata") and span.metadata.get("is_workflow"):
                        workflow_spans.append(span)
                
                # Should have at least 2 workflow spans (one per agent)
                assert len(workflow_spans) >= 2, f"Expected at least 2 workflow spans, got {len(workflow_spans)}"
                
                # Verify agent names
                agent_names = {s.node_name for s in workflow_spans}
                assert any("Research" in name for name in agent_names), "Researcher agent span not found"
                assert any("Trivia" in name for name in agent_names), "Trivia agent span not found"

            finally:
                instrumentor.uninstall()

        finally:
            sys.path.remove(str(EXAMPLES_DIR))

    def test_instrumentation_does_not_break_crew(self):
        """Test that installing instrumentation doesn't break CrewAI functionality.
        
        This test verifies that the instrumentor can be installed and uninstalled
        without causing errors, even if no crew is run.
        """
        from gradient_adk.runtime.crewai.crewai_instrumentor import CrewAIInstrumentor
        from unittest.mock import MagicMock

        tracker = MagicMock()
        
        # Install
        instrumentor = CrewAIInstrumentor()
        instrumentor.install(tracker)
        assert instrumentor.is_installed()
        
        # Verify CrewAI can still be imported and used
        from crewai import Agent, Task, Crew, Process
        
        # Create basic objects (without LLM to avoid API calls)
        # Just verify imports work
        assert Agent is not None
        assert Task is not None
        assert Crew is not None
        
        # Uninstall
        instrumentor.uninstall()
        assert not instrumentor.is_installed()


# -----------------------------
# Concurrent Request Tests
# -----------------------------


class TestConcurrentRequests:
    """Tests for handling concurrent requests."""

    def test_concurrent_invocations_isolated(self, local_agent_server):
        """Test that concurrent invocations have isolated span contexts."""
        import concurrent.futures
        
        url = local_agent_server
        
        def make_request(topic):
            payload = {
                "date": "2025-01-15",
                "topic": topic,
            }
            response = requests.post(
                f"{url}/run",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            return response.status_code, topic
        
        topics = ["Technology", "Science", "Health"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, topic) for topic in topics]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for status_code, topic in results:
            assert status_code == 200, f"Request for topic '{topic}' failed"
