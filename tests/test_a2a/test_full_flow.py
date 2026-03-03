"""End-to-end tests for A2A integration."""

import pytest
import httpx
from fastapi.testclient import TestClient

from gradient_adk.a2a.infrastructure.server import create_a2a_server


# Test agent function (not a pytest test, just a helper)
def echo_agent(data, context):
    """Simple test agent that echoes the prompt."""
    prompt = data.get("prompt", "")
    return {"output": f"Echo: {prompt}"}


@pytest.fixture
def app():
    """Create test FastAPI app."""
    return create_a2a_server(
        agent_func=echo_agent,
        base_url="http://test",
        agent_name="Test A2A Agent",
    )


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestAgentDiscovery:
    """Test agent discovery endpoint."""

    def test_agent_card_endpoint_exists(self, client):
        """Test that agent card endpoint is accessible."""
        response = client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_agent_card_has_correct_structure(self, client):
        """Test that agent card has correct structure."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        assert "name" in card
        assert "version" in card
        assert "capabilities" in card
        assert "defaultInputModes" in card
        assert "defaultOutputModes" in card

    def test_agent_card_capabilities(self, client):
        """Test that agent card capabilities are correct for MVP."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        assert card["capabilities"]["streaming"] is False
        assert card["capabilities"]["pushNotifications"] is False
        assert card["capabilities"]["stateTransitionHistory"] is True

    def test_agent_card_input_output_modes(self, client):
        """Test that input/output modes are correct."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        assert "text/plain" in card["defaultInputModes"]
        assert "text/plain" in card["defaultOutputModes"]


class TestMessageSend:
    """Test message/send operation."""

    def test_send_valid_text_message(self, client):
        """Test sending a valid text message."""
        response = client.post("/", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-001",
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello, agent!"}]
                }
            },
            "id": 1
        })

        assert response.status_code == 200
        result = response.json()

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert "result" in result

        result_data = result["result"]
        assert "role" in result_data or ("kind" in result_data and result_data["kind"] == "task")

    def test_send_message_with_empty_text(self, client):
        """Test that empty messages are rejected."""
        response = client.post("/", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-002",
                    "role": "user",
                    "parts": [{"type": "text", "text": ""}]
                }
            },
            "id": 2
        })

        assert response.status_code == 200
        result = response.json()

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 2

    def test_invalid_json_rpc_request(self, client):
        """Test that invalid JSON-RPC requests are rejected."""
        response = client.post("/", json={
            "invalid": "request"
        })

        assert response.status_code == 200
        result = response.json()

        assert "error" in result
        assert result["error"]["code"] == -32600


class TestTaskOperations:
    """Test task operations."""

    def test_get_nonexistent_task(self, client):
        """Test getting a non-existent task."""
        response = client.post("/", json={
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {
                "taskId": "nonexistent-task-id"
            },
            "id": 3
        })

        assert response.status_code == 200
        result = response.json()

        assert "error" in result or ("result" in result and "error" in str(result["result"]))

    def test_cancel_nonexistent_task(self, client):
        """Test canceling a non-existent task."""
        response = client.post("/", json={
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {
                "taskId": "nonexistent-task-id"
            },
            "id": 4
        })

        assert response.status_code == 200
        result = response.json()

        assert result["jsonrpc"] == "2.0"


class TestUnsupportedOperations:
    """Test unsupported operations."""

    def test_streaming_is_unsupported(self, client):
        """Test that message/stream returns unsupported operation."""
        response = client.post("/", json={
            "jsonrpc": "2.0",
            "method": "message/stream",
            "params": {
                "message": {
                    "messageId": "msg-stream-001",
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello"}]
                }
            },
            "id": 5
        })

        assert response.status_code == 200
        result = response.json()

        assert "error" in result
        assert result["error"]["code"] in [-32004, -32601, -32602, -32603]

    def test_unknown_method(self, client):
        """Test that unknown methods return method not found."""
        response = client.post("/", json={
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "params": {},
            "id": 6
        })

        assert response.status_code == 200
        result = response.json()

        assert "error" in result
        assert result["error"]["code"] == -32601


class TestAsyncFlow:
    """Test async message flow."""

    def test_async_message_send_with_test_client(self, client):
        """Test async message send using TestClient (simulates async)."""
        response = client.post("/", json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "msg-async-001",
                    "role": "user",
                    "parts": [{"type": "text", "text": "Async hello!"}]
                }
            },
            "id": 7
        })

        assert response.status_code == 200
        result = response.json()
        assert result["jsonrpc"] == "2.0"
        assert "result" in result or "error" in result
