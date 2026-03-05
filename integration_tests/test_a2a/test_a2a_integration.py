import pytest


def test_message_send(a2a_server):
    """Test message/send operation bridges to Gradient agent."""
    response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "message_id": "msg-1",
                    "kind": "message",
                }
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    result = data["result"]

    assert isinstance(result, dict)

    if result.get("kind") == "task":
        assert "id" in result
        assert "status" in result
        assert result["status"]["state"] == "completed"

        response_text = None
        if result["status"].get("message") and "parts" in result["status"]["message"]:
            response_text = result["status"]["message"]["parts"][0]["text"]
        elif result.get("history"):
            for msg in reversed(result["history"]):
                if msg.get("role") == "agent" and "parts" in msg:
                    response_text = msg["parts"][0]["text"]
                    break

        assert response_text is not None, "Agent response not found in task"
        assert "Echo: Hello" in response_text
    elif "parts" in result:
        assert "Echo: Hello" in result["parts"][0]["text"]
    else:
        pytest.fail(f"Unexpected result structure: {result}")


def test_tasks_get(a2a_server):
    """Test tasks/get operation."""
    send_response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test"}],
                    "message_id": "msg-1",
                    "kind": "message",
                }
            },
        },
    )
    task_id = send_response.json()["result"]["id"]

    response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tasks/get",
            "params": {"id": task_id},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["id"] == task_id


def test_agent_card_discovery(a2a_server):
    """Test agent card discovery endpoint."""
    response = a2a_server.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Gradient A2A Agent"
    assert "capabilities" in data
    assert data["capabilities"]["streaming"] is False
    assert data["capabilities"]["pushNotifications"] is False
    assert data.get("supportsAuthenticatedExtendedCard") is False


def test_agent_card_reflects_configured_url():
    """Test that AgentCard URL reflects the base_url parameter."""
    from gradient_adk.a2a import create_a2a_server
    from fastapi.testclient import TestClient

    async def mock_agent(data: dict, context) -> str:
        return "test"

    app = create_a2a_server(mock_agent, base_url="https://my-agent.example.com")
    client = TestClient(app)

    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.json()
    assert card["url"] == "https://my-agent.example.com"

    app_default = create_a2a_server(mock_agent)
    client_default = TestClient(app_default)

    response_default = client_default.get("/.well-known/agent-card.json")
    assert response_default.status_code == 200
    card_default = response_default.json()
    assert card_default["url"] == "http://localhost:8000"


def test_unsupported_streaming(a2a_server):
    """Test that SDK automatically returns UnsupportedOperationError for streaming."""
    response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "message_id": "msg-1",
                    "kind": "message",
                }
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603
    assert "not supported" in data["error"]["message"].lower()


def test_unsupported_push_notifications(a2a_server):
    """Test that SDK automatically returns UnsupportedOperationError for push notifications."""
    response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tasks/pushNotificationConfig/set",
            "params": {
                "task_id": "task1",
                "pushNotificationConfig": {
                    "url": "https://example.com",
                    "token": "token",
                },
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32603
    assert "not supported" in data["error"]["message"].lower()


def test_tasks_get_with_history_length(a2a_server):
    """Test tasks/get with historyLength parameter (SDK handles automatically)."""
    send_response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test"}],
                    "message_id": "msg-1",
                    "kind": "message",
                }
            },
        },
    )
    task_id = send_response.json()["result"]["id"]

    response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tasks/get",
            "params": {"id": task_id, "historyLength": 1},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["id"] == task_id


def test_tasks_cancel(a2a_server):
    """Test tasks/cancel operation (SDK handles race conditions automatically)."""
    send_response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test"}],
                    "message_id": "msg-1",
                    "kind": "message",
                }
            },
        },
    )
    task_id = send_response.json()["result"]["id"]

    response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        },
    )
    assert response.status_code == 200
    data = response.json()
    if "result" in data:
        assert data["result"]["status"]["state"] == "canceled"
    elif "error" in data:
        assert data["error"]["code"] in [-32002, -32603]
        assert "cancel" in data["error"]["message"].lower() or "terminal" in data["error"]["message"].lower()
    else:
        pytest.fail(f"Unexpected response: {data}")


def test_tasks_cancel_already_completed(a2a_server):
    """Test that canceling an already-completed task is idempotent (req 3.4)."""
    send_response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Done fast"}],
                    "message_id": "msg-1",
                    "kind": "message",
                }
            },
        },
    )
    task_id = send_response.json()["result"]["id"]
    assert send_response.json()["result"]["status"]["state"] == "completed"

    # Cancel a completed task — should not error
    cancel_response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        },
    )
    assert cancel_response.status_code == 200
    data = cancel_response.json()
    # Either succeeds silently or returns a protocol error — both are valid
    if "result" in data:
        assert data["result"]["status"]["state"] in ("completed", "canceled")
    elif "error" in data:
        # Task already in terminal state — protocol-correct rejection
        pass

    # Cancel again — idempotent
    cancel_response2 = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        },
    )
    assert cancel_response2.status_code == 200


def test_tasks_cancel_race_with_slow_agent(slow_a2a_server):
    """Test cancel vs completion race with a slow agent (req 7: explicit race test).

    Sends a message to a slow agent (5s sleep), then immediately cancels.
    The outcome should be either 'canceled' or 'completed' — both are valid
    under the A2A spec's best-effort cancellation semantics.
    """
    def send_message():
        return slow_a2a_server.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Slow task"}],
                        "message_id": "msg-race",
                        "kind": "message",
                    }
                },
            },
        )

    def cancel_task(task_id):
        return slow_a2a_server.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "2",
                "method": "tasks/cancel",
                "params": {"id": task_id},
            },
        )

    # Send the message (may block or return immediately depending on SDK behavior)
    send_response = send_message()
    assert send_response.status_code == 200
    send_data = send_response.json()
    assert "result" in send_data
    task_id = send_data["result"]["id"]

    # Immediately attempt cancel
    cancel_response = cancel_task(task_id)
    assert cancel_response.status_code == 200
    cancel_data = cancel_response.json()

    # Verify task final state — either completed or canceled is acceptable
    get_response = slow_a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tasks/get",
            "params": {"id": task_id},
        },
    )
    assert get_response.status_code == 200
    task_state = get_response.json()["result"]["status"]["state"]
    assert task_state in ("completed", "canceled", "failed"), (
        f"Task in unexpected state '{task_state}' after cancel race"
    )


def test_text_only_validation_rejects_file_part(a2a_server):
    """Test that MVP rejects FilePart per spec requirement."""
    response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "file", "file": {"uri": "https://example.com/file.txt"}}],
                    "message_id": "msg-1",
                    "kind": "message",
                }
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602
    assert "text/plain" in data["error"]["message"].lower()


def test_text_only_validation_rejects_data_part(a2a_server):
    """Test that MVP rejects DataPart per spec requirement."""
    response = a2a_server.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "data", "data": {"key": "value"}}],
                    "message_id": "msg-1",
                    "kind": "message",
                }
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602
    assert "text/plain" in data["error"]["message"].lower()
