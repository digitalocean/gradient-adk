"""
Tests for the gradient_agent package.
"""

import pytest
from fastapi.testclient import TestClient
from gradient_agent import entrypoint, get_app


def test_basic_entrypoint():
    """Test basic entrypoint functionality."""

    @entrypoint
    def test_agent(prompt: str) -> str:
        return f"Response to: {prompt}"

    app = get_app()
    client = TestClient(app)

    # Test completion endpoint
    response = client.post("/completions", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["completion"] == "Response to: Hello"

    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_entrypoint_with_optional_params():
    """Test entrypoint with optional parameters."""

    @entrypoint
    def test_agent(prompt: str, max_tokens: int = 100) -> str:
        return f"Response to: {prompt} [max_tokens={max_tokens}]"

    app = get_app()
    client = TestClient(app)

    # Test with default parameters
    response = client.post("/completions", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "max_tokens=100" in data["completion"]

    # Test with custom parameters
    response = client.post("/completions", json={"prompt": "Hello", "max_tokens": 50})
    assert response.status_code == 200
    data = response.json()
    assert "max_tokens=50" in data["completion"]


def test_entrypoint_dict_return():
    """Test entrypoint that returns a dict."""

    @entrypoint
    def test_agent(prompt: str) -> dict:
        return {
            "completion": f"Response to: {prompt}",
            "metadata": {"model": "test", "tokens": 10},
        }

    app = get_app()
    client = TestClient(app)

    response = client.post("/completions", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["completion"] == "Response to: Hello"
    assert data["metadata"]["model"] == "test"
