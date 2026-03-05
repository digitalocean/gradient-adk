import asyncio

import pytest
from fastapi.testclient import TestClient


# Mock Gradient agent function
async def mock_gradient_agent(data: dict, context) -> str:
    """Mock Gradient agent for testing."""
    return f"Echo: {data.get('prompt', '')}"


async def slow_gradient_agent(data: dict, context) -> str:
    """Slow agent for cancel-race testing."""
    await asyncio.sleep(5)
    return f"Echo: {data.get('prompt', '')}"


@pytest.fixture
def a2a_server():
    """Create A2A server for testing."""
    from gradient_adk.a2a import create_a2a_server
    app = create_a2a_server(mock_gradient_agent)
    return TestClient(app)


@pytest.fixture
def slow_a2a_server():
    """Create A2A server with slow agent for race-condition testing."""
    from gradient_adk.a2a import create_a2a_server
    app = create_a2a_server(slow_gradient_agent)
    return TestClient(app)
