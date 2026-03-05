"""A2A protocol v0.3.0 integration for Gradient agents.

Public API:
    create_a2a_server: Create an A2A-enabled FastAPI server for a Gradient agent
"""

from gradient_adk.a2a.infrastructure.server import create_a2a_server

__all__ = [
    "create_a2a_server",
]
