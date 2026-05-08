"""A2A protocol v0.3.0 integration for Gradient agents.

Keep server-only imports lazy so domain-only consumers and tests do not need the
optional HTTP-server bits of the A2A SDK just to import `gradient_adk.a2a.*`.
"""

from __future__ import annotations

from typing import Any


def create_a2a_server(*args: Any, **kwargs: Any):
    from gradient_adk.a2a.infrastructure.server import create_a2a_server as _impl

    return _impl(*args, **kwargs)


__all__ = ["create_a2a_server"]
