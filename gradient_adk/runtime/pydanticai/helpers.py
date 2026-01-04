"""
PydanticAI instrumentation helpers for Gradient ADK.

This module provides backward-compatible functions that delegate to the
centralized InstrumentorRegistry in gradient_adk.runtime.helpers.

For new code, prefer using:
    from gradient_adk.runtime.helpers import capture_all, get_tracker, registry
"""

from __future__ import annotations
from typing import Optional

from gradient_adk.runtime.digitalocean_tracker import DigitalOceanTracesTracker

# Environment variable to disable PydanticAI instrumentation
DISABLE_PYDANTICAI_INSTRUMENTOR_ENV = "GRADIENT_DISABLE_PYDANTICAI_INSTRUMENTOR"


def _is_pydanticai_instrumentation_disabled() -> bool:
    """Check if PydanticAI instrumentation is disabled via environment variable."""
    import os
    val = os.environ.get(DISABLE_PYDANTICAI_INSTRUMENTOR_ENV, "").lower()
    return val in ("true", "1", "yes")


def _is_pydanticai_available() -> bool:
    """Check if pydantic-ai is installed and available."""
    try:
        from pydantic_ai import Agent
        return True
    except ImportError:
        return False


def capture_pydanticai() -> Optional[DigitalOceanTracesTracker]:
    """
    Install DO tracing for PydanticAI exactly once.
    Must be called BEFORE agent creation/execution to capture spans.
    
    Can be disabled by setting GRADIENT_DISABLE_PYDANTICAI_INSTRUMENTOR=true
    
    Returns:
        The tracker instance if instrumentation was installed, None otherwise.
    
    Note: This function now delegates to the centralized registry.
    For new code, prefer using capture_all() from gradient_adk.runtime.helpers.
    """
    from gradient_adk.runtime.helpers import registry, _register_pydanticai

    # Ensure pydanticai is registered
    if "pydanticai" not in registry._registrations:
        _register_pydanticai()

    # Install and return tracker
    return registry.install("pydanticai")


def get_pydanticai_tracker() -> Optional[DigitalOceanTracesTracker]:
    """
    Get the PydanticAI tracker instance if available.
    
    Note: This function now returns the shared tracker from the registry.
    For new code, prefer using get_tracker() from gradient_adk.runtime.helpers.
    """
    from gradient_adk.runtime.helpers import registry
    return registry.get_tracker()