"""
Unified Gradient Agent package providing both the SDK (decorator, runtime)
and the CLI (gradient command).
"""

from .decorator import entrypoint, get_app, run_server  # SDK exports
from .runtime import get_runtime_manager  # runtime manager accessor

__all__ = [
    "entrypoint",
    "get_app",
    "run_server",
    "get_runtime_manager",
]

__version__ = "0.2.0"  # bumped for unified package release
