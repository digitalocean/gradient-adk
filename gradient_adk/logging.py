"""
Structured logging configuration for gradient-adk.

This module provides centralized logging configuration using structlog,
with support for verbose mode controlled by the GRADIENT_VERBOSE environment variable.
"""

import os
import sys
import structlog
from typing import Any, Dict


def configure_logging(force_verbose: bool = False) -> None:
    """Configure structured logging for the gradient-adk package.

    Args:
        force_verbose: Override environment variable to force verbose mode
    """
    # Check verbose mode from environment or parameter
    verbose_mode = force_verbose or os.getenv("GRADIENT_VERBOSE") == "1"

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _gradient_renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set logging level based on verbose mode
    import logging

    # Set up root logger
    root_logger = logging.getLogger()

    if verbose_mode:
        root_logger.setLevel(logging.DEBUG)
        # Set gradient_agents logger to DEBUG
        gradient_logger = logging.getLogger("gradient_agents")
        gradient_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)
        # Set gradient_agents logger to INFO
        gradient_logger = logging.getLogger("gradient_agents")
        gradient_logger.setLevel(logging.INFO)

        # Suppress httpx INFO logs (like HTTP requests) unless in verbose mode
        logging.getLogger("httpx").setLevel(logging.WARNING)

    # Add console handler if not present
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)


def _gradient_renderer(logger, method_name: str, event_dict: Dict[str, Any]) -> str:
    """Custom renderer for gradient-adk log messages."""
    # Extract components
    level = event_dict.get("level", "").upper()
    logger_name = event_dict.get("logger", "")
    event = event_dict.get("event", "")

    # Create component prefix based on logger name
    if "traces" in logger_name or "runtime" in logger_name:
        component = "TRACES"
    elif "langgraph" in logger_name:
        component = "LANGGRAPH"
    elif "digitalocean" in logger_name:
        component = "DO"
    elif "cli" in logger_name:
        component = "CLI"
    else:
        component = "GRADIENT"

    # Format the message
    prefix = f"[{component}]"

    # Add success/error indicators
    if level == "INFO" and ("✓" in event or "success" in event.lower()):
        prefix += " ✓"
    elif level == "ERROR" or level == "CRITICAL":
        prefix += " ✗"
    elif level == "WARNING":
        prefix += " ⚠"
    elif level == "DEBUG":
        prefix += " 🔍"

    # Build final message
    message = f"{prefix} {event}"

    # Add extra fields if present
    extra_fields = {
        k: v
        for k, v in event_dict.items()
        if k not in ["level", "logger", "event", "timestamp"]
    }

    if extra_fields:
        extra_str = " | ".join(f"{k}={v}" for k, v in extra_fields.items())
        message += f" | {extra_str}"

    return message


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger for the given name.

    Args:
        name: Logger name (defaults to calling module)

    Returns:
        Configured structlog logger
    """
    if name is None:
        # Get caller's module name
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "gradient_agents")

    return structlog.get_logger(name)


def is_verbose_mode() -> bool:
    """Check if verbose mode is enabled.

    Returns:
        True if GRADIENT_VERBOSE=1, False otherwise
    """
    return os.getenv("GRADIENT_VERBOSE") == "1"


# Configure logging on module import
configure_logging()
