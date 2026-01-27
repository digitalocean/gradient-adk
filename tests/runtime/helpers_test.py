"""Tests for the centralized InstrumentorRegistry in gradient_adk.runtime.helpers."""

import pytest
import os
from unittest.mock import MagicMock, patch

from gradient_adk.runtime.helpers import (
    InstrumentorRegistry,
    registry,
    capture_all,
    get_tracker,
    register_all_instrumentors,
    _register_langgraph,
    _register_pydanticai,
    _is_tracing_disabled,
)


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def fresh_registry():
    """Create a fresh registry instance for testing."""
    return InstrumentorRegistry()


@pytest.fixture
def mock_instrumentor():
    """Create a mock instrumentor that follows the protocol."""
    inst = MagicMock()
    inst.install = MagicMock()
    inst.uninstall = MagicMock()
    inst.is_installed = MagicMock(return_value=False)
    return inst


# -----------------------------
# Registry Basic Tests
# -----------------------------


def test_registry_initial_state(fresh_registry):
    """Test that a fresh registry starts with no tracker and no instrumentors."""
    assert fresh_registry.get_tracker() is None
    assert fresh_registry.get_installed_names() == []
    assert not fresh_registry.is_installed("anything")


def test_register_adds_instrumentor(fresh_registry, mock_instrumentor):
    """Test that register() adds an instrumentor to the registry."""
    fresh_registry.register(
        name="test",
        env_disable_var="TEST_DISABLE",
        availability_check=lambda: True,
        instrumentor_factory=lambda: mock_instrumentor,
    )

    assert "test" in fresh_registry._registrations


def test_install_without_api_token_returns_none(fresh_registry, mock_instrumentor):
    """Test that install returns None when no API token is available."""
    fresh_registry.register(
        name="test",
        env_disable_var="TEST_DISABLE",
        availability_check=lambda: True,
        instrumentor_factory=lambda: mock_instrumentor,
    )

    # Ensure no API token
    with patch.dict(os.environ, {}, clear=True):
        result = fresh_registry.install("test")

    assert result is None
    mock_instrumentor.install.assert_not_called()


def test_install_when_disabled_returns_none(fresh_registry, mock_instrumentor):
    """Test that install returns None when disabled via env var."""
    fresh_registry.register(
        name="test",
        env_disable_var="TEST_DISABLE",
        availability_check=lambda: True,
        instrumentor_factory=lambda: mock_instrumentor,
    )

    with patch.dict(os.environ, {"TEST_DISABLE": "true"}):
        result = fresh_registry.install("test")

    assert result is None
    mock_instrumentor.install.assert_not_called()


def test_install_when_unavailable_returns_none(fresh_registry, mock_instrumentor):
    """Test that install returns None when framework is unavailable."""
    fresh_registry.register(
        name="test",
        env_disable_var="TEST_DISABLE",
        availability_check=lambda: False,  # Not available
        instrumentor_factory=lambda: mock_instrumentor,
    )

    result = fresh_registry.install("test")

    assert result is None
    mock_instrumentor.install.assert_not_called()


def test_install_nonexistent_returns_none(fresh_registry):
    """Test that install returns None for unregistered instrumentor."""
    result = fresh_registry.install("nonexistent")
    assert result is None


def test_is_env_disabled_variations(fresh_registry):
    """Test environment variable disable check with various values."""
    # True values
    for val in ["true", "TRUE", "1", "yes", "YES"]:
        with patch.dict(os.environ, {"TEST_VAR": val}):
            assert fresh_registry._is_env_disabled("TEST_VAR") is True

    # False values
    for val in ["false", "FALSE", "0", "no", "", "anything"]:
        with patch.dict(os.environ, {"TEST_VAR": val}):
            assert fresh_registry._is_env_disabled("TEST_VAR") is False

    # Missing var
    with patch.dict(os.environ, {}, clear=True):
        assert fresh_registry._is_env_disabled("MISSING_VAR") is False


def test_is_installed_returns_correct_state(fresh_registry):
    """Test is_installed returns correct boolean state."""
    assert not fresh_registry.is_installed("test")

    # Simulate installation by adding to _instrumentors directly
    mock_inst = MagicMock()
    fresh_registry._instrumentors["test"] = mock_inst

    assert fresh_registry.is_installed("test")


def test_get_installed_names_returns_list(fresh_registry):
    """Test get_installed_names returns list of installed instrumentor names."""
    assert fresh_registry.get_installed_names() == []

    # Simulate installations
    fresh_registry._instrumentors["test1"] = MagicMock()
    fresh_registry._instrumentors["test2"] = MagicMock()

    names = fresh_registry.get_installed_names()
    assert sorted(names) == ["test1", "test2"]


def test_uninstall_calls_uninstall_method(fresh_registry):
    """Test that uninstall calls the instrumentor's uninstall method."""
    mock_inst = MagicMock()
    fresh_registry._instrumentors["test"] = mock_inst

    fresh_registry.uninstall("test")

    mock_inst.uninstall.assert_called_once()
    assert "test" not in fresh_registry._instrumentors


def test_uninstall_nonexistent_is_safe(fresh_registry):
    """Test that uninstalling non-existent instrumentor doesn't raise."""
    # Should not raise
    fresh_registry.uninstall("nonexistent")


def test_uninstall_all_clears_all_instrumentors(fresh_registry):
    """Test that uninstall_all clears all instrumentors."""
    mock1 = MagicMock()
    mock2 = MagicMock()
    fresh_registry._instrumentors["test1"] = mock1
    fresh_registry._instrumentors["test2"] = mock2

    fresh_registry.uninstall_all()

    mock1.uninstall.assert_called_once()
    mock2.uninstall.assert_called_once()
    assert fresh_registry.get_installed_names() == []


def test_install_is_idempotent(fresh_registry, mock_instrumentor):
    """Test that installing the same instrumentor twice is a no-op."""
    # Simulate already installed
    fresh_registry._instrumentors["test"] = mock_instrumentor

    fresh_registry.register(
        name="test",
        env_disable_var="TEST_DISABLE",
        availability_check=lambda: True,
        instrumentor_factory=lambda: mock_instrumentor,
    )

    # Second install should not call install again
    result = fresh_registry.install("test")

    # Should return the tracker (which may be None), but not call install
    mock_instrumentor.install.assert_not_called()


# -----------------------------
# Registration Function Tests
# -----------------------------


def test_register_langgraph_adds_to_registry():
    """Test that _register_langgraph adds langgraph to registry."""
    test_registry = InstrumentorRegistry()

    # Patch the global registry temporarily
    with patch("gradient_adk.runtime.helpers.registry", test_registry):
        _register_langgraph()

    assert "langgraph" in test_registry._registrations
    assert test_registry._registrations["langgraph"]["env_disable_var"] == "GRADIENT_DISABLE_LANGGRAPH_INSTRUMENTOR"


def test_register_pydanticai_adds_to_registry():
    """Test that _register_pydanticai adds pydanticai to registry."""
    test_registry = InstrumentorRegistry()

    # Patch the global registry temporarily
    with patch("gradient_adk.runtime.helpers.registry", test_registry):
        _register_pydanticai()

    assert "pydanticai" in test_registry._registrations
    assert test_registry._registrations["pydanticai"]["env_disable_var"] == "GRADIENT_DISABLE_PYDANTICAI_INSTRUMENTOR"


def test_register_all_instrumentors_registers_both():
    """Test that register_all_instrumentors registers both frameworks."""
    test_registry = InstrumentorRegistry()

    with patch("gradient_adk.runtime.helpers.registry", test_registry):
        register_all_instrumentors()

    assert "langgraph" in test_registry._registrations
    assert "pydanticai" in test_registry._registrations


# -----------------------------
# Global Registry Tests
# -----------------------------


def test_global_registry_exists():
    """Test that the global registry instance exists."""
    assert registry is not None
    assert isinstance(registry, InstrumentorRegistry)


def test_get_tracker_returns_global_tracker():
    """Test that get_tracker returns the global registry's tracker."""
    # Note: tracker may be None if not initialized, but function should work
    result = get_tracker()
    assert result == registry.get_tracker()


# -----------------------------
# Availability Check Tests
# -----------------------------


def test_langgraph_availability_check():
    """Test langgraph availability check function."""
    test_registry = InstrumentorRegistry()

    with patch("gradient_adk.runtime.helpers.registry", test_registry):
        _register_langgraph()

    check = test_registry._registrations["langgraph"]["availability_check"]

    # Since we're running tests with langgraph installed, it should be available
    assert check() is True


def test_pydanticai_availability_check():
    """Test pydanticai availability check function."""
    test_registry = InstrumentorRegistry()

    with patch("gradient_adk.runtime.helpers.registry", test_registry):
        _register_pydanticai()

    check = test_registry._registrations["pydanticai"]["availability_check"]

    # Since we're running tests with pydantic-ai installed, it should be available
    assert check() is True


# -----------------------------
# DISABLE_TRACES Tests
# -----------------------------


def test_is_tracing_disabled_true_values():
    """Test _is_tracing_disabled returns True for truthy env var values."""
    for val in ["true", "TRUE", "True", "1", "yes", "YES", "Yes"]:
        with patch.dict(os.environ, {"DISABLE_TRACES": val}):
            assert _is_tracing_disabled() is True, f"Expected True for DISABLE_TRACES={val}"


def test_is_tracing_disabled_false_values():
    """Test _is_tracing_disabled returns False for falsy env var values."""
    for val in ["false", "FALSE", "0", "no", "", "anything", "disabled"]:
        with patch.dict(os.environ, {"DISABLE_TRACES": val}):
            assert _is_tracing_disabled() is False, f"Expected False for DISABLE_TRACES={val}"


def test_is_tracing_disabled_missing_var():
    """Test _is_tracing_disabled returns False when env var is not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert _is_tracing_disabled() is False


def test_ensure_tracker_returns_none_when_tracing_disabled(fresh_registry):
    """Test that _ensure_tracker returns None when DISABLE_TRACES=1, even with valid API token."""
    with patch.dict(os.environ, {"DISABLE_TRACES": "1", "DIGITALOCEAN_API_TOKEN": "test-token"}):
        result = fresh_registry._ensure_tracker()
    
    assert result is None
    assert fresh_registry._tracker is None


def test_install_all_returns_none_when_tracing_disabled():
    """Test that install_all returns None and installs nothing when DISABLE_TRACES=1."""
    test_registry = InstrumentorRegistry()
    
    with patch("gradient_adk.runtime.helpers.registry", test_registry):
        with patch.dict(os.environ, {"DISABLE_TRACES": "1", "DIGITALOCEAN_API_TOKEN": "test-token"}):
            register_all_instrumentors()
            result = test_registry.install_all()
    
    assert result is None
    assert test_registry.get_installed_names() == []