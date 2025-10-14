"""
Unit tests for the decorator module.
"""

import pytest
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
import json

from gradient_adk.decorator import entrypoint, run_server
from gradient_adk.streaming import StreamingResponse as GradientStreamingResponse


class TestEntrypointDecorator:
    """Test cases for the entrypoint decorator."""

    def test_entrypoint_validates_function_parameters(self):
        """Test that entrypoint validates function has exactly 2 parameters."""

        # Test function with no parameters
        with pytest.raises(
            ValueError, match="must have exactly 2 parameters.*but has 0"
        ):

            @entrypoint
            def no_params():
                pass

        # Test function with 1 parameter
        with pytest.raises(
            ValueError, match="must have exactly 2 parameters.*but has 1"
        ):

            @entrypoint
            def one_param(data):
                pass

        # Test function with 3 parameters
        with pytest.raises(
            ValueError, match="must have exactly 2 parameters.*but has 3"
        ):

            @entrypoint
            def three_params(data, context, extra):
                pass

    def test_entrypoint_accepts_valid_function(self):
        """Test that entrypoint accepts function with exactly 2 parameters."""

        @entrypoint
        def valid_func(data, context):
            return {"success": True}

        # Should not raise any exception
        assert callable(valid_func)

    @patch("sys._getframe")
    def test_entrypoint_injects_app_into_module(self, mock_getframe):
        """Test that entrypoint injects FastAPI app into module globals."""

        # Mock the frame and globals
        mock_frame = Mock()
        mock_globals = {}
        mock_frame.f_globals = mock_globals
        mock_getframe.return_value = mock_frame

        @entrypoint
        def test_func(data, context):
            return {"test": True}

        # Check that 'app' was injected into globals
        assert "app" in mock_globals
        assert isinstance(mock_globals["app"], FastAPI)
        assert "test_func" in mock_globals["app"].title

    @patch("sys._getframe")
    def test_entrypoint_returns_original_function(self, mock_getframe):
        """Test that entrypoint returns the original function."""

        # Mock the frame
        mock_frame = Mock()
        mock_frame.f_globals = {}
        mock_getframe.return_value = mock_frame

        def original_func(data, context):
            return {"original": True}

        decorated_func = entrypoint(original_func)

        # Should return the original function
        assert decorated_func is original_func

    @patch("sys._getframe")
    def test_fastapi_app_configuration(self, mock_getframe):
        """Test that FastAPI app is configured correctly."""

        mock_frame = Mock()
        mock_globals = {}
        mock_frame.f_globals = mock_globals
        mock_getframe.return_value = mock_frame

        @entrypoint
        def my_agent(data, context):
            return {"message": "hello"}

        app = mock_globals["app"]

        # Check app configuration
        assert "my_agent" in app.title
        assert "Gradient ADK build agent" in app.description
        assert app.version == "1.0.0"


class TestFastAPIEndpoints:
    """Test cases for the generated FastAPI endpoints."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_runtime_manager = Mock()
        self.mock_context = Mock()

        # Create a test app
        with patch("sys._getframe") as mock_getframe:
            mock_frame = Mock()
            self.mock_globals = {}
            mock_frame.f_globals = self.mock_globals
            mock_getframe.return_value = mock_frame

            @entrypoint
            def test_agent(data, context):
                return {"echo": data}

            self.test_func = test_agent
            self.app = self.mock_globals["app"]
            self.client = TestClient(self.app)

    @patch("gradient_adk.decorator.get_runtime_manager")
    @patch("gradient_adk.decorator.get_current_context")
    @pytest.mark.asyncio
    async def test_completions_endpoint_success(
        self, mock_get_context, mock_get_runtime_manager
    ):
        """Test successful request to /completions endpoint."""

        mock_get_runtime_manager.return_value = self.mock_runtime_manager
        mock_get_context.return_value = self.mock_context

        # Mock successful execution
        expected_result = {"message": "success", "data": {"key": "value"}}
        self.mock_runtime_manager.run_entrypoint = AsyncMock(
            return_value=expected_result
        )

        # Make request
        response = self.client.post("/completions", json={"key": "value"})

        # Assertions
        assert response.status_code == 200
        assert response.json() == expected_result

        # Verify runtime manager was called correctly
        self.mock_runtime_manager.run_entrypoint.assert_called_once_with(
            self.test_func, {"key": "value"}, self.mock_context
        )

    @patch("gradient_adk.decorator.get_runtime_manager")
    @patch("gradient_adk.decorator.get_current_context")
    def test_completions_endpoint_invalid_json(
        self, mock_get_context, mock_get_runtime_manager
    ):
        """Test /completions endpoint with invalid JSON."""

        mock_get_runtime_manager.return_value = self.mock_runtime_manager
        mock_get_context.return_value = self.mock_context

        # Make request with invalid JSON
        response = self.client.post(
            "/completions",
            content="invalid json",  # Use content instead of data
            headers={"Content-Type": "application/json"},
        )

        # Due to the error handling in the decorator, this will be 500
        # The decorator catches the HTTPException and logs it, then returns 500
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"

    @patch("gradient_adk.decorator.get_runtime_manager")
    @patch("gradient_adk.decorator.get_current_context")
    @pytest.mark.asyncio
    async def test_completions_endpoint_runtime_error(
        self, mock_get_context, mock_get_runtime_manager
    ):
        """Test /completions endpoint when runtime manager raises exception."""

        mock_get_runtime_manager.return_value = self.mock_runtime_manager
        mock_get_context.return_value = self.mock_context

        # Mock runtime manager to raise exception
        self.mock_runtime_manager.run_entrypoint = AsyncMock(
            side_effect=Exception("Runtime error")
        )

        # Make request
        response = self.client.post("/completions", json={"test": "data"})

        # Should return 500 error
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"

    @patch("gradient_adk.decorator.get_runtime_manager")
    @patch("gradient_adk.decorator.get_current_context")
    @pytest.mark.asyncio
    async def test_completions_endpoint_streaming_response(
        self, mock_get_context, mock_get_runtime_manager
    ):
        """Test /completions endpoint with streaming response."""

        mock_get_runtime_manager.return_value = self.mock_runtime_manager
        mock_get_context.return_value = self.mock_context

        # Create mock streaming response
        async def mock_content():
            yield b"chunk1"
            yield b"chunk2"

        mock_streaming_response = GradientStreamingResponse(
            content=mock_content(),
            media_type="text/plain",
            headers={"X-Custom": "header"},
        )

        self.mock_runtime_manager.run_entrypoint = AsyncMock(
            return_value=mock_streaming_response
        )

        # Make request
        response = self.client.post("/completions", json={"stream": True})

        # Should return streaming response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.headers["X-Custom"] == "header"

    def test_health_endpoint(self):
        """Test the /health endpoint."""

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["entrypoint"] == "test_agent"


class TestRunServer:
    """Test cases for the run_server function."""

    @patch("gradient_adk.decorator.uvicorn.run")
    def test_run_server_default_params(self, mock_uvicorn_run):
        """Test run_server with default parameters."""

        mock_app = Mock(spec=FastAPI)

        run_server(mock_app)

        mock_uvicorn_run.assert_called_once_with(mock_app, host="0.0.0.0", port=8080)

    @patch("gradient_adk.decorator.uvicorn.run")
    def test_run_server_custom_params(self, mock_uvicorn_run):
        """Test run_server with custom parameters."""

        mock_app = Mock(spec=FastAPI)

        run_server(mock_app, host="127.0.0.1", port=3000, reload=True, debug=True)

        mock_uvicorn_run.assert_called_once_with(
            mock_app, host="127.0.0.1", port=3000, reload=True, debug=True
        )


class TestIntegration:
    """Integration tests for the decorator functionality."""

    @patch("gradient_adk.decorator.get_runtime_manager")
    @patch("gradient_adk.decorator.get_current_context")
    def test_full_decorator_workflow(self, mock_get_context, mock_get_runtime_manager):
        """Test complete workflow from decoration to endpoint call."""

        mock_runtime_manager = Mock()
        mock_context = Mock()
        mock_get_runtime_manager.return_value = mock_runtime_manager
        mock_get_context.return_value = mock_context

        # Create a real module globals dict by using the test module's globals
        # This avoids the complex mocking of sys._getframe
        test_globals = {}

        with patch("sys._getframe") as mock_getframe:
            mock_frame = Mock()
            mock_frame.f_globals = test_globals
            mock_getframe.return_value = mock_frame

            @entrypoint
            def integration_agent(data, context):
                return {"processed": data, "context_id": context.id}

        # Mock runtime execution
        expected_result = {"processed": {"input": "test"}, "context_id": "ctx-123"}
        mock_runtime_manager.run_entrypoint = AsyncMock(return_value=expected_result)
        mock_context.id = "ctx-123"

        # Test the generated app
        app = test_globals["app"]

        # Use a simpler client setup to avoid logging issues
        with TestClient(app) as client:
            # Make request
            response = client.post("/completions", json={"input": "test"})

        # Verify response
        assert response.status_code == 200
        assert response.json() == expected_result

        # Verify runtime manager was called
        mock_runtime_manager.run_entrypoint.assert_called_once_with(
            integration_agent, {"input": "test"}, mock_context
        )

    @patch("sys._getframe")
    def test_multiple_decorated_functions(self, mock_getframe):
        """Test that multiple functions can be decorated independently."""

        # Setup different mock globals for each function
        mock_frame1 = Mock()
        mock_globals1 = {}
        mock_frame1.f_globals = mock_globals1

        mock_frame2 = Mock()
        mock_globals2 = {}
        mock_frame2.f_globals = mock_globals2

        mock_getframe.side_effect = [mock_frame1, mock_frame2]

        @entrypoint
        def agent1(data, context):
            return {"agent": "1"}

        @entrypoint
        def agent2(data, context):
            return {"agent": "2"}

        # Both should have their own apps
        assert "app" in mock_globals1
        assert "app" in mock_globals2
        assert mock_globals1["app"] is not mock_globals2["app"]

        # Apps should have different titles
        assert "agent1" in mock_globals1["app"].title
        assert "agent2" in mock_globals2["app"].title


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("sys._getframe") as mock_getframe:
            mock_frame = Mock()
            mock_globals = {}
            mock_frame.f_globals = mock_globals
            mock_getframe.return_value = mock_frame

            @entrypoint
            def error_test_agent(data, context):
                return {"test": True}

            self.app = mock_globals["app"]
            self.client = TestClient(self.app)

    @patch("gradient_adk.decorator.get_runtime_manager")
    @patch("gradient_adk.decorator.get_current_context")
    @patch("gradient_adk.decorator.logger")
    def test_exception_logging(
        self, mock_logger, mock_get_context, mock_get_runtime_manager
    ):
        """Test that exceptions are properly logged."""

        mock_runtime_manager = Mock()
        mock_get_runtime_manager.return_value = mock_runtime_manager
        mock_get_context.return_value = Mock()

        # Mock runtime manager to raise exception
        test_exception = Exception("Test error")
        mock_runtime_manager.run_entrypoint = AsyncMock(side_effect=test_exception)

        # Make request
        response = self.client.post("/completions", json={"test": "data"})

        # Verify error response
        assert response.status_code == 500

        # Verify logging was called
        mock_logger.error.assert_called_once_with(
            "Error in entrypoint", error="Test error", exc_info=True
        )
