"""
Gradient entrypoint decorator for creating FastAPI agents.
"""

from __future__ import annotations
import functools
import inspect
from typing import Any, Callable, Dict, Optional, Union, get_type_hints
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import uvicorn

# Import runtime system
from .runtime.manager import get_runtime_manager


class CompletionRequest(BaseModel):
    """Standard completion request model."""

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    """Standard completion response model."""

    completion: str
    metadata: Optional[Dict[str, Any]] = None


class EntrypointRegistry:
    """Global registry for entrypoint functions."""

    def __init__(self):
        self._function: Optional[Callable] = None
        self._app: Optional[FastAPI] = None

    def register(self, func: Callable) -> FastAPI:
        """Register an entrypoint function and create the FastAPI app."""
        self._function = func

        # Create FastAPI app
        self._app = FastAPI(
            title="Gradient Agent",
            description="AI Agent powered by Gradient",
            version="1.0.0",
        )

        # Get function signature for validation
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        @self._app.post("/completions", response_model=CompletionResponse)
        async def completions(request: CompletionRequest) -> CompletionResponse:
            """Handle completion requests."""
            try:
                # Get runtime manager
                runtime_manager = get_runtime_manager()

                # Start request tracking
                runtime_manager.start_request(
                    entrypoint_name=func.__name__,
                    inputs={
                        "prompt": request.prompt,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "metadata": request.metadata,
                    },
                )

                try:
                    # Prepare arguments for the entrypoint function
                    kwargs = {}

                    # Map request fields to function parameters
                    for param_name, param in sig.parameters.items():
                        if param_name == "prompt":
                            kwargs[param_name] = request.prompt
                        elif param_name == "max_tokens":
                            if request.max_tokens is not None:
                                kwargs[param_name] = request.max_tokens
                        elif param_name == "temperature":
                            if request.temperature is not None:
                                kwargs[param_name] = request.temperature
                        elif param_name == "metadata":
                            if request.metadata is not None:
                                kwargs[param_name] = request.metadata
                        else:
                            # Check if parameter has a default value
                            if param.default is not inspect.Parameter.empty:
                                continue
                            else:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Required parameter '{param_name}' not provided in request",
                                )

                    # Call the entrypoint function
                    result = func(**kwargs)

                    # Handle different return types
                    if isinstance(result, str):
                        response = CompletionResponse(completion=result)
                    elif isinstance(result, dict):
                        if "completion" in result:
                            response = CompletionResponse(
                                completion=result["completion"],
                                metadata=result.get("metadata"),
                            )
                        else:
                            raise HTTPException(
                                status_code=500,
                                detail="Function must return a string or dict with 'completion' key",
                            )
                    else:
                        # Try to convert to string
                        response = CompletionResponse(completion=str(result))

                    # End request tracking with success
                    runtime_manager.end_request(
                        outputs={
                            "completion": response.completion,
                            "metadata": response.metadata,
                        }
                    )

                    return response

                except Exception as e:
                    # End request tracking with error
                    runtime_manager.end_request(error=str(e))
                    raise HTTPException(status_code=500, detail=str(e))

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self._app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "service": "gradient-agent"}

        @self._app.get("/")
        async def root():
            """Root endpoint with basic info."""
            return {
                "service": "gradient-agent",
                "entrypoint": func.__name__ if func else None,
                "endpoints": ["/completions", "/health"],
            }

        return self._app

    def get_app(self) -> FastAPI:
        """Get the registered FastAPI app."""
        if self._app is None:
            raise RuntimeError(
                "No entrypoint function decorated. Use @entrypoint decorator first."
            )
        return self._app


# Global registry instance
_registry = EntrypointRegistry()


def entrypoint(func: Callable) -> Callable:
    """
    Decorator to mark a function as the agent entrypoint.

    The decorated function should accept parameters that match CompletionRequest
    fields and return either a string or a dict with 'completion' key.

    Example:
        @entrypoint
        def my_agent(prompt: str, max_tokens: int = 100) -> str:
            return f"Response to: {prompt}"
    """
    # Register the function and create the app
    _registry.register(func)

    # Return the original function unchanged
    return func


def get_app() -> FastAPI:
    """Get the FastAPI app instance."""
    return _registry.get_app()


def run_server(host: str = "0.0.0.0", port: int = 8080, **kwargs):
    """
    Run the FastAPI server with the decorated entrypoint.

    Args:
        host: Host to bind to
        port: Port to bind to
        **kwargs: Additional arguments to pass to uvicorn.run()
    """
    app = get_app()
    uvicorn.run(app, host=host, port=port, **kwargs)
