"""
Gradient entrypoint decorator for creating FastAPI agents.
"""

from __future__ import annotations
import inspect
from typing import Any, Callable, Optional
from fastapi import FastAPI, HTTPException, Request
import uvicorn

from .runtime.manager import get_runtime_manager
from .runtime.context import get_current_context


# Responses will be plain dict objects; no pydantic model or wrapper.


class EntrypointRegistry:
    """Global registry for entrypoint functions."""

    def __init__(self):
        self._function: Optional[Callable] = None
        self._app: Optional[FastAPI] = None

    def register(self, func: Callable) -> FastAPI:
        """Register an entrypoint function and create the FastAPI app."""
        # Validate that the function has exactly 2 parameters
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if len(params) != 2:
            raise ValueError(
                f"Entrypoint function '{func.__name__}' must have exactly 2 parameters (data, context), "
                f"but has {len(params)}: {params}"
            )

        self._function = func

        self._app = FastAPI(
            title="Gradient Agent",
            description="AI Agent powered by Gradient",
            version="1.0.0",
        )

        @self._app.post("/completions", response_model=None)
        async def completions(req: Request):
            runtime_manager = get_runtime_manager()
            try:
                # Get raw JSON body without any transformation
                try:
                    body = await req.json()
                except Exception:
                    body = {}

                runtime_manager.start_request(
                    entrypoint_name=func.__name__, inputs=body
                )

                # Call user function with data and context - user function must accept exactly 2 parameters
                try:
                    # Get the current context for the second parameter
                    context = get_current_context()
                    result = func(body, context)
                except Exception as e:
                    runtime_manager.end_request(error=str(e))
                    raise HTTPException(status_code=500, detail=str(e)) from e

                # Return result exactly as user provided it
                runtime_manager.end_request(outputs=result)
                return result
            except HTTPException:
                raise
            except Exception as e:
                runtime_manager.end_request(error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self._app.get("/health")
        async def health():
            return {"status": "healthy", "service": "gradient-agent"}

        @self._app.get("/")
        async def root():
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

    The decorated function must accept exactly 2 parameters:
    1. data: The request data (dict)
    2. context: The request context object (can be named anything)

    Example:
        @entrypoint
        def my_agent(data, ctx):
            return f"Processing: {data}"

        @entrypoint
        def my_other_agent(request_data, request_context):
            return {"result": request_data["query"]}
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
