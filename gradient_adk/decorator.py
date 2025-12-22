"""
Gradient ADK entrypoint decorator.

Provides the @entrypoint decorator that creates a FastAPI app for agent functions,
with automatic support for both regular and streaming responses.
"""

from __future__ import annotations
import inspect
import json
from typing import Callable, Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
import uvicorn

from .logging import get_logger

logger = get_logger(__name__)

from gradient_adk.runtime.langgraph.helpers import capture_graph, get_tracker

capture_graph()


class _StreamingIteratorWithTracking:
    """
    Async iterator that wraps a user's async generator for streaming responses.

    This uses direct __anext__ calls instead of nested generators to avoid
    buffering issues that can occur with generator-within-generator patterns.
    """

    def __init__(self, user_generator, tracker, entrypoint_name: str):
        self._gen = user_generator
        self._tracker = tracker
        self._entrypoint = entrypoint_name
        self._collected: List[str] = []
        self._finished = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._finished:
            raise StopAsyncIteration

        try:
            chunk = await self._gen.__anext__()

            # Convert chunk to string
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8", errors="replace")
            elif isinstance(chunk, dict):
                chunk_str = json.dumps(chunk)
            elif chunk is None:
                # Skip None values, get next chunk
                return await self.__anext__()
            else:
                chunk_str = str(chunk)

            self._collected.append(chunk_str)
            return chunk_str

        except StopAsyncIteration:
            # Stream complete - submit tracking data
            self._finished = True
            await self._submit_tracking(error=None)
            raise

        except Exception as e:
            # Error during streaming
            self._finished = True
            await self._submit_tracking(error=str(e))
            raise

    async def _submit_tracking(self, error: Optional[str]) -> None:
        """Submit collected stream data to tracker."""
        if not self._tracker:
            return

        try:
            self._tracker._req["outputs"] = "".join(self._collected)
            if error:
                self._tracker._req["error"] = error
            await self._tracker._submit()
        except Exception:
            # Never break streaming due to tracking errors
            pass


def entrypoint(func: Callable) -> Callable:
    """
    Decorator that creates a FastAPI app and exposes it as `app` in the caller module.

    The decorated function can accept either (data) or (data, context).

    For streaming responses, use an async generator function:

        @entrypoint
        async def my_agent(payload):
            async for chunk in some_stream:
                yield chunk

    For regular responses, use a normal async function:

        @entrypoint
        async def my_agent(payload):
            return {"result": "Hello"}

    The decorator automatically detects async generators and handles streaming.
    """
    sig = inspect.signature(func)
    num_params = len(sig.parameters)

    if num_params < 1 or num_params > 2:
        raise ValueError(f"{func.__name__} must accept (data) or (data, context)")

    is_async_generator = inspect.isasyncgenfunction(func)

    fastapi_app = FastAPI(title=f"Gradient Agent - {func.__name__}", version="1.0.0")

    @fastapi_app.on_event("shutdown")
    async def _shutdown():
        """Flush pending trace submissions on shutdown."""
        try:
            tr = get_tracker()
            if tr and hasattr(tr, "aclose"):
                await tr.aclose()
        except Exception:
            pass

    @fastapi_app.post("/run")
    async def run(req: Request):
        """Main agent invocation endpoint."""
        # Parse request body
        try:
            body = await req.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        is_evaluation = "evaluation-id" in req.headers

        # Initialize tracker
        tr = None
        try:
            tr = get_tracker()
            if tr:
                tr.on_request_start(func.__name__, body, is_evaluation=is_evaluation)
        except Exception:
            pass

        # Handle streaming responses (async generator functions)
        if is_async_generator:
            try:
                if num_params == 1:
                    user_gen = func(body)
                else:
                    user_gen = func(body, None)
            except Exception as e:
                if tr:
                    try:
                        tr.on_request_end(outputs=None, error=str(e))
                    except Exception:
                        pass
                logger.error("Error creating generator", error=str(e), exc_info=True)
                raise HTTPException(status_code=500, detail="Internal server error")

            # Wrap in tracking iterator
            streaming_iter = _StreamingIteratorWithTracking(user_gen, tr, func.__name__)

            return FastAPIStreamingResponse(
                streaming_iter,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Handle regular (non-streaming) responses
        try:
            if num_params == 1:
                if inspect.iscoroutinefunction(func):
                    result = await func(body)
                else:
                    result = func(body)
            else:
                if inspect.iscoroutinefunction(func):
                    result = await func(body, None)
                else:
                    result = func(body, None)
        except Exception as e:
            if tr:
                try:
                    tr.on_request_end(outputs=None, error=str(e))
                except Exception:
                    pass
            logger.error("Error in /run", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

        # Regular response with tracking
        trace_id = None
        if tr:
            try:
                tr.on_request_end(outputs=result, error=None)
                if is_evaluation:
                    trace_id = await tr.submit_and_get_trace_id()
            except Exception:
                pass

        if trace_id:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                content=result,
                headers={"X-Gradient-Trace-Id": trace_id},
            )

        return result

    @fastapi_app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "entrypoint": func.__name__}

    # Expose fastapi_app in caller's module for `uvicorn main:fastapi_app`
    import sys

    sys._getframe(1).f_globals["fastapi_app"] = fastapi_app

    return func


def run_server(fastapi_app: FastAPI, host: str = "0.0.0.0", port: int = 8080, **kwargs):
    """Run the FastAPI server with uvicorn."""
    uvicorn.run(fastapi_app, host=host, port=port, **kwargs)
