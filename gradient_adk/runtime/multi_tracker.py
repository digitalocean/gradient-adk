from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

from .digitalocean_tracker import DigitalOceanTracesTracker


class MultiTracker:
    def __init__(self, trackers: Iterable[Any]) -> None:
        self._trackers = [tracker for tracker in trackers if tracker is not None]

    def on_request_start(
        self,
        entrypoint: str,
        inputs: dict[str, Any],
        is_evaluation: bool = False,
        session_id: Optional[str] = None,
        parent_context: Any = None,
        evaluation_run_uuid: Optional[str] = None,
    ) -> None:
        for tracker in self._trackers:
            tracker.on_request_start(
                entrypoint,
                inputs,
                is_evaluation=is_evaluation,
                session_id=session_id,
                parent_context=parent_context,
                evaluation_run_uuid=evaluation_run_uuid,
            )

    def on_request_end(self, outputs: Any | None, error: Optional[str]) -> None:
        for tracker in self._trackers:
            tracker.on_request_end(outputs=outputs, error=error)

    def on_node_start(self, node: Any) -> None:
        for tracker in self._trackers:
            tracker.on_node_start(node)

    def on_node_end(self, node: Any, outputs: Any | None) -> None:
        for tracker in self._trackers:
            tracker.on_node_end(node, outputs)

    def on_node_error(self, node: Any, error: BaseException) -> None:
        for tracker in self._trackers:
            tracker.on_node_error(node, error)

    async def submit_and_get_trace_id(self) -> Optional[str]:
        legacy_trace_id: Optional[str] = None
        fallback_trace_id: Optional[str] = None

        for tracker in self._trackers:
            trace_id = await tracker.submit_and_get_trace_id()
            if trace_id and fallback_trace_id is None:
                fallback_trace_id = trace_id
            if trace_id and isinstance(tracker, DigitalOceanTracesTracker):
                legacy_trace_id = trace_id

        return legacy_trace_id or fallback_trace_id

    async def aclose(self) -> None:
        await asyncio.gather(
            *(tracker.aclose() for tracker in self._trackers),
            return_exceptions=True,
        )
