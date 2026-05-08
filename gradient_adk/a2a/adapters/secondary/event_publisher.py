"""Secondary adapter - publishes events to A2A SDK."""

from a2a.server.tasks import TaskUpdater
from a2a.server.events import EventQueue
from a2a.types import TaskState
from a2a.helpers import new_text_message

from gradient_adk.a2a.domain.models import DomainMessage, ValidationError, TransformationError
from gradient_adk.a2a.domain.ports import EventPort


TASK_STATE_COMPLETED = TaskState.Value("TASK_STATE_COMPLETED")
TASK_STATE_FAILED = TaskState.Value("TASK_STATE_FAILED")
TASK_STATE_CANCELED = TaskState.Value("TASK_STATE_CANCELED")


class A2AEventPublisher:
    """Publishes events to SDK EventQueue. Implements EventPort."""

    def __init__(self, event_queue: EventQueue, context_id: str, task_id: str):
        """
        Args:
            event_queue: SDK event queue
            context_id: Context identifier
            task_id: Task identifier
        """
        self.task_updater = TaskUpdater(event_queue, task_id, context_id)
        self.context_id = context_id
        self.task_id = task_id

    async def publish_completed(
        self,
        task_id: str,
        message: DomainMessage,
    ) -> None:
        """Publish task completion."""
        sdk_message = new_text_message(
            text=message.text,
            context_id=self.context_id,
            task_id=task_id,
        )

        await self.task_updater.update_status(
            TASK_STATE_COMPLETED,
            message=sdk_message,
        )

    async def publish_failed(
        self,
        task_id: str,
        error: ValidationError | TransformationError,
    ) -> None:
        """Publish task failure."""
        error_text = f"Agent error: {error.message}"
        sdk_message = new_text_message(
            text=error_text,
            context_id=self.context_id,
            task_id=task_id,
        )

        await self.task_updater.update_status(
            TASK_STATE_FAILED,
            message=sdk_message,
        )

    async def publish_canceled(self, task_id: str) -> None:
        """Publish task cancellation."""
        await self.task_updater.update_status(TASK_STATE_CANCELED)
