"""Domain ports - abstract interfaces."""

from typing import Protocol, Any
from gradient_adk.a2a.domain.models import GradientInput, DomainMessage, ValidationError, TransformationError


class AgentPort(Protocol):
    """Port for executing Gradient agents."""

    async def execute(
        self,
        agent_input: GradientInput
    ) -> Any:
        """
        Execute agent with given input.
        Returns: Agent result (any type)
        """
        ...


class EventPort(Protocol):
    """Port for publishing events."""

    async def publish_completed(
        self,
        task_id: str,
        message: DomainMessage,
    ) -> None:
        """Publish task completion event."""
        ...

    async def publish_failed(
        self,
        task_id: str,
        error: ValidationError | TransformationError,
    ) -> None:
        """Publish task failure event."""
        ...

    async def publish_canceled(self, task_id: str) -> None:
        """Publish task cancellation."""
        ...
