"""Dependency injection - compose the application."""

from typing import Any

from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

from gradient_adk.a2a.domain.validation import MessageValidator
from gradient_adk.a2a.domain.transformation import MessageTransformer, OutputExtractor
from gradient_adk.a2a.adapters.primary.a2a_executor import A2AExecutorAdapter
from gradient_adk.a2a.adapters.secondary.agent_client import GradientAgentClient
from gradient_adk.a2a.adapters.secondary.event_publisher import A2AEventPublisher
from gradient_adk.a2a.infrastructure.validating_handler import ValidatingRequestHandler


def compose_executor(
    agent_func: Any,
    input_key: str = "prompt",
    output_keys: list[str] | None = None,
) -> A2AExecutorAdapter:
    """
    Compose all dependencies and create executor.

    Args:
        agent_func: Gradient agent function
        input_key: Key for agent input (default: "prompt")
        output_keys: Keys to try for output

    Returns:
        Fully composed A2AExecutorAdapter
    """
    transformer = MessageTransformer(input_key=input_key)
    output_extractor = OutputExtractor(output_keys=output_keys)

    agent_client = GradientAgentClient(agent_func, input_key=input_key)

    def event_publisher_factory(event_queue, context_id, task_id):
        return A2AEventPublisher(event_queue, context_id, task_id)

    executor = A2AExecutorAdapter(
        transformer=transformer,
        output_extractor=output_extractor,
        agent_client=agent_client,
        event_publisher_factory=event_publisher_factory,
    )

    return executor


def compose_request_handler(
    agent_func: Any,
    agent_card: AgentCard,
    input_key: str = "prompt",
    output_keys: list[str] | None = None,
) -> ValidatingRequestHandler:
    """
    Compose request handler with validation and execution.

    This creates a ValidatingRequestHandler that:
    1. Validates messages using domain validator (before task creation)
    2. Creates tasks for valid messages
    3. Executes via A2AExecutorAdapter

    Args:
        agent_func: Gradient agent function
        input_key: Key for agent input (default: "prompt")
        output_keys: Keys to try for output

    Returns:
        Fully composed ValidatingRequestHandler
    """
    executor = compose_executor(
        agent_func,
        input_key=input_key,
        output_keys=output_keys,
    )

    validator = MessageValidator()

    return ValidatingRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
        validator=validator,
    )
