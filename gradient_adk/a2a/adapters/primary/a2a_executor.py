"""Primary adapter - implements A2A SDK interface."""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue

from gradient_adk.a2a.domain.models import GradientOutput, TransformationError
from gradient_adk.a2a.domain.transformation import MessageTransformer, OutputExtractor
from gradient_adk.a2a.domain.ports import AgentPort
from gradient_adk.logging import get_logger

logger = get_logger(__name__)


class A2AExecutorAdapter(AgentExecutor):
    """
    Implements SDK's AgentExecutor interface.

    Orchestrates transformation and execution.
    Note: Validation happens in ValidatingRequestHandler BEFORE tasks are created.
    """

    def __init__(
        self,
        transformer: MessageTransformer,
        output_extractor: OutputExtractor,
        agent_client: AgentPort,
        event_publisher_factory,
    ):
        """All dependencies injected via constructor."""
        self.transformer = transformer
        self.output_extractor = output_extractor
        self.agent_client = agent_client
        self.event_publisher_factory = event_publisher_factory

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute agent for A2A request.

        Flow:
        1. Extract user input (already validated by ValidatingRequestHandler)
        2. Transform to Gradient format
        3. Execute agent
        4. Extract output
        5. Publish result

        Note: No validation here - ValidatingRequestHandler validates BEFORE
        creating tasks, so we can assume input is valid.

        Exception Handling:
            Catches all exceptions from agent execution (agent functions can raise
            any exception type). Logs full exception with traceback for debugging
            and publishes failure event before re-raising to maintain exception
            propagation.
        """
        event_publisher = self.event_publisher_factory(
            event_queue,
            context.context_id,
            context.task_id
        )

        try:
            user_text = context.get_user_input()

            gradient_input = self.transformer.to_gradient_input_from_text(
                user_text,
                session_id=context.context_id,
            )

            agent_result = await self.agent_client.execute(gradient_input)

            text_result = self.output_extractor.extract_text(agent_result)
            if text_result.is_err():
                await event_publisher.publish_failed(
                    context.task_id,
                    text_result.error,
                )
                return

            output = GradientOutput(text=text_result.value)
            response_message = self.transformer.from_gradient_output(output)

            await event_publisher.publish_completed(
                context.task_id,
                response_message,
            )

        except Exception as e:
            logger.error(
                "Agent execution failed",
                task_id=context.task_id,
                context_id=context.context_id,
                exception_type=type(e).__name__,
                exc_info=True,
            )

            exception_type = type(e).__name__
            error_message = f"Agent execution failed: {exception_type}: {str(e)}"
            await event_publisher.publish_failed(
                context.task_id,
                TransformationError(
                    message=error_message,
                    code="AGENT_EXECUTION_ERROR"
                ),
            )
            raise

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel task execution (best-effort)."""
        event_publisher = self.event_publisher_factory(
            event_queue,
            context.context_id,
            context.task_id
        )
        await event_publisher.publish_canceled(context.task_id)
