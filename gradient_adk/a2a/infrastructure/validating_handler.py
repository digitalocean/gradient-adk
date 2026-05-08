"""Custom request handler with domain validation.

This handler validates messages BEFORE creating tasks, ensuring that
validation errors are returned as JSON-RPC errors rather than creating
failed tasks.
"""

from typing import Any

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.context import ServerCallContext
from a2a.types import AgentCard, Message, Task, SendMessageRequest, InvalidParamsError

from gradient_adk.a2a.domain.validation import MessageValidator
from gradient_adk.a2a.domain.models import DomainMessage, MessageRole


class ValidatingRequestHandler(DefaultRequestHandler):
    """
    Extends SDK's DefaultRequestHandler to validate messages before task creation.

    Architecture:
        Request -> ValidatingHandler.validate() -> DefaultRequestHandler.on_message_send()
                        |                                    |
                  Domain Validator                    Create Task -> Execute

    This ensures:
    - Validation happens at the right time (before task creation)
    - Invalid requests return JSON-RPC errors (not failed tasks)
    - Domain validation rules are enforced consistently
    - Executor can assume input is valid
    """

    def __init__(
        self,
        agent_executor: Any,
        task_store: Any,
        agent_card: AgentCard,
        validator: MessageValidator,
    ):
        """
        Initialize validating request handler.

        Args:
            agent_executor: The agent executor to use
            task_store: The task store for persistence
            validator: Domain validator for business rules
        """
        super().__init__(agent_executor, task_store, agent_card)
        self.validator = validator

    async def on_message_send(
        self,
        params: SendMessageRequest,
        context: ServerCallContext | None = None,
    ) -> Message | Task:
        """
        Validate message before delegating to parent handler.

        Args:
            params: The message send parameters
            context: The server call context

        Returns:
            Task or Message response

        Raises:
            ServerError: If validation fails (returned as JSON-RPC error)

        Error Code Mapping:
            Domain error codes are included in the error message in format:
            "[DOMAIN_CODE] Error message"

            Domain codes:
            - UNSUPPORTED_CONTENT_TYPE: Message contains file/data parts (not text-only)
            - EMPTY_MESSAGE: Message text is empty or whitespace-only

            JSON-RPC error code is always -32602 (Invalid params) for validation errors.
        """
        domain_message = self._to_domain_message(params.message)

        validation_result = self.validator.validate_all(domain_message)

        if validation_result.is_err():
            error = validation_result.error
            error_message = f"[{error.code}] {error.message}"
            raise InvalidParamsError(message=error_message)

        return await super().on_message_send(params, context)

    def _to_domain_message(self, message: Message) -> DomainMessage:
        """
        Convert SDK Message to DomainMessage for validation.

        Args:
            message: SDK message

        Returns:
            Domain message with extracted text and part flags
        """
        text_parts = [
            part.text
            for part in message.parts
            if part.WhichOneof("content") == "text"
        ]
        text = " ".join(text_parts) if text_parts else ""

        has_file = any(
            part.WhichOneof("content") in {"raw", "url"} for part in message.parts
        )
        has_data = any(
            part.WhichOneof("content") == "data" for part in message.parts
        )

        return DomainMessage(
            text=text,
            role=MessageRole.USER,
            has_file_parts=has_file,
            has_data_parts=has_data,
        )
