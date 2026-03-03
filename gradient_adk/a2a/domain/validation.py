"""Domain validation logic."""

from gradient_adk.a2a.domain.models import DomainMessage, ValidationError, Result, Ok, Err


class MessageValidator:
    """Validates domain messages according to business rules."""

    def __init__(self):
        """No dependencies - pure validation."""
        pass

    def validate_text_only(
        self,
        message: DomainMessage
    ) -> Result[DomainMessage, ValidationError]:
        """
        Validate that message contains only text parts.
        MVP supports text/plain only.
        """
        if not message.is_text_only:
            return Err(ValidationError(
                message="Only text/plain input is supported in MVP. "
                        "FilePart and DataPart are not supported.",
                code="UNSUPPORTED_CONTENT_TYPE"
            ))

        return Ok(message)

    def validate_non_empty(
        self,
        message: DomainMessage
    ) -> Result[DomainMessage, ValidationError]:
        """Validate that message text is not empty."""
        if not message.text.strip():
            return Err(ValidationError(
                message="Message text cannot be empty",
                code="EMPTY_MESSAGE"
            ))

        return Ok(message)

    def validate_all(
        self,
        message: DomainMessage
    ) -> Result[DomainMessage, ValidationError]:
        """Run all validations. Returns first error or Ok."""
        result = self.validate_text_only(message)
        if result.is_err():
            return result

        result = self.validate_non_empty(message)
        if result.is_err():
            return result

        return Ok(message)
