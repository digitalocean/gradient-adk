"""Domain transformation logic."""

from typing import Any

from gradient_adk.a2a.domain.models import (
    DomainMessage,
    GradientInput,
    GradientOutput,
    TransformationError,
    MessageRole,
    Result,
    Ok,
    Err,
)


class MessageTransformer:
    """Transforms messages between A2A and Gradient formats."""

    def __init__(self, input_key: str = "prompt"):
        """
        Args:
            input_key: Key to use in Gradient input dict
        """
        self.input_key = input_key

    def to_gradient_input(
        self,
        message: DomainMessage,
        session_id: str | None = None,
    ) -> GradientInput:
        """Transform domain message to Gradient input format."""
        return GradientInput(
            prompt=message.text,
            session_id=session_id,
        )

    def to_gradient_input_from_text(
        self,
        text: str,
        session_id: str | None = None,
    ) -> GradientInput:
        """Create Gradient input from text string directly."""
        return GradientInput(
            prompt=text,
            session_id=session_id,
        )

    def from_gradient_output(
        self,
        output: GradientOutput
    ) -> DomainMessage:
        """Transform Gradient output to domain message."""
        return DomainMessage(
            text=output.text,
            role=MessageRole.AGENT,
            has_file_parts=False,
            has_data_parts=False,
        )


class OutputExtractor:
    """Extracts output from various Gradient agent result formats."""

    def __init__(self, output_keys: list[str] | None = None):
        """
        Args:
            output_keys: Keys to try in order
        """
        self.output_keys = output_keys or ["output", "response", "result"]

    def extract_text(self, result: Any) -> Result[str, TransformationError]:
        """
        Extract text from agent result.
        Handles: string, dict, other types.
        """
        if isinstance(result, str):
            return Ok(result)

        if isinstance(result, dict):
            for key in self.output_keys:
                if key in result and result[key]:
                    return Ok(str(result[key]))
            return Ok(str(result))

        try:
            return Ok(str(result))
        except Exception as e:
            return Err(TransformationError(
                message=f"Failed to convert result to text: {str(e)}",
                code="TEXT_EXTRACTION_FAILED"
            ))
