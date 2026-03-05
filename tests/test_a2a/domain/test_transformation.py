"""Unit tests for domain transformation logic."""

import pytest
from gradient_adk.a2a.domain.models import (
    DomainMessage,
    MessageRole,
    GradientInput,
    GradientOutput,
)
from gradient_adk.a2a.domain.transformation import MessageTransformer, OutputExtractor


class TestMessageTransformer:
    """Tests for MessageTransformer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = MessageTransformer(input_key="prompt")

    def test_to_gradient_input_creates_correct_format(self):
        """Test transformation to Gradient input format."""
        message = DomainMessage(
            text="What is 2+2?",
            role=MessageRole.USER,
        )

        result = self.transformer.to_gradient_input(message, session_id="session-123")

        assert isinstance(result, GradientInput)
        assert result.prompt == "What is 2+2?"
        assert result.session_id == "session-123"

    def test_to_gradient_input_without_session_id(self):
        """Test transformation without session ID."""
        message = DomainMessage(
            text="Hello",
            role=MessageRole.USER,
        )

        result = self.transformer.to_gradient_input(message)

        assert isinstance(result, GradientInput)
        assert result.prompt == "Hello"
        assert result.session_id is None

    def test_from_gradient_output_creates_agent_message(self):
        """Test transformation from Gradient output."""
        output = GradientOutput(text="The answer is 4", metadata=None)

        result = self.transformer.from_gradient_output(output)

        assert isinstance(result, DomainMessage)
        assert result.text == "The answer is 4"
        assert result.role == MessageRole.AGENT
        assert result.has_file_parts is False
        assert result.has_data_parts is False


class TestOutputExtractor:
    """Tests for OutputExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = OutputExtractor(output_keys=["output", "response", "result"])

    def test_extract_text_from_string_result(self):
        """Test extracting text from string result."""
        result = self.extractor.extract_text("Hello, world!")

        assert result.is_ok()
        assert result.value == "Hello, world!"

    def test_extract_text_from_dict_with_output_key(self):
        """Test extracting text from dict with 'output' key."""
        agent_result = {"output": "Response text"}

        result = self.extractor.extract_text(agent_result)

        assert result.is_ok()
        assert result.value == "Response text"

    def test_extract_text_from_dict_with_response_key(self):
        """Test extracting text from dict with 'response' key."""
        agent_result = {"response": "Response text"}

        result = self.extractor.extract_text(agent_result)

        assert result.is_ok()
        assert result.value == "Response text"

    def test_extract_text_from_dict_with_result_key(self):
        """Test extracting text from dict with 'result' key."""
        agent_result = {"result": "Response text"}

        result = self.extractor.extract_text(agent_result)

        assert result.is_ok()
        assert result.value == "Response text"

    def test_extract_text_uses_key_priority(self):
        """Test that extractor uses first matching key in priority order."""
        agent_result = {
            "output": "First choice",
            "response": "Second choice",
            "result": "Third choice",
        }

        result = self.extractor.extract_text(agent_result)

        assert result.is_ok()
        assert result.value == "First choice"

    def test_extract_text_from_dict_without_known_keys(self):
        """Test extracting from dict without known keys converts whole dict."""
        agent_result = {"foo": "bar", "baz": "qux"}

        result = self.extractor.extract_text(agent_result)

        assert result.is_ok()
        assert "foo" in result.value
        assert "bar" in result.value

    def test_extract_text_from_integer(self):
        """Test extracting text from integer."""
        result = self.extractor.extract_text(42)

        assert result.is_ok()
        assert result.value == "42"

    def test_extract_text_from_none(self):
        """Test extracting text from None."""
        result = self.extractor.extract_text(None)

        assert result.is_ok()
        assert result.value == "None"

    def test_extract_text_skips_empty_values(self):
        """Test that extractor skips empty values in dict."""
        agent_result = {
            "output": "",
            "response": "Valid response",
        }

        result = self.extractor.extract_text(agent_result)

        assert result.is_ok()
        assert result.value == "Valid response"
