"""Unit tests for domain validation logic."""

import pytest
from gradient_adk.a2a.domain.models import DomainMessage, MessageRole
from gradient_adk.a2a.domain.validation import MessageValidator


class TestMessageValidator:
    """Tests for MessageValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = MessageValidator()

    def test_validate_text_only_accepts_text_only_message(self):
        """Test that validator accepts text-only messages."""
        message = DomainMessage(
            text="Hello, world!",
            role=MessageRole.USER,
            has_file_parts=False,
            has_data_parts=False,
        )

        result = self.validator.validate_text_only(message)

        assert result.is_ok()
        assert result.value == message

    def test_validate_text_only_rejects_message_with_file_parts(self):
        """Test that validator rejects messages with file parts."""
        message = DomainMessage(
            text="Hello",
            role=MessageRole.USER,
            has_file_parts=True,
        )

        result = self.validator.validate_text_only(message)

        assert result.is_err()
        assert result.error.code == "UNSUPPORTED_CONTENT_TYPE"
        assert "FilePart" in result.error.message

    def test_validate_text_only_rejects_message_with_data_parts(self):
        """Test that validator rejects messages with data parts."""
        message = DomainMessage(
            text="Hello",
            role=MessageRole.USER,
            has_data_parts=True,
        )

        result = self.validator.validate_text_only(message)

        assert result.is_err()
        assert result.error.code == "UNSUPPORTED_CONTENT_TYPE"
        assert "DataPart" in result.error.message

    def test_validate_non_empty_accepts_non_empty_message(self):
        """Test that validator accepts non-empty messages."""
        message = DomainMessage(
            text="Hello, world!",
            role=MessageRole.USER,
        )

        result = self.validator.validate_non_empty(message)

        assert result.is_ok()
        assert result.value == message

    def test_validate_non_empty_rejects_empty_message(self):
        """Test that validator rejects empty messages."""
        message = DomainMessage(
            text="",
            role=MessageRole.USER,
        )

        result = self.validator.validate_non_empty(message)

        assert result.is_err()
        assert result.error.code == "EMPTY_MESSAGE"

    def test_validate_non_empty_rejects_whitespace_only_message(self):
        """Test that validator rejects whitespace-only messages."""
        message = DomainMessage(
            text="   \n\t  ",
            role=MessageRole.USER,
        )

        result = self.validator.validate_non_empty(message)

        assert result.is_err()
        assert result.error.code == "EMPTY_MESSAGE"

    def test_validate_all_accepts_valid_message(self):
        """Test that validate_all accepts valid messages."""
        message = DomainMessage(
            text="Hello, world!",
            role=MessageRole.USER,
            has_file_parts=False,
            has_data_parts=False,
        )

        result = self.validator.validate_all(message)

        assert result.is_ok()
        assert result.value == message

    def test_validate_all_rejects_file_parts_first(self):
        """Test that validate_all checks text-only first (before non-empty check)."""
        message = DomainMessage(
            text="",
            role=MessageRole.USER,
            has_file_parts=True,
        )

        result = self.validator.validate_all(message)

        assert result.is_err()
        assert result.error.code == "UNSUPPORTED_CONTENT_TYPE"

    def test_validate_all_rejects_empty_text_after_text_only(self):
        """Test that validate_all checks non-empty after text-only validation."""
        message = DomainMessage(
            text="",
            role=MessageRole.USER,
            has_file_parts=False,
            has_data_parts=False,
        )

        result = self.validator.validate_all(message)

        assert result.is_err()
        assert result.error.code == "EMPTY_MESSAGE"
