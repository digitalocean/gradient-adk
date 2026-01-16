"""Unit tests for CLI output formatting."""
import json
import io
import sys
from unittest.mock import patch, MagicMock

import pytest
import typer

from gradient_adk.cli.cli import (
    OutputFormat,
    output_json,
    output_json_error,
)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_output_format_values(self):
        """Test that OutputFormat has the expected values."""
        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"

    def test_output_format_is_string_enum(self):
        """Test that OutputFormat values can be used as strings."""
        assert str(OutputFormat.TEXT) == "OutputFormat.TEXT"
        assert OutputFormat.TEXT == "text"
        assert OutputFormat.JSON == "json"


class TestOutputJson:
    """Tests for output_json function."""

    def test_output_json_to_stdout(self, capsys):
        """Test that output_json writes to stdout by default."""
        data = {"status": "success", "value": 42}
        output_json(data)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == data

    def test_output_json_to_custom_file(self):
        """Test that output_json can write to a custom file."""
        data = {"key": "value"}
        buffer = io.StringIO()
        output_json(data, file=buffer)

        buffer.seek(0)
        parsed = json.loads(buffer.read())
        assert parsed == data

    def test_output_json_formats_with_indent(self, capsys):
        """Test that output_json formats with indentation."""
        data = {"nested": {"key": "value"}}
        output_json(data)

        captured = capsys.readouterr()
        # Check that indentation is present (2 spaces)
        assert '  "nested"' in captured.out

    def test_output_json_handles_datetime(self, capsys):
        """Test that output_json handles datetime objects."""
        from datetime import datetime

        now = datetime(2025, 1, 15, 12, 0, 0)
        data = {"timestamp": now}
        output_json(data)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["timestamp"] == "2025-01-15 12:00:00"


class TestOutputJsonError:
    """Tests for output_json_error function."""

    def test_output_json_error_writes_to_stderr(self, capsys):
        """Test that output_json_error writes to stderr."""
        with pytest.raises(typer.Exit) as exc_info:
            output_json_error("Test error message")

        assert exc_info.value.exit_code == 1

        captured = capsys.readouterr()
        assert captured.out == ""  # Nothing to stdout
        parsed = json.loads(captured.err)
        assert parsed["status"] == "error"
        assert parsed["error"] == "Test error message"

    def test_output_json_error_custom_exit_code(self, capsys):
        """Test that output_json_error uses custom exit code."""
        with pytest.raises(typer.Exit) as exc_info:
            output_json_error("Test error", exit_code=2)

        assert exc_info.value.exit_code == 2

    def test_output_json_error_format(self, capsys):
        """Test the JSON error format structure."""
        with pytest.raises(typer.Exit):
            output_json_error("Something went wrong")

        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert "status" in parsed
        assert "error" in parsed
        assert parsed["status"] == "error"


class TestDeployCommandJsonOutput:
    """Tests for deploy command JSON output (mocked)."""

    def test_deploy_json_error_outputs_valid_json(self):
        """Test that deploy with --output json returns valid JSON error on failure."""
        from typer.testing import CliRunner
        from gradient_adk.cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["agent", "deploy", "--output", "json", "--skip-validation"],
            env={"DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.exit_code == 1
        # The output should contain JSON formatted error
        # Note: CliRunner captures both stdout and stderr in result.output
        try:
            parsed = json.loads(result.output)
            assert parsed["status"] == "error"
            assert "error" in parsed
            # The error message should be present (could be config, auth, or API related)
            assert len(parsed["error"]) > 0
        except json.JSONDecodeError:
            # If the first JSON parse fails, the test fails
            pytest.fail(f"Expected valid JSON output, got: {result.output}")


class TestLogsCommandJsonOutput:
    """Tests for logs command JSON output (mocked)."""

    def test_logs_json_error_outputs_valid_json(self):
        """Test that logs with --output json returns valid JSON error on failure."""
        from typer.testing import CliRunner
        from gradient_adk.cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["agent", "logs", "--output", "json"],
            env={"DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.exit_code == 1
        # The output should contain JSON formatted error
        try:
            parsed = json.loads(result.output)
            assert parsed["status"] == "error"
            assert "error" in parsed
            # The error message should be present (could be config or auth related)
            assert len(parsed["error"]) > 0
        except json.JSONDecodeError:
            # If the first JSON parse fails, the test fails
            pytest.fail(f"Expected valid JSON output, got: {result.output}")


class TestDeployHelpOutput:
    """Tests for deploy command help output."""

    def test_deploy_help_shows_output_option(self):
        """Test that deploy --help shows the --output option."""
        from typer.testing import CliRunner
        from gradient_adk.cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["agent", "deploy", "--help"])

        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output
        assert "json" in result.output.lower()


class TestLogsHelpOutput:
    """Tests for logs command help output."""

    def test_logs_help_shows_output_option(self):
        """Test that logs --help shows the --output option."""
        from typer.testing import CliRunner
        from gradient_adk.cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["agent", "logs", "--help"])

        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output
        assert "json" in result.output.lower()