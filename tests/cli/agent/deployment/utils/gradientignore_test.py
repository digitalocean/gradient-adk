"""Tests for .gradientignore loading and parsing."""

import pytest
from pathlib import Path
import tempfile
import shutil

from gradient_adk.cli.agent.deployment.utils.gradientignore import (
    load_gradientignore,
    ensure_gradientignore,
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_GRADIENTIGNORE_CONTENT,
    GRADIENTIGNORE_PATH,
)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_gradientignore_"))
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


class TestLoadGradientignore:
    """Tests for load_gradientignore."""

    def test_returns_defaults_when_file_missing(self, temp_project_dir):
        """When no .gradientignore exists, return the default patterns."""
        patterns = load_gradientignore(temp_project_dir)
        assert patterns == DEFAULT_EXCLUDE_PATTERNS

    def test_returns_defaults_when_gradient_dir_missing(self, temp_project_dir):
        """When .gradient/ dir doesn't exist at all, return defaults."""
        patterns = load_gradientignore(temp_project_dir)
        assert patterns == DEFAULT_EXCLUDE_PATTERNS

    def test_parses_patterns_from_file(self, temp_project_dir):
        """Patterns are read line by line from the file."""
        gradient_dir = temp_project_dir / ".gradient"
        gradient_dir.mkdir()
        ignore_file = gradient_dir / ".gradientignore"
        ignore_file.write_text("*.zip\n__pycache__/\nenv/\n")

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == ["*.zip", "__pycache__/", "env/"]

    def test_strips_comments(self, temp_project_dir):
        """Lines starting with # are ignored."""
        gradient_dir = temp_project_dir / ".gradient"
        gradient_dir.mkdir()
        ignore_file = gradient_dir / ".gradientignore"
        ignore_file.write_text("# This is a comment\n*.zip\n# Another comment\nenv/\n")

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == ["*.zip", "env/"]

    def test_strips_blank_lines(self, temp_project_dir):
        """Blank lines (empty or whitespace-only) are ignored."""
        gradient_dir = temp_project_dir / ".gradient"
        gradient_dir.mkdir()
        ignore_file = gradient_dir / ".gradientignore"
        ignore_file.write_text("*.zip\n\n   \nenv/\n\n")

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == ["*.zip", "env/"]

    def test_strips_whitespace_from_patterns(self, temp_project_dir):
        """Leading/trailing whitespace is stripped from each pattern."""
        gradient_dir = temp_project_dir / ".gradient"
        gradient_dir.mkdir()
        ignore_file = gradient_dir / ".gradientignore"
        ignore_file.write_text("  *.zip  \n  env/  \n")

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == ["*.zip", "env/"]

    def test_empty_file_returns_empty_list(self, temp_project_dir):
        """An empty .gradientignore means exclude nothing."""
        gradient_dir = temp_project_dir / ".gradient"
        gradient_dir.mkdir()
        ignore_file = gradient_dir / ".gradientignore"
        ignore_file.write_text("")

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == []

    def test_comments_only_file_returns_empty_list(self, temp_project_dir):
        """A file with only comments means exclude nothing."""
        gradient_dir = temp_project_dir / ".gradient"
        gradient_dir.mkdir()
        ignore_file = gradient_dir / ".gradientignore"
        ignore_file.write_text("# Only comments here\n# Nothing else\n")

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == []

    def test_parses_default_content_correctly(self, temp_project_dir):
        """The auto-generated default content round-trips through the parser."""
        gradient_dir = temp_project_dir / ".gradient"
        gradient_dir.mkdir()
        ignore_file = gradient_dir / ".gradientignore"
        ignore_file.write_text(DEFAULT_GRADIENTIGNORE_CONTENT)

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == DEFAULT_EXCLUDE_PATTERNS


class TestEnsureGradientignore:
    """Tests for ensure_gradientignore."""

    def test_creates_file_when_missing(self, temp_project_dir):
        """Creates .gradient/.gradientignore with default content."""
        ensure_gradientignore(temp_project_dir)

        ignore_file = temp_project_dir / GRADIENTIGNORE_PATH
        assert ignore_file.exists()
        assert ignore_file.read_text() == DEFAULT_GRADIENTIGNORE_CONTENT

    def test_creates_gradient_dir_if_missing(self, temp_project_dir):
        """Creates .gradient/ directory if it doesn't exist."""
        ensure_gradientignore(temp_project_dir)

        assert (temp_project_dir / ".gradient").is_dir()

    def test_does_not_overwrite_existing_file(self, temp_project_dir):
        """Does not overwrite a user-modified .gradientignore."""
        gradient_dir = temp_project_dir / ".gradient"
        gradient_dir.mkdir()
        ignore_file = gradient_dir / ".gradientignore"
        custom_content = "*.zip\nmy_custom_pattern/\n"
        ignore_file.write_text(custom_content)

        ensure_gradientignore(temp_project_dir)

        assert ignore_file.read_text() == custom_content
