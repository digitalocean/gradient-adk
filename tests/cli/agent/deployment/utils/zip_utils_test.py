"""Tests for DirectoryZipCreator with .gradientignore integration."""

import pytest
import zipfile
from pathlib import Path
import tempfile
import shutil

from gradient_adk.cli.agent.deployment.utils.zip_utils import DirectoryZipCreator
from gradient_adk.cli.agent.deployment.utils.gradientignore import (
    load_gradientignore,
    ensure_gradientignore,
    DEFAULT_EXCLUDE_PATTERNS,
)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_zip_"))
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def _create_sample_project(project_dir: Path) -> None:
    """Populate a project dir with typical agent files and artifacts to exclude."""
    # Files that should be included
    (project_dir / "main.py").write_text("print('hello')\n")
    (project_dir / "requirements.txt").write_text("gradient-adk\n")
    (project_dir / ".env").write_text("SECRET=value\n")
    agents_dir = project_dir / "agents"
    agents_dir.mkdir()
    (agents_dir / "__init__.py").write_text("")
    (agents_dir / "my_agent.py").write_text("# agent code\n")

    # Config
    gradient_dir = project_dir / ".gradient"
    gradient_dir.mkdir()
    (gradient_dir / "agent.yml").write_text("agent_name: test\n")

    # Files/dirs that should be excluded by default patterns
    pycache = project_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-313.pyc").write_bytes(b"\x00compiled")

    git_dir = project_dir / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config")

    env_dir = project_dir / "env"
    env_dir.mkdir()
    (env_dir / "pyvenv.cfg").write_text("venv config")

    venv_dir = project_dir / "venv"
    venv_dir.mkdir()
    (venv_dir / "pyvenv.cfg").write_text("venv config")

    dot_venv_dir = project_dir / ".venv"
    dot_venv_dir.mkdir()
    (dot_venv_dir / "pyvenv.cfg").write_text("venv config")

    pytest_cache = project_dir / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "v" / "cache").mkdir(parents=True)

    dist_dir = project_dir / "dist"
    dist_dir.mkdir()
    (dist_dir / "package-0.1.tar.gz").write_bytes(b"\x00pkg")


def _get_zip_names(zip_path: Path) -> set[str]:
    """Return the set of file names inside a zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return set(zf.namelist())


class TestDirectoryZipCreatorWithGradientignore:
    """Tests that zip creation properly respects .gradientignore patterns."""

    def test_zip_with_gradientignore_file(self, temp_project_dir):
        """When .gradientignore exists, its patterns control what is excluded."""
        _create_sample_project(temp_project_dir)

        # Write a .gradientignore that only excludes __pycache__/ and .git/
        ignore_file = temp_project_dir / ".gradient" / ".gradientignore"
        ignore_file.write_text("__pycache__/\n.git/\n")

        patterns = load_gradientignore(temp_project_dir)
        creator = DirectoryZipCreator(exclude_patterns=patterns)
        zip_path = temp_project_dir / "test.zip"
        creator.create_zip(temp_project_dir, zip_path)

        names = _get_zip_names(zip_path)

        # Should be included (not in ignore)
        assert "main.py" in names
        assert "requirements.txt" in names
        assert "agents/my_agent.py" in names
        # env/ is NOT excluded by this custom ignore
        assert "env/pyvenv.cfg" in names
        assert "venv/pyvenv.cfg" in names

        # Should be excluded
        assert not any("__pycache__" in n for n in names)
        assert not any(".git" in n for n in names)

    def test_zip_without_gradientignore_file(self, temp_project_dir):
        """When no .gradientignore exists, default patterns are used."""
        _create_sample_project(temp_project_dir)

        # No .gradientignore written — load_gradientignore returns defaults
        patterns = load_gradientignore(temp_project_dir)
        assert patterns == DEFAULT_EXCLUDE_PATTERNS

        creator = DirectoryZipCreator(exclude_patterns=patterns)
        zip_path = temp_project_dir / "output.zip"
        creator.create_zip(temp_project_dir, zip_path)

        names = _get_zip_names(zip_path)

        # Should be included
        assert "main.py" in names
        assert "requirements.txt" in names
        assert ".env" in names
        assert "agents/__init__.py" in names
        assert "agents/my_agent.py" in names
        assert ".gradient/agent.yml" in names

        # Should be excluded by default patterns
        assert not any("__pycache__" in n for n in names)
        assert not any(".git/" in n or n == ".git/config" for n in names)
        assert not any(n.startswith("env/") for n in names)
        assert not any(n.startswith("venv/") for n in names)
        assert not any(n.startswith(".venv/") for n in names)
        assert not any(n.startswith(".pytest_cache/") for n in names)
        assert not any(n.startswith("dist/") for n in names)
        assert not any(n.endswith(".pyc") for n in names)

    def test_zip_with_autogenerated_gradientignore(self, temp_project_dir):
        """ensure_gradientignore creates a file that produces the same defaults."""
        _create_sample_project(temp_project_dir)
        ensure_gradientignore(temp_project_dir)

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == DEFAULT_EXCLUDE_PATTERNS

        creator = DirectoryZipCreator(exclude_patterns=patterns)
        zip_path = temp_project_dir / "output.zip"
        creator.create_zip(temp_project_dir, zip_path)

        names = _get_zip_names(zip_path)

        # Same result as without the file
        assert "main.py" in names
        assert not any("__pycache__" in n for n in names)
        assert not any(n.startswith("env/") for n in names)

    def test_zip_with_empty_gradientignore_excludes_nothing(self, temp_project_dir):
        """An empty .gradientignore means nothing is excluded (except the zip itself)."""
        _create_sample_project(temp_project_dir)

        ignore_file = temp_project_dir / ".gradient" / ".gradientignore"
        ignore_file.write_text("")

        patterns = load_gradientignore(temp_project_dir)
        assert patterns == []

        creator = DirectoryZipCreator(exclude_patterns=patterns)
        zip_path = temp_project_dir / "output.zip"
        creator.create_zip(temp_project_dir, zip_path)

        names = _get_zip_names(zip_path)

        # Everything should be included now
        assert "main.py" in names
        assert any("__pycache__" in n for n in names)
        assert any(n.startswith("env/") for n in names)
        assert any(n.startswith(".git/") for n in names)

    def test_zip_with_custom_pattern(self, temp_project_dir):
        """Users can add custom patterns to exclude project-specific files."""
        _create_sample_project(temp_project_dir)

        # Create a data directory the user wants to exclude
        data_dir = temp_project_dir / "test_data"
        data_dir.mkdir()
        (data_dir / "large_file.csv").write_text("lots,of,data\n")

        ignore_file = temp_project_dir / ".gradient" / ".gradientignore"
        ignore_file.write_text("*.zip\n__pycache__/\n.git/\ntest_data/\n")

        patterns = load_gradientignore(temp_project_dir)
        creator = DirectoryZipCreator(exclude_patterns=patterns)
        zip_path = temp_project_dir / "output.zip"
        creator.create_zip(temp_project_dir, zip_path)

        names = _get_zip_names(zip_path)

        assert "main.py" in names
        assert not any("test_data" in n for n in names)
        assert not any("__pycache__" in n for n in names)


class TestDirectoryZipCreatorBasic:
    """Basic zip creation tests."""

    def test_creates_valid_zip(self, temp_project_dir):
        """Created zip is a valid zip file."""
        (temp_project_dir / "file.txt").write_text("hello")

        creator = DirectoryZipCreator(exclude_patterns=[])
        zip_path = temp_project_dir / "output.zip"
        result = creator.create_zip(temp_project_dir, zip_path)

        assert result == zip_path
        assert zip_path.exists()
        assert zipfile.is_zipfile(zip_path)

    def test_raises_for_nonexistent_source(self, temp_project_dir):
        """Raises ValueError when source directory doesn't exist."""
        creator = DirectoryZipCreator()
        with pytest.raises(ValueError, match="does not exist"):
            creator.create_zip(
                temp_project_dir / "nonexistent", temp_project_dir / "out.zip"
            )

    def test_raises_for_file_as_source(self, temp_project_dir):
        """Raises ValueError when source path is a file, not a directory."""
        file_path = temp_project_dir / "not_a_dir.txt"
        file_path.write_text("hello")

        creator = DirectoryZipCreator()
        with pytest.raises(ValueError, match="not a directory"):
            creator.create_zip(file_path, temp_project_dir / "out.zip")

    def test_zip_excludes_itself(self, temp_project_dir):
        """The output zip file is not included inside itself."""
        (temp_project_dir / "file.txt").write_text("hello")

        creator = DirectoryZipCreator(exclude_patterns=["*.zip"])
        zip_path = temp_project_dir / "output.zip"
        creator.create_zip(temp_project_dir, zip_path)

        names = _get_zip_names(zip_path)
        assert "output.zip" not in names
        assert "file.txt" in names

    def test_preserves_subdirectory_structure(self, temp_project_dir):
        """Subdirectory paths are preserved in the zip archive."""
        sub = temp_project_dir / "a" / "b"
        sub.mkdir(parents=True)
        (sub / "deep.py").write_text("# deep\n")

        creator = DirectoryZipCreator(exclude_patterns=[])
        zip_path = temp_project_dir / "output.zip"
        creator.create_zip(temp_project_dir, zip_path)

        names = _get_zip_names(zip_path)
        assert "a/b/deep.py" in names
