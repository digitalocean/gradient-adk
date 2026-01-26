"""Tests for Python environment detection."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from gradient_adk.cli.agent.deployment.python_environment_detector import (
    PythonEnvironmentDetector,
    PythonEnvironmentDetectionError,
    SUPPORTED_PYTHON_VERSIONS,
)
from gradient_adk.digital_ocean_api.models import (
    PythonDependencyFile,
    PythonPackageManager,
    PythonVersion,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def detector() -> PythonEnvironmentDetector:
    """Create a PythonEnvironmentDetector instance."""
    return PythonEnvironmentDetector()


class TestDependencyFileDetection:
    """Tests for dependency file detection."""

    def test_detect_requirements_txt_only(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection when only requirements.txt exists."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        result = detector._detect_dependency_file(temp_dir)

        assert result == PythonDependencyFile.PYTHON_DEPENDENCY_FILE_REQUIREMENTS_TXT

    def test_detect_pyproject_toml_only(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection when only pyproject.toml exists."""
        (temp_dir / "pyproject.toml").write_text('[project]\nname = "test"\n')

        result = detector._detect_dependency_file(temp_dir)

        assert result == PythonDependencyFile.PYTHON_DEPENDENCY_FILE_PYPROJECT_TOML

    def test_detect_both_files_warns_and_uses_requirements(
        self, temp_dir: Path, detector: PythonEnvironmentDetector, caplog
    ):
        """Test that both files existing warns and uses requirements.txt."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")
        (temp_dir / "pyproject.toml").write_text('[project]\nname = "test"\n')

        with caplog.at_level("WARNING"):
            result = detector._detect_dependency_file(temp_dir)

        assert result == PythonDependencyFile.PYTHON_DEPENDENCY_FILE_REQUIREMENTS_TXT
        assert "Both requirements.txt and pyproject.toml found" in caplog.text

    def test_detect_no_dependency_file_raises_error(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test that missing dependency files raises an error."""
        with pytest.raises(PythonEnvironmentDetectionError) as exc_info:
            detector._detect_dependency_file(temp_dir)

        assert "No dependency file found" in str(exc_info.value)
        assert "requirements.txt" in str(exc_info.value)
        assert "pyproject.toml" in str(exc_info.value)


class TestPythonVersionDetection:
    """Tests for Python version detection."""

    def test_detect_from_python_version_file(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection from .python-version file."""
        (temp_dir / ".python-version").write_text("3.12\n")
        # Need a dependency file for full detection
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_12

    def test_detect_from_python_version_file_with_patch(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection from .python-version file with patch version."""
        (temp_dir / ".python-version").write_text("3.11.5\n")
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_11

    def test_detect_from_python_version_file_with_prefix(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection from .python-version file with python- prefix."""
        (temp_dir / ".python-version").write_text("python-3.13\n")
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_13

    def test_detect_from_pyproject_requires_python(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection from pyproject.toml requires-python."""
        (temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\nrequires-python = ">=3.10"\n'
        )

        result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_10

    def test_detect_from_pyproject_requires_python_caret(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection from pyproject.toml with caret version."""
        (temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\nrequires-python = "^3.11"\n'
        )

        result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_11

    def test_detect_from_pyproject_requires_python_tilde(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection from pyproject.toml with tilde version."""
        (temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\nrequires-python = "~=3.12"\n'
        )

        result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_12

    def test_detect_from_pyproject_exact_version(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection from pyproject.toml with exact version."""
        (temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\nrequires-python = "==3.13"\n'
        )

        result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_13

    def test_detect_fallback_to_runtime(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test fallback to current runtime Python version."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        # Mock sys.version_info with a SimpleNamespace to preserve attribute access
        mock_version_info = SimpleNamespace(major=3, minor=12)
        with patch.object(sys, "version_info", mock_version_info):
            result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_12

    def test_python_version_file_takes_precedence(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test that .python-version takes precedence over pyproject.toml."""
        (temp_dir / ".python-version").write_text("3.10\n")
        (temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\nrequires-python = ">=3.13"\n'
        )
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_10

    def test_unsupported_python_version_raises_error(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test that unsupported Python version raises an error."""
        (temp_dir / ".python-version").write_text("3.9\n")
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        with pytest.raises(PythonEnvironmentDetectionError) as exc_info:
            detector._detect_python_version(temp_dir)

        assert "3.9 is not supported" in str(exc_info.value)
        assert "3.10" in str(exc_info.value)

    def test_unsupported_python_version_too_old(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test that Python 2.x raises an error."""
        (temp_dir / ".python-version").write_text("2.7\n")
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        with pytest.raises(PythonEnvironmentDetectionError) as exc_info:
            detector._detect_python_version(temp_dir)

        assert "2.7 is not supported" in str(exc_info.value)

    def test_unsupported_runtime_version_raises_error(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test that unsupported runtime version raises an error."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        # Mock sys.version_info with a SimpleNamespace to preserve attribute access
        mock_version_info = SimpleNamespace(major=3, minor=9)
        with patch.object(sys, "version_info", mock_version_info):
            with pytest.raises(PythonEnvironmentDetectionError) as exc_info:
                detector._detect_python_version(temp_dir)

        assert "3.9 is not supported" in str(exc_info.value)


class TestPackageManagerDetection:
    """Tests for package manager detection."""

    def test_detect_uv_from_lock_file(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test UV detection from uv.lock file."""
        (temp_dir / "uv.lock").write_text("# uv lock file\n")
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        result = detector._detect_package_manager(temp_dir)

        assert result == PythonPackageManager.PYTHON_PACKAGE_MANAGER_UV

    def test_detect_uv_from_tool_uv_section(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test UV detection from [tool.uv] in pyproject.toml."""
        (temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\n\n[tool.uv]\ndev-dependencies = []\n'
        )

        result = detector._detect_package_manager(temp_dir)

        assert result == PythonPackageManager.PYTHON_PACKAGE_MANAGER_UV

    def test_detect_default_pip(
        self, temp_dir: Path, detector: PythonEnvironmentDetector, caplog
    ):
        """Test default to pip when no UV indicators."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        with caplog.at_level("WARNING"):
            result = detector._detect_package_manager(temp_dir)

        assert result == PythonPackageManager.PYTHON_PACKAGE_MANAGER_PIP
        assert "defaulting to pip" in caplog.text

    def test_detect_pip_with_pyproject_no_uv(
        self, temp_dir: Path, detector: PythonEnvironmentDetector, caplog
    ):
        """Test pip detection with pyproject.toml but no [tool.uv]."""
        (temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\n\n[tool.ruff]\nline-length = 88\n'
        )

        with caplog.at_level("WARNING"):
            result = detector._detect_package_manager(temp_dir)

        assert result == PythonPackageManager.PYTHON_PACKAGE_MANAGER_PIP
        assert "defaulting to pip" in caplog.text

    def test_uv_lock_takes_precedence(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test that uv.lock takes precedence over [tool.uv] check."""
        (temp_dir / "uv.lock").write_text("# uv lock file\n")
        (temp_dir / "pyproject.toml").write_text('[project]\nname = "test"\n')

        result = detector._detect_package_manager(temp_dir)

        assert result == PythonPackageManager.PYTHON_PACKAGE_MANAGER_UV


class TestFullDetection:
    """Tests for full environment detection."""

    def test_full_detection_requirements_txt(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test full detection with requirements.txt."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")
        (temp_dir / ".python-version").write_text("3.12\n")

        result = detector.detect(temp_dir)

        assert result.dependency_file == PythonDependencyFile.PYTHON_DEPENDENCY_FILE_REQUIREMENTS_TXT
        assert result.python_version == PythonVersion.PYTHON_VERSION_3_12
        assert result.package_manager == PythonPackageManager.PYTHON_PACKAGE_MANAGER_PIP

    def test_full_detection_pyproject_with_uv(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test full detection with pyproject.toml and UV."""
        (temp_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\nrequires-python = ">=3.11"\n\n[tool.uv]\n'
        )

        result = detector.detect(temp_dir)

        assert result.dependency_file == PythonDependencyFile.PYTHON_DEPENDENCY_FILE_PYPROJECT_TOML
        assert result.python_version == PythonVersion.PYTHON_VERSION_3_11
        assert result.package_manager == PythonPackageManager.PYTHON_PACKAGE_MANAGER_UV

    def test_full_detection_with_all_supported_versions(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test detection works for all supported Python versions."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        for (major, minor), expected_version in SUPPORTED_PYTHON_VERSIONS.items():
            (temp_dir / ".python-version").write_text(f"{major}.{minor}\n")

            result = detector.detect(temp_dir)

            assert result.python_version == expected_version

    def test_full_detection_error_no_dependency_file(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test that full detection fails without dependency file."""
        with pytest.raises(PythonEnvironmentDetectionError) as exc_info:
            detector.detect(temp_dir)

        assert "No dependency file found" in str(exc_info.value)

    def test_full_detection_error_unsupported_version(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test that full detection fails with unsupported Python version."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")
        (temp_dir / ".python-version").write_text("3.8\n")

        with pytest.raises(PythonEnvironmentDetectionError) as exc_info:
            detector.detect(temp_dir)

        assert "3.8 is not supported" in str(exc_info.value)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_python_version_file(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test handling of empty .python-version file."""
        (temp_dir / ".python-version").write_text("")
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        # Should fall back to runtime version
        mock_version_info = SimpleNamespace(major=3, minor=12)
        with patch.object(sys, "version_info", mock_version_info):
            result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_12

    def test_malformed_python_version_file(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test handling of malformed .python-version file."""
        (temp_dir / ".python-version").write_text("invalid version string\n")
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        # Should fall back to runtime version
        mock_version_info = SimpleNamespace(major=3, minor=11)
        with patch.object(sys, "version_info", mock_version_info):
            result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_11

    def test_pyproject_without_requires_python(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test handling of pyproject.toml without requires-python."""
        (temp_dir / "pyproject.toml").write_text('[project]\nname = "test"\n')

        # Should fall back to runtime version
        mock_version_info = SimpleNamespace(major=3, minor=13)
        with patch.object(sys, "version_info", mock_version_info):
            result = detector._detect_python_version(temp_dir)

        assert result == PythonVersion.PYTHON_VERSION_3_13

    def test_unreadable_file_handling(
        self, temp_dir: Path, detector: PythonEnvironmentDetector
    ):
        """Test handling of unreadable files gracefully."""
        (temp_dir / "requirements.txt").write_text("gradient-adk\n")

        # Create a .python-version file and make it unreadable
        python_version_file = temp_dir / ".python-version"
        python_version_file.write_text("3.12\n")

        # Patch read_text to simulate read error
        original_read = Path.read_text

        def mock_read_text(self):
            if self.name == ".python-version":
                raise PermissionError("Permission denied")
            return original_read(self)

        mock_version_info = SimpleNamespace(major=3, minor=11)
        with patch.object(Path, "read_text", mock_read_text):
            with patch.object(sys, "version_info", mock_version_info):
                result = detector._detect_python_version(temp_dir)

        # Should fall back to runtime
        assert result == PythonVersion.PYTHON_VERSION_3_11