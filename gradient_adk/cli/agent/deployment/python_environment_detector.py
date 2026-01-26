"""Python environment detection for agent deployments."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from gradient_adk.logging import get_logger
from gradient_adk.digital_ocean_api.models import (
    PythonDependencyFile,
    PythonEnvironmentConfig,
    PythonPackageManager,
    PythonVersion,
)

logger = get_logger(__name__)

# Supported Python versions mapping
SUPPORTED_PYTHON_VERSIONS = {
    (3, 10): PythonVersion.PYTHON_VERSION_3_10,
    (3, 11): PythonVersion.PYTHON_VERSION_3_11,
    (3, 12): PythonVersion.PYTHON_VERSION_3_12,
    (3, 13): PythonVersion.PYTHON_VERSION_3_13,
    (3, 14): PythonVersion.PYTHON_VERSION_3_14,
}


class PythonEnvironmentDetectionError(Exception):
    """Raised when Python environment detection fails."""

    pass


class PythonEnvironmentDetector:
    """Detects Python environment configuration from a source directory."""

    def detect(self, source_dir: Path) -> PythonEnvironmentConfig:
        """Detect Python environment configuration from the source directory.

        Args:
            source_dir: The source directory to analyze

        Returns:
            PythonEnvironmentConfig with detected settings

        Raises:
            PythonEnvironmentDetectionError: If detection fails due to missing
                dependency files or unsupported Python version
        """
        dependency_file = self._detect_dependency_file(source_dir)
        python_version = self._detect_python_version(source_dir)
        package_manager = self._detect_package_manager(source_dir)

        return PythonEnvironmentConfig(
            python_version=python_version,
            package_manager=package_manager,
            dependency_file=dependency_file,
        )

    def _detect_dependency_file(self, source_dir: Path) -> PythonDependencyFile:
        """Detect the dependency file type.

        Args:
            source_dir: The source directory to analyze

        Returns:
            PythonDependencyFile enum value

        Raises:
            PythonEnvironmentDetectionError: If no dependency file is found
        """
        requirements_txt = source_dir / "requirements.txt"
        pyproject_toml = source_dir / "pyproject.toml"

        has_requirements = requirements_txt.exists()
        has_pyproject = pyproject_toml.exists()

        if has_requirements and has_pyproject:
            logger.warning(
                "Both requirements.txt and pyproject.toml found. "
                "Using requirements.txt as the dependency file."
            )
            return PythonDependencyFile.PYTHON_DEPENDENCY_FILE_REQUIREMENTS_TXT

        if has_requirements:
            logger.debug("Detected dependency file: requirements.txt")
            return PythonDependencyFile.PYTHON_DEPENDENCY_FILE_REQUIREMENTS_TXT

        if has_pyproject:
            logger.debug("Detected dependency file: pyproject.toml")
            return PythonDependencyFile.PYTHON_DEPENDENCY_FILE_PYPROJECT_TOML

        raise PythonEnvironmentDetectionError(
            "No dependency file found. Please create either requirements.txt or pyproject.toml "
            "in your project directory."
        )

    def _detect_python_version(self, source_dir: Path) -> PythonVersion:
        """Detect the Python version.

        Priority:
        1. .python-version file
        2. pyproject.toml requires-python
        3. Current runtime Python version

        Args:
            source_dir: The source directory to analyze

        Returns:
            PythonVersion enum value

        Raises:
            PythonEnvironmentDetectionError: If Python version is not supported
        """
        # Try .python-version file first
        version = self._parse_python_version_file(source_dir)
        if version:
            return self._validate_and_return_version(version, ".python-version file")

        # Try pyproject.toml requires-python
        version = self._parse_pyproject_python_version(source_dir)
        if version:
            return self._validate_and_return_version(version, "pyproject.toml")

        # Fall back to current runtime
        version = (sys.version_info.major, sys.version_info.minor)
        return self._validate_and_return_version(version, "current runtime")

    def _parse_python_version_file(
        self, source_dir: Path
    ) -> Optional[Tuple[int, int]]:
        """Parse Python version from .python-version file.

        Args:
            source_dir: The source directory to analyze

        Returns:
            Tuple of (major, minor) version or None if not found
        """
        python_version_file = source_dir / ".python-version"
        if not python_version_file.exists():
            return None

        try:
            content = python_version_file.read_text().strip()
            # Handle formats like "3.12", "3.12.1", "python-3.12"
            match = re.search(r"(\d+)\.(\d+)", content)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        except Exception as e:
            logger.debug(f"Failed to parse .python-version file: {e}")

        return None

    def _parse_pyproject_python_version(
        self, source_dir: Path
    ) -> Optional[Tuple[int, int]]:
        """Parse Python version from pyproject.toml requires-python.

        Args:
            source_dir: The source directory to analyze

        Returns:
            Tuple of (major, minor) version or None if not found
        """
        pyproject_toml = source_dir / "pyproject.toml"
        if not pyproject_toml.exists():
            return None

        try:
            content = pyproject_toml.read_text()

            # Look for requires-python in various formats
            # e.g., requires-python = ">=3.12" or requires-python = "^3.12"
            match = re.search(
                r'requires-python\s*=\s*["\']([^"\']+)["\']', content
            )
            if match:
                version_spec = match.group(1)
                # Extract the version number from specs like ">=3.12", "^3.12", "~=3.12", "==3.12"
                version_match = re.search(r"(\d+)\.(\d+)", version_spec)
                if version_match:
                    return (int(version_match.group(1)), int(version_match.group(2)))

            # Also check for python_requires in [project] section (PEP 621)
            match = re.search(
                r'python_requires\s*=\s*["\']([^"\']+)["\']', content
            )
            if match:
                version_spec = match.group(1)
                version_match = re.search(r"(\d+)\.(\d+)", version_spec)
                if version_match:
                    return (int(version_match.group(1)), int(version_match.group(2)))

        except Exception as e:
            logger.debug(f"Failed to parse pyproject.toml for Python version: {e}")

        return None

    def _validate_and_return_version(
        self, version: Tuple[int, int], source: str
    ) -> PythonVersion:
        """Validate Python version and return the enum value.

        Args:
            version: Tuple of (major, minor) version
            source: Description of where the version was detected from

        Returns:
            PythonVersion enum value

        Raises:
            PythonEnvironmentDetectionError: If version is not supported
        """
        if version in SUPPORTED_PYTHON_VERSIONS:
            logger.debug(
                f"Detected Python version {version[0]}.{version[1]} from {source}"
            )
            return SUPPORTED_PYTHON_VERSIONS[version]

        supported_versions = ", ".join(
            f"{v[0]}.{v[1]}" for v in sorted(SUPPORTED_PYTHON_VERSIONS.keys())
        )
        raise PythonEnvironmentDetectionError(
            f"Python version {version[0]}.{version[1]} is not supported. "
            f"Supported versions: {supported_versions}"
        )

    def _detect_package_manager(self, source_dir: Path) -> PythonPackageManager:
        """Detect the package manager to use.

        Priority:
        1. uv.lock file present -> UV
        2. pyproject.toml with [tool.uv] section -> UV
        3. Default to PIP

        Args:
            source_dir: The source directory to analyze

        Returns:
            PythonPackageManager enum value
        """
        # Check for uv.lock file
        uv_lock = source_dir / "uv.lock"
        if uv_lock.exists():
            logger.debug("Detected package manager: uv (uv.lock file found)")
            return PythonPackageManager.PYTHON_PACKAGE_MANAGER_UV

        # Check for [tool.uv] in pyproject.toml
        pyproject_toml = source_dir / "pyproject.toml"
        if pyproject_toml.exists():
            try:
                content = pyproject_toml.read_text()
                if "[tool.uv]" in content:
                    logger.debug(
                        "Detected package manager: uv ([tool.uv] section found)"
                    )
                    return PythonPackageManager.PYTHON_PACKAGE_MANAGER_UV
            except Exception as e:
                logger.debug(f"Failed to read pyproject.toml for UV detection: {e}")

        # Default to pip
        logger.warning(
            "Could not determine package manager, defaulting to pip. "
            "To use uv, add a uv.lock file or [tool.uv] section to pyproject.toml."
        )
        return PythonPackageManager.PYTHON_PACKAGE_MANAGER_PIP