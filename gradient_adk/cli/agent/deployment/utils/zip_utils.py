"""Utilities for creating zip archives."""

from __future__ import annotations
import zipfile
from pathlib import Path
from typing import Protocol

from gradient_adk.logging import get_logger

logger = get_logger(__name__)


class ZipCreator(Protocol):
    """Protocol for creating zip archives."""

    def create_zip(self, source_dir: Path, output_path: Path) -> Path:
        """Create a zip archive from a directory."""
        ...


class DirectoryZipCreator:
    """Creates zip archives from directories, excluding certain patterns."""

    def __init__(self, exclude_patterns: list[str] | None = None):
        """Initialize the zip creator.

        Args:
            exclude_patterns: List of patterns to exclude (e.g., ['*.zip', 'env/', '__pycache__/'])
        """
        self.exclude_patterns = exclude_patterns or [
            "*.zip",
            "env/",
            "__pycache__/",
            ".git/",
        ]

    def create_zip(self, source_dir: Path, output_path: Path) -> Path:
        """Create a zip archive from a directory, excluding certain patterns.

        Args:
            source_dir: Directory to zip
            output_path: Path where the zip file should be created

        Returns:
            Path to the created zip file

        Raises:
            ValueError: If source_dir doesn't exist or isn't a directory
            Exception: If zip creation fails
        """
        if not source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        if not source_dir.is_dir():
            raise ValueError(f"Source path is not a directory: {source_dir}")

        logger.debug(f"Source directory: {source_dir}")
        logger.debug(f"Output path: {output_path}")
        logger.debug(f"Exclude patterns: {self.exclude_patterns}")

        file_count = 0
        try:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in source_dir.rglob("*"):
                    if file_path.is_file() and not self._should_exclude(
                        file_path, source_dir
                    ):
                        try:
                            arcname = file_path.relative_to(source_dir)
                            # Convert to forward slashes for cross-platform compatibility
                            arcname_str = str(arcname).replace("\\", "/")
                            logger.debug(f"Adding to zip: {arcname_str}")
                            zipf.write(file_path, arcname_str)
                            file_count += 1
                        except Exception as e:
                            logger.error(f"Failed to add {file_path} to zip: {e}")
                            raise Exception(
                                f"Failed to add {file_path} to zip: {e}"
                            ) from e

            logger.debug(f"Successfully created zip with {file_count} files")

            # Verify the zip was created
            if not output_path.exists():
                raise Exception(f"Zip file was not created at {output_path}")

            zip_size = output_path.stat().st_size
            logger.debug(f"Zip file size: {zip_size} bytes")

            if zip_size == 0:
                raise Exception(f"Created zip file is empty: {output_path}")

        except Exception as e:
            # Clean up partial zip file if it exists
            if output_path.exists():
                try:
                    output_path.unlink()
                    logger.debug(f"Cleaned up partial zip file: {output_path}")
                except Exception:
                    pass
            raise

        return output_path

    def _should_exclude(self, file_path: Path, source_dir: Path) -> bool:
        """Check if a file should be excluded based on patterns.

        Args:
            file_path: The file to check
            source_dir: The source directory (for calculating relative paths)

        Returns:
            True if the file should be excluded
        """
        relative_path = file_path.relative_to(source_dir)
        path_str = str(relative_path)

        for pattern in self.exclude_patterns:
            # Simple pattern matching
            if pattern.endswith("/"):
                # Directory pattern - check if any parent matches
                dir_pattern = pattern.rstrip("/")
                if dir_pattern in relative_path.parts:
                    return True
            elif pattern.startswith("*"):
                # Extension pattern
                if path_str.endswith(pattern[1:]):
                    return True
            else:
                # Exact match
                if pattern == path_str or pattern in relative_path.parts:
                    return True

        return False
