"""S3 utilities for file uploads."""

from __future__ import annotations
import httpx
from typing import Protocol
from pathlib import Path

from gradient_adk.logging import get_logger

logger = get_logger(__name__)


class S3Uploader(Protocol):
    """Protocol for S3 file upload operations."""

    async def upload_file(self, file_path: Path, presigned_url: str) -> None:
        """Upload a file to S3 using a presigned URL."""
        ...


class HttpxS3Uploader:
    """S3 file uploader using httpx."""

    async def upload_file(self, file_path: Path, presigned_url: str) -> None:
        """Upload a file to S3 using a presigned URL.

        Args:
            file_path: Path to the file to upload
            presigned_url: The presigned S3 URL

        Raises:
            Exception: If the upload fails
        """
        logger.debug(f"Uploading file to S3: {file_path}")

        # Validate file exists before attempting upload
        if not file_path.exists():
            raise FileNotFoundError(f"File not found for upload: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            # Read file content synchronously (since we're in asyncio.to_thread context anyway)
            file_content = file_path.read_bytes()
            logger.debug(f"Read {len(file_content)} bytes from {file_path.name}")
        except Exception as e:
            raise Exception(f"Failed to read file {file_path}: {e}") from e

        async with httpx.AsyncClient(
            timeout=300.0
        ) as client:  # 5 minute timeout for large files
            try:
                response = await client.put(
                    presigned_url,
                    content=file_content,
                    headers={"Content-Type": "application/zip"},
                )

                logger.debug(f"S3 upload response status: {response.status_code}")

                if response.status_code not in (200, 204):
                    raise Exception(
                        f"Failed to upload file to S3: {response.status_code} - {response.text}"
                    )

                logger.debug(f"Successfully uploaded {file_path.name} to S3")
            except httpx.HTTPError as e:
                raise Exception(f"HTTP error during S3 upload: {e}") from e
            except Exception as e:
                raise Exception(f"Unexpected error during S3 upload: {e}") from e
