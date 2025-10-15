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

        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                file_content = f.read()

            response = await client.put(
                presigned_url,
                content=file_content,
                headers={"Content-Type": "application/zip"},
            )

            if response.status_code not in (200, 204):
                raise Exception(
                    f"Failed to upload file to S3: {response.status_code} - {response.text}"
                )

        logger.info(f"Successfully uploaded {file_path.name}")
