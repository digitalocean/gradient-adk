from __future__ import annotations
import asyncio
import time
from pathlib import Path
from typing import Protocol

from gradient_adk.logging import get_logger
from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI
from gradient_adk.digital_ocean_api.models import (
    AgentDeploymentCodeArtifact,
    CreateAgentDeploymentFileUploadPresignedURLInput,
    CreateAgentWorkspaceInput,
    CreateAgentWorkspaceDeploymentInput,
    CreateAgentDeploymentReleaseInput,
    PresignedUrlFile,
    ReleaseStatus,
)
from gradient_adk.digital_ocean_api.errors import DOAPIClientError

from .utils.zip_utils import ZipCreator, DirectoryZipCreator
from .utils.s3_utils import S3Uploader, HttpxS3Uploader

logger = get_logger(__name__)


class DeployService(Protocol):
    """Protocol for agent deployment operations."""

    async def deploy_agent(
        self,
        agent_workspace_name: str,
        agent_deployment_name: str,
        source_dir: Path,
        project_id: str,
        api_token: str,
    ) -> None:
        """Deploy the agent to the configured environment."""
        ...


class AgentDeployService:
    """Service for deploying agents with proper dependency injection."""

    def __init__(
        self,
        client: AsyncDigitalOceanGenAI,
        zip_creator: ZipCreator | None = None,
        s3_uploader: S3Uploader | None = None,
        polling_interval_sec: float = 10.0,
        max_polling_time_sec: float = 600.0,  # 10 minutes
    ):
        """Initialize the deploy service.

        Args:
            client: AsyncDigitalOceanGenAI client for API calls
            zip_creator: Zip creator for packaging code (defaults to DirectoryZipCreator)
            s3_uploader: S3 uploader for uploading files (defaults to HttpxS3Uploader)
            polling_interval_sec: How often to poll for release status
            max_polling_time_sec: Maximum time to wait for deployment
        """
        self.client = client
        self.zip_creator = zip_creator or DirectoryZipCreator()
        self.s3_uploader = s3_uploader or HttpxS3Uploader()
        self.polling_interval_sec = polling_interval_sec
        self.max_polling_time_sec = max_polling_time_sec

    async def deploy_agent(
        self,
        agent_workspace_name: str,
        agent_deployment_name: str,
        source_dir: Path,
        project_id: str,
        api_token: str,
    ) -> None:
        """Deploy an agent to the platform.

        This orchestrates the full deployment workflow:
        1. Check if workspace/deployment exists
        2. Ensure .env has DIGITALOCEAN_API_TOKEN
        3. Zip the source code
        4. Get presigned URL and upload to S3
        5. Create workspace/deployment/release as needed
        6. Poll until deployment is ready

        Args:
            agent_workspace_name: Name of the agent workspace
            agent_deployment_name: Name of the deployment
            source_dir: Directory containing the agent code
            project_id: DigitalOcean project ID
            api_token: DigitalOcean API token to include in .env

        Raises:
            ValueError: If source directory doesn't exist
            Exception: If deployment fails
        """
        logger.info(
            "Starting agent deployment",
            workspace=agent_workspace_name,
            deployment=agent_deployment_name,
        )

        #: Check if workspace and deployment exist
        workspace_exists, deployment_exists = await self._check_existing_resources(
            agent_workspace_name, agent_deployment_name
        )

        # Create zip archive (this will handle .env setup)
        zip_path = await self._create_deployment_zip(
            source_dir, agent_deployment_name, api_token
        )

        try:
            # Get presigned URL and upload
            code_artifact = await self._upload_code_artifact(zip_path)

            # Create workspace/deployment/release
            release_uuid = await self._create_or_update_deployment(
                workspace_exists=workspace_exists,
                deployment_exists=deployment_exists,
                agent_workspace_name=agent_workspace_name,
                agent_deployment_name=agent_deployment_name,
                code_artifact=code_artifact,
                project_id=project_id,
            )

            # Poll for deployment completion
            await self._poll_release_status(release_uuid)

            logger.info(
                "✅ Agent deployed successfully!",
                workspace=agent_workspace_name,
                deployment=agent_deployment_name,
            )

        finally:
            # Cleanup zip file
            if zip_path.exists():
                zip_path.unlink()
                logger.debug(f"Cleaned up zip file: {zip_path}")

    async def _check_existing_resources(
        self, agent_workspace_name: str, agent_deployment_name: str
    ) -> tuple[bool, bool]:
        """Check if workspace and deployment already exist.

        Args:
            agent_workspace_name: Name of the workspace to check
            agent_deployment_name: Name of the deployment to check

        Returns:
            Tuple of (workspace_exists, deployment_exists)
        """
        logger.info("Checking existing resources...")

        # Check if workspace exists
        workspaces_output = await self.client.list_agent_workspaces()
        workspace_exists = any(
            ws.name == agent_workspace_name for ws in workspaces_output.agent_workspaces
        )

        if not workspace_exists:
            logger.info(
                f"Workspace '{agent_workspace_name}' does not exist - will create new"
            )
            return False, False

        logger.info(f"Workspace '{agent_workspace_name}' exists")

        # Check if deployment exists
        try:
            await self.client.get_agent_workspace_deployment(
                agent_workspace_name=agent_workspace_name,
                agent_deployment_name=agent_deployment_name,
            )
            logger.info(
                f"Deployment '{agent_deployment_name}' exists - will create new release"
            )
            return True, True
        except DOAPIClientError as e:
            if e.status_code == 404:
                logger.info(
                    f"Deployment '{agent_deployment_name}' does not exist - will create new"
                )
                return True, False
            raise

    async def _create_deployment_zip(
        self, source_dir: Path, agent_deployment_name: str, api_token: str
    ) -> Path:
        """Create a zip archive of the source directory.

        This method ensures that the .env file contains DIGITALOCEAN_API_TOKEN
        before creating the zip archive.

        Args:
            source_dir: Directory to zip
            agent_deployment_name: Name for the zip file
            api_token: DigitalOcean API token to add to .env

        Returns:
            Path to the created zip file
        """
        logger.info(f"Creating deployment package from {source_dir}")

        # Ensure .env has DIGITALOCEAN_API_TOKEN
        env_file = source_dir / ".env"
        env_token_line = f"DIGITALOCEAN_API_TOKEN={api_token}\n"

        if env_file.exists():
            # Read existing .env
            env_content = env_file.read_text()

            # Check if DIGITALOCEAN_API_TOKEN is already set
            if "DIGITALOCEAN_API_TOKEN=" not in env_content:
                # Append to existing file
                logger.debug("Adding DIGITALOCEAN_API_TOKEN to existing .env file")
                with env_file.open("a") as f:
                    # Ensure there's a newline before our addition if file doesn't end with one
                    if env_content and not env_content.endswith("\n"):
                        f.write("\n")
                    f.write(env_token_line)
            else:
                logger.debug("DIGITALOCEAN_API_TOKEN already exists in .env file")
        else:
            # Create new .env file
            logger.debug("Creating new .env file with DIGITALOCEAN_API_TOKEN")
            env_file.write_text(env_token_line)

        zip_path = source_dir / f"{agent_deployment_name}.zip"

        # Run in thread pool to avoid blocking
        await asyncio.to_thread(self.zip_creator.create_zip, source_dir, zip_path)

        return zip_path

    async def _upload_code_artifact(
        self, zip_path: Path
    ) -> AgentDeploymentCodeArtifact:
        """Upload code artifact to S3 and return artifact metadata.

        Args:
            zip_path: Path to the zip file to upload

        Returns:
            AgentDeploymentCodeArtifact with upload details
        """
        logger.info("Requesting presigned URL for upload...")

        file_size = zip_path.stat().st_size

        # Get presigned URL
        presigned_input = CreateAgentDeploymentFileUploadPresignedURLInput(
            file=PresignedUrlFile(
                file_name=zip_path.name,
                file_size=file_size,
            )
        )

        presigned_output = (
            await self.client.create_agent_deployment_file_upload_presigned_url(
                presigned_input
            )
        )

        logger.info("Uploading code artifact...")

        # Upload to S3
        await self.s3_uploader.upload_file(
            zip_path, presigned_output.upload.presigned_url
        )

        # Return artifact metadata
        return AgentDeploymentCodeArtifact(
            agent_code_file_path=zip_path.name,
            stored_object_key=presigned_output.upload.object_key,
            size_in_bytes=file_size,
        )

    async def _create_or_update_deployment(
        self,
        workspace_exists: bool,
        deployment_exists: bool,
        agent_workspace_name: str,
        agent_deployment_name: str,
        code_artifact: AgentDeploymentCodeArtifact,
        project_id: str,
    ) -> str:
        """Create or update the deployment based on what exists.

        Args:
            workspace_exists: Whether the workspace exists
            deployment_exists: Whether the deployment exists
            agent_workspace_name: Name of the workspace
            agent_deployment_name: Name of the deployment
            code_artifact: Code artifact metadata
            project_id: Project ID

        Returns:
            UUID of the created release
        """
        if not workspace_exists:
            # Create new workspace (which includes first deployment)
            logger.info(f"Creating new workspace '{agent_workspace_name}'...")
            workspace_input = CreateAgentWorkspaceInput(
                agent_workspace_name=agent_workspace_name,
                agent_deployment_name=agent_deployment_name,
                agent_deployment_code_artifact=code_artifact,
                project_id=project_id,
            )
            workspace_output = await self.client.create_agent_workspace(workspace_input)

            # Get the first deployment's release UUID
            if workspace_output.agent_workspace.deployments:
                first_deployment = workspace_output.agent_workspace.deployments[0]
                if first_deployment.latest_release:
                    return first_deployment.latest_release.uuid

            raise Exception("Created workspace but no release UUID found")

        elif not deployment_exists:
            # Create new deployment in existing workspace
            logger.info(f"Creating new deployment '{agent_deployment_name}'...")
            deployment_input = CreateAgentWorkspaceDeploymentInput(
                agent_workspace_name=agent_workspace_name,
                agent_deployment_name=agent_deployment_name,
                agent_deployment_code_artifact=code_artifact,
            )
            deployment_output = await self.client.create_agent_workspace_deployment(
                deployment_input
            )

            if deployment_output.agent_workspace_deployment.latest_release:
                return deployment_output.agent_workspace_deployment.latest_release.uuid

            raise Exception("Created deployment but no release UUID found")

        else:
            # Create new release for existing deployment
            logger.info(f"Creating new release for existing deployment...")
            release_input = CreateAgentDeploymentReleaseInput(
                agent_workspace_name=agent_workspace_name,
                agent_deployment_name=agent_deployment_name,
                agent_deployment_code_artifact=code_artifact,
            )
            release_output = await self.client.create_agent_deployment_release(
                release_input
            )

            return release_output.agent_deployment_release.uuid

    async def _poll_release_status(self, release_uuid: str) -> None:
        """Poll the release status until it's running or failed.

        Args:
            release_uuid: UUID of the release to poll

        Raises:
            Exception: If deployment fails or times out
        """
        import sys

        logger.info(f"Monitoring deployment progress (UUID: {release_uuid})...")
        print()  # Add blank line for better formatting

        start_time = time.time()
        last_status = None
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_index = 0
        last_poll_time = 0.0
        current_status = None
        release = None

        # Status to emoji/message mapping
        status_display = {
            ReleaseStatus.RELEASE_STATUS_BUILDING: (
                "⏳",
                "Building",
            ),
            ReleaseStatus.RELEASE_STATUS_WAITING_FOR_DEPLOYMENT: (
                "⏳",
                "Waiting for deployment",
            ),
            ReleaseStatus.RELEASE_STATUS_DEPLOYING: ("📦", "Deploying"),
            ReleaseStatus.RELEASE_STATUS_RUNNING: ("✅", "Running"),
            ReleaseStatus.RELEASE_STATUS_FAILED: ("❌", "Failed"),
            ReleaseStatus.RELEASE_STATUS_UNKNOWN: ("❓", "Initializing"),
        }

        def format_elapsed(seconds: float) -> str:
            """Format elapsed time as MM:SS."""
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"

        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            # Check timeout
            if elapsed > self.max_polling_time_sec:
                print("\r" + " " * 80 + "\r", end="")  # Clear line
                raise Exception(
                    f"Deployment timed out after {self.max_polling_time_sec}s"
                )

            # Poll API if it's time (every polling_interval_sec)
            if current_time - last_poll_time >= self.polling_interval_sec:
                release_output = await self.client.get_agent_deployment_release(
                    release_uuid
                )
                release = release_output.agent_deployment_release
                current_status = release.status

                # Handle None status
                if current_status is None:
                    current_status = ReleaseStatus.RELEASE_STATUS_UNKNOWN

                # Log status changes (clear line first to avoid glitches)
                if current_status != last_status:
                    print("\r" + " " * 80 + "\r", end="", flush=True)  # Clear line
                    logger.debug(
                        f"Release status changed to: {current_status.value if hasattr(current_status, 'value') else current_status}"
                    )
                    last_status = current_status

                last_poll_time = current_time

                # Check terminal states
                if current_status == ReleaseStatus.RELEASE_STATUS_RUNNING:
                    print("\r" + " " * 80 + "\r", end="")  # Clear line
                    print(
                        f"✅ Deployment completed successfully! [{format_elapsed(elapsed)}]"
                    )
                    if release.url:
                        print(f"🔗 Access URL: {release.url}")
                    print()
                    return

                if current_status == ReleaseStatus.RELEASE_STATUS_FAILED:
                    print("\r" + " " * 80 + "\r", end="")  # Clear line
                    error_msg = release.error_msg or "Unknown error"
                    raise Exception(f"Deployment failed: {error_msg}")

                if current_status == ReleaseStatus.RELEASE_STATUS_UNDEPLOYMENT_FAILED:
                    print("\r" + " " * 80 + "\r", end="")  # Clear line
                    error_msg = release.error_msg or "Unknown error"
                    raise Exception(f"Undeployment failed: {error_msg}")

            # Update spinner and display (every iteration for smooth animation)
            if current_status:
                emoji, message = status_display.get(
                    current_status,
                    (
                        "⚙️",
                        (
                            current_status.value
                            if hasattr(current_status, "value")
                            else str(current_status)
                        ),
                    ),
                )
            else:
                emoji, message = ("⚙️", "Starting")

            spinner = spinner_chars[spinner_index % len(spinner_chars)]
            elapsed_str = format_elapsed(elapsed)

            # Print status line with spinner
            status_line = f"{spinner} {emoji} {message}... [{elapsed_str}]"
            print(f"\r{status_line}", end="", flush=True)

            spinner_index += 1

            # Short sleep for smooth animation (0.1 seconds)
            await asyncio.sleep(0.1)
