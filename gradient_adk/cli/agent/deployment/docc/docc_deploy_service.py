from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from gradient_adk.logging import get_logger
from gradient_adk.cli.config.agent_config_manager import DOCCConfig
from .dockerfile_generator import generate_dockerfile
from .manifest_generator import generate_docc_manifest, write_docc_manifest

logger = get_logger(__name__)


class DOCCDeployError(Exception):
    """Raised when a DOCC deployment operation fails."""


class DOCCDeployService:
    """Orchestrates building, pushing, and deploying an ADK agent to DOCC.

    The high-level flow:
        1. Generate a Dockerfile (if one does not already exist).
        2. Build the Docker image.
        3. Push the image to the internal registry.
        4. Generate a DOCC manifest.
        5. Deploy via ``docc deploy``.
        6. Poll ``docc deploy-status`` until the rollout completes.
    """

    def __init__(
        self,
        docc_config: DOCCConfig,
        quiet: bool = False,
        deploy_timeout_sec: float = 600.0,
    ) -> None:
        self.docc_config = docc_config
        self.quiet = quiet
        self.deploy_timeout_sec = deploy_timeout_sec

    async def deploy_agent(
        self,
        agent_name: str,
        agent_environment: str,
        source_dir: Path,
        env_vars: dict[str, str] | None = None,
    ) -> str:
        """Run the full DOCC deployment pipeline.

        Args:
            agent_name: Agent workspace name.
            agent_environment: Deployment name (used as the image tag suffix).
            source_dir: Directory containing the agent source code.
            env_vars: Optional env vars to bake into the DOCC manifest.

        Returns:
            The DOCC application service address (``<app>.<namespace>.svc``).

        Raises:
            DOCCDeployError: If any step of the pipeline fails.
        """
        self._check_prerequisites()

        if not self.quiet:
            print(f"Starting DOCC deployment for {agent_name}/{agent_environment}...")

        # 1. Generate Dockerfile
        dockerfile_path = generate_dockerfile(source_dir)
        if not self.quiet:
            print(f"  Dockerfile: {dockerfile_path}")

        # 2. Build image
        image_tag = self._compute_image_tag(source_dir)
        image_uri = f"{self.docc_config.image_registry}/{agent_name}:{image_tag}"
        await self._build_image(source_dir, image_uri)

        # 3. Push image
        await self._push_image(image_uri)

        # 4. Generate DOCC manifest
        manifest = generate_docc_manifest(
            agent_name=agent_name,
            image_uri=image_uri,
            docc_config=self.docc_config,
            env_vars=env_vars,
        )
        manifest_path = source_dir / ".gradient" / "docc-manifest.json"
        write_docc_manifest(manifest, manifest_path)
        if not self.quiet:
            print(f"  Manifest written to {manifest_path}")

        # 5. Deploy via docc CLI
        docc_app_name = manifest["application"]["name"]
        await self._docc_deploy(manifest_path)

        # 6. Wait for rollout
        await self._poll_deploy_status(docc_app_name)

        service_address = (
            f"{docc_app_name}.{self.docc_config.namespace}.svc"
        )
        if not self.quiet:
            print(f"\n✅ DOCC deployment complete!")
            print(f"   Service address: {service_address}")

        return service_address

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_prerequisites(self) -> None:
        """Verify that ``docker`` and ``docc`` CLIs are available."""
        for tool in ("docker", "docc"):
            if shutil.which(tool) is None:
                raise DOCCDeployError(
                    f"'{tool}' CLI not found on PATH. "
                    f"Please install it before deploying to DOCC."
                )

    def _compute_image_tag(self, source_dir: Path) -> str:
        """Derive a deterministic short hash from the source directory contents."""
        hasher = hashlib.sha256()
        for p in sorted(source_dir.rglob("*")):
            if p.is_file() and ".git" not in p.parts:
                hasher.update(p.read_bytes())
        return hasher.hexdigest()[:12]

    async def _build_image(self, source_dir: Path, image_uri: str) -> None:
        if not self.quiet:
            print(f"  Building image {image_uri} ...")

        proc = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", image_uri, ".",
            cwd=str(source_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise DOCCDeployError(
                f"Docker build failed (exit {proc.returncode}):\n"
                f"{stderr.decode()}"
            )
        logger.debug("Docker build succeeded", image_uri=image_uri)

    async def _push_image(self, image_uri: str) -> None:
        if not self.quiet:
            print(f"  Pushing image {image_uri} ...")

        proc = await asyncio.create_subprocess_exec(
            "docker", "push", image_uri,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise DOCCDeployError(
                f"Docker push failed (exit {proc.returncode}):\n"
                f"{stderr.decode()}"
            )
        logger.debug("Docker push succeeded", image_uri=image_uri)

    async def _docc_deploy(self, manifest_path: Path) -> None:
        if not self.quiet:
            print(f"  Running docc deploy ...")

        cmd = [
            "docc",
            "--context", self.docc_config.context,
            "deploy",
            "--force",
            str(manifest_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise DOCCDeployError(
                f"docc deploy failed (exit {proc.returncode}):\n"
                f"{stderr.decode()}\n{stdout.decode()}"
            )
        if not self.quiet:
            print(f"  docc deploy submitted successfully")
        logger.debug("docc deploy succeeded", output=stdout.decode())

    async def _poll_deploy_status(self, app_name: str) -> None:
        """Poll ``docc deploy-status`` until success or timeout."""
        if not self.quiet:
            print(f"  Waiting for deployment rollout of '{app_name}' ...")

        start = time.monotonic()
        poll_interval = 10.0

        while True:
            elapsed = time.monotonic() - start
            if elapsed > self.deploy_timeout_sec:
                raise DOCCDeployError(
                    f"DOCC deployment timed out after {self.deploy_timeout_sec}s"
                )

            proc = await asyncio.create_subprocess_exec(
                "docc",
                "--context", self.docc_config.context,
                "deploy-status",
                "-n", self.docc_config.namespace,
                app_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode().strip()

            if proc.returncode == 0:
                logger.debug("docc deploy-status succeeded", output=output)
                return

            logger.debug(
                "docc deploy-status not ready yet",
                exit_code=proc.returncode,
                output=output,
                elapsed_sec=elapsed,
            )
            await asyncio.sleep(poll_interval)
