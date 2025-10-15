"""Service for managing agent traces and opening Galileo console."""

from __future__ import annotations
import webbrowser
import urllib.parse
import httpx
from typing import Protocol
from gradient_adk.logging import get_logger
from gradient_adk.digital_ocean_api.client_async import AsyncDigitalOceanGenAI
from gradient_adk.digital_ocean_api.models import (
    GetAgentWorkspaceDeploymentOutput,
    TracingServiceJWTOutput,
)
import tempfile
import base64
import html
import os
import threading

logger = get_logger(__name__)


class TracesService(Protocol):
    """Protocol for traces service."""

    async def open_traces_console(
        self, agent_workspace_name: str, agent_deployment_name: str
    ) -> None:
        """Open the DigitalOcean traces console in the browser."""
        ...

    async def get_runtime_logs(
        self, agent_workspace_name: str, agent_deployment_name: str
    ) -> str:
        """Get runtime logs for an agent deployment."""
        ...


class GalileoTracesService:
    """Service for opening Galileo traces console."""

    def __init__(self, client: AsyncDigitalOceanGenAI):
        """Initialize the traces service.

        Args:
            client: AsyncDigitalOceanGenAI client for making API calls
        """
        self.client = client

    def _build_galileo_redirect_url(
        self, base_url: str, project_id: str, logstream_id: str
    ) -> str:
        """Build the Galileo redirect URL.

        Args:
            base_url: Base URL of the Galileo service
            project_id: Galileo project ID
            logstream_id: Log stream ID

        Returns:
            The constructed redirect URL
        """
        # Remove trailing slash from base_url if present
        base_url = base_url.rstrip("/")
        return f"{base_url}/white-label-login"

    def _open_galileo_console(
        self,
        base_url: str,
        access_token: str,
        logstream_id: str,
        project_id: str,
        workspace_name: str,
        agent_name: str,
    ) -> None:
        """
        Open Galileo console by triggering a real POST in the user's browser.

        Strategy:
        1) Build a small HTML page with a <form method="post"> to the redirect URL.
        2) Auto-submit it via JS so the browser performs a POST navigation.
        3) Prefer a data: URL so nothing is written to disk; fall back to a temp file.

        Notes:
        - This keeps the token out of the query string.
        - The token/value will still be present in the local HTML page the browser loads.
            Use short-lived tokens.
        """
        redirect_url = self._build_galileo_redirect_url(
            base_url, project_id, logstream_id
        )

        fields = {
            "accessToken": access_token,
            "logstreamId": logstream_id,
            "projectId": project_id,
            "workspaceName": workspace_name,
            "agentName": agent_name,
        }

        inputs_html = "\n".join(
            f'<input type="hidden" name="{html.escape(k, quote=True)}" '
            f'value="{html.escape(v or "", quote=True)}" />'
            for k, v in fields.items()
            if v is not None
        )

        page = f"""<!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Redirecting…</title>
    </head>
    <body>
        <form id="f" action="{html.escape(redirect_url, quote=True)}" method="post">
        {inputs_html}
        </form>
        <script>document.getElementById('f').submit();</script>
        <noscript>
        <p>JavaScript is required to continue. Click the button:</p>
        <button form="f" type="submit">Continue</button>
        </noscript>
    </body>
    </html>"""

        # Option A: try a data: URL (no filesystem footprint)
        try:
            data_url = "data:text/html;base64," + base64.b64encode(
                page.encode("utf-8")
            ).decode("ascii")
            opened = webbrowser.open(data_url)
            if opened:
                return
        except Exception:
            pass

        # Option B: fall back to a temp file and clean it up shortly after
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".html", encoding="utf-8"
            ) as f:
                f.write(page)
                tmp_path = f.name
            webbrowser.open(f"file://{tmp_path}")

            # best-effort cleanup after 2 minutes
            def _cleanup(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

            if tmp_path:
                threading.Timer(120, _cleanup, args=(tmp_path,)).start()
        except Exception:
            # If even this fails, at least print a hint
            print(
                "Unable to open traces browser. You can view traces in the DigitalOcean console."
            )

    async def get_runtime_logs(
        self, agent_workspace_name: str, agent_deployment_name: str
    ) -> str:
        """Get runtime logs for an agent deployment.

        Args:
            agent_workspace_name: Name of the agent workspace
            agent_deployment_name: Name of the agent deployment

        Returns:
            str: The runtime logs content
        """
        # Get the runtime logs URL from the API
        logs_output = await self.client.get_agent_workspace_deployment_runtime_logs(
            agent_workspace_name=agent_workspace_name,
            agent_deployment_name=agent_deployment_name,
        )

        print(logs_output)

        # Fetch the logs from the live URL
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(logs_output.live_url)
            response.raise_for_status()
            return response.text

    async def open_traces_console(
        self, agent_workspace_name: str, agent_deployment_name: str
    ) -> None:
        """Open the Galileo traces console in the browser.

        This method:
        1. Lists agent workspaces to find the workspace UUID
        2. Gets agent workspace deployment details
        3. Gets tracing token
        4. Opens Galileo console with the credentials

        Args:
            agent_workspace_name: Name of the agent workspace
            agent_deployment_name: Name of the agent deployment
        """
        # List agent workspaces to find the workspace UUID
        workspaces_output = await self.client.list_agent_workspaces()
        workspace = next(
            (
                ws
                for ws in workspaces_output.agent_workspaces
                if ws.name == agent_workspace_name
            ),
            None,
        )

        if workspace is None:
            raise ValueError(f"Agent workspace '{agent_workspace_name}' not found")

        workspace_uuid = workspace.uuid
        logger.info(
            "Found agent workspace",
            workspace_name=agent_workspace_name,
            workspace_uuid=workspace_uuid,
        )

        # Get agent workspace deployment
        deployment_output: GetAgentWorkspaceDeploymentOutput = (
            await self.client.get_agent_workspace_deployment(
                agent_workspace_name=agent_workspace_name,
                agent_deployment_name=agent_deployment_name,
            )
        )

        deployment = deployment_output.agent_workspace_deployment
        logging_config = deployment.logging_config

        # Get tracing token using workspace UUID
        tracing_token: TracingServiceJWTOutput = await self.client.get_tracing_token(
            agent_workspace_uuid=workspace_uuid,
            agent_deployment_name=agent_deployment_name,
        )

        # Open Galileo console
        self._open_galileo_console(
            base_url=tracing_token.base_url,
            access_token=tracing_token.access_token,
            logstream_id=logging_config.log_stream_id,
            project_id=logging_config.galileo_project_id,
            workspace_name=agent_workspace_name,
            agent_name=agent_deployment_name,
        )
