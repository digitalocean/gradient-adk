from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict

from gradient_adk.cli.config.agent_config_manager import DOCCConfig


def generate_docc_manifest(
    agent_name: str,
    image_uri: str,
    docc_config: DOCCConfig,
    env_vars: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Build a DOCC application manifest dict for an ADK agent.

    Args:
        agent_name: The agent workspace name (used as the DOCC app name).
        image_uri: Fully qualified Docker image URI including tag.
        docc_config: DOCC-specific configuration from agent.yml.
        env_vars: Optional environment variables to inject into the container.

    Returns:
        A dict representing a valid DOCC manifest JSON.
    """
    service_id = docc_config.service_id or str(uuid.uuid4())

    # Truncate name to 28 chars (DOCC limit) and prefix with "adk-"
    docc_app_name = f"adk-{agent_name}"[:28]

    container_env: Dict[str, str] = {
        "PORT": "8080",
    }
    if env_vars:
        container_env.update(env_vars)

    manifest: Dict[str, Any] = {
        "$schema": "https://docc-schema.internal.digitalocean.com/manifest.json",
        "service_id": service_id,
        "maintainer": docc_config.maintainer,
        "application": {
            "name": docc_app_name,
            "namespace": docc_config.namespace,
            "scale": docc_config.scale,
            "fault_domain": "node",
            "auto_tls": True,
            "stdout_logging": True,
            "containers": {
                "agent": {
                    "image": image_uri,
                    "init": False,
                    "ports": [
                        {"port": 8080, "protocol": "TCP"},
                    ],
                    "env": container_env,
                    "resources": {
                        "request": {
                            "memory": docc_config.resources_request_memory,
                            "cpu": docc_config.resources_request_cpu,
                        },
                        "limit": {
                            "memory": docc_config.resources_limit_memory,
                            "cpu": docc_config.resources_limit_cpu,
                        },
                    },
                    "check": {
                        "ready": {
                            "path": "/health",
                            "port": 8080,
                            "start_after": "30s",
                            "interval": "10s",
                            "timeout": "5s",
                        },
                        "health": {
                            "path": "/health",
                            "port": 8080,
                            "start_after": "30s",
                            "interval": "15s",
                            "timeout": "5s",
                        },
                    },
                },
            },
            "metrics": [
                {"path": "/metrics", "port": 8080},
            ],
        },
    }

    if docc_config.github_acl:
        manifest["github_acl"] = docc_config.github_acl

    if docc_config.mtls_acl:
        manifest["mtls_acl"] = docc_config.mtls_acl

    return manifest


def write_docc_manifest(manifest: Dict[str, Any], output_path: Path) -> Path:
    """Serialize a DOCC manifest dict to a JSON file.

    Args:
        manifest: The manifest dict (as returned by ``generate_docc_manifest``).
        output_path: Destination file path.

    Returns:
        The path the manifest was written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path
