from __future__ import annotations
import os
from typing import Optional
from gradient_adk.cli.config.yaml_agent_config_manager import YamlAgentConfigManager
from gradient_adk.runtime.langgraph.langgraph_instrumentor import LangGraphInstrumentor
from gradient_adk.runtime.digitalocean_tracker import DigitalOceanTracesTracker
from gradient_adk.digital_ocean_api import AsyncDigitalOceanGenAI
from gradient_adk.runtime.network_interceptor import setup_digitalocean_interception

_TRACKER: Optional[DigitalOceanTracesTracker] = None
_INSTALLED = False

config_reader = YamlAgentConfigManager()


def capture_graph() -> None:
    """Install DO tracing for LangGraph exactly once.
    Must be called BEFORE graph.add_node/compile to capture spans.
    """
    global _TRACKER, _INSTALLED
    if _INSTALLED and _TRACKER:
        return _TRACKER

    try:
        api_token = os.environ["DIGITALOCEAN_API_TOKEN"]
    except Exception as e:
        # Only enable DO tracing if we have an API token
        return
    ws = config_reader.get_agent_name()
    dep = config_reader.get_agent_environment()

    _TRACKER = DigitalOceanTracesTracker(
        client=AsyncDigitalOceanGenAI(api_token=api_token),
        agent_workspace_name=ws,
        agent_deployment_name=dep,
    )
    setup_digitalocean_interception()
    LangGraphInstrumentor().install(_TRACKER)
    _INSTALLED = True


def get_tracker() -> Optional[DigitalOceanTracesTracker]:
    return _TRACKER
