from __future__ import annotations

import os
from typing import Any, Dict, Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"
DEFAULT_MAX_EVENT_BYTES = 256 * 1024


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes"}


def get_package_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("gradient-adk")
    except Exception:
        try:
            from gradient_adk import __version__

            return __version__
        except Exception:
            return "unknown"


def get_otlp_traces_endpoint() -> str:
    traces_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if traces_endpoint:
        return traces_endpoint

    base_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    return f"{base_endpoint.rstrip('/')}/v1/traces"


def build_resource_attributes(
    *,
    agent_workspace_name: str,
    agent_deployment_name: str,
    team_id: Optional[str] = None,
    org_id: Optional[str] = None,
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {
        "service.name": "adk-agent",
        "service.version": get_package_version(),
        "do.trace.visibility": "customer",
        # The collector is authoritative in production, but these help
        # local/dev runs where no resource processor is present.
        "agent_workspace_name": agent_workspace_name,
        "agent_deployment_name": agent_deployment_name,
    }
    if team_id:
        attributes["team_id"] = str(team_id)
    if org_id:
        attributes["org_id"] = str(org_id)
    return attributes


def create_tracer_provider(
    *,
    agent_workspace_name: str,
    agent_deployment_name: str,
    team_id: Optional[str] = None,
    org_id: Optional[str] = None,
) -> TracerProvider:
    resource = Resource.create(
        build_resource_attributes(
            agent_workspace_name=agent_workspace_name,
            agent_deployment_name=agent_deployment_name,
            team_id=team_id,
            org_id=org_id,
        )
    )

    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=ALWAYS_ON),
    )
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=get_otlp_traces_endpoint()),
        export_timeout_millis=5000,
        max_queue_size=2048,
        max_export_batch_size=512,
    )
    provider.add_span_processor(processor)
    return provider
