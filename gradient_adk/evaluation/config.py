"""
Evaluation configuration — loads ``.gradient/eval.yml``, merges with CLI overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class EvalConfig:
    """Resolved evaluation configuration."""

    judge_model: str = "openai/openai-gpt-oss-120b"
    judge_base_url: str = "https://inference.do-ai.run/v1"
    judge_api_key_env: str = "GRADIENT_MODEL_ACCESS_KEY"
    preset: str = "basic"
    thresholds: Dict[str, float] = field(default_factory=dict)
    dataset: Optional[str] = None
    concurrency: int = 5
    timeout: int = 30
    # Per-metric judge model overrides: metric_name -> model string
    metric_judge_models: Dict[str, str] = field(default_factory=dict)

    @property
    def judge_api_key(self) -> Optional[str]:
        return os.environ.get(self.judge_api_key_env)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_YAML_PATH = Path(".gradient") / "eval.yml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config file, returning empty dict if missing."""
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _apply_yaml(cfg: EvalConfig, data: Dict[str, Any]) -> None:
    """Mutate *cfg* with values from a parsed YAML dict."""
    for key in ("judge_model", "judge_base_url", "judge_api_key_env", "preset", "dataset"):
        if key in data:
            setattr(cfg, key, data[key])

    if "concurrency" in data:
        cfg.concurrency = int(data["concurrency"])
    if "timeout" in data:
        cfg.timeout = int(data["timeout"])

    # Top-level thresholds dict
    if isinstance(data.get("thresholds"), dict):
        cfg.thresholds.update(
            {k: float(v) for k, v in data["thresholds"].items()}
        )

    # Per-metric overrides (may include threshold + judge_model)
    if isinstance(data.get("metrics"), dict):
        for metric_name, metric_cfg in data["metrics"].items():
            if not isinstance(metric_cfg, dict):
                continue
            if "threshold" in metric_cfg:
                cfg.thresholds[metric_name] = float(metric_cfg["threshold"])
            if "judge_model" in metric_cfg:
                cfg.metric_judge_models[metric_name] = metric_cfg["judge_model"]


def load_eval_config(
    *,
    yaml_path: Optional[Path] = None,
    cli_preset: Optional[str] = None,
    cli_judge_model: Optional[str] = None,
    cli_threshold: Optional[float] = None,
    cli_dataset: Optional[str] = None,
) -> EvalConfig:
    """Build an EvalConfig with precedence: CLI > YAML > defaults."""
    cfg = EvalConfig()

    # Layer 1: YAML
    yaml_data = _load_yaml(yaml_path or _YAML_PATH)
    _apply_yaml(cfg, yaml_data)

    # Layer 2: CLI overrides
    if cli_preset is not None:
        cfg.preset = cli_preset
    if cli_judge_model is not None:
        cfg.judge_model = cli_judge_model
    if cli_dataset is not None:
        cfg.dataset = cli_dataset
    if cli_threshold is not None:
        # CLI threshold applies as a global default for every metric
        for name in cfg.thresholds:
            cfg.thresholds[name] = cli_threshold
        # Also store as a sentinel so the runner can use it as fallback
        cfg.thresholds.setdefault("__global__", cli_threshold)

    return cfg
