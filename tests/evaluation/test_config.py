"""Tests for gradient_adk.evaluation.config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gradient_adk.evaluation.config import EvalConfig, load_eval_config


class TestDefaults:
    def test_default_values(self):
        cfg = EvalConfig()
        assert cfg.judge_model == "openai/openai-gpt-oss-120b"
        assert cfg.judge_base_url == "https://inference.do-ai.run/v1"
        assert cfg.judge_api_key_env == "GRADIENT_MODEL_ACCESS_KEY"
        assert cfg.preset == "basic"
        assert cfg.thresholds == {}
        assert cfg.dataset is None
        assert cfg.concurrency == 5
        assert cfg.timeout == 30

    def test_missing_yaml_returns_defaults(self, tmp_path):
        cfg = load_eval_config(yaml_path=tmp_path / "nonexistent.yml")
        assert cfg.preset == "basic"
        assert cfg.judge_model == "openai/openai-gpt-oss-120b"


class TestYamlLoading:
    def test_yaml_overrides_defaults(self, tmp_path):
        yaml_file = tmp_path / "eval.yml"
        yaml_file.write_text(textwrap.dedent("""\
            judge_model: "custom-model"
            preset: "rag"
            concurrency: 10
            timeout: 60
            thresholds:
              answer_relevancy: 0.7
              faithfulness: 0.8
        """))
        cfg = load_eval_config(yaml_path=yaml_file)
        assert cfg.judge_model == "custom-model"
        assert cfg.preset == "rag"
        assert cfg.concurrency == 10
        assert cfg.timeout == 60
        assert cfg.thresholds["answer_relevancy"] == 0.7
        assert cfg.thresholds["faithfulness"] == 0.8

    def test_per_metric_judge_override(self, tmp_path):
        yaml_file = tmp_path / "eval.yml"
        yaml_file.write_text(textwrap.dedent("""\
            metrics:
              faithfulness:
                threshold: 0.9
                judge_model: "openai/big-model"
        """))
        cfg = load_eval_config(yaml_path=yaml_file)
        assert cfg.thresholds["faithfulness"] == 0.9
        assert cfg.metric_judge_models["faithfulness"] == "openai/big-model"

    def test_empty_yaml_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "eval.yml"
        yaml_file.write_text("")
        cfg = load_eval_config(yaml_path=yaml_file)
        assert cfg.preset == "basic"


class TestCLIOverrides:
    def test_cli_preset_overrides_yaml(self, tmp_path):
        yaml_file = tmp_path / "eval.yml"
        yaml_file.write_text("preset: rag\n")
        cfg = load_eval_config(yaml_path=yaml_file, cli_preset="agent")
        assert cfg.preset == "agent"

    def test_cli_judge_model_overrides_yaml(self, tmp_path):
        yaml_file = tmp_path / "eval.yml"
        yaml_file.write_text('judge_model: "yaml-model"\n')
        cfg = load_eval_config(yaml_path=yaml_file, cli_judge_model="cli-model")
        assert cfg.judge_model == "cli-model"

    def test_cli_dataset(self):
        cfg = load_eval_config(
            yaml_path=Path("/nonexistent"),
            cli_dataset="test.csv",
        )
        assert cfg.dataset == "test.csv"

    def test_cli_threshold_applies_globally(self, tmp_path):
        yaml_file = tmp_path / "eval.yml"
        yaml_file.write_text(textwrap.dedent("""\
            thresholds:
              answer_relevancy: 0.7
        """))
        cfg = load_eval_config(yaml_path=yaml_file, cli_threshold=0.9)
        assert cfg.thresholds["answer_relevancy"] == 0.9
        assert cfg.thresholds["__global__"] == 0.9
