"""Tests for YamlAgentConfigManager multi-deployment support."""

import pytest
from pathlib import Path
import tempfile
import shutil
import yaml

from gradient_adk.cli.config.yaml_agent_config_manager import YamlAgentConfigManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test config files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_config_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def config_manager(temp_dir, monkeypatch):
    """Create a config manager pointing to the temp directory."""
    monkeypatch.chdir(temp_dir)
    return YamlAgentConfigManager()


class TestOldFormatBackwardsCompat:
    """Tests for backwards compatibility with old config format."""

    def test_get_deployment_names_returns_empty_for_old_format(self, config_manager, temp_dir):
        """Old format without deployments section returns empty list."""
        config = {
            "agent_name": "my-agent",
            "agent_environment": "main",
            "entrypoint_file": "main.py",
        }
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        (temp_dir / ".gradient" / "agent.yml").write_text(yaml.safe_dump(config))

        assert config_manager.get_deployment_names() == []

    def test_old_format_getters_still_work(self, config_manager, temp_dir):
        """Old format getters continue to work for backwards compat."""
        config = {
            "agent_name": "my-agent",
            "agent_environment": "main",
            "entrypoint_file": "main.py",
            "description": "My description",
        }
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        (temp_dir / ".gradient" / "agent.yml").write_text(yaml.safe_dump(config))

        assert config_manager.get_agent_name() == "my-agent"
        assert config_manager.get_agent_environment() == "main"
        assert config_manager.get_entrypoint_file() == "main.py"
        assert config_manager.get_description() == "My description"


class TestNewFormatDeployments:
    """Tests for new deployments section format."""

    def test_get_deployment_names_returns_deployment_keys(self, config_manager, temp_dir):
        """New format returns list of deployment names."""
        config = {
            "agent_name": "my-agent",
            "deployments": {
                "prod": {"entrypoint_file": "main.py"},
                "staging": {"entrypoint_file": "staging.py"},
            },
        }
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        (temp_dir / ".gradient" / "agent.yml").write_text(yaml.safe_dump(config))

        names = config_manager.get_deployment_names()
        assert set(names) == {"prod", "staging"}

    def test_get_config_for_deployment_merges_correctly(self, config_manager, temp_dir):
        """Deployment config merges top-level agent_name with deployment values."""
        config = {
            "agent_name": "my-agent",
            "deployments": {
                "prod": {
                    "entrypoint_file": "main.py",
                    "description": "Production deployment",
                },
            },
        }
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        (temp_dir / ".gradient" / "agent.yml").write_text(yaml.safe_dump(config))

        merged = config_manager.get_config_for_deployment("prod")
        assert merged["agent_name"] == "my-agent"
        assert merged["agent_environment"] == "prod"  # Key becomes environment
        assert merged["entrypoint_file"] == "main.py"
        assert merged["description"] == "Production deployment"

    def test_get_config_for_deployment_returns_none_for_missing(self, config_manager, temp_dir):
        """Returns None for non-existent deployment."""
        config = {
            "agent_name": "my-agent",
            "deployments": {
                "prod": {"entrypoint_file": "main.py"},
            },
        }
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        (temp_dir / ".gradient" / "agent.yml").write_text(yaml.safe_dump(config))

        assert config_manager.get_config_for_deployment("nonexistent") is None

    def test_deployment_name_becomes_agent_environment(self, config_manager, temp_dir):
        """The deployment key is used as the agent_environment."""
        config = {
            "agent_name": "my-agent",
            "deployments": {
                "production": {"entrypoint_file": "main.py"},
            },
        }
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        (temp_dir / ".gradient" / "agent.yml").write_text(yaml.safe_dump(config))

        merged = config_manager.get_config_for_deployment("production")
        assert merged["agent_environment"] == "production"


class TestAddDeployment:
    """Tests for add_deployment method."""

    def test_add_deployment_creates_deployments_section(self, config_manager, temp_dir):
        """Adding deployment to config without deployments section creates it."""
        # Create initial config without deployments
        config = {"agent_name": "my-agent"}
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        config_file = temp_dir / ".gradient" / "agent.yml"
        config_file.write_text(yaml.safe_dump(config))

        # Create entrypoint file
        (temp_dir / "main.py").write_text("from gradient_adk import entrypoint\n@entrypoint\nasync def agent(data, ctx): pass")

        config_manager.add_deployment("prod", "main.py", "Production")

        # Reload and check
        new_config = yaml.safe_load(config_file.read_text())
        assert "deployments" in new_config
        assert "prod" in new_config["deployments"]
        assert new_config["deployments"]["prod"]["entrypoint_file"] == "main.py"
        assert new_config["deployments"]["prod"]["description"] == "Production"

    def test_add_deployment_to_existing_deployments(self, config_manager, temp_dir):
        """Adding deployment appends to existing deployments section."""
        config = {
            "agent_name": "my-agent",
            "deployments": {
                "prod": {"entrypoint_file": "main.py"},
            },
        }
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        config_file = temp_dir / ".gradient" / "agent.yml"
        config_file.write_text(yaml.safe_dump(config))

        # Create entrypoint files
        (temp_dir / "main.py").write_text("from gradient_adk import entrypoint\n@entrypoint\nasync def agent(data, ctx): pass")
        (temp_dir / "staging.py").write_text("from gradient_adk import entrypoint\n@entrypoint\nasync def agent(data, ctx): pass")

        config_manager.add_deployment("staging", "staging.py")

        # Reload and check
        new_config = yaml.safe_load(config_file.read_text())
        assert "prod" in new_config["deployments"]
        assert "staging" in new_config["deployments"]
        assert new_config["deployments"]["staging"]["entrypoint_file"] == "staging.py"


class TestSaveConfigNewFormat:
    """Tests for _save_config generating new format."""

    def test_save_config_uses_deployments_format(self, config_manager, temp_dir):
        """_save_config creates config with deployments section."""
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        (temp_dir / "main.py").write_text("from gradient_adk import entrypoint\n@entrypoint\nasync def agent(data, ctx): pass")

        config_manager._save_config(
            agent_name="my-agent",
            agent_environment="main",
            entrypoint_file="main.py",
            description="Test description",
        )

        config_file = temp_dir / ".gradient" / "agent.yml"
        config = yaml.safe_load(config_file.read_text())

        assert config["agent_name"] == "my-agent"
        assert "deployments" in config
        assert "main" in config["deployments"]
        assert config["deployments"]["main"]["entrypoint_file"] == "main.py"
        assert config["deployments"]["main"]["description"] == "Test description"
        # Old format keys should NOT exist at top level
        assert "agent_environment" not in config
        assert "entrypoint_file" not in config


class TestEmptyAndMissingConfig:
    """Tests for edge cases with missing or empty configs."""

    def test_get_deployment_names_returns_empty_for_no_config(self, config_manager):
        """Returns empty list when no config file exists."""
        assert config_manager.get_deployment_names() == []

    def test_get_config_for_deployment_returns_none_for_no_config(self, config_manager):
        """Returns None when no config file exists."""
        assert config_manager.get_config_for_deployment("prod") is None

    def test_get_deployment_names_returns_empty_for_empty_deployments(self, config_manager, temp_dir):
        """Returns empty list when deployments section is empty."""
        config = {
            "agent_name": "my-agent",
            "deployments": {},
        }
        (temp_dir / ".gradient").mkdir(exist_ok=True)
        (temp_dir / ".gradient" / "agent.yml").write_text(yaml.safe_dump(config))

        assert config_manager.get_deployment_names() == []
