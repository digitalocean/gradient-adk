"""
Integration tests for the `gradient agent evaluate` CLI command.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


class TestADKAgentsEvaluateDatasetValidation:
    """Tests for evaluate command dataset validation."""

    @pytest.fixture
    def echo_agent_dir(self):
        """Get the path to the echo agent directory."""
        return Path(__file__).parent.parent / "example_agents" / "echo_agent"

    @pytest.fixture
    def setup_valid_agent(self, echo_agent_dir):
        """
        Setup a temporary directory with a valid agent structure.
        Yields the temp directory path and cleans up after.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy the echo agent main.py
            shutil.copy(echo_agent_dir / "main.py", temp_path / "main.py")

            # Create .gradient directory and config
            gradient_dir = temp_path / ".gradient"
            gradient_dir.mkdir()

            config = {
                "agent_name": "test-echo-agent",
                "agent_environment": "main",
                "entrypoint_file": "main.py",
            }

            with open(gradient_dir / "agent.yml", "w") as f:
                yaml.safe_dump(config, f)

            # Create requirements.txt
            requirements_path = temp_path / "requirements.txt"
            requirements_path.write_text("gradient-adk\n")

            yield temp_path

    @pytest.mark.cli
    def test_evaluate_missing_dataset_file(self, setup_valid_agent):
        """
        Test that evaluate fails with helpful error when dataset file doesn't exist.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        logger.info(f"Testing evaluate with missing dataset file in {temp_path}")

        result = subprocess.run(
            [
                "gradient", "agent", "evaluate",
                "--test-case-name", "test-case",
                "--dataset-file", "nonexistent.csv",
                "--categories", "correctness",
                "--no-interactive",
            ],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Evaluate should have failed with missing dataset"

        combined_output = result.stdout + result.stderr
        assert "file not found" in combined_output.lower() or "not found" in combined_output.lower(), \
            f"Expected error about missing file, got: {combined_output}"

        logger.info("Correctly failed with missing dataset file")

    @pytest.mark.cli
    def test_evaluate_missing_query_column(self, setup_valid_agent):
        """
        Test that evaluate fails with helpful error when dataset is missing query column.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Create a CSV without the required 'query' column
        dataset_path = temp_path / "bad_dataset.csv"
        dataset_path.write_text("""prompt,expected_response
"hello","world"
""")

        logger.info(f"Testing evaluate with dataset missing query column in {temp_path}")

        result = subprocess.run(
            [
                "gradient", "agent", "evaluate",
                "--test-case-name", "test-case",
                "--dataset-file", str(dataset_path),
                "--categories", "correctness",
                "--no-interactive",
            ],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Evaluate should have failed with missing query column"

        combined_output = result.stdout + result.stderr
        assert "missing required column" in combined_output.lower() or "query" in combined_output.lower(), \
            f"Expected error about missing query column, got: {combined_output}"

        logger.info("Correctly failed with missing query column")

    @pytest.mark.cli
    def test_evaluate_invalid_json_in_query(self, setup_valid_agent):
        """
        Test that evaluate fails with helpful error when query column has invalid JSON.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Create a CSV with invalid JSON in the query column
        dataset_path = temp_path / "bad_json_dataset.csv"
        dataset_path.write_text("""query,expected_response
"this is not valid JSON",10%
"{""valid"": ""json""}",20%
"also not valid JSON {",30%
""")

        logger.info(f"Testing evaluate with invalid JSON in query column in {temp_path}")

        result = subprocess.run(
            [
                "gradient", "agent", "evaluate",
                "--test-case-name", "test-case",
                "--dataset-file", str(dataset_path),
                "--categories", "correctness",
                "--no-interactive",
            ],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Evaluate should have failed with invalid JSON"

        combined_output = result.stdout + result.stderr
        # Should mention which rows have invalid JSON
        assert "row 2" in combined_output.lower() or "row 4" in combined_output.lower(), \
            f"Expected error mentioning specific row numbers, got: {combined_output}"
        assert "invalid json" in combined_output.lower(), \
            f"Expected error about invalid JSON, got: {combined_output}"

        logger.info("Correctly failed with invalid JSON in query column")

    @pytest.mark.cli
    def test_evaluate_empty_query_value(self, setup_valid_agent):
        """
        Test that evaluate fails with helpful error when query column has empty values.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Create a CSV with empty query values
        dataset_path = temp_path / "empty_query_dataset.csv"
        dataset_path.write_text("""query,expected_response
"{""query"": ""hello""}",10%
"",20%
"{""query"": ""world""}",30%
""")

        logger.info(f"Testing evaluate with empty query values in {temp_path}")

        result = subprocess.run(
            [
                "gradient", "agent", "evaluate",
                "--test-case-name", "test-case",
                "--dataset-file", str(dataset_path),
                "--categories", "correctness",
                "--no-interactive",
            ],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Evaluate should have failed with empty query value"

        combined_output = result.stdout + result.stderr
        # Should mention empty value
        assert "empty" in combined_output.lower() or "row 3" in combined_output.lower(), \
            f"Expected error about empty value, got: {combined_output}"

        logger.info("Correctly failed with empty query value")

    @pytest.mark.cli
    def test_evaluate_valid_dataset(self, setup_valid_agent):
        """
        Test that evaluate accepts a valid dataset file.
        Note: This will fail at the API call stage, but should pass validation.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Create a valid dataset
        dataset_path = temp_path / "valid_dataset.csv"
        dataset_path.write_text("""query,expected_response
"{""query"": ""How much did revenue increase?""}",10%
"{""query"": ""What was the total?""}",22b
""")

        logger.info(f"Testing evaluate with valid dataset in {temp_path}")

        result = subprocess.run(
            [
                "gradient", "agent", "evaluate",
                "--test-case-name", "test-case",
                "--dataset-file", str(dataset_path),
                "--categories", "correctness",
                "--no-interactive",
            ],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        combined_output = result.stdout + result.stderr
        # Should not fail on dataset validation - will fail later at API call
        assert "dataset validation failed" not in combined_output.lower(), \
            f"Dataset validation should have passed, got: {combined_output}"
        
        # The command will fail at API stage, but that's OK - we're testing dataset validation
        logger.info("Dataset validation passed (command may fail at API stage)")

    @pytest.mark.cli
    def test_evaluate_non_csv_file(self, setup_valid_agent):
        """
        Test that evaluate fails with helpful error when file is not a CSV.
        """
        logger = logging.getLogger(__name__)
        temp_path = setup_valid_agent

        # Create a non-CSV file
        dataset_path = temp_path / "dataset.txt"
        dataset_path.write_text("This is not a CSV file")

        logger.info(f"Testing evaluate with non-CSV file in {temp_path}")

        result = subprocess.run(
            [
                "gradient", "agent", "evaluate",
                "--test-case-name", "test-case",
                "--dataset-file", str(dataset_path),
                "--categories", "correctness",
                "--no-interactive",
            ],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DIGITALOCEAN_API_TOKEN": "test-token"},
        )

        assert result.returncode != 0, "Evaluate should have failed with non-CSV file"

        combined_output = result.stdout + result.stderr
        assert "csv" in combined_output.lower(), \
            f"Expected error about CSV requirement, got: {combined_output}"

        logger.info("Correctly failed with non-CSV file")


class TestADKAgentsEvaluateHelp:
    """Tests for evaluate command help."""

    @pytest.mark.cli
    def test_evaluate_help(self):
        """
        Test that 'gradient agent evaluate --help' shows proper usage.
        """
        logger = logging.getLogger(__name__)

        result = subprocess.run(
            ["gradient", "agent", "evaluate", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, "Help command should succeed"

        combined_output = result.stdout + result.stderr
        # Check for expected options in help
        assert "--test-case-name" in combined_output, "Should show --test-case-name option"
        assert "--dataset-file" in combined_output, "Should show --dataset-file option"
        assert "--categories" in combined_output, "Should show --categories option"
        assert "--star-metric-name" in combined_output, "Should show --star-metric-name option"

        logger.info("Help output is correct")