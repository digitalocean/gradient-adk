"""Tests for gradient_adk.evaluation.dataset."""

from __future__ import annotations

import pytest

from gradient_adk.evaluation.dataset import parse_dataset


def _write_csv(tmp_path, name, content):
    """Write raw CSV content (no dedent — caller controls whitespace)."""
    p = tmp_path / name
    p.write_text(content)
    return p


class TestParseDataset:
    def test_query_only(self, tmp_path):
        # JSON strings: "hello", "world"  — in CSV: """hello""", """world"""
        f = _write_csv(tmp_path, "data.csv",
            'query\n'
            '"""hello"""\n'
            '"""world"""\n'
        )
        result = parse_dataset(f)
        assert len(result.rows) == 2
        assert result.rows[0].query == "hello"
        assert result.rows[1].query == "world"
        assert result.available_columns == set()

    def test_query_json_object(self, tmp_path):
        # query is a JSON object: {"msg": "hi"}
        f = _write_csv(tmp_path, "data.csv",
            'query\n'
            '"{""msg"": ""hi""}"\n'
        )
        result = parse_dataset(f)
        assert result.rows[0].query == {"msg": "hi"}

    def test_all_columns(self, tmp_path):
        f = _write_csv(tmp_path, "data.csv",
            'query,expected_output,expected_context,expected_tools\n'
            '"""hello""",goodbye,"[""ctx1"", ""ctx2""]","[""tool_a""]"\n'
        )
        result = parse_dataset(f)
        assert len(result.rows) == 1
        row = result.rows[0]
        assert row.query == "hello"
        assert row.expected_output == "goodbye"
        assert row.expected_context == ["ctx1", "ctx2"]
        assert row.expected_tools == ["tool_a"]
        assert result.available_columns == {
            "expected_output", "expected_context", "expected_tools"
        }

    def test_detects_available_columns(self, tmp_path):
        f = _write_csv(tmp_path, "data.csv",
            'query,expected_output\n'
            '"""hi""",bye\n'
        )
        result = parse_dataset(f)
        assert result.available_columns == {"expected_output"}

    def test_missing_query_column_raises(self, tmp_path):
        f = _write_csv(tmp_path, "data.csv", "input,output\nhello,world\n")
        with pytest.raises(ValueError, match="Missing required column: 'query'"):
            parse_dataset(f)

    def test_empty_query_raises(self, tmp_path):
        # Second row has an explicitly empty query field
        f = _write_csv(tmp_path, "data.csv", 'query\n"""hello"""\n""\n')
        with pytest.raises(ValueError, match="empty value in 'query'"):
            parse_dataset(f)

    def test_invalid_json_in_query_raises(self, tmp_path):
        f = _write_csv(tmp_path, "data.csv", "query\nnot-json\n")
        with pytest.raises(ValueError, match="invalid JSON in 'query'"):
            parse_dataset(f)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            parse_dataset(tmp_path / "missing.csv")

    def test_non_csv_raises(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("query\nhello\n")
        with pytest.raises(ValueError, match="CSV file"):
            parse_dataset(txt_file)

    def test_json_list_columns(self, tmp_path):
        f = _write_csv(tmp_path, "data.csv",
            'query,expected_context,expected_tools\n'
            '"""q1""","[""a"", ""b""]","[""tool1"", ""tool2""]"\n'
        )
        result = parse_dataset(f)
        assert result.rows[0].expected_context == ["a", "b"]
        assert result.rows[0].expected_tools == ["tool1", "tool2"]

    def test_no_data_rows_raises(self, tmp_path):
        f = _write_csv(tmp_path, "data.csv", "query\n")
        with pytest.raises(ValueError, match="no data rows"):
            parse_dataset(f)

    def test_optional_columns_left_empty(self, tmp_path):
        """Optional columns present in header but empty values."""
        f = _write_csv(tmp_path, "data.csv",
            'query,expected_output\n'
            '"""q1""",\n'
        )
        result = parse_dataset(f)
        assert result.rows[0].expected_output is None
