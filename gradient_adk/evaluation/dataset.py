"""
CSV dataset parsing and validation for local evaluation.

Extends the validation pattern from ``cli/agent/evaluation_service.py``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DatasetRow:
    """One parsed row from the evaluation dataset."""

    query: Any  # parsed JSON value
    expected_output: Optional[str] = None
    expected_context: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None


@dataclass
class ParsedDataset:
    """Result of parsing an evaluation CSV."""

    rows: List[DatasetRow] = field(default_factory=list)
    available_columns: Set[str] = field(default_factory=set)


# Optional columns and whether their value is a JSON list
_OPTIONAL_COLUMNS: Dict[str, bool] = {
    "expected_output": False,
    "expected_context": True,
    "expected_tools": True,
}


def parse_dataset(file_path: Path) -> ParsedDataset:
    """Parse and validate a CSV dataset.

    Raises ``ValueError`` with a user-friendly message on any problem.
    """
    if not file_path.exists():
        raise ValueError(f"Dataset file not found: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Dataset must be a CSV file, got: {file_path.suffix}")

    rows: List[DatasetRow] = []
    available_columns: Set[str] = set()

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no header row")

        if "query" not in reader.fieldnames:
            raise ValueError(
                f"Missing required column: 'query'. "
                f"Found columns: {', '.join(reader.fieldnames)}"
            )

        # Detect which optional columns are present
        for col in _OPTIONAL_COLUMNS:
            if col in reader.fieldnames:
                available_columns.add(col)

        for row_num, row in enumerate(reader, start=2):
            query_raw = row.get("query", "").strip()
            if not query_raw:
                raise ValueError(f"Row {row_num}: empty value in 'query' column")

            try:
                query_parsed = json.loads(query_raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Row {row_num}: invalid JSON in 'query' column: {e}"
                ) from e

            dr = DatasetRow(query=query_parsed)

            # Parse optional columns
            for col, is_json_list in _OPTIONAL_COLUMNS.items():
                raw = row.get(col, "").strip() if col in available_columns else ""
                if not raw:
                    continue
                if is_json_list:
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Row {row_num}: invalid JSON in '{col}' column: {e}"
                        ) from e
                    if not isinstance(parsed, list):
                        raise ValueError(
                            f"Row {row_num}: '{col}' must be a JSON list"
                        )
                    setattr(dr, col, parsed)
                else:
                    setattr(dr, col, raw)

            rows.append(dr)

    if not rows:
        raise ValueError("Dataset has no data rows")

    return ParsedDataset(rows=rows, available_columns=available_columns)
