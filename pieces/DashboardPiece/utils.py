"""Helper utilities for DashboardPiece (pandas-dependent). Kept separate so models.py can be loaded without pandas (Domino validation)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SCENARIO_COLUMNS = ["scenario", "scenario_name", "case", "variant"]


def safe_read_table(path_value: str | None, file_format: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """Read a CSV/parquet safely and return (DataFrame, error_message)."""
    if not path_value:
        return pd.DataFrame(), "file path not provided"

    file_path = Path(path_value)
    if not file_path.is_file():
        return pd.DataFrame(), f"file not found: {path_value}"

    inferred_format = (file_format or file_path.suffix.lstrip(".")).lower()

    try:
        if inferred_format == "parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as exc:
        return pd.DataFrame(), f"failed to parse {inferred_format or 'table'}: {exc}"

    for candidate in ("datetime", "timestamp", "date_time", "time"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate], errors="coerce")

    return df, None


def dataframe_to_json_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows to JSON-safe dicts, including datetimes."""
    if df.empty:
        return []

    converted = df.copy()
    for column in converted.columns:
        if pd.api.types.is_datetime64_any_dtype(converted[column]):
            converted[column] = converted[column].dt.strftime("%Y-%m-%dT%H:%M:%S")

    converted = converted.where(pd.notnull(converted), None)
    return converted.to_dict(orient="records")


def extract_scenarios(*dfs: pd.DataFrame) -> list[str]:
    """Extract scenario names from any known scenario-like column."""
    scenarios: set[str] = set()
    for df in dfs:
        if df.empty:
            continue
        lower_map = {column.lower(): column for column in df.columns}
        for scenario_col in SCENARIO_COLUMNS:
            if scenario_col in lower_map:
                source_col = lower_map[scenario_col]
                values = df[source_col].dropna().astype(str).str.strip()
                scenarios.update(v for v in values if v)
                break

    return sorted(scenarios)
