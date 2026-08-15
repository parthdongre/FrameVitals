from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from framevitals.contracts import (
    infer_contract as _infer_contract,
    load_contract,
    validate_contract,
)
from framevitals.drift_analysis import compare_datasets
from framevitals.loader import load_dataset
from framevitals.pipeline import run_full_analysis


VALID_MODES = {
    "quick",
    "standard",
    "deep",
    "research",
}

DataInput = str | Path | pd.DataFrame
ContractInput = dict | str | Path


def _validated_path(value: str | Path, *, label: str = "Dataset") -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file for {label.lower()}, got: {path}")
    return path


def _load_input(value: DataInput, *, label: str) -> tuple[pd.DataFrame, str]:
    if isinstance(value, pd.DataFrame):
        if value.empty:
            raise ValueError(f"{label} DataFrame is empty.")
        return value.copy(), "<dataframe>"

    if isinstance(value, (str, Path)):
        path = _validated_path(value, label=label)
        df = load_dataset(path)
        if df.empty:
            raise ValueError(f"{label} dataset is empty: {path}")
        return df, path.name

    raise TypeError(
        f"{label} must be a pandas DataFrame or a path to a supported dataset."
    )


def analyze(
    data: DataInput,
    *,
    target: str | None = None,
    mode: str = "standard",
    artifacts: bool = False,
) -> dict:
    """Analyze a tabular dataset with FrameVitals.

    ``data`` may be a pandas DataFrame or a path to CSV, TSV, Excel, or JSON.
    Reusable library calls do not persist cleaned/chart artifacts unless
    ``artifacts=True`` is supplied.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid analysis mode '{mode}'. "
            f"Choose from: {', '.join(sorted(VALID_MODES))}"
        )

    dataset_id = f"fv_{uuid4().hex[:12]}"

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Dataset DataFrame is empty.")
        return run_full_analysis(
            dataset_id=dataset_id,
            dataframe=data,
            original_filename="<dataframe>",
            analysis_mode=mode,
            target_column=target,
            skip_ai=True,
            write_artifacts=artifacts,
        )

    if isinstance(data, (str, Path)):
        path = _validated_path(data)
        return run_full_analysis(
            dataset_id=dataset_id,
            file_path=path,
            original_filename=path.name,
            analysis_mode=mode,
            target_column=target,
            skip_ai=True,
            write_artifacts=artifacts,
        )

    raise TypeError(
        "data must be a pandas DataFrame or a path to a supported dataset."
    )


def compare(
    reference: DataInput,
    current: DataInput,
    *,
    columns: list[str] | None = None,
    max_columns: int = 30,
) -> dict:
    """Compare reference and current datasets for distribution drift."""
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    ref_df, ref_name = _load_input(reference, label="Reference")
    cur_df, cur_name = _load_input(current, label="Current")

    result = compare_datasets(
        ref_df,
        cur_df,
        columns=columns,
        max_columns=max_columns,
    )
    result["reference_name"] = ref_name
    result["current_name"] = cur_name
    return result


def infer_contract(
    reference: DataInput,
    *,
    missing_tolerance: float = 0.05,
    duplicate_tolerance: float = 0.02,
    max_allowed_values: int = 20,
) -> dict:
    """Infer a reusable data-health contract from a known-good dataset."""
    df, source_name = _load_input(reference, label="Reference")
    return _infer_contract(
        df,
        source_name=source_name,
        missing_tolerance=missing_tolerance,
        duplicate_tolerance=duplicate_tolerance,
        max_allowed_values=max_allowed_values,
    )


def validate(
    data: DataInput,
    contract: ContractInput,
) -> dict:
    """Validate a dataset against a FrameVitals contract.

    ``contract`` may be the dictionary returned by :func:`infer_contract` or a
    path to a JSON contract written by the CLI.
    """
    df, source_name = _load_input(data, label="Dataset")

    if isinstance(contract, dict):
        contract_value = contract
    elif isinstance(contract, (str, Path)):
        contract_value = load_contract(contract)
    else:
        raise TypeError("contract must be a dictionary or a path to a JSON contract.")

    return validate_contract(
        df,
        contract_value,
        source_name=source_name,
    )
