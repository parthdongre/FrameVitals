from pathlib import Path
from uuid import uuid4

from framevitals.pipeline import run_full_analysis


VALID_MODES = {
    "quick",
    "standard",
    "deep",
    "research",
}


def analyze(
    file_path: str | Path,
    *,
    target: str | None = None,
    mode: str = "standard",
) -> dict:
    """
    Analyze a tabular dataset with FrameVitals.

    Parameters
    ----------
    file_path:
        Path to a CSV, TSV, Excel, or JSON dataset.

    target:
        Optional supervised-learning target column.

    mode:
        Analysis depth. One of:
        ``quick``, ``standard``, ``deep``, or ``research``.

    Returns
    -------
    dict
        Structured FrameVitals analysis results.

    Examples
    --------
    >>> import framevitals as fv
    >>> report = fv.analyze("customers.csv")
    >>> print(report["health"]["overall_score"])
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, got: {path}"
        )

    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid analysis mode '{mode}'. "
            f"Choose from: {', '.join(sorted(VALID_MODES))}"
        )

    dataset_id = f"fv_{uuid4().hex[:12]}"

    return run_full_analysis(
        dataset_id=dataset_id,
        file_path=path,
        original_filename=path.name,
        analysis_mode=mode,
        target_column=target,
        skip_ai=True,
    )
