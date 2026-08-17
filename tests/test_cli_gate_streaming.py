import json

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from framevitals.cli import main
from framevitals.sources import ParquetSource


def _write_parquet(path, *, rows: int, shift: float = 0.0) -> None:
    frame = pd.DataFrame({
        "value": np.arange(rows, dtype=np.float64) + shift,
        "group": [f"g-{index % 5}" for index in range(rows)],
    })
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        row_group_size=777,
    )


def test_cli_reference_gate_streams_parquet(tmp_path, monkeypatch, capsys):
    reference = tmp_path / "reference.parquet"
    current = tmp_path / "current.parquet"
    _write_parquet(reference, rows=60_000)
    _write_parquet(current, rows=72_000, shift=50.0)

    def fail_load(self):
        raise AssertionError("CLI reference gate must not fully materialize Parquet inputs")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "gate",
            str(current),
            "--reference",
            str(reference),
            "--format",
            "json",
        ],
    )

    exit_code = main()
    assert exit_code in {0, 1}
    payload = json.loads(capsys.readouterr().out)
    execution = payload["execution"]["drift"]
    assert execution["full_materialization"] is False
    assert execution["reference"]["source_rows"] == 60_000
    assert execution["reference"]["sample_rows"] == 50_000
    assert execution["current"]["source_rows"] == 72_000
    assert execution["current"]["sample_rows"] == 50_000
    assert execution["components"]["value_distributions"] == "bounded_row_sample"
