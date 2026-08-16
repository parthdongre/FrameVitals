import json

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from framevitals.cli_entry import main
from framevitals.sources import ParquetSource


def test_installed_cli_analyze_streams_parquet(tmp_path, monkeypatch, capsys):
    path = tmp_path / "cli-stream.parquet"
    frame = pd.DataFrame({
        "value": np.arange(12_000, dtype=np.float64),
        "other": np.arange(12_000, dtype=np.float64) * 2.0,
        "group": [f"g-{index % 5}" for index in range(12_000)],
    })
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        row_group_size=777,
    )

    def fail_load(self):
        raise AssertionError("installed CLI analyze must not materialize full Parquet")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "analyze",
            str(path),
            "--mode",
            "quick",
            "--no-artifacts",
            "--workers",
            "1",
            "--format",
            "json",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["shape"] == {"rows": len(frame), "columns": 3}
    assert payload["execution"]["streaming"]["enabled"] is True
    assert payload["execution"]["streaming"]["full_materialization"] is False
    assert payload["execution"]["streaming"]["working_sample_rows"] == 5_000
