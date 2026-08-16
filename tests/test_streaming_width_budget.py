import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

import framevitals
from framevitals.streaming_profile import (
    STREAM_BATCH_CELL_BUDGET,
    STREAM_BATCH_SIZE,
    STREAM_SAMPLE_CELL_BUDGET,
    STREAM_SAMPLE_ROWS,
    _width_aware_row_limit,
)


def test_width_aware_limits_preserve_normal_width_defaults():
    columns = 105

    assert _width_aware_row_limit(
        STREAM_BATCH_SIZE,
        columns,
        cell_budget=STREAM_BATCH_CELL_BUDGET,
    ) == STREAM_BATCH_SIZE
    assert _width_aware_row_limit(
        STREAM_SAMPLE_ROWS,
        columns,
        cell_budget=STREAM_SAMPLE_CELL_BUDGET,
    ) == STREAM_SAMPLE_ROWS


def test_width_aware_limits_clamp_ten_thousand_columns():
    columns = 10_000

    assert _width_aware_row_limit(
        STREAM_BATCH_SIZE,
        columns,
        cell_budget=STREAM_BATCH_CELL_BUDGET,
    ) == 3_200
    assert _width_aware_row_limit(
        STREAM_SAMPLE_ROWS,
        columns,
        cell_budget=STREAM_SAMPLE_CELL_BUDGET,
    ) == 600


def test_public_profile_discloses_width_limited_execution():
    rows = 100
    columns = 500
    values = np.arange(rows, dtype=np.float64)
    table = pa.table({f"c{index}": values + index for index in range(columns)})

    result = framevitals.profile(table)
    streaming = result["streaming_metadata"]

    assert result["shape"] == {"rows": rows, "columns": columns}
    assert streaming["full_materialization"] is False
    assert streaming["width_limited"] is True
    assert streaming["requested_batch_size"] == STREAM_BATCH_SIZE
    assert streaming["batch_size"] == STREAM_BATCH_CELL_BUDGET // columns
    assert streaming["requested_sample_rows"] == STREAM_SAMPLE_ROWS
    assert streaming["sample_row_limit"] == STREAM_SAMPLE_CELL_BUDGET // columns
    assert streaming["sample_cells_retained"] == rows * columns
