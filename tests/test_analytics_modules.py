import pandas as pd

from framevitals.deep_statistics_v2 import (
    run_deep_statistics_v2,
)
from framevitals.text_profile import (
    profile_text_columns,
)
from framevitals.time_series import (
    detect_and_analyze_time_series,
)


def test_deep_statistics():
    df = pd.DataFrame({
        "age": [
            20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31,
        ],
        "income": [
            30000, 32000, 34000, 36000,
            38000, 40000, 42000, 44000,
            46000, 48000, 50000, 52000,
        ],
        "city": [
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
        ],
    })

    result = run_deep_statistics_v2(df)

    assert result["version"] == "v2"

    assert "age" in result["numeric_columns"]
    assert "income" in result["numeric_columns"]
    assert "city" in result["categorical_columns"]

    assert result["summary"]["numeric_count"] == 2
