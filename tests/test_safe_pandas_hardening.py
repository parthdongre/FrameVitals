import pandas as pd

from framevitals.safe_pandas import safe_eval


def test_safe_eval_cannot_mutate_callers_dataframe():
    frame = pd.DataFrame({"keep": [1, 2], "drop_me": [3, 4]})

    result = safe_eval("df.drop(columns=['drop_me'], inplace=True)", frame)

    assert result["ok"] is True
    assert list(frame.columns) == ["keep", "drop_me"]


def test_safe_eval_rejects_pandas_query_string_surface():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    result = safe_eval("df.query('x > 1')", frame)
    assert result["ok"] is False
    assert "Disallowed" in result["error"]


def test_safe_eval_rejects_deprecated_applymap_surface():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    result = safe_eval("df.applymap(abs)", frame)
    assert result["ok"] is False
    assert "Disallowed" in result["error"]
