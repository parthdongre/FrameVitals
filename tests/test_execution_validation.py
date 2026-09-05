import pandas as pd
import pytest

from framevitals.execution import (
    ExecutionPolicy,
    derive_execution_budget,
    derive_streaming_profile_column_limit,
    deterministic_sample_frame,
)


@pytest.mark.parametrize(
    "value",
    [True, 2.5, "10"],
)
def test_execution_policy_rejects_non_integer_caps(value):
    with pytest.raises(ValueError, match="max_sample_rows"):
        ExecutionPolicy(max_sample_rows=value)


def test_execution_budget_rejects_boolean_and_fractional_shapes():
    with pytest.raises(ValueError, match="rows"):
        derive_execution_budget(True, 10)
    with pytest.raises(ValueError, match="columns"):
        derive_streaming_profile_column_limit(10, 2.5)


def test_deterministic_sample_rejects_invalid_control_types():
    frame = pd.DataFrame({"x": range(5)})
    with pytest.raises(ValueError, match="max_rows"):
        deterministic_sample_frame(frame, True)
    with pytest.raises(TypeError, match="DataFrame"):
        deterministic_sample_frame([1, 2, 3], 2)
