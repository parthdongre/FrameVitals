import pytest

pa = pytest.importorskip("pyarrow")
native = pytest.importorskip("framevitals._native")


def test_native_arrow_batch_accumulator_profiles_int16_without_numpy_bridge():
    accumulator_type = getattr(native, "ArrowBatchProfileAccumulator", None)
    assert accumulator_type is not None

    batch = pa.record_batch(
        [
            pa.array([1, 2, None, 4], type=pa.int16()),
            pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
            pa.array(["a", "b", "c", "d"]),
        ],
        names=["small", "wide", "label"],
    )
    accumulator = accumulator_type()
    accumulator.update(batch)
    payload = dict(accumulator.snapshot())
    profiles = {
        str(name): dict(value)
        for name, value in dict(payload["profiles"]).items()
    }

    assert payload["rows"] == 4
    assert profiles["small"]["count"] == 3
    assert profiles["small"]["missing"] == 1
    assert profiles["small"]["mean"] == pytest.approx(7 / 3)
    assert profiles["small"]["minimum"] == 1.0
    assert profiles["small"]["maximum"] == 4.0
    assert profiles["wide"]["mean"] == pytest.approx(25.0)
    assert "label" in payload["skipped_columns"]
