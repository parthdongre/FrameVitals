//! Python bindings for FrameVitals native kernels.

use framevitals_core::sketches::NumericSketchState;
use framevitals_core::NumericState;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn exact_state_dict<'py>(
    py: Python<'py>,
    state: &NumericState,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("count", state.count)?;
    payload.set_item("missing", state.missing)?;
    payload.set_item("infinite", state.infinite)?;
    payload.set_item(
        "mean",
        if state.count > 0 { Some(state.mean) } else { None },
    )?;
    payload.set_item("variance", state.variance())?;
    payload.set_item("std", state.standard_deviation())?;
    payload.set_item("minimum", state.minimum)?;
    payload.set_item("maximum", state.maximum)?;
    Ok(payload)
}

fn checked_buffer<'py>(values: &Bound<'py, PyAny>) -> PyResult<PyBuffer<f64>> {
    let buffer = PyBuffer::<f64>::get(values)?;
    if buffer.dimensions() != 1 {
        return Err(PyBufferError::new_err(
            "FrameVitals native numeric kernels require a 1D float64 buffer.",
        ));
    }
    if !buffer.is_c_contiguous() {
        return Err(PyBufferError::new_err(
            "FrameVitals native numeric kernels require a C-contiguous float64 buffer.",
        ));
    }
    Ok(buffer)
}

#[pyfunction]
fn numeric_state_f64(py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let buffer = checked_buffer(values)?;
    let slice = buffer.as_slice(py).ok_or_else(|| {
        PyBufferError::new_err(
            "FrameVitals could not borrow the supplied float64 buffer as a contiguous slice.",
        )
    })?;

    let mut state = NumericState::default();
    for value in slice {
        state.observe(Some(value.get()));
    }
    let payload = exact_state_dict(py, &state)?;
    payload.set_item("backend", "rust")?;
    payload.set_item("observations", buffer.item_count())?;
    Ok(payload.unbind())
}

#[pyfunction]
#[pyo3(signature = (values, stream_id = 0))]
fn numeric_profile_f64(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    stream_id: u64,
) -> PyResult<Py<PyDict>> {
    let buffer = checked_buffer(values)?;
    let slice = buffer.as_slice(py).ok_or_else(|| {
        PyBufferError::new_err(
            "FrameVitals could not borrow the supplied float64 buffer as a contiguous slice.",
        )
    })?;

    let mut state = NumericState::default();
    let mut sketches = NumericSketchState::default();

    for (index, value) in slice.iter().enumerate() {
        let value = value.get();
        state.observe(Some(value));
        sketches.observe(value, stream_id, index as u64);
    }

    let payload = exact_state_dict(py, &state)?;
    payload.set_item("backend", "rust")?;
    payload.set_item("observations", buffer.item_count())?;
    payload.set_item(
        "cardinality_estimate",
        sketches.cardinality.estimate().round() as u64,
    )?;

    let quantiles = PyDict::new(py);
    for (name, q) in [
        ("p01", 0.01),
        ("p05", 0.05),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p95", 0.95),
        ("p99", 0.99),
    ] {
        quantiles.set_item(name, sketches.quantiles.quantile(q))?;
    }
    quantiles.set_item(
        "relative_accuracy",
        sketches.quantiles.relative_accuracy(),
    )?;
    payload.set_item("quantiles", quantiles)?;
    payload.set_item("heavy_hitters", sketches.heavy_hitters.candidates())?;
    payload.set_item("reservoir", sketches.reservoir.values())?;
    Ok(payload.unbind())
}

#[pyfunction]
fn backend_info() -> (&'static str, &'static str) {
    ("rust", env!("CARGO_PKG_VERSION"))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(numeric_state_f64, module)?)?;
    module.add_function(wrap_pyfunction!(numeric_profile_f64, module)?)?;
    module.add_function(wrap_pyfunction!(backend_info, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
