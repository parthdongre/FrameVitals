use framevitals_core::categorical_sketches::{CategoricalSketchState, StableByteHasher};
use pyo3::buffer::{Element, PyBuffer, ReadOnlyCell};
use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

trait OffsetValue: Element + Copy {
    fn as_i64(self) -> i64;
}

impl OffsetValue for i32 {
    fn as_i64(self) -> i64 {
        i64::from(self)
    }
}

impl OffsetValue for i64 {
    fn as_i64(self) -> i64 {
        self
    }
}

fn is_valid(validity: Option<&[ReadOnlyCell<u8>]>, index: usize) -> bool {
    let Some(validity) = validity else {
        return true;
    };
    let byte_index = index / 8;
    if byte_index >= validity.len() {
        return false;
    }
    let bit = index % 8;
    validity[byte_index].get() & (1_u8 << bit) != 0
}

fn update_utf8_buffers<T: OffsetValue>(
    py: Python<'_>,
    state: &mut CategoricalSketchState,
    data: &Bound<'_, PyAny>,
    offsets: &Bound<'_, PyAny>,
    length: usize,
    validity: Option<&Bound<'_, PyAny>>,
    array_offset: usize,
) -> PyResult<()> {
    let data_buffer = PyBuffer::<u8>::get(data)?;
    let offsets_buffer = PyBuffer::<T>::get(offsets)?;
    let validity_buffer = validity.map(PyBuffer::<u8>::get).transpose()?;

    if !data_buffer.is_c_contiguous() || !offsets_buffer.is_c_contiguous() {
        return Err(PyBufferError::new_err(
            "FrameVitals string kernels require contiguous Arrow data/offset buffers.",
        ));
    }
    if let Some(buffer) = &validity_buffer {
        if !buffer.is_c_contiguous() {
            return Err(PyBufferError::new_err(
                "FrameVitals string kernels require a contiguous validity bitmap.",
            ));
        }
    }

    let data_slice = data_buffer.as_slice(py).ok_or_else(|| {
        PyBufferError::new_err("Could not borrow Arrow UTF8 data as a byte slice.")
    })?;
    let offsets_slice = offsets_buffer.as_slice(py).ok_or_else(|| {
        PyBufferError::new_err("Could not borrow Arrow UTF8 offsets as a typed slice.")
    })?;
    let validity_slice = validity_buffer
        .as_ref()
        .and_then(|buffer| buffer.as_slice(py));

    let required_offsets = array_offset
        .checked_add(length)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| PyBufferError::new_err("Arrow string offset range overflowed."))?;
    if required_offsets > offsets_slice.len() {
        return Err(PyBufferError::new_err(
            "Arrow string offsets are shorter than the requested array range.",
        ));
    }

    let label_limit = state.heavy_hitters.max_label_bytes();
    for local_index in 0..length {
        let logical_index = array_offset + local_index;
        if !is_valid(validity_slice, logical_index) {
            state.observe_missing();
            continue;
        }

        let start = offsets_slice[logical_index].get().as_i64();
        let end = offsets_slice[logical_index + 1].get().as_i64();
        if start < 0 || end < start {
            return Err(PyBufferError::new_err(
                "Arrow string offsets contain an invalid range.",
            ));
        }
        let start = usize::try_from(start)
            .map_err(|_| PyBufferError::new_err("Arrow string offset is out of range."))?;
        let end = usize::try_from(end)
            .map_err(|_| PyBufferError::new_err("Arrow string offset is out of range."))?;
        if end > data_slice.len() {
            return Err(PyBufferError::new_err(
                "Arrow string offset exceeds the UTF8 data buffer.",
            ));
        }

        let mut hasher = StableByteHasher::new();
        let mut label = Vec::with_capacity((end - start).min(label_limit));
        for cell in &data_slice[start..end] {
            let byte = cell.get();
            hasher.update(byte);
            if label.len() < label_limit {
                label.push(byte);
            }
        }
        state.observe_hashed(hasher.finish(), &label);
    }
    Ok(())
}

#[pyclass]
pub(crate) struct StringAccumulator {
    state: CategoricalSketchState,
}

#[pymethods]
impl StringAccumulator {
    #[new]
    fn new() -> Self {
        Self {
            state: CategoricalSketchState::default(),
        }
    }

    #[pyo3(signature = (data, offsets, length, validity=None, array_offset=0))]
    fn update_utf8(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        offsets: &Bound<'_, PyAny>,
        length: usize,
        validity: Option<&Bound<'_, PyAny>>,
        array_offset: usize,
    ) -> PyResult<()> {
        update_utf8_buffers::<i32>(
            py,
            &mut self.state,
            data,
            offsets,
            length,
            validity,
            array_offset,
        )
    }

    #[pyo3(signature = (data, offsets, length, validity=None, array_offset=0))]
    fn update_large_utf8(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        offsets: &Bound<'_, PyAny>,
        length: usize,
        validity: Option<&Bound<'_, PyAny>>,
        array_offset: usize,
    ) -> PyResult<()> {
        update_utf8_buffers::<i64>(
            py,
            &mut self.state,
            data,
            offsets,
            length,
            validity,
            array_offset,
        )
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item("backend", "rust")?;
        payload.set_item("count", self.state.count)?;
        payload.set_item("missing", self.state.missing)?;
        payload.set_item("observations", self.state.count + self.state.missing)?;
        payload.set_item(
            "cardinality_estimate",
            self.state.cardinality.estimate().round() as u64,
        )?;
        payload.set_item("heavy_hitters", self.state.heavy_hitters.candidates())?;
        payload.set_item("cardinality_method", "hyperloglog")?;
        payload.set_item("heavy_hitter_method", "misra_gries_candidates")?;
        payload.set_item("heavy_hitter_count_semantics", "lower_bound")?;
        payload.set_item(
            "max_retained_label_bytes",
            self.state.heavy_hitters.max_label_bytes(),
        )?;
        Ok(payload.unbind())
    }

    fn reset(&mut self) {
        self.state = CategoricalSketchState::default();
    }

    #[getter]
    fn observations(&self) -> u64 {
        self.state.count + self.state.missing
    }
}
