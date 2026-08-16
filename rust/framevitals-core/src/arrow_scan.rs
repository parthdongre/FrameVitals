//! Arrow-native batch scanning for FrameVitals.
//!
//! The scanner consumes Arrow `RecordBatch` values incrementally and produces
//! mergeable numeric states. It never materializes a Python object per cell and
//! does not require a complete dataset to reside in memory.

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, UInt32Array,
    UInt64Array,
};
use arrow_schema::DataType;

use crate::NumericState;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchNumericStates {
    pub rows: u64,
    pub states: BTreeMap<String, NumericState>,
    pub skipped_columns: BTreeSet<String>,
}

impl BatchNumericStates {
    /// Merge states from another Arrow batch/partition without retaining either
    /// batch's raw observations.
    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        self.rows += other.rows;
        self.skipped_columns.extend(other.skipped_columns);

        for (name, state) in other.states {
            self.states
                .entry(name)
                .and_modify(|current| *current = current.merge(state))
                .or_insert(state);
        }
        self
    }
}

macro_rules! scan_primitive {
    ($array:expr) => {{
        let array = $array;
        let mut state = NumericState::default();
        for index in 0..array.len() {
            if array.is_null(index) {
                state.observe(None);
            } else {
                state.observe(Some(array.value(index) as f64));
            }
        }
        state
    }};
}

/// Scan the supported primitive numeric columns of an Arrow `RecordBatch`.
///
/// Unsupported types are reported in `skipped_columns` rather than failing the
/// complete batch, allowing future semantic/text/sketch scanners to process
/// those columns independently.
pub fn scan_record_batch(batch: &RecordBatch) -> BatchNumericStates {
    let mut output = BatchNumericStates {
        rows: batch.num_rows() as u64,
        ..BatchNumericStates::default()
    };

    for (index, field) in batch.schema().fields().iter().enumerate() {
        let name = field.name().clone();
        let array = batch.column(index);

        let state = match field.data_type() {
            DataType::Float64 => array
                .as_any()
                .downcast_ref::<Float64Array>()
                .map(|values| scan_primitive!(values)),
            DataType::Float32 => array
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|values| scan_primitive!(values)),
            DataType::Int64 => array
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|values| scan_primitive!(values)),
            DataType::Int32 => array
                .as_any()
                .downcast_ref::<Int32Array>()
                .map(|values| scan_primitive!(values)),
            DataType::UInt64 => array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .map(|values| scan_primitive!(values)),
            DataType::UInt32 => array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .map(|values| scan_primitive!(values)),
            _ => None,
        };

        match state {
            Some(state) => {
                output.states.insert(name, state);
            }
            None => {
                output.skipped_columns.insert(name);
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};

    use super::scan_record_batch;

    fn batch(values: Vec<Option<f64>>, integers: Vec<Option<i64>>) -> RecordBatch {
        RecordBatch::try_from_iter(vec![
            ("value", Arc::new(Float64Array::from(values)) as ArrayRef),
            ("count", Arc::new(Int64Array::from(integers)) as ArrayRef),
            (
                "label",
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])) as ArrayRef,
            ),
        ])
        .expect("valid test batch")
    }

    #[test]
    fn scans_supported_numeric_columns_and_reports_others() {
        let batch = batch(
            vec![Some(1.0), None, Some(f64::INFINITY)],
            vec![Some(10), Some(20), None],
        );
        let state = scan_record_batch(&batch);

        assert_eq!(state.rows, 3);
        assert_eq!(state.states.len(), 2);
        assert!(state.skipped_columns.contains("label"));

        let value = state.states.get("value").unwrap();
        assert_eq!(value.count, 1);
        assert_eq!(value.missing, 1);
        assert_eq!(value.infinite, 1);
        assert_eq!(value.mean, 1.0);

        let count = state.states.get("count").unwrap();
        assert_eq!(count.count, 2);
        assert_eq!(count.missing, 1);
        assert_eq!(count.mean, 15.0);
    }

    #[test]
    fn merges_arrow_batches_into_one_dataset_state() {
        let left = batch(
            vec![Some(1.0), Some(2.0), None],
            vec![Some(1), Some(2), Some(3)],
        );
        let right = batch(
            vec![Some(4.0), Some(8.0), Some(16.0)],
            vec![Some(4), Some(5), Some(6)],
        );

        let merged = scan_record_batch(&left).merge(scan_record_batch(&right));
        let value = merged.states.get("value").unwrap();
        let count = merged.states.get("count").unwrap();

        assert_eq!(merged.rows, 6);
        assert_eq!(value.count, 5);
        assert_eq!(value.missing, 1);
        assert_eq!(value.minimum, Some(1.0));
        assert_eq!(value.maximum, Some(16.0));
        assert!((value.mean - 6.2).abs() < 1e-12);

        assert_eq!(count.count, 6);
        assert_eq!(count.minimum, Some(1.0));
        assert_eq!(count.maximum, Some(6.0));
        assert!((count.mean - 3.5).abs() < 1e-12);
    }
}
