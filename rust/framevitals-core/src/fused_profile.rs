//! Fused Arrow numeric profiling.
//!
//! One pass over each supported numeric Arrow column updates exact moments and
//! bounded sketches together. This avoids a separate full-column scan for every
//! statistic and produces states that can be merged across batches/partitions.

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, UInt32Array,
    UInt64Array,
};
use arrow_schema::DataType;

use crate::sketches::NumericSketchState;
use crate::NumericState;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NumericProfileState {
    pub moments: NumericState,
    pub sketches: NumericSketchState,
}

impl NumericProfileState {
    pub fn observe(&mut self, value: Option<f64>, stream_id: u64, sequence: u64) {
        self.moments.observe(value);
        if let Some(value) = value {
            self.sketches.observe(value, stream_id, sequence);
        }
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            moments: self.moments.merge(other.moments),
            sketches: self.sketches.merge(other.sketches),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchProfileState {
    pub rows: u64,
    pub profiles: BTreeMap<String, NumericProfileState>,
    pub skipped_columns: BTreeSet<String>,
}

impl BatchProfileState {
    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        self.rows += other.rows;
        self.skipped_columns.extend(other.skipped_columns);
        for (name, profile) in other.profiles {
            self.profiles
                .entry(name)
                .and_modify(|current| *current = current.clone().merge(profile.clone()))
                .or_insert(profile);
        }
        self
    }
}

macro_rules! scan_profile_primitive {
    ($array:expr, $stream_id:expr) => {{
        let array = $array;
        let mut profile = NumericProfileState::default();
        for index in 0..array.len() {
            if array.is_null(index) {
                profile.observe(None, $stream_id, index as u64);
            } else {
                profile.observe(Some(array.value(index) as f64), $stream_id, index as u64);
            }
        }
        profile
    }};
}

/// Scan supported primitive numeric columns in one fused pass.
///
/// `partition_id` must be stable and distinct for independently scanned
/// partitions when deterministic reservoirs are expected to merge without
/// priority collisions. Moments and the other sketches do not depend on it.
pub fn profile_record_batch(batch: &RecordBatch, partition_id: u64) -> BatchProfileState {
    let mut output = BatchProfileState {
        rows: batch.num_rows() as u64,
        ..BatchProfileState::default()
    };

    for (index, field) in batch.schema().fields().iter().enumerate() {
        let name = field.name().clone();
        let array = batch.column(index);
        let stream_id = partition_id
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(index as u64);

        let profile = match field.data_type() {
            DataType::Float64 => array
                .as_any()
                .downcast_ref::<Float64Array>()
                .map(|values| scan_profile_primitive!(values, stream_id)),
            DataType::Float32 => array
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|values| scan_profile_primitive!(values, stream_id)),
            DataType::Int64 => array
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|values| scan_profile_primitive!(values, stream_id)),
            DataType::Int32 => array
                .as_any()
                .downcast_ref::<Int32Array>()
                .map(|values| scan_profile_primitive!(values, stream_id)),
            DataType::UInt64 => array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .map(|values| scan_profile_primitive!(values, stream_id)),
            DataType::UInt32 => array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .map(|values| scan_profile_primitive!(values, stream_id)),
            _ => None,
        };

        match profile {
            Some(profile) => {
                output.profiles.insert(name, profile);
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

    use super::profile_record_batch;

    fn batch(values: Vec<Option<f64>>, counts: Vec<Option<i64>>) -> RecordBatch {
        let rows = values.len();
        assert_eq!(counts.len(), rows);
        let labels: Vec<Option<&str>> = (0..rows)
            .map(|index| if index % 2 == 0 { Some("x") } else { None })
            .collect();
        RecordBatch::try_from_iter(vec![
            ("value", Arc::new(Float64Array::from(values)) as ArrayRef),
            ("count", Arc::new(Int64Array::from(counts)) as ArrayRef),
            ("label", Arc::new(StringArray::from(labels)) as ArrayRef),
        ])
        .expect("valid batch")
    }

    #[test]
    fn fused_profile_updates_moments_and_sketches_together() {
        let batch = batch(
            vec![Some(1.0), Some(2.0), None, Some(2.0), Some(f64::INFINITY)],
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        );
        let state = profile_record_batch(&batch, 7);
        let value = state.profiles.get("value").unwrap();

        assert_eq!(state.rows, 5);
        assert_eq!(value.moments.count, 3);
        assert_eq!(value.moments.missing, 1);
        assert_eq!(value.moments.infinite, 1);
        assert!((value.moments.mean - 5.0 / 3.0).abs() < 1e-12);
        assert!(value.sketches.cardinality.estimate() > 1.5);
        assert_eq!(value.sketches.quantiles.count(), 3);
        assert!(!value.sketches.reservoir.is_empty());
        assert!(state.skipped_columns.contains("label"));
    }

    #[test]
    fn fused_partition_merge_preserves_compact_profile_semantics() {
        let left = batch(
            (0..1_000).map(|value| Some(value as f64)).collect(),
            (0..1_000).map(|value| Some(value as i64)).collect(),
        );
        let right = batch(
            (1_000..2_000).map(|value| Some(value as f64)).collect(),
            (1_000..2_000).map(|value| Some(value as i64)).collect(),
        );

        let merged = profile_record_batch(&left, 1).merge(profile_record_batch(&right, 2));
        let value = merged.profiles.get("value").unwrap();

        assert_eq!(merged.rows, 2_000);
        assert_eq!(value.moments.count, 2_000);
        assert!((value.moments.mean - 999.5).abs() < 1e-12);
        assert!(value.sketches.cardinality.estimate() > 1_800.0);
        let median = value.sketches.quantiles.quantile(0.5).unwrap();
        assert!(median > 900.0 && median < 1_100.0);
        assert_eq!(value.sketches.reservoir.len(), 256);
    }
}
