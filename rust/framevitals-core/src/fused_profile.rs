//! Fused Arrow numeric profiling.
//!
//! One pass over each supported numeric Arrow column updates exact moments and
//! the one sketch consumed by the streaming profile: mergeable log quantiles.
//! Richer HLL/heavy-hitter/reservoir sketches remain available through the
//! standalone numeric profiling APIs, but the full-stream dataframe profiler
//! avoids paying for statistics it never reads.

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::{
    Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, RecordBatch,
    UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::DataType;

use crate::sketches::LogQuantileSketch;
use crate::NumericState;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NumericProfileState {
    pub moments: NumericState,
    pub quantiles: LogQuantileSketch,
}

impl NumericProfileState {
    pub fn observe(&mut self, value: Option<f64>) {
        self.moments.observe(value);
        if let Some(value) = value {
            if value.is_finite() {
                self.quantiles.observe(value);
            }
        }
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            moments: self.moments.merge(other.moments),
            quantiles: self.quantiles.merge(other.quantiles),
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
    ($array:expr) => {{
        let array = $array;
        let mut profile = NumericProfileState::default();
        for index in 0..array.len() {
            if array.is_null(index) {
                profile.observe(None);
            } else {
                profile.observe(Some(array.value(index) as f64));
            }
        }
        profile
    }};
}

/// Scan supported primitive numeric columns in one fused pass.
///
/// ``partition_id`` is retained in the public signature for compatibility with
/// earlier fused-profile callers. Moments and log quantiles are deterministic
/// without partition-specific state.
pub fn profile_record_batch(batch: &RecordBatch, _partition_id: u64) -> BatchProfileState {
    let mut output = BatchProfileState {
        rows: batch.num_rows() as u64,
        ..BatchProfileState::default()
    };

    for (index, field) in batch.schema().fields().iter().enumerate() {
        let name = field.name().clone();
        let array = batch.column(index);

        let profile = match field.data_type() {
            DataType::Float64 => array
                .as_any()
                .downcast_ref::<Float64Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::Float32 => array
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::Int64 => array
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::Int32 => array
                .as_any()
                .downcast_ref::<Int32Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::Int16 => array
                .as_any()
                .downcast_ref::<Int16Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::Int8 => array
                .as_any()
                .downcast_ref::<Int8Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::UInt64 => array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::UInt32 => array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::UInt16 => array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .map(|values| scan_profile_primitive!(values)),
            DataType::UInt8 => array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .map(|values| scan_profile_primitive!(values)),
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

    use arrow_array::{
        ArrayRef, Float64Array, Int16Array, Int64Array, RecordBatch, StringArray, UInt8Array,
    };

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
    fn fused_profile_updates_moments_and_quantiles_together() {
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
        assert_eq!(value.quantiles.count(), 3);
        assert!(value.quantiles.quantile(0.5).is_some());
        assert!(state.skipped_columns.contains("label"));
    }

    #[test]
    fn fused_profile_supports_compact_integer_arrays() {
        let batch = RecordBatch::try_from_iter(vec![
            (
                "small_signed",
                Arc::new(Int16Array::from(vec![Some(-2), Some(4), None, Some(8)])) as ArrayRef,
            ),
            (
                "small_unsigned",
                Arc::new(UInt8Array::from(vec![Some(1), Some(2), Some(3), Some(4)])) as ArrayRef,
            ),
        ])
        .expect("valid compact integer batch");

        let state = profile_record_batch(&batch, 1);
        let signed = state.profiles.get("small_signed").unwrap();
        let unsigned = state.profiles.get("small_unsigned").unwrap();

        assert_eq!(signed.moments.count, 3);
        assert_eq!(signed.moments.missing, 1);
        assert_eq!(signed.moments.minimum, Some(-2.0));
        assert_eq!(signed.moments.maximum, Some(8.0));
        assert_eq!(unsigned.moments.count, 4);
        assert_eq!(unsigned.moments.mean, 2.5);
    }

    #[test]
    fn fused_partition_merge_preserves_profile_semantics() {
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
        let median = value.quantiles.quantile(0.5).unwrap();
        assert!(median > 900.0 && median < 1_100.0);
    }
}
