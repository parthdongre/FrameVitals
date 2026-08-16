//! Native streaming primitives for FrameVitals.
//!
//! This crate intentionally starts with small, well-tested kernels whose
//! semantics match the Python reference implementation. Higher-level Arrow,
//! sketch, graph, and Python bindings can build on these primitives without
//! coupling the analysis engine to pandas.

#[cfg(feature = "arrow")]
pub mod arrow_scan;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericState {
    pub count: u64,
    pub missing: u64,
    pub infinite: u64,
    pub mean: f64,
    pub m2: f64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

impl Default for NumericState {
    fn default() -> Self {
        Self {
            count: 0,
            missing: 0,
            infinite: 0,
            mean: 0.0,
            m2: 0.0,
            minimum: None,
            maximum: None,
        }
    }
}

impl NumericState {
    /// Observe an optional numeric value using Welford's online algorithm.
    ///
    /// `None` and NaN are counted as missing. Positive/negative infinity are
    /// counted separately and excluded from finite moments, matching the
    /// current Python `NumericColumnState` semantics.
    pub fn observe(&mut self, value: Option<f64>) {
        let Some(value) = value else {
            self.missing += 1;
            return;
        };

        if value.is_nan() {
            self.missing += 1;
            return;
        }
        if !value.is_finite() {
            self.infinite += 1;
            return;
        }

        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        self.minimum = Some(match self.minimum {
            Some(current) => current.min(value),
            None => value,
        });
        self.maximum = Some(match self.maximum {
            Some(current) => current.max(value),
            None => value,
        });
    }

    /// Scan a slice into one compact state without retaining observations.
    pub fn from_values(values: &[Option<f64>]) -> Self {
        let mut state = Self::default();
        for &value in values {
            state.observe(value);
        }
        state
    }

    /// Merge independently computed partition states using the parallel
    /// variance formula. No raw observations are required.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        let missing = self.missing + other.missing;
        let infinite = self.infinite + other.infinite;

        if self.count == 0 {
            return Self {
                missing,
                infinite,
                ..other
            };
        }
        if other.count == 0 {
            return Self {
                missing,
                infinite,
                ..self
            };
        }

        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let total_f = total as f64;
        let left_f = self.count as f64;
        let right_f = other.count as f64;

        let mean = self.mean + delta * right_f / total_f;
        let m2 = self.m2
            + other.m2
            + delta * delta * left_f * right_f / total_f;

        Self {
            count: total,
            missing,
            infinite,
            mean,
            m2,
            minimum: match (self.minimum, other.minimum) {
                (Some(left), Some(right)) => Some(left.min(right)),
                (left @ Some(_), None) => left,
                (None, right @ Some(_)) => right,
                (None, None) => None,
            },
            maximum: match (self.maximum, other.maximum) {
                (Some(left), Some(right)) => Some(left.max(right)),
                (left @ Some(_), None) => left,
                (None, right @ Some(_)) => right,
                (None, None) => None,
            },
        }
    }

    #[must_use]
    pub fn variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m2 / (self.count - 1) as f64)
        }
    }

    #[must_use]
    pub fn standard_deviation(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }
}

#[cfg(test)]
mod tests {
    use super::NumericState;

    fn assert_close(left: f64, right: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!((left - right).abs() <= 1e-12 * scale, "{left} != {right}");
    }

    #[test]
    fn scan_tracks_missing_infinite_and_moments() {
        let values = [
            Some(1.0),
            Some(2.0),
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(4.0),
        ];
        let state = NumericState::from_values(&values);

        assert_eq!(state.count, 3);
        assert_eq!(state.missing, 2);
        assert_eq!(state.infinite, 1);
        assert_eq!(state.minimum, Some(1.0));
        assert_eq!(state.maximum, Some(4.0));
        assert_close(state.mean, 7.0 / 3.0);
        assert_close(state.variance().unwrap(), 7.0 / 3.0);
    }

    #[test]
    fn partition_merge_matches_single_pass() {
        let values: Vec<Option<f64>> = (0..10_000)
            .map(|value| {
                if value % 97 == 0 {
                    None
                } else {
                    Some(value as f64 * 0.25 - 100.0)
                }
            })
            .collect();

        let full = NumericState::from_values(&values);
        let left = NumericState::from_values(&values[..3_333]);
        let middle = NumericState::from_values(&values[3_333..7_777]);
        let right = NumericState::from_values(&values[7_777..]);
        let merged = left.merge(middle).merge(right);

        assert_eq!(merged.count, full.count);
        assert_eq!(merged.missing, full.missing);
        assert_eq!(merged.infinite, full.infinite);
        assert_eq!(merged.minimum, full.minimum);
        assert_eq!(merged.maximum, full.maximum);
        assert_close(merged.mean, full.mean);
        assert_close(merged.m2, full.m2);
        assert_close(merged.variance().unwrap(), full.variance().unwrap());
    }

    #[test]
    fn merge_preserves_nonfinite_counts_for_empty_partition() {
        let finite = NumericState::from_values(&[Some(1.0), Some(2.0)]);
        let nonfinite = NumericState::from_values(&[None, Some(f64::NEG_INFINITY)]);
        let merged = finite.merge(nonfinite);

        assert_eq!(merged.count, 2);
        assert_eq!(merged.missing, 1);
        assert_eq!(merged.infinite, 1);
        assert_eq!(merged.minimum, Some(1.0));
        assert_eq!(merged.maximum, Some(2.0));
    }

    #[test]
    fn constant_values_have_zero_sample_variance() {
        let state = NumericState::from_values(&[Some(3.0), Some(3.0), Some(3.0)]);
        assert_eq!(state.variance(), Some(0.0));
        assert_eq!(state.standard_deviation(), Some(0.0));
    }
}
