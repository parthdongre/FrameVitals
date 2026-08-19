//! Native streaming primitives for FrameVitals.
//!
//! This crate intentionally starts with small, well-tested kernels whose
//! semantics match the Python reference implementation. Higher-level Arrow,
//! sketch, graph, and Python bindings can build on these primitives without
//! coupling the analysis engine to pandas.

#[cfg(feature = "arrow")]
pub mod arrow_scan;
pub mod categorical_sketches;
#[cfg(feature = "arrow")]
pub mod fused_profile;
pub mod sketches;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericState {
    pub count: u64,
    pub missing: u64,
    pub infinite: u64,
    pub mean: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
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
            m3: 0.0,
            m4: 0.0,
            minimum: None,
            maximum: None,
        }
    }
}

impl NumericState {
    /// Observe an optional numeric value using stable online central moments
    /// through fourth order.
    ///
    /// `None` and NaN are counted as missing. Positive/negative infinity are
    /// counted separately and excluded from finite moments, matching the
    /// Python `NumericColumnState` semantics.
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

        let previous_count = self.count;
        self.count += 1;

        let count = self.count as f64;
        let previous_count_f = previous_count as f64;
        let delta = value - self.mean;
        let delta_n = delta / count;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * previous_count_f;

        // M4/M3 depend on the previous lower-order moments, so update from the
        // highest order down before advancing M2 and the mean.
        self.m4 += term1 * delta_n2 * (count * count - 3.0 * count + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (count - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        self.mean += delta_n;

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

    /// Merge independently computed partition states through fourth central
    /// moment. No raw observations are required.
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
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;
        let total2 = total_f * total_f;
        let total3 = total2 * total_f;

        let mean = self.mean + delta * right_f / total_f;
        let m2 = self.m2 + other.m2 + delta2 * left_f * right_f / total_f;
        let m3 = self.m3
            + other.m3
            + delta3 * left_f * right_f * (left_f - right_f) / total2
            + 3.0 * delta * (left_f * other.m2 - right_f * self.m2) / total_f;
        let m4 = self.m4
            + other.m4
            + delta4 * left_f * right_f * (left_f * left_f - left_f * right_f + right_f * right_f)
                / total3
            + 6.0 * delta2 * (left_f * left_f * other.m2 + right_f * right_f * self.m2) / total2
            + 4.0 * delta * (left_f * other.m3 - right_f * self.m3) / total_f;

        Self {
            count: total,
            missing,
            infinite,
            mean,
            m2,
            m3,
            m4,
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

    /// Bias-corrected Fisher-Pearson sample skewness, matching pandas/SciPy
    /// `bias=False` semantics used by FrameVitals' deep-statistics report.
    #[must_use]
    pub fn skewness(&self) -> Option<f64> {
        if self.count < 3 || self.m2 <= 0.0 {
            return None;
        }
        let n = self.count as f64;
        let population_skew = n.sqrt() * self.m3 / self.m2.powf(1.5);
        Some((n * (n - 1.0)).sqrt() / (n - 2.0) * population_skew)
    }

    /// Bias-corrected Fisher excess kurtosis, matching pandas/SciPy
    /// `fisher=True, bias=False` semantics.
    #[must_use]
    pub fn excess_kurtosis(&self) -> Option<f64> {
        if self.count < 4 || self.m2 <= 0.0 {
            return None;
        }
        let n = self.count as f64;
        let population_excess = n * self.m4 / (self.m2 * self.m2) - 3.0;
        Some((n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * population_excess + 6.0))
    }
}

#[cfg(test)]
mod tests {
    use super::NumericState;

    fn assert_close(left: f64, right: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!((left - right).abs() <= 1e-10 * scale, "{left} != {right}");
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
        assert!(state.skewness().is_some());
        assert!(state.excess_kurtosis().is_none());
    }

    #[test]
    fn partition_merge_matches_single_pass_through_fourth_moment() {
        let values: Vec<Option<f64>> = (0..10_000)
            .map(|value| {
                if value % 97 == 0 {
                    None
                } else {
                    let x = value as f64 * 0.025 - 100.0;
                    Some(x * x.signum() + (value % 11) as f64)
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
        assert_close(merged.m3, full.m3);
        assert_close(merged.m4, full.m4);
        assert_close(merged.variance().unwrap(), full.variance().unwrap());
        assert_close(merged.skewness().unwrap(), full.skewness().unwrap());
        assert_close(
            merged.excess_kurtosis().unwrap(),
            full.excess_kurtosis().unwrap(),
        );
    }

    #[test]
    fn known_shape_statistics_match_reference_values() {
        let state = NumericState::from_values(&[
            Some(1.0),
            Some(2.0),
            Some(2.0),
            Some(3.0),
            Some(9.0),
            Some(12.0),
        ]);

        assert_close(state.skewness().unwrap(), 1.069_287_452_144_894_3);
        assert_close(state.excess_kurtosis().unwrap(), -0.796_319_305_259_674_9);
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
    fn constant_values_have_zero_variance_and_undefined_shape() {
        let state = NumericState::from_values(&[Some(3.0), Some(3.0), Some(3.0), Some(3.0)]);
        assert_eq!(state.variance(), Some(0.0));
        assert_eq!(state.standard_deviation(), Some(0.0));
        assert_eq!(state.skewness(), None);
        assert_eq!(state.excess_kurtosis(), None);
    }
}
