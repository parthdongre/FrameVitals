//! Mergeable bounded-memory sketches used by the native FrameVitals profiler.
//!
//! These structures intentionally avoid retaining raw observations. Each sketch
//! can be built per Arrow batch/partition and merged later, which makes the same
//! semantics usable for local streaming, parallel scans, and distributed jobs.

use std::cmp::Ordering;
use std::collections::BTreeMap;

const DEFAULT_ZERO_THRESHOLD: f64 = 1.0e-12;

#[inline]
pub(crate) fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline]
fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperLogLog {
    precision: u8,
    registers: Vec<u8>,
}

impl HyperLogLog {
    pub fn new(precision: u8) -> Self {
        assert!((4..=16).contains(&precision), "HLL precision must be 4..=16");
        Self {
            precision,
            registers: vec![0; 1usize << precision],
        }
    }

    pub fn observe_f64(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.observe_hash(mix64(canonical_f64_bits(value)));
    }

    pub fn observe_hash(&mut self, hash: u64) {
        let index = (hash >> (64 - self.precision)) as usize;
        let remainder = hash << self.precision;
        let max_rank = 64 - u32::from(self.precision) + 1;
        let rank = (remainder.leading_zeros() + 1).min(max_rank) as u8;
        self.registers[index] = self.registers[index].max(rank);
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        assert_eq!(self.precision, other.precision, "HLL precision mismatch");
        for (left, right) in self.registers.iter_mut().zip(other.registers) {
            *left = (*left).max(right);
        }
        self
    }

    #[must_use]
    pub fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let alpha = match self.registers.len() {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let harmonic_sum: f64 = self
            .registers
            .iter()
            .map(|register| 2.0_f64.powi(-i32::from(*register)))
            .sum();
        let raw = alpha * m * m / harmonic_sum;
        let zero_registers = self.registers.iter().filter(|&&value| value == 0).count();

        if raw <= 2.5 * m && zero_registers > 0 {
            m * (m / zero_registers as f64).ln()
        } else {
            raw
        }
    }

    pub fn precision(&self) -> u8 {
        self.precision
    }

    pub fn bytes_used(&self) -> usize {
        self.registers.len()
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new(12)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogQuantileSketch {
    relative_accuracy: f64,
    log_gamma: f64,
    zero_threshold: f64,
    negative: BTreeMap<i32, u64>,
    positive: BTreeMap<i32, u64>,
    zero_count: u64,
    count: u64,
}

impl LogQuantileSketch {
    pub fn new(relative_accuracy: f64) -> Self {
        assert!(
            relative_accuracy > 0.0 && relative_accuracy < 1.0,
            "relative accuracy must be between 0 and 1"
        );
        let gamma = (1.0 + relative_accuracy) / (1.0 - relative_accuracy);
        Self {
            relative_accuracy,
            log_gamma: gamma.ln(),
            zero_threshold: DEFAULT_ZERO_THRESHOLD,
            negative: BTreeMap::new(),
            positive: BTreeMap::new(),
            zero_count: 0,
            count: 0,
        }
    }

    fn key(&self, magnitude: f64) -> i32 {
        (magnitude.ln() / self.log_gamma).floor() as i32
    }

    fn representative(&self, key: i32) -> f64 {
        ((key as f64 + 0.5) * self.log_gamma).exp()
    }

    pub fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        if value.abs() <= self.zero_threshold {
            self.zero_count += 1;
        } else if value < 0.0 {
            *self.negative.entry(self.key(-value)).or_insert(0) += 1;
        } else {
            *self.positive.entry(self.key(value)).or_insert(0) += 1;
        }
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        assert!(
            (self.relative_accuracy - other.relative_accuracy).abs() <= f64::EPSILON,
            "quantile sketch accuracy mismatch"
        );
        self.count += other.count;
        self.zero_count += other.zero_count;
        for (key, count) in other.negative {
            *self.negative.entry(key).or_insert(0) += count;
        }
        for (key, count) in other.positive {
            *self.positive.entry(key).or_insert(0) += count;
        }
        self
    }

    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.count == 0 || !(0.0..=1.0).contains(&q) {
            return None;
        }
        let target = (q * (self.count - 1) as f64).floor() as u64;
        let mut seen = 0_u64;

        for (key, count) in self.negative.iter().rev() {
            if target < seen + count {
                return Some(-self.representative(*key));
            }
            seen += count;
        }
        if target < seen + self.zero_count {
            return Some(0.0);
        }
        seen += self.zero_count;
        for (key, count) in &self.positive {
            if target < seen + count {
                return Some(self.representative(*key));
            }
            seen += count;
        }
        self.positive
            .last_key_value()
            .map(|(key, _)| self.representative(*key))
            .or_else(|| {
                self.negative
                    .first_key_value()
                    .map(|(key, _)| -self.representative(*key))
            })
            .or(Some(0.0))
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn bin_count(&self) -> usize {
        self.negative.len() + self.positive.len() + usize::from(self.zero_count > 0)
    }

    pub fn relative_accuracy(&self) -> f64 {
        self.relative_accuracy
    }
}

impl Default for LogQuantileSketch {
    fn default() -> Self {
        Self::new(0.01)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeavyHittersSketch {
    capacity: usize,
    counters: BTreeMap<u64, u64>,
}

impl HeavyHittersSketch {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "heavy-hitter capacity must be positive");
        Self {
            capacity,
            counters: BTreeMap::new(),
        }
    }

    pub fn observe_f64(&mut self, value: f64) {
        if value.is_finite() {
            self.observe_weighted(canonical_f64_bits(value), 1);
        }
    }

    pub fn observe_weighted(&mut self, key: u64, weight: u64) {
        if weight == 0 {
            return;
        }
        if let Some(counter) = self.counters.get_mut(&key) {
            *counter += weight;
            return;
        }
        if self.counters.len() < self.capacity {
            self.counters.insert(key, weight);
            return;
        }

        let minimum = self.counters.values().copied().min().unwrap_or(0);
        let decrement = minimum.min(weight);
        self.counters.retain(|_, count| {
            *count -= decrement;
            *count > 0
        });
        let remaining = weight - decrement;
        if remaining > 0 {
            self.counters.insert(key, remaining);
        }
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        assert_eq!(self.capacity, other.capacity, "heavy-hitter capacity mismatch");
        for (key, count) in other.counters {
            self.observe_weighted(key, count);
        }
        self
    }

    pub fn candidates(&self) -> Vec<(f64, u64)> {
        let mut values: Vec<(f64, u64)> = self
            .counters
            .iter()
            .map(|(bits, count)| (f64::from_bits(*bits), *count))
            .collect();
        values.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| left.0.total_cmp(&right.0))
        });
        values
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for HeavyHittersSketch {
    fn default() -> Self {
        Self::new(32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReservoirEntry {
    pub priority: u64,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PriorityReservoir {
    capacity: usize,
    entries: Vec<ReservoirEntry>,
}

impl PriorityReservoir {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "reservoir capacity must be positive");
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
        }
    }

    pub fn observe(&mut self, value: f64, stream_id: u64, sequence: u64) {
        if !value.is_finite() {
            return;
        }
        let value_hash = mix64(canonical_f64_bits(value));
        let priority = mix64(value_hash ^ mix64(stream_id) ^ mix64(sequence));
        let entry = ReservoirEntry { priority, value };

        if self.entries.len() < self.capacity {
            self.entries.push(entry);
            return;
        }
        if let Some((index, largest)) = self
            .entries
            .iter()
            .enumerate()
            .max_by_key(|(_, current)| current.priority)
        {
            if priority < largest.priority {
                self.entries[index] = entry;
            }
        }
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        assert_eq!(self.capacity, other.capacity, "reservoir capacity mismatch");
        self.entries.extend(other.entries);
        self.entries.sort_by_key(|entry| entry.priority);
        self.entries.truncate(self.capacity);
        self
    }

    pub fn values(&self) -> Vec<f64> {
        let mut entries = self.entries.clone();
        entries.sort_by(|left, right| {
            left.value
                .partial_cmp(&right.value)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.priority.cmp(&right.priority))
        });
        entries.into_iter().map(|entry| entry.value).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for PriorityReservoir {
    fn default() -> Self {
        Self::new(256)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NumericSketchState {
    pub cardinality: HyperLogLog,
    pub quantiles: LogQuantileSketch,
    pub heavy_hitters: HeavyHittersSketch,
    pub reservoir: PriorityReservoir,
}

impl NumericSketchState {
    pub fn observe(&mut self, value: f64, stream_id: u64, sequence: u64) {
        if !value.is_finite() {
            return;
        }
        self.cardinality.observe_f64(value);
        self.quantiles.observe(value);
        self.heavy_hitters.observe_f64(value);
        self.reservoir.observe(value, stream_id, sequence);
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            cardinality: self.cardinality.merge(other.cardinality),
            quantiles: self.quantiles.merge(other.quantiles),
            heavy_hitters: self.heavy_hitters.merge(other.heavy_hitters),
            reservoir: self.reservoir.merge(other.reservoir),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HeavyHittersSketch, HyperLogLog, LogQuantileSketch, NumericSketchState,
        PriorityReservoir,
    };

    #[test]
    fn hll_estimates_cardinality_and_merges_partitions() {
        let mut left = HyperLogLog::new(12);
        let mut right = HyperLogLog::new(12);
        for value in 0..10_000 {
            if value < 5_000 {
                left.observe_f64(value as f64);
            } else {
                right.observe_f64(value as f64);
            }
        }
        let estimate = left.merge(right).estimate();
        assert!((estimate - 10_000.0).abs() / 10_000.0 < 0.08);
    }

    #[test]
    fn logarithmic_quantiles_are_mergeable_and_bounded() {
        let mut left = LogQuantileSketch::new(0.01);
        let mut right = LogQuantileSketch::new(0.01);
        for value in 1..=10_000 {
            if value <= 5_000 {
                left.observe(value as f64);
            } else {
                right.observe(value as f64);
            }
        }
        let merged = left.merge(right);
        let median = merged.quantile(0.5).unwrap();
        assert!((median - 5_000.0).abs() / 5_000.0 < 0.03);
        assert!(merged.bin_count() < 1_000);
    }

    #[test]
    fn heavy_hitters_retain_dominant_candidates_after_merge() {
        let mut left = HeavyHittersSketch::new(8);
        let mut right = HeavyHittersSketch::new(8);
        for _ in 0..100 {
            left.observe_f64(7.0);
        }
        for value in 0..50 {
            left.observe_f64(value as f64);
            right.observe_f64((value + 50) as f64);
        }
        for _ in 0..80 {
            right.observe_f64(7.0);
        }
        let candidates = left.merge(right).candidates();
        assert_eq!(candidates.first().map(|item| item.0), Some(7.0));
    }

    #[test]
    fn priority_reservoir_is_bounded_and_partition_mergeable() {
        let mut left = PriorityReservoir::new(32);
        let mut right = PriorityReservoir::new(32);
        for value in 0..1_000_u64 {
            if value < 500 {
                left.observe(value as f64, 1, value);
            } else {
                right.observe(value as f64, 2, value - 500);
            }
        }
        let merged = left.merge(right);
        assert_eq!(merged.len(), 32);
        assert_eq!(merged.values().len(), 32);
    }

    #[test]
    fn combined_numeric_sketch_merges_without_raw_values() {
        let mut left = NumericSketchState::default();
        let mut right = NumericSketchState::default();
        for value in 0..2_000_u64 {
            if value < 1_000 {
                left.observe(value as f64, 11, value);
            } else {
                right.observe(value as f64, 12, value - 1_000);
            }
        }
        let merged = left.merge(right);
        assert!(merged.cardinality.estimate() > 1_800.0);
        assert!(merged.quantiles.quantile(0.5).unwrap() > 900.0);
        assert_eq!(merged.reservoir.len(), 256);
    }
}
