//! Mergeable bounded-memory sketches for UTF-8 categorical values.

use std::collections::BTreeMap;

use crate::sketches::{mix64, HyperLogLog};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const DEFAULT_HEAVY_HITTER_CAPACITY: usize = 32;
const DEFAULT_LABEL_BYTES: usize = 256;

/// Incremental stable byte hasher used by Arrow buffer consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StableByteHasher {
    hash: u64,
    length: u64,
}

impl StableByteHasher {
    #[must_use]
    pub fn new() -> Self {
        Self {
            hash: FNV_OFFSET_BASIS,
            length: 0,
        }
    }

    pub fn update(&mut self, byte: u8) {
        self.hash ^= u64::from(byte);
        self.hash = self.hash.wrapping_mul(FNV_PRIME);
        self.length = self.length.wrapping_add(1);
    }

    #[must_use]
    pub fn finish(self) -> u64 {
        mix64(self.hash ^ mix64(self.length))
    }
}

impl Default for StableByteHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StringCandidate {
    label: Vec<u8>,
    count: u64,
}

/// Misra-Gries style bounded candidate tracker keyed by a stable full-value hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringHeavyHittersSketch {
    capacity: usize,
    max_label_bytes: usize,
    counters: BTreeMap<u64, StringCandidate>,
}

impl StringHeavyHittersSketch {
    #[must_use]
    pub fn new(capacity: usize, max_label_bytes: usize) -> Self {
        assert!(capacity > 0, "heavy-hitter capacity must be positive");
        assert!(max_label_bytes > 0, "max_label_bytes must be positive");
        Self {
            capacity,
            max_label_bytes,
            counters: BTreeMap::new(),
        }
    }

    pub fn observe_hashed(&mut self, hash: u64, label: &[u8], weight: u64) {
        if weight == 0 {
            return;
        }
        if let Some(candidate) = self.counters.get_mut(&hash) {
            candidate.count = candidate.count.saturating_add(weight);
            return;
        }
        if self.counters.len() < self.capacity {
            self.counters.insert(
                hash,
                StringCandidate {
                    label: label[..label.len().min(self.max_label_bytes)].to_vec(),
                    count: weight,
                },
            );
            return;
        }

        let minimum = self
            .counters
            .values()
            .map(|candidate| candidate.count)
            .min()
            .unwrap_or(0);
        let decrement = minimum.min(weight);
        self.counters.retain(|_, candidate| {
            candidate.count -= decrement;
            candidate.count > 0
        });

        let remaining = weight - decrement;
        if remaining > 0 {
            self.counters.insert(
                hash,
                StringCandidate {
                    label: label[..label.len().min(self.max_label_bytes)].to_vec(),
                    count: remaining,
                },
            );
        }
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        assert_eq!(self.capacity, other.capacity, "heavy-hitter capacity mismatch");
        assert_eq!(
            self.max_label_bytes, other.max_label_bytes,
            "heavy-hitter label limit mismatch"
        );
        for (hash, candidate) in other.counters {
            self.observe_hashed(hash, &candidate.label, candidate.count);
        }
        self
    }

    #[must_use]
    pub fn candidates(&self) -> Vec<(String, u64)> {
        let mut values: Vec<(String, u64)> = self
            .counters
            .values()
            .map(|candidate| {
                (
                    String::from_utf8_lossy(&candidate.label).into_owned(),
                    candidate.count,
                )
            })
            .collect();
        values.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        values
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn max_label_bytes(&self) -> usize {
        self.max_label_bytes
    }
}

impl Default for StringHeavyHittersSketch {
    fn default() -> Self {
        Self::new(DEFAULT_HEAVY_HITTER_CAPACITY, DEFAULT_LABEL_BYTES)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalSketchState {
    pub count: u64,
    pub missing: u64,
    pub cardinality: HyperLogLog,
    pub heavy_hitters: StringHeavyHittersSketch,
}

impl Default for CategoricalSketchState {
    fn default() -> Self {
        Self {
            count: 0,
            missing: 0,
            cardinality: HyperLogLog::default(),
            heavy_hitters: StringHeavyHittersSketch::default(),
        }
    }
}

impl CategoricalSketchState {
    pub fn observe_missing(&mut self) {
        self.missing = self.missing.saturating_add(1);
    }

    pub fn observe_hashed(&mut self, hash: u64, label: &[u8]) {
        self.count = self.count.saturating_add(1);
        self.cardinality.observe_hash(hash);
        self.heavy_hitters.observe_hashed(hash, label, 1);
    }

    pub fn observe_bytes(&mut self, value: Option<&[u8]>) {
        let Some(value) = value else {
            self.observe_missing();
            return;
        };
        let mut hasher = StableByteHasher::new();
        for &byte in value {
            hasher.update(byte);
        }
        self.observe_hashed(hasher.finish(), value);
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            count: self.count.saturating_add(other.count),
            missing: self.missing.saturating_add(other.missing),
            cardinality: self.cardinality.merge(other.cardinality),
            heavy_hitters: self.heavy_hitters.merge(other.heavy_hitters),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CategoricalSketchState, StableByteHasher, StringHeavyHittersSketch};

    #[test]
    fn stable_hash_is_incremental_and_deterministic() {
        let mut left = StableByteHasher::new();
        for byte in b"framevitals" {
            left.update(*byte);
        }
        let mut right = StableByteHasher::new();
        for chunk in [b"frame".as_slice(), b"vitals".as_slice()] {
            for byte in chunk {
                right.update(*byte);
            }
        }
        assert_eq!(left.finish(), right.finish());
    }

    #[test]
    fn categorical_state_tracks_cardinality_missing_and_heavy_hitters() {
        let mut state = CategoricalSketchState::default();
        for _ in 0..100 {
            state.observe_bytes(Some(b"pune"));
        }
        for _ in 0..40 {
            state.observe_bytes(Some(b"mumbai"));
        }
        state.observe_bytes(Some(b"nagpur"));
        state.observe_bytes(None);

        assert_eq!(state.count, 141);
        assert_eq!(state.missing, 1);
        assert!((state.cardinality.estimate() - 3.0).abs() < 1.0);
        assert_eq!(
            state.heavy_hitters.candidates().first().map(|item| item.0.as_str()),
            Some("pune")
        );
    }

    #[test]
    fn categorical_states_merge_without_raw_strings() {
        let mut left = CategoricalSketchState::default();
        let mut right = CategoricalSketchState::default();
        for _ in 0..60 {
            left.observe_bytes(Some(b"alpha"));
        }
        for _ in 0..80 {
            right.observe_bytes(Some(b"alpha"));
        }
        left.observe_bytes(Some(b"left"));
        right.observe_bytes(Some(b"right"));
        right.observe_missing();

        let merged = left.merge(right);
        assert_eq!(merged.count, 142);
        assert_eq!(merged.missing, 1);
        assert_eq!(
            merged.heavy_hitters.candidates().first().map(|item| item.0.as_str()),
            Some("alpha")
        );
    }

    #[test]
    fn labels_are_bounded_even_when_values_are_long() {
        let mut sketch = StringHeavyHittersSketch::new(4, 8);
        let value = b"abcdefghijklmnopqrstuvwxyz";
        let mut hasher = StableByteHasher::new();
        for byte in value {
            hasher.update(*byte);
        }
        sketch.observe_hashed(hasher.finish(), value, 5);
        assert_eq!(sketch.candidates()[0].0, "abcdefgh");
        assert_eq!(sketch.max_label_bytes(), 8);
    }
}
