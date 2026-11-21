//! Prefill profiles indexed by exact chunk size and bucketed context.
use super::{Estimate, Timing};
use std::collections::{BTreeMap, BTreeSet};

fn bucket(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

#[derive(Default)]
struct Curves {
    cells: BTreeMap<(usize, usize), (Timing, u64)>,
    clock: u64,
}
impl Curves {
    fn observe(&mut self, chunk: usize, context: usize, seconds: f64) {
        let key = (bucket(context), chunk);
        self.clock += 1;
        if self.cells.len() == 256 && !self.cells.contains_key(&key) {
            let old = *self
                .cells
                .iter()
                .min_by_key(|(_, (_, last))| last)
                .unwrap()
                .0;
            self.cells.remove(&old);
        }
        let (timing, last) = self.cells.entry(key).or_default();
        *last = self.clock;
        timing.observe_recurring(seconds);
    }
}
/// Greedy registered prefill profiles and checkpoint policy.
pub(crate) struct Profiles {
    shapes: BTreeMap<usize, Estimate>,
    weight_pass: Option<f64>,
    timings: Curves,
    stages: BTreeMap<(usize, usize, bool), f64>,
    restore_s_per_byte: f64,
    dynamic: Option<DynamicPrefill>,
}

struct DynamicPrefill {
    capacity: usize,
    seconds_per_row: Option<f64>,
}

impl Profiles {
    pub fn new(shapes: impl IntoIterator<Item = usize>) -> Self {
        Self {
            shapes: shapes
                .into_iter()
                .chain([1])
                .map(|n| (n, Estimate::default()))
                .collect(),
            weight_pass: None,
            timings: Default::default(),
            stages: BTreeMap::new(),
            restore_s_per_byte: 1. / 20e9,
            dynamic: None,
        }
    }

    pub fn with_dynamic_capacity(mut self, capacity: usize) -> Self {
        self.dynamic = Some(DynamicPrefill {
            capacity,
            seconds_per_row: None,
        });
        self
    }

    pub fn observe(&mut self, chunk: usize, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            if (1..=8).contains(&chunk) {
                self.weight_pass = Some(
                    self.weight_pass
                        .map_or(seconds, |old| old * 0.8 + seconds * 0.2),
                );
            }
            self.shapes.entry(chunk).or_default().observe(seconds);
        }
    }

    pub fn observe_at(&mut self, chunk: usize, context: usize, seconds: f64) {
        if !seconds.is_finite() || seconds <= 0. {
            return;
        }
        self.observe(chunk, seconds);
        let weight_pass_s = self.weight_pass_s();
        if let Some(dynamic) = &mut self.dynamic
            && (128..=dynamic.capacity).contains(&chunk)
            && seconds > weight_pass_s
        {
            // Fit the variable work after one weight pass. Small tails are
            // dominated by launch cost and give unstable throughput samples.
            let sample = (seconds - weight_pass_s) / chunk as f64;
            let weight = 0.2 * (chunk as f64 / 1024.).min(1.);
            dynamic.seconds_per_row = Some(
                dynamic
                    .seconds_per_row
                    .map_or(sample, |old| old * (1. - weight) + sample * weight),
            );
        }
        self.timings.observe(chunk, context, seconds);
    }

    pub fn chunk_upper(&self, chunk: usize, context: usize) -> f64 {
        self.timings
            .cells
            .get(&(bucket(context), chunk))
            .map(|(t, _)| t.upper())
            .unwrap_or_else(|| self.chunk_cost(chunk) * 1.2)
    }

    pub fn observe_stage(&mut self, chunk: usize, context: usize, warm: bool, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            // Charge the slowest observed layer in this geometry. Draft warm
            // runs in a separate stage with a different amount of work.
            let peak = self
                .stages
                .entry((bucket(chunk), bucket(context), warm))
                .or_default();
            *peak = peak.max(seconds);
        }
    }

    pub fn stage_upper(&self, chunk: usize, context: usize, count: usize, warm: bool) -> f64 {
        self.stages
            .get(&(bucket(chunk), bucket(context), warm))
            .copied()
            .unwrap_or_else(|| (self.chunk_upper(chunk, context) / count.max(1) as f64).max(0.001))
    }
    pub fn bounded_chunk_at(&self, remaining: usize, context: usize, budget_ms: f64) -> usize {
        let fitting = self
            .shapes
            .keys()
            .copied()
            .filter(|&n| {
                n <= remaining
                    && self.chunk_upper(n, context.saturating_add(n)) * 1000. <= budget_ms
            })
            .max()
            .unwrap_or(1);
        // A cold measurement or indivisible tail/head cost can make every
        // compiled tile exceed the soft budget. Repeated one-token weight
        // passes are then much worse than one minimum useful tile. Explore
        // that tile again so later measurements can correct the estimate.
        let minimum = self
            .shapes
            .keys()
            .rev()
            .copied()
            .find(|&n| n > 1 && n <= 16 && n <= remaining);
        minimum
            .filter(|&n| {
                fitting == 1
                    && self.chunk_upper(n, context.saturating_add(n)) / (n as f64)
                        < self.chunk_upper(1, context.saturating_add(1)) * 0.5
            })
            .unwrap_or(fitting)
    }
    pub fn observe_restore(&mut self, bytes: usize, seconds: f64) {
        if bytes > 0 && seconds.is_finite() && seconds > 0. {
            self.restore_s_per_byte = self.restore_s_per_byte * 0.8 + seconds / bytes as f64 * 0.2;
        }
    }
    pub fn restore_cost(&self, bytes: usize) -> f64 {
        bytes as f64 * self.restore_s_per_byte
    }
    fn weight_pass_s(&self) -> f64 {
        self.weight_pass.unwrap_or(0.1)
    }
    pub fn chunk_cost(&self, tokens: usize) -> f64 {
        self.shapes
            .get(&tokens)
            .filter(|estimate| estimate.samples > 0)
            .map(|estimate| estimate.mean)
            .unwrap_or_else(|| {
                if self.dynamic.is_some() {
                    self.span(tokens)
                } else {
                    self.weight_pass_s() * (tokens as f64 / 64.).max(1.)
                }
            })
    }
    /// Largest real profile within the estimated budget; one token guarantees
    /// progress even when no profile fits. This is a soft scheduling target.
    #[cfg(test)]
    pub fn bounded_chunk(&self, remaining: usize, budget_ms: f64) -> usize {
        self.shapes
            .keys()
            .copied()
            .filter(|&n| n <= remaining && self.chunk_cost(n) * 1000. <= budget_ms)
            .max()
            .unwrap_or(1)
    }
    fn span(&self, mut tokens: usize) -> f64 {
        if let Some(dynamic) = &self.dynamic {
            // The native plan executes one exact-sized forward pass per
            // chunk. Measurements of other lengths are not chunk boundaries.
            let passes = tokens.div_ceil(dynamic.capacity);
            let weight_pass_s = self.weight_pass_s();
            let per_row = dynamic.seconds_per_row.unwrap_or(weight_pass_s / 64.);
            return passes as f64 * weight_pass_s + tokens as f64 * per_row;
        }
        let mut total = 0.;
        for &chunk in self.shapes.keys().rev() {
            let count = tokens / chunk;
            tokens %= chunk;
            total += count as f64 * self.chunk_cost(chunk);
        }
        total
    }
    pub fn remaining(&self, mut offset: usize, end: usize) -> f64 {
        let mut total = 0.;
        while offset < end {
            let boundary = ((offset / 8192 + 1) * 8192).min(end);
            total += self.span(boundary - offset);
            offset = boundary;
        }
        total
    }
    pub fn score(&self, prefix: usize, end: usize, bytes: usize) -> Option<f64> {
        let cost = self.remaining(prefix, end) + self.restore_cost(bytes);
        // A complete head-bearing checkpoint always avoids model execution.
        (prefix == end || cost < self.remaining(0, end) * 0.95).then_some(cost)
    }
    #[cfg(test)]
    pub fn admit(
        &self,
        prefix: usize,
        end: usize,
        bytes: usize,
        start: usize,
        checkpoints: &BTreeSet<usize>,
    ) -> bool {
        self.admit_with_reuse(prefix, end, bytes, start, checkpoints, false)
    }
    pub fn admit_with_reuse(
        &self,
        prefix: usize,
        end: usize,
        bytes: usize,
        start: usize,
        checkpoints: &BTreeSet<usize>,
        reused: bool,
    ) -> bool {
        if prefix <= start || self.score(prefix, end, bytes).is_none() {
            return false;
        }
        if prefix >= 64 && (2..=8).contains(&end.saturating_sub(prefix)) {
            // The boundary before <think> can be reused after template rendering.
            return true;
        }
        let mut trial = checkpoints.clone();
        trial.insert(prefix);
        let (mut previous, mut split) = (start, 0.);
        for &point in trial.range((
            std::ops::Bound::Excluded(start),
            std::ops::Bound::Excluded(end),
        )) {
            split += self.remaining(previous, point);
            previous = point;
        }
        split += self.remaining(previous, end);
        let baseline = self.remaining(start, end);
        // Do not create a speculative boundary by destroying an efficient
        // large chunk. Periodic checkpoints are already chunk boundaries.
        // Account from the restored endpoint and include already admitted
        // boundaries, rather than evaluating each split against a cold run.
        let extra = (split - baseline).max(0.);
        extra <= (baseline * 0.1).max(0.05)
            || (reused
                && baseline - self.remaining(prefix, end) > extra + self.restore_cost(bytes) * 2.)
    }
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn stage_buckets_reuse_observations_across_layers_and_dynamic_lengths() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(600, 4096, 0.2);
        for chunk in 513..=640 {
            for layer in 0..64 {
                let seconds = if chunk == 513 && layer == 0 {
                    0.008
                } else {
                    0.002
                };
                costs.observe_stage(chunk, 4096, false, seconds);
            }
        }
        costs.observe_stage(600, 4096, true, 0.07);
        assert_eq!(costs.stages.len(), 2);
        assert_eq!(costs.timings.cells.len(), 1);
        assert_eq!(costs.stage_upper(900, 4096, 64, false), 0.008);
        assert_eq!(costs.stage_upper(900, 4096, 64, true), 0.07);
    }

    #[test]
    fn cold_stage_uses_the_fitted_dynamic_cost_and_a_millisecond_floor() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(4, 4, 0.005);
        costs.observe_at(2048, 2048, 0.21);
        let fitted = 0.005 + (0.21 - 0.005) * 1024. / 2048.;
        assert!((costs.chunk_cost(1024) - fitted).abs() < 1e-12);
        assert!((costs.stage_upper(1024, 4096, 64, false) - fitted * 1.2 / 64.).abs() < 1e-12);
        assert_eq!(costs.stage_upper(4, 4096, 64, false), 0.001);
    }

    #[test]
    fn near_end_boundaries_are_admitted_before_a_short_pass_is_measured() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(2048, 2048, 0.4);
        let checkpoints = BTreeSet::new();
        for tail in 2..=8 {
            assert!(costs.admit(2048 - tail, 2048, 100 << 20, 0, &checkpoints));
        }
        assert!(!costs.admit(2039, 2048, 100 << 20, 0, &checkpoints));
        costs.observe_restore(100 << 20, 100.);
        assert!(!costs.admit(2044, 2048, 100 << 20, 0, &checkpoints));
    }
    #[test]
    fn one_token_tail_uses_the_split_cost_and_confirmed_reuse() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(2048, 2048, 0.8);
        let checkpoints = BTreeSet::new();
        let bytes = 100 << 20;
        for suffix in [256, 312, 768] {
            let start = 2048 - suffix;
            assert!(costs.score(2047, 2048, bytes).is_some());
            assert!(!costs.admit(2047, 2048, bytes, start, &checkpoints));
            for tail in 2..=8 {
                assert!(costs.admit(2048 - tail, 2048, bytes, start, &checkpoints));
            }
        }
        assert!(costs.admit_with_reuse(2047, 2048, bytes, 2048 - 768, &checkpoints, true));
        costs.observe_at(1, 2048, 0.01);
        assert!(costs.admit(2047, 2048, bytes, 2048 - 312, &checkpoints));
    }
    #[test]
    fn first_short_pass_replaces_the_cold_estimate() {
        let mut costs = Profiles::new([2048]).with_dynamic_capacity(4096);
        costs.observe_at(4, 2048, f64::NAN);
        assert_eq!(costs.weight_pass_s(), 0.1);
        costs.observe_at(4, 2048, 0.005);
        assert_eq!(costs.weight_pass_s(), 0.005);
        costs.observe_at(8, 2048, 0.01);
        assert!((costs.weight_pass_s() - 0.006).abs() < 1e-12);
    }
    #[test]
    fn dynamic_restore_prices_the_real_tail_instead_of_observed_partitions() {
        let mut costs = Profiles::new([4096]).with_dynamic_capacity(4096);
        costs.observe_at(1203, 4508, 1.3);
        costs.observe_at(21, 4508, 0.4);
        let ancestor = costs.score(3303, 4508, 150 << 20).unwrap();
        let parent = costs.score(3970, 4508, 180 << 20).unwrap();
        assert!(parent < ancestor);
        for tail in [1, 537, 538, 1203, 4096] {
            assert!(costs.remaining(0, tail) < costs.remaining(0, tail + 1));
        }
        let two_passes = costs.remaining(0, 4097);
        let split = costs.remaining(0, 4096) + costs.remaining(0, 1);
        assert!((two_passes - split).abs() < 1e-12);
    }
    #[test]
    fn confirmed_reuse_pays_for_a_checkpoint_but_not_an_unprofitable_split() {
        let mut costs = Profiles::new((2..=16).chain([128, 512, 2048, 4096]));
        for n in 1..16 {
            costs.observe(n, 0.1);
        }
        for (n, seconds) in [
            (16, 0.075),
            (128, 0.21),
            (512, 0.6),
            (2048, 1.7),
            (4096, 3.),
        ] {
            costs.observe(n, seconds);
        }
        let checkpoints = BTreeSet::from([8192]);
        assert!(!costs.admit(3584, 8223, 160 << 20, 0, &checkpoints));
        assert!(costs.admit_with_reuse(3584, 8223, 160 << 20, 0, &checkpoints, true));
        assert!(!costs.admit_with_reuse(2403, 8223, 160 << 20, 0, &checkpoints, true));
        costs.observe_restore(160 << 20, 100.);
        assert!(!costs.admit_with_reuse(3584, 8223, 160 << 20, 0, &checkpoints, true));
    }
    #[test]
    fn rejects_small_prefixes_that_destroy_large_chunks() {
        let costs = Profiles::new([2, 4, 8, 512, 2048]);
        let checkpoints = BTreeSet::new();
        for end in [512, 2048, 8192, 32768] {
            for prefix in [1, 26] {
                assert!(costs.score(prefix, end, 160 << 20).is_none());
                assert!(!costs.admit(prefix, end, 160 << 20, 0, &checkpoints));
            }
            assert!(costs.score(end, end, 2 << 30).is_some());
        }
        assert!(costs.admit(512, 520, 180 << 20, 0, &checkpoints));
        assert!(costs.score(527, 536, 180 << 20).is_some());
        assert!(costs.admit(8192, 32768, 450 << 20, 0, &checkpoints));
        assert!(costs.admit(26, 46, 160 << 20, 0, &checkpoints));
        assert!(costs.admit(62, 71, 160 << 20, 0, &checkpoints));
        assert!(!costs.admit(62, 71, 160 << 20, 47, &checkpoints));
        assert!(costs.admit(44, 200, 160 << 20, 0, &checkpoints));
        assert!(costs.admit(82, 200, 160 << 20, 0, &checkpoints));
        assert!(!costs.admit(82, 200, 160 << 20, 0, &BTreeSet::from([44])));
    }
    #[test]
    fn measurements_change_selection_and_account_for_restore() {
        let mut costs = Profiles::new([8, 512]);
        assert!(costs.score(8, 512, 160 << 20).is_none());
        costs.observe(512, 5.);
        costs.observe(8, 0.02);
        assert!(costs.score(8, 512, 160 << 20).is_some());
        assert!(costs.admit(8, 512, 160 << 20, 0, &BTreeSet::new()));
        costs.observe_restore(160 << 20, 100.);
        assert!(costs.score(8, 512, 160 << 20).is_none());
    }
    #[test]
    fn mixed_chunks_follow_measured_cost_and_never_pad_a_tail() {
        let mut costs = Profiles::new([16, 128, 512, 2048, 4096]);
        assert_eq!(costs.bounded_chunk(8192, 200.), 128);
        costs.observe(16, 0.04);
        costs.observe(128, 0.25);
        costs.observe(512, 0.6);
        assert_eq!(costs.bounded_chunk(8192, 200.), 16);
        assert_eq!(costs.bounded_chunk(8192, 700.), 512);
        assert_eq!(costs.bounded_chunk(15, 200.), 1);
        assert_eq!(costs.bounded_chunk(8192, 1.), 1);
        assert_eq!(costs.bounded_chunk(8192, f64::INFINITY), 4096);
    }
    #[test]
    fn context_buckets_do_not_reuse_a_short_attention_timing_for_a_long_prompt() {
        let mut costs = Profiles::new([128, 512]);
        costs.observe_at(128, 1024, 0.1);
        costs.observe_at(128, 8192, 0.5);
        assert_eq!(costs.bounded_chunk_at(8192, 896, 200.), 128);
        assert_eq!(costs.bounded_chunk_at(8192, 8064, 200.), 1);
        costs.observe_at(512, 8192, f64::NAN);
        assert!(costs.chunk_upper(512, 8192) > 0.);
    }
    #[test]
    fn soft_budget_does_not_lock_prefill_into_one_token_weight_passes() {
        let mut costs = Profiles::new([16, 128]);
        costs.observe_at(16, 16, 0.4);
        costs.observe_at(128, 128, 0.8);
        assert_eq!(costs.bounded_chunk_at(512, 0, 200.), 16);
        // No padding across a tail, and no large tile forced beyond a budget.
        assert_eq!(costs.bounded_chunk_at(15, 0, 1.), 1);
        assert_eq!(costs.bounded_chunk_at(512, 0, 1000.), 128);
        costs.observe_at(16, 16, 5.);
        assert_eq!(costs.bounded_chunk_at(512, 0, 1.), 1);
    }
    #[test]
    fn exact_tail_profiles_do_not_turn_soft_budget_fallback_into_tiny_passes() {
        let mut costs = Profiles::new((2..=16).chain([128]));
        for n in 2..=16 {
            costs.observe_at(n, n, 0.4);
        }
        assert_eq!(costs.bounded_chunk_at(512, 0, 1.), 16);
        assert_eq!(costs.bounded_chunk_at(15, 0, 1.), 15);
        assert_eq!(costs.bounded_chunk_at(1, 0, 1.), 1);
    }
}
