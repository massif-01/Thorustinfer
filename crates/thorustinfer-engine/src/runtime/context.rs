//! Plan service limits without altering the compiled model's buffer geometry.
use super::requests::GenerationInput;
use super::*;
use crate::execution::{ContextLimit, ContextOptions, ContextPlan};
impl ModelRuntime {
    pub(crate) fn context_resident(&self, features: usize) -> [usize; 3] {
        let mut bytes = self.execution.context_resident();
        if let Some(draft) = &self.external_draft {
            for (n, other) in bytes.iter_mut().zip(draft.execution.context_resident()) {
                *n += other;
            }
        }
        bytes[2] += features;
        bytes
    }
    pub(crate) fn configure_context(
        &mut self,
        options: ContextOptions,
        scheduler: &crate::scheduler::Options,
    ) -> Result<ContextPlan> {
        let package = self
            .external_draft
            .as_deref()
            .map_or(self.manifest.max_context, |d| {
                d.manifest.max_context.min(self.manifest.max_context)
            });
        let owners = options.context_concurrency.map_or(1, usize::from);
        if owners > scheduler.max_active {
            return Err("context-concurrency exceeds max-active-requests".into());
        }
        let requested = options.kv_bytes()?;
        let manual = match options.context {
            Some(ContextLimit::Tokens(n)) => Some(n.get()),
            _ => None,
        };
        if manual.is_some_and(|n| n > package) {
            return Err(format!("Requested context exceeds package limit {package}"));
        }
        let mut plan = ContextPlan {
            package,
            effective: manual.unwrap_or(package),
            owners,
            requested_bytes: requested,
            pool_bytes: 0,
            planned_growth_bytes: 0,
            other_worker_bytes: 0,
            ..Default::default()
        };
        if requested == 0
            && options.context != Some(ContextLimit::Auto)
            && options.context_concurrency.is_none()
        {
            self.effective_context = plan.effective;
            return Ok(plan);
        }
        let pool = if requested > 0 {
            let g = self.execution.kv_granularity()?;
            if requested < g {
                return Err(format!("KV budget {requested} is below page size {g}"));
            }
            requested / g * g
        } else {
            0
        };
        let available = self
            .execution
            .free_bytes()?
            .min(crate::scheduler::usable_host_bytes()?)
            .saturating_sub(scheduler.memory_reserve_bytes);
        if pool > available {
            return Err(format!(
                "KV pool requires {pool} bytes, {available} bytes available after system reserve"
            ));
        }
        // Workspace request sizing uses the same monotone high-water extent
        // and all text-program dependencies as runtime admission.
        let input = GenerationInput {
            input_tokens: vec![],
            images: vec![],
            audios: vec![],
            max_new_tokens: 1,
            sampling: Default::default(),
            prefix_hints: vec![],
        };
        let mut programs = vec!["decode", "prefill", "head"];
        if let Some(spec) = &self.manifest.vision {
            programs.extend(spec.plans.iter().map(|p| p.program.as_str()));
        }
        if self.manifest.audio.is_some() {
            programs.extend(["audio_stem", "audio_encoder"]);
        }
        let required = |n| -> Result<(u128, u128, u128)> {
            let (mut fixed, mut kv, mut shared) = (0u128, 0u128, 0u128);
            for model in std::iter::once(self as &Self).chain(self.external_draft.as_deref()) {
                let limits = model.request_limits(&input, n)?;
                fixed += model.execution.additional_state_bytes(&limits)? as u128
                    + model.execution.request_fixed_bytes(&limits)? as u128 * (owners - 1) as u128;
                kv += model.execution.reserved_kv_bytes(n)? as u128 * owners as u128;
                let text = ["decode", "prefill", "head"];
                let required = if model.owner_id == self.owner_id {
                    &programs[..]
                } else {
                    &text[..]
                };
                shared += model.pending_workspace(n, owners, required)? as u128;
            }
            if let Some(block) = &self.block_draft {
                let limits = self.request_limits(&input, n)?;
                fixed += self.execution.request_fixed_bytes(&limits)? as u128 * owners as u128;
                kv += self.execution.reserved_kv_bytes(n)? as u128 * owners as u128;
                shared += block.temporary_bytes(n)? as u128
                    + block.feature_bytes(n)? as u128 * (owners - 1) as u128;
            }
            Ok((kv, fixed, shared))
        };
        let fits = |n| -> Result<bool> {
            let (kv, fixed, shared) = required(n)?;
            let other = fixed + shared;
            Ok((pool == 0 || kv <= pool as u128)
                && other + if pool > 0 { pool as u128 } else { kv } <= available as u128)
        };
        let (mut low, mut high) = (0, package);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if fits(mid)? {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        plan.effective = manual.unwrap_or(low);
        if low == 0 || plan.effective > low {
            return Err(format!(
                "Context budget fits at most {low} tokens for {owners} owners; pool {pool}, available {available} bytes"
            ));
        }
        let (kv, fixed, shared) = required(plan.effective)?;
        plan.growth = [
            if pool == 0 { kv as usize } else { 0 },
            fixed as usize,
            shared as usize,
        ];
        plan.planned_growth_bytes = plan.growth.iter().sum();
        if pool > 0 {
            plan.used = self.execution.reserve_kv_pool(pool)?;
            if let Some(draft) = &mut self.external_draft {
                self.execution.share_kv_pool(&mut draft.execution)?;
            }
            plan.pool_bytes = pool;
        }
        plan.baseline = self.context_resident(0);
        plan.observe(plan.baseline);
        self.effective_context = plan.effective;
        Ok(plan)
    }
}
