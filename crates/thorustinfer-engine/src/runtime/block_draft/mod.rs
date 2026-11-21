//! Parallel block proposals. Only target-verified rows reach request state.
mod store;
use super::*;
use crate::execution::SpecMethod;
use half::bf16;
use rayon::prelude::*;
use serde::Deserialize;
use std::path::Path;
use store::Store;

#[derive(Deserialize)]
struct Transformer {
    hidden_size: usize,
    intermediate_size: usize,
    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    layer_types: Vec<String>,
    rope_parameters: Rope,
    sliding_window: Option<usize>,
    is_causal: Option<bool>,
}
#[derive(Deserialize)]
struct Rope {
    rope_theta: f32,
}
#[derive(Deserialize, Default)]
struct Layout {
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
}
#[derive(Deserialize)]
struct Header {
    architectures: Vec<String>,
    dflash_config: Option<Layout>,
    aux_hidden_state_layer_ids: Option<Vec<usize>>,
    block_size: Option<usize>,
    mask_token_id: Option<u32>,
    sample_from_anchor: Option<bool>,
    sliding_window_non_causal: Option<bool>,
    conv_kernel_size: Option<usize>,
    conv_group_size: Option<usize>,
    selector_rank: Option<usize>,
    selector_top_k: Option<usize>,
    markov_rank: Option<usize>,
    confidence_head_with_markov: Option<bool>,
    markov_head_type: Option<String>,
}
pub(crate) struct BlockDraft {
    pub(super) layers: Vec<usize>,
    pub(super) block: usize,
    pub(super) mask: u32,
    pub(super) from_anchor: bool,
    method: SpecMethod,
    net: Transformer,
    store: Store,
    sliding_noncausal: bool,
    group: usize,
    taps: usize,
    rank: usize,
    top: usize,
}
fn rounded(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn scalar(x: u16) -> f32 {
    bf16::from_bits(x).to_f32()
}
fn rms(values: &mut [f32], weight: &[u16], eps: f32) {
    for row in values.chunks_exact_mut(weight.len()) {
        let scale = (eps + row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32)
            .sqrt()
            .recip();
        for (x, &w) in row.iter_mut().zip(weight) {
            *x = rounded(rounded(*x * scale) * scalar(w));
        }
    }
}
impl BlockDraft {
    pub(crate) fn load(
        root: &Path,
        method: SpecMethod,
        hidden: usize,
        vocab: usize,
        depth: usize,
        checkpoint: &(String, String),
        verify: bool,
    ) -> Result<Self> {
        let raw = std::fs::read(root.join("config.json")).map_err(|e| e.to_string())?;
        let config: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        let h: Header = serde_json::from_value(config.clone()).map_err(|e| e.to_string())?;
        let net: Transformer = serde_json::from_value(
            config
                .get("transformer_layer_config")
                .unwrap_or(&config)
                .clone(),
        )
        .map_err(|e| e.to_string())?;
        let arch = match method {
            SpecMethod::Dflash => "DFlashDraftModel",
            SpecMethod::Dflash2 => "DFlash2DraftModel",
            SpecMethod::Dspark => "DSparkDraftModel",
            _ => return Err("Expected a block draft method".into()),
        };
        let (block, mask, layers) = if let Some(l) = h.dflash_config {
            (l.block_size, l.mask_token_id, l.target_layer_ids)
        } else {
            let layers = h
                .aux_hidden_state_layer_ids
                .ok_or("Missing draft hidden-state indices")?
                .into_iter()
                .map(|i| {
                    i.checked_sub(1)
                        .ok_or("Embedding-layer taps are unsupported")
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            (
                h.block_size.ok_or("Missing draft block size")?,
                h.mask_token_id.ok_or("Missing draft mask ID")?,
                layers,
            )
        };
        if h.architectures != [arch]
            || net.hidden_size != hidden
            || net.vocab_size != vocab
            || block < 2
            || mask as usize >= vocab
            || layers.is_empty()
            || layers.iter().any(|&i| i >= depth)
            || net.head_dim == 0
            || !net.head_dim.is_multiple_of(2)
            || net.num_key_value_heads == 0
            || !net
                .num_attention_heads
                .is_multiple_of(net.num_key_value_heads)
            || net.num_hidden_layers != net.layer_types.len()
            || !net.rope_parameters.rope_theta.is_finite()
            || net.rope_parameters.rope_theta <= 0.
        {
            return Err(
                "Draft checkpoint does not match target geometry, features, or method".into(),
            );
        }
        let taps = h.conv_kernel_size.unwrap_or(0);
        let group = h.conv_group_size.unwrap_or(0);
        let rank = h.selector_rank.or(h.markov_rank).unwrap_or(0);
        let top = h.selector_top_k.unwrap_or(0);
        if (method == SpecMethod::Dflash2
            && (group == 0
                || !hidden.is_multiple_of(group)
                || taps == 0
                || taps > block
                || top == 0
                || top > vocab
                || rank == 0))
            || (method == SpecMethod::Dspark
                && (rank == 0
                    || h.markov_head_type.as_deref() != Some("vanilla")
                    || h.confidence_head_with_markov != Some(true)))
        {
            return Err("Unsupported block selector or Markov head contract".into());
        }
        let store = Store::load(root, &raw, checkpoint, verify)?;
        let draft = Self {
            store,
            net,
            layers,
            block,
            mask,
            method,
            group,
            taps,
            rank,
            top,
            from_anchor: h.sample_from_anchor.unwrap_or(false),
            sliding_noncausal: h.sliding_window_non_causal.unwrap_or(false),
        };
        draft.check_weights()?;
        Ok(draft)
    }
    pub(super) fn feature_bytes(&self, n: usize) -> Result<usize> {
        n.checked_mul(self.net.hidden_size)
            .and_then(|v| v.checked_mul(self.layers.len()))
            .and_then(|v| v.checked_mul(4))
            .ok_or("Feature history overflow".into())
    }
    pub(super) fn temporary_bytes(&self, context: usize) -> Result<usize> {
        let t = &self.net;
        // During attention the feature history, its projection, packed KV
        // input, keys and values are simultaneously live in unified memory.
        let state = t
            .hidden_size
            .checked_mul(self.layers.len() + 2)
            .and_then(|n| n.checked_add(2 * t.num_key_value_heads * t.head_dim))
            .and_then(|n| n.checked_mul(context))
            .ok_or("Block feature workspace overflow")?;
        let rows = self
            .block
            .checked_mul(2 * t.vocab_size + 3 * t.intermediate_size + 6 * t.hidden_size)
            .ok_or("Block projection workspace overflow")?;
        state
            .checked_add(rows)
            .and_then(|n| n.checked_mul(4))
            .ok_or("Block workspace overflow".into())
    }
    fn check_weights(&self) -> Result<()> {
        let t = &self.net;
        let h = t.hidden_size;
        let d = t.head_dim;
        let q = d * t.num_attention_heads;
        let kv = d * t.num_key_value_heads;
        let ff = t.intermediate_size;
        self.store.check("fc.weight", &[h, h * self.layers.len()])?;
        for name in ["hidden_norm.weight", "norm.weight"] {
            self.store.check(name, &[h])?;
        }
        for i in 0..t.num_hidden_layers {
            let p = format!("layers.{i}.");
            for (name, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![q, h]),
                ("self_attn.k_proj.weight", vec![kv, h]),
                ("self_attn.v_proj.weight", vec![kv, h]),
                ("self_attn.o_proj.weight", vec![h, q]),
                ("self_attn.k_norm.weight", vec![d]),
                ("self_attn.q_norm.weight", vec![d]),
                ("mlp.gate_proj.weight", vec![ff, h]),
                ("mlp.up_proj.weight", vec![ff, h]),
                ("mlp.down_proj.weight", vec![h, ff]),
            ] {
                self.store.check(&(p.clone() + name), &shape)?;
            }
            if self.method == SpecMethod::Dflash2 {
                for c in ["attention_conv", "mlp_conv"] {
                    self.store.check(
                        &format!("{p}{c}.kernel_projection.weight"),
                        &[2 * self.taps * (h / self.group), h],
                    )?;
                    self.store
                        .check(&format!("{p}{c}.base_kernel"), &[2, self.taps, h])?;
                }
            }
        }
        if self.method == SpecMethod::Dflash2 {
            for n in ["predecessor", "successor"] {
                self.store.check(
                    &format!("candidate_selector.{n}_codebook"),
                    &[t.vocab_size, self.rank],
                )?;
            }
            self.store.check(
                "candidate_selector.hidden_projection.weight",
                &[self.rank, h],
            )?;
        } else if self.method == SpecMethod::Dspark {
            for n in ["w1", "w2"] {
                self.store.check(
                    &format!("markov_head.markov_{n}.weight"),
                    &[t.vocab_size, self.rank],
                )?;
            }
            self.store
                .check("confidence_head.proj.weight", &[1, h + self.rank])?;
            self.store.check("confidence_head.proj.bias", &[1])?;
        }
        Ok(())
    }
    fn normalized(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let mut y = x.to_vec();
        rms(&mut y, self.store.host(name)?, self.net.rms_norm_eps);
        Ok(y)
    }
    fn conv(&self, name: &str, x: &[f32], delta: &[f32], part: usize) -> Result<Vec<f32>> {
        if self.taps == 0 {
            return Ok(x.to_vec());
        }
        let h = self.net.hidden_size;
        let groups = h / self.group;
        let w = self.store.host(&format!("{name}.base_kernel"))?;
        Ok((0..x.len())
            .map(|i| {
                let pos = i / h;
                let ch = i % h;
                let mut y = 0.;
                for tap in 0..self.taps.min(pos + 1) {
                    let coeff = rounded(
                        scalar(w[(part * self.taps + tap) * h + ch])
                            + delta
                                [((pos * 2 + part) * self.taps + tap) * groups + ch / self.group],
                    );
                    y = rounded(y + rounded(x[i - tap * h] * coeff));
                }
                y
            })
            .collect())
    }
    fn sublayer(
        &self,
        p: &str,
        conv: &str,
        norm: &str,
        x: &mut [f32],
        run: impl FnOnce(&[f32]) -> Result<Vec<f32>>,
    ) -> Result<()> {
        let mut normalized = self.normalized(x, &format!("{p}.{norm}.weight"))?;
        let c = format!("{p}.{conv}");
        let delta = if self.taps > 0 {
            self.store
                .linear(&format!("{c}.kernel_projection.weight"), &normalized)?
        } else {
            vec![]
        };
        if self.taps > 0 {
            normalized = self.conv(&c, &normalized, &delta, 0)?;
        }
        let mut output = run(&normalized)?;
        if self.taps > 0 {
            output = self.conv(&c, &output, &delta, 1)?;
        }
        for (v, y) in x.iter_mut().zip(output) {
            *v = rounded(*v + y);
        }
        Ok(())
    }
    fn rotate(&self, x: &mut [f32], heads: usize, first: usize) {
        let d = self.net.head_dim;
        for (row, v) in x.chunks_exact_mut(d).enumerate() {
            let pos = (first + row / heads) as f32;
            for j in 0..d / 2 {
                let angle = pos
                    * self
                        .net
                        .rope_parameters
                        .rope_theta
                        .powf(-2. * j as f32 / d as f32);
                let (sin, cos) = angle.sin_cos();
                let (sin, cos) = (rounded(sin), rounded(cos));
                let (a, b) = (v[j], v[j + d / 2]);
                v[j] = rounded(rounded(a * cos) - rounded(b * sin));
                v[j + d / 2] = rounded(rounded(b * cos) + rounded(a * sin));
            }
        }
    }
    fn attention(&self, i: usize, x: &[f32], context: &[f32]) -> Result<Vec<f32>> {
        let t = &self.net;
        let d = t.head_dim;
        let h = t.hidden_size;
        let c = context.len() / h;
        let p = format!("layers.{i}.self_attn");
        let mut q = self.store.linear(&format!("{p}.q_proj.weight"), x)?;
        let kv = [context, x].concat();
        let mut k = self.store.linear(&format!("{p}.k_proj.weight"), &kv)?;
        let v = self.store.linear(&format!("{p}.v_proj.weight"), &kv)?;
        rms(
            &mut q,
            self.store.host(&format!("{p}.q_norm.weight"))?,
            t.rms_norm_eps,
        );
        rms(
            &mut k,
            self.store.host(&format!("{p}.k_norm.weight"))?,
            t.rms_norm_eps,
        );
        self.rotate(&mut q, t.num_attention_heads, c);
        self.rotate(&mut k, t.num_key_value_heads, 0);
        let sliding = t.layer_types[i] == "sliding_attention";
        let causal = t.is_causal.unwrap_or(sliding && !self.sliding_noncausal);
        let total = c + x.len() / h;
        let mut out = vec![0.; q.len()];
        out.par_chunks_mut(d).enumerate().for_each(|(row, o)| {
            let pos = c + row / t.num_attention_heads;
            let head =
                (row % t.num_attention_heads) / (t.num_attention_heads / t.num_key_value_heads);
            let window = if sliding {
                t.sliding_window.unwrap_or(total)
            } else {
                total
            };
            let first = (pos + 1).saturating_sub(window);
            let last = if causal {
                pos + 1
            } else {
                total.min(pos.saturating_add(window))
            };
            let query = &q[row * d..(row + 1) * d];
            let mut scores = Vec::with_capacity(last - first);
            for kpos in first..last {
                let base = (kpos * t.num_key_value_heads + head) * d;
                scores.push(
                    query
                        .iter()
                        .zip(&k[base..base + d])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / (d as f32).sqrt(),
                );
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z = scores.iter().map(|s| (s - max).exp()).sum::<f32>();
            for (j, s) in scores.iter().enumerate() {
                let prob = (s - max).exp() / z;
                let base = ((first + j) * t.num_key_value_heads + head) * d;
                for (dst, value) in o.iter_mut().zip(&v[base..base + d]) {
                    *dst += prob * value;
                }
            }
            for a in o {
                *a = rounded(*a);
            }
        });
        self.store.linear(&format!("{p}.o_proj.weight"), &out)
    }
    pub(super) fn forward(&self, features: &[f32], embedding: &[f32]) -> Result<Vec<f32>> {
        let mut ctx = self.store.linear("fc.weight", features)?;
        rms(
            &mut ctx,
            self.store.host("hidden_norm.weight")?,
            self.net.rms_norm_eps,
        );
        let mut x = embedding.iter().copied().map(rounded).collect::<Vec<_>>();
        for i in 0..self.net.num_hidden_layers {
            let p = format!("layers.{i}");
            self.sublayer(&p, "attention_conv", "input_layernorm", &mut x, |v| {
                self.attention(i, v, &ctx)
            })?;
            #[cfg(test)]
            probe(&format!("{p}.attention"), &x)?;
            self.sublayer(&p, "mlp_conv", "post_attention_layernorm", &mut x, |v| {
                let mut gate = self.store.linear(&format!("{p}.mlp.gate_proj.weight"), v)?;
                let up = self.store.linear(&format!("{p}.mlp.up_proj.weight"), v)?;
                for (a, b) in gate.iter_mut().zip(up) {
                    *a = rounded(rounded(*a / (1. + (-*a).exp())) * b);
                }
                self.store
                    .linear(&format!("{p}.mlp.down_proj.weight"), &gate)
            })?;
            #[cfg(test)]
            probe(&format!("{p}.mlp"), &x)?;
        }
        rms(
            &mut x,
            self.store.host("norm.weight")?,
            self.net.rms_norm_eps,
        );
        Ok(x)
    }
    pub(super) fn propose(
        &self,
        hidden: &[f32],
        logits: &[f32],
        anchor: u32,
        limit: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<u32>> {
        let (v, h, r) = (self.net.vocab_size, self.net.hidden_size, self.rank);
        if logits.len() != hidden.len() / h * v || anchor as usize >= v {
            return Err("Block draft logits extent mismatch".into());
        }
        let projected = if self.method == SpecMethod::Dflash2 {
            self.store
                .linear("candidate_selector.hidden_projection.weight", hidden)?
        } else {
            vec![]
        };
        let mut prev = anchor as usize;
        let mut ids = vec![];
        for (pos, unary) in logits.chunks_exact(v).take(limit).enumerate() {
            let mut scores = unary.to_vec();
            let id = match self.method {
                SpecMethod::Dflash2 => {
                    let pred = &self.store.host("candidate_selector.predecessor_codebook")?
                        [prev * r..(prev + 1) * r];
                    let succ = self.store.host("candidate_selector.successor_codebook")?;
                    let context = pred
                        .iter()
                        .zip(&projected[pos * r..(pos + 1) * r])
                        .map(|(&a, b)| rounded(scalar(a) * b))
                        .collect::<Vec<_>>();
                    let mut top = (0..v).collect::<Vec<_>>();
                    let rank =
                        |a: &usize, b: &usize| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b));
                    if self.top < v {
                        top.select_nth_unstable_by(self.top, rank);
                        top.truncate(self.top);
                    }
                    top.into_iter()
                        .max_by(|&a, &b| {
                            let edge = |i| {
                                scores[i]
                                    + rounded(
                                        context
                                            .iter()
                                            .zip(&succ[i * r..(i + 1) * r])
                                            .map(|(&x, &w)| rounded(x * scalar(w)))
                                            .sum(),
                                    )
                            };
                            edge(a).total_cmp(&edge(b)).then(b.cmp(&a))
                        })
                        .ok_or("Empty transition candidates")?
                }
                SpecMethod::Dspark => {
                    let pred = self.store.host("markov_head.markov_w1.weight")?
                        [prev * r..(prev + 1) * r]
                        .iter()
                        .map(|&x| scalar(x))
                        .collect::<Vec<_>>();
                    let bias = self.store.linear("markov_head.markov_w2.weight", &pred)?;
                    for (s, b) in scores.iter_mut().zip(bias) {
                        *s += b;
                    }
                    if let Some(t) = threshold {
                        let features = hidden[pos * h..(pos + 1) * h].iter().chain(&pred);
                        let weights = self.store.host("confidence_head.proj.weight")?;
                        let z = features
                            .zip(weights)
                            .map(|(x, &w)| x * scalar(w))
                            .sum::<f32>()
                            + scalar(self.store.host("confidence_head.proj.bias")?[0]);
                        if !ids.is_empty() && (1. + (-z).exp()).recip() < t {
                            break;
                        }
                    }
                    best(&scores)?
                }
                _ => best(&scores)?,
            };
            ids.push(id as u32);
            prev = id;
        }
        Ok(ids)
    }
}
#[cfg(test)]
fn probe(name: &str, values: &[f32]) -> Result<()> {
    if let Ok(dir) = std::env::var("THORUSTINFER_DRAFT_REFERENCE_OUTPUT") {
        let path = Path::new(&dir);
        std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
        std::fs::write(
            path.join(format!("{name}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn best(values: &[f32]) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then(j.cmp(i)))
        .map(|(i, _)| i)
        .ok_or("Empty block draft logits".into())
}

pub(super) struct FeatureHistory {
    slot: Option<usize>,
    abandoned: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    captured: usize,
    values: Vec<f32>,
}
impl FeatureHistory {
    pub(super) fn release(&mut self, model: &mut ModelRuntime) -> Result<()> {
        if let Some(slot) = self.slot {
            model
                .execution
                .release_sequence(slot, &model.manifest.reset_buffers)?;
            self.slot = None;
        }
        Ok(())
    }
}
impl Drop for FeatureHistory {
    fn drop(&mut self) {
        if let Some(slot) = self.slot
            && let Ok(mut abandoned) = self.abandoned.lock()
        {
            abandoned.push(slot);
        }
    }
}

impl ModelRuntime {
    /// Capture committed rows in request order, including a cached prompt.
    /// The shadow sequence never consumes an unverified draft token.
    pub(super) fn block_candidates(
        &mut self,
        req: &mut requests::RequestState,
        limit: usize,
    ) -> Result<Vec<u32>> {
        let draft = self.block_draft.as_ref().ok_or("Missing block draft")?;
        let history = &req.core.history[..req.core.history.len() - 1];
        let anchor = *req.history.last().ok_or("Missing block anchor")?;
        let layout = self
            .manifest
            .batch_layout
            .as_ref()
            .ok_or("Missing layer layout")?;
        let h = layout.hidden;
        let layers = draft.layers.clone();
        let rows = draft
            .block
            .min(self.effective_context.saturating_sub(history.len()));
        let input = requests::GenerationInput {
            input_tokens: vec![],
            images: vec![],
            audios: vec![],
            max_new_tokens: 1,
            sampling: Default::default(),
            prefix_hints: vec![],
        };
        let capacity = (req.core.input.len() + req.core.limit)
            .saturating_add(draft.block - 1)
            .min(self.effective_context);
        if req.features.is_none() {
            let limits = self.request_limits(&input, capacity)?;
            let slot = self.execution.lease_sequence(&limits)?;
            req.features = Some(FeatureHistory {
                slot: Some(slot),
                abandoned: self.abandoned.clone(),
                captured: 0,
                values: vec![],
            });
            self.execution
                .reset_sequence_reusing_pages(&self.manifest.reset_buffers)?;
            self.prepare_visual_capacity(history, &[], capacity, &|| false)?;
        }
        let state = req.features.as_mut().unwrap();
        self.execution
            .activate_sequence(state.slot.ok_or("Released feature owner")?)?;
        let features = &mut state.values;
        let capture_at = Instant::now();
        let prior = state.captured;
        let mut capture_s = 0.;
        let result = (|| {
            let mut offset = state.captured;
            features.resize(
                history
                    .len()
                    .checked_mul(h)
                    .and_then(|n| n.checked_mul(layers.len()))
                    .ok_or("Feature storage overflow")?,
                0.,
            );
            while offset < history.len() {
                let count = (history.len() - offset).min(self.manifest.chunk_tokens);
                let plan = self
                    .manifest
                    .prefill_plan(count)
                    .cloned()
                    .ok_or("Missing staged feature prefill")?;
                if plan.stages.len() != layout.layers.len() {
                    return Err("Draft taps require individual target layer stages".into());
                }
                let chunk = &history[offset..offset + count];
                self.upload_ids(&self.manifest.input, chunk)?;
                self.prepare_prefill_inputs(&plan.prefill_program, chunk, &history[..offset])?;
                self.upload_segment_controls(count)?;
                for (i, stage) in plan.stages.iter().enumerate() {
                    self.launch_rows(stage, count, ExecutionPhase::Prefill)?;
                    if let Some(index) = layers.iter().position(|&tap| tap == i) {
                        let x = self.execution.download_bytes("Hidden", count * h * 2)?;
                        let r = self.execution.download_bytes("R0", count * h * 4)?;
                        for row in 0..count {
                            let dest = ((offset + row) * layers.len() + index) * h;
                            for col in 0..h {
                                let at = row * h + col;
                                let hidden =
                                    half::f16::from_le_bytes([x[at * 2], x[at * 2 + 1]]).to_f32();
                                let residual =
                                    f32::from_le_bytes(r[at * 4..at * 4 + 4].try_into().unwrap());
                                features[dest + col] = rounded(hidden + residual);
                            }
                        }
                    }
                }
                offset += count;
            }
            state.captured = history.len();
            capture_s = capture_at.elapsed().as_secs_f64();
            let mut noise = vec![draft.mask; rows];
            noise[0] = anchor;
            self.upload_ids(&self.manifest.input, &noise)?;
            self.prepare_prefill_inputs("prefill", &noise, history)?;
            self.upload_segment_controls(rows)?;
            self.launch_rows("qwen_rows/begin", rows, ExecutionPhase::Prefill)?;
            let embedding = floats(
                &self.execution.download_bytes("Hidden", rows * h * 2)?,
                crate::artifact::Dtype::F16,
            );
            let hidden = draft.forward(features, &embedding)?;
            let begin = usize::from(!draft.from_anchor);
            let count = limit.min(rows - begin);
            let selected = &hidden[begin * h..(begin + count) * h];
            // The two actual target profiles share the tied compressed head.
            let encoded = selected
                .iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect::<Vec<_>>();
            self.execution.upload_bytes("HeadHidden", &encoded)?;
            self.upload_segment_controls(count)?;
            self.launch_rows("qwen_rows/head", count, ExecutionPhase::Decode)?;
            let logits = floats(
                &self
                    .execution
                    .download_bytes("HeadLogits", count * self.manifest.vocab * 4)?,
                crate::artifact::Dtype::F32,
            );
            let candidates = draft.propose(selected, &logits, anchor, count, self.confidence)?;
            #[cfg(test)]
            if let Ok(root) = std::env::var("THORUSTINFER_DRAFT_REFERENCE_OUTPUT") {
                let root = Path::new(&root);
                std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
                for (name, values) in [
                    ("features", features.as_slice()),
                    ("embedding", embedding.as_slice()),
                    ("hidden", hidden.as_slice()),
                    ("logits", logits.as_slice()),
                ] {
                    let bytes = values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
                    std::fs::write(root.join(format!("{name}.f32")), bytes)
                        .map_err(|e| e.to_string())?;
                }
                let evidence = serde_json::json!({"layers":layers,"history":history,"anchor":anchor,"candidates":candidates,"block":rows,"confidence":self.confidence});
                std::fs::write(root.join("round.json"), evidence.to_string())
                    .map_err(|e| e.to_string())?;
            }
            Ok(candidates)
        })();
        let captured = state.captured - prior;
        req.mtp.feature_capture_tokens += captured;
        req.mtp.feature_capture_s += capture_s;
        self.execution.activate_sequence(req.slot)?;
        result
    }
}

#[test]
#[ignore = "Requires target and block draft packages on SM110a"]
fn export_block_draft_reference() {
    #[derive(Deserialize)]
    struct Fixture {
        target: std::path::PathBuf,
        draft: std::path::PathBuf,
        method: SpecMethod,
        input: Vec<u32>,
        width: usize,
    }
    let f: Fixture = crate::model::read(Path::new(
        &std::env::var("THORUSTINFER_BLOCK_FIXTURE").unwrap(),
    ))
    .unwrap();
    let mut model = ModelRuntime::load_with_options(
        &f.target,
        LoadOptions {
            spec_method: Some(f.method),
            ..Default::default()
        },
    )
    .unwrap();
    model.attach_block(&f.draft, f.method, None, false).unwrap();
    let mut req = model
        .start_request(
            requests::GenerationInput {
                input_tokens: f.input,
                images: vec![],
                audios: vec![],
                max_new_tokens: f.width + 4,
                sampling: crate::sampling::Options {
                    temperature: 0.,
                    ..Default::default()
                },
                prefix_hints: vec![],
            },
            &|| false,
        )
        .unwrap();
    while req.prefilling {
        model
            .advance_requests(&mut [&mut req], &Default::default())
            .unwrap();
    }
    let ids = model.block_candidates(&mut req, f.width).unwrap();
    assert_eq!(ids.len(), f.width);
    model.finish_request(&mut req, false).unwrap();
}
