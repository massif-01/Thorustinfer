//! Compare the verifier with teacher-forced ordinary rows on the same target.
use super::*;

#[test]
#[ignore = "Requires an installed target verifier and exclusive CUDA access"]
fn fixed_history_target_verification() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        model: std::path::PathBuf,
        input: Vec<u32>,
        depth: usize,
        output: std::path::PathBuf,
    }
    let f: Fixture = crate::model::read(std::path::Path::new(
        &std::env::var("THORUSTINFER_TARGET_VERIFICATION_FIXTURE").unwrap(),
    ))
    .unwrap();
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            mtp_drafts: Some(f.depth),
            ..Default::default()
        },
    )
    .unwrap();
    let start = |model: &mut ModelRuntime| {
        let mut state = model
            .start_request(
                GenerationInput {
                    input_tokens: f.input.clone(),
                    images: vec![],
                    audios: vec![],
                    prefix_hints: vec![],
                    max_new_tokens: f.depth * 3 + 8,
                    sampling: crate::sampling::Options {
                        temperature: 0.,
                        ..Default::default()
                    },
                },
                &|| false,
            )
            .unwrap();
        while state.prefilling {
            model
                .advance_requests(&mut [&mut state], &Default::default())
                .unwrap();
        }
        state
    };
    let ordinary = |model: &mut ModelRuntime, state: &mut RequestState| {
        model
            .with_request(state, |m, r| {
                m.execute_target_tail(r)?;
                m.commit_ordinary_token(r)
            })
            .unwrap()[0]
    };
    let state_bytes = |m: &ModelRuntime, pos: usize| {
        let mut values = std::collections::BTreeMap::new();
        for name in m
            .manifest
            .reset_buffers
            .iter()
            .filter(|n| n.starts_with('L'))
        {
            let stride = m
                .manifest
                .kv_cache
                .as_ref()
                .and_then(|kv| kv.buffers.get(name));
            let count = stride.map_or(m.execution.sizes[name], |stride| stride * pos);
            values.insert(
                name.clone(),
                m.execution.download_bytes(name, count).unwrap(),
            );
        }
        values
    };
    let mut baseline = start(&mut model);
    let reference = (0..f.depth + 5)
        .map(|_| ordinary(&mut model, &mut baseline))
        .collect::<Vec<_>>();
    model.finish_request(&mut baseline, false).unwrap();
    let mut report = vec![];
    let spec = model.manifest.mtp.clone().unwrap();
    for wrong in [None, Some(0), Some(f.depth / 2)] {
        let mut drafts = reference[..f.depth].to_vec();
        if let Some(i) = wrong {
            drafts[i] = (drafts[i] + 1) % model.manifest.vocab as u32;
        }
        let mut serial = start(&mut model);
        let mut serial_logits = vec![];
        let mut serial_ids = vec![];
        for row in 0..=f.depth {
            serial_ids.push(ordinary(&mut model, &mut serial));
            let raw = model
                .execution
                .download_bytes(&model.manifest.logits, model.manifest.vocab * 4)
                .unwrap();
            serial_logits.extend(floats(&raw, crate::artifact::Dtype::F32));
            if let Some(&id) = drafts.get(row) {
                *serial.history.last_mut().unwrap() = id;
                model
                    .execution
                    .upload_sequence_ids(serial.slot, &model.manifest.token, &[id])
                    .unwrap();
            }
        }
        model.finish_request(&mut serial, false).unwrap();
        let accepted = wrong.unwrap_or(f.depth);
        let mut reference_state = start(&mut model);
        for _ in 0..=accepted {
            ordinary(&mut model, &mut reference_state);
        }
        let expected_state = state_bytes(&model, reference_state.history.len() - 1);
        model.finish_request(&mut reference_state, false).unwrap();
        let mut state = start(&mut model);
        let position = state.history.len() - 1;
        let round = PreparedRound {
            plan: spec.verification_plan(f.depth + 1).unwrap(),
            position,
            remaining: state.limit - state.generated,
            vocab: model.manifest.vocab,
            host_sampling: false,
            is_lookup: true,
            verification: std::iter::once(*state.history.last().unwrap())
                .chain(drafts.iter().copied())
                .collect(),
            drafts,
            proposals: vec![],
        };
        model.verify_single_round(&mut state, &round).unwrap();
        let raw = model
            .execution
            .download_bytes(&spec.verification_logits, serial_logits.len() * 4)
            .unwrap();
        let actual_logits = floats(&raw, crate::artifact::Dtype::F32);
        let mut differences = vec![];
        for (idx, (a, b)) in actual_logits
            .chunks(model.manifest.vocab)
            .zip(serial_logits.chunks(model.manifest.vocab))
            .enumerate()
        {
            let top = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|(i, a), (j, b)| a.total_cmp(b).then(j.cmp(i)))
                    .unwrap()
                    .0 as u32
            };
            assert_eq!(top(a), serial_ids[idx], "verification logits at row {idx}");
            assert_eq!(top(b), serial_ids[idx], "ordinary logits at row {idx}");
            let mut max = 0f32;
            let mut sum = 0f64;
            for (&x, &y) in a.iter().zip(b) {
                assert!(x.is_finite() && y.is_finite());
                max = max.max((x - y).abs());
                sum += f64::from(x - y).powi(2);
            }
            differences.push(
                serde_json::json!({"max_abs_logit":max, "rms_logit":(sum/a.len() as f64).sqrt()}),
            );
        }
        let commit = model.commit_verified_round(&mut state, round).unwrap();
        let accepted = wrong.unwrap_or(f.depth);
        assert_eq!(commit, reference[..accepted + 1]);
        assert_eq!(state.mtp.accepted_draft_tokens, accepted);
        assert_eq!(
            model.read_control(&model.manifest.position).unwrap() as usize,
            position + commit.len()
        );
        let actual_state = state_bytes(&model, position + commit.len());
        let mut gdn_l2 = 0.0f64;
        for (name, expected) in &expected_state {
            let actual = &actual_state[name];
            if name.ends_with("_State") {
                let (mut error, mut norm) = (0f64, 0f64);
                for (a, b) in actual
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(expected.as_chunks::<4>().0)
                {
                    let (x, y) = (f32::from_le_bytes(*a) as f64, f32::from_le_bytes(*b) as f64);
                    error += (x - y) * (x - y);
                    norm += y * y;
                }
                let l2 = (error / norm.max(1e-20)).sqrt();
                assert!(l2 < 1e-4, "{name} relative L2 {l2}");
                gdn_l2 = gdn_l2.max(l2);
            } else {
                assert!(expected == actual, "KV/conv state mismatch: {name}");
            }
        }
        let continued = (0..3)
            .map(|_| ordinary(&mut model, &mut state))
            .collect::<Vec<_>>();
        assert_eq!(continued, reference[commit.len()..commit.len() + 3]);
        report.push(serde_json::json!({"wrong_at":wrong,"accepted":accepted,"committed":commit,"continued":continued,"rows":differences,"compared_state_buffers":expected_state.len(),"max_gdn_relative_l2":gdn_l2}));
        model.finish_request(&mut state, false).unwrap();
    }
    std::fs::write(f.output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}
