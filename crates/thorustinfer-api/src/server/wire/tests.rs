use super::*;
fn source(kind: Kind, stream: bool) -> Value {
    if kind.messages() {
        json!({"model":"m","messages":[{"role":"user","content":"hello"}],"max_tokens":12,"stream":stream,"tools":[{"name":"read","input_schema":{"type":"object"}}]})
    } else {
        json!({"model":"m","input":"hello","stream":stream,"tools":[{"type":"function","name":"read","parameters":{"type":"object"}},{"type":"custom","name":"patch","format":{"type":"text"}}]})
    }
}
#[test]
fn tool_history_is_paired_by_id_and_additional_text_follows_results() {
    for kind in [Kind::Responses, Kind::Messages, Kind::Count] {
        let mut v = source(kind, false);
        if kind.messages() {
            v["messages"] = json!([
            {"role":"user","content":"read both"},
            {"role":"assistant","content":[{"type":"text","text":"checking"},{"type":"tool_use","id":"a","name":"read","input":{"file":"x"}},{"type":"tool_use","id":"b","name":"read","input":{"file":"y"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"Y"},{"type":"tool_result","tool_use_id":"a","content":"X"},{"type":"text","text":"compare"}]}]);
        } else {
            v["input"] = json!([{"role":"user","content":"read both"},{"type":"message","role":"assistant","content":"checking"},
            {"type":"function_call","id":"item1","call_id":"a","name":"read","arguments":"{\"file\":\"x\"}"},{"type":"function_call","id":"item2","call_id":"b","name":"read","arguments":"{\"file\":\"y\"}"},
            {"type":"function_call_output","call_id":"b","output":"Y"},{"type":"function_call_output","call_id":"a","output":"X"},{"role":"user","content":"compare"}]);
        }
        let (v, _) = translate(kind, v).unwrap();
        let m = v["messages"].as_array().unwrap();
        assert_eq!(m.len(), 5);
        assert_eq!(m[1]["tool_calls"][1]["id"], "b");
        assert_eq!(m[2]["content"], "X");
        assert_eq!(m[3]["content"], "Y");
        assert_eq!(text_of(&m[4]["content"]), "compare");
    }
}
fn text_of(v: &Value) -> &str {
    v.as_str().unwrap_or_else(|| v[0]["text"].as_str().unwrap())
}
fn delta(v: Value) -> Value {
    json!({"choices":[{"delta":v}]})
}
fn complete(reason: &str) -> Value {
    json!({"finish_reason":reason,"exhausted":reason=="length","usage":{"prompt_tokens":13,"completion_tokens":5,"total_tokens":18,"prompt_tokens_details":{"cached_tokens":0}}})
}
#[test]
fn responses_custom_bytes_survive_stream_final_item_and_next_input() {
    let raw = "*** Begin Patch\n @@ \tλ\\\"  \n*** End Patch\n";
    let mut v = source(Kind::Responses, true);
    let (_, w) = translate(Kind::Responses, v.clone()).unwrap();
    let mut e = response::Encoder::new(w, "i".into(), 1, 13);
    let mut events = e.start();
    events.extend(e.chunk(&delta(json!({"content":"before"}))).unwrap());
    events.extend(e.chunk(&delta(json!({"tool_calls":[{"index":0,"id":"call-a","function":{"name":"patch","arguments":json!({"input":raw}).to_string()}}]}))).unwrap());
    events.extend(e.chunk(&delta(json!({"validated_tool_index":0}))).unwrap());
    events.extend(e.chunk(&delta(json!({"content":"after"}))).unwrap());
    let (r, end) = e.finish(&complete("tool_calls")).unwrap();
    events.extend(end);
    let item = &r["output"][1];
    assert_eq!(item["input"], raw);
    assert_eq!(item["call_id"], "call-a");
    assert_ne!(item["id"], item["call_id"]);
    assert_eq!(events.last().unwrap()["response"], r);
    for (n, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], n)
    }
    assert_eq!(
        events
            .iter()
            .find(|e| e["type"] == "response.custom_tool_call_input.delta")
            .unwrap()["delta"],
        raw
    );
    assert_eq!(r["output"][2]["content"][0]["text"], "after");
    v["input"] = json!([{"role":"user","content":"patch it"},item,{"type":"custom_tool_call_output","call_id":"call-a","output":"ok"}]);
    let (chat, _) = translate(Kind::Responses, v).unwrap();
    let args: Value = serde_json::from_str(
        chat["messages"][1]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(args["input"], raw);
}
#[test]
fn terminal_metadata_controls_usage_stops_and_incomplete_tools() {
    for kind in [Kind::Responses, Kind::Messages] {
        let (_, wire) = translate(kind, source(kind, true)).unwrap();
        let mut e = response::Encoder::new(wire.clone(), "i".into(), 1, 13);
        let opening = e.chunk(&delta(json!({"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{\"file\":"}}]}))).unwrap();
        let (r, events) = e.finish(&complete("length")).unwrap();
        if kind.messages() {
            assert_eq!(r["stop_reason"], "max_tokens");
            assert_eq!(r["content"], json!([]));
        } else {
            assert_eq!(r["status"], "incomplete");
            let item = &r["output"][0];
            assert_eq!(r["output"].as_array().unwrap().len(), 1);
            assert_eq!(opening[0]["item"]["id"], item["id"]);
            assert_eq!(opening[0]["item"]["call_id"], item["call_id"]);
            assert_eq!(opening[1]["delta"], item["arguments"]);
            assert_eq!(item["arguments"], "{\"file\":");
            assert_eq!(item["status"], "incomplete");
            assert_eq!(events[0]["type"], "response.output_item.done");
            assert_eq!(events[0]["item"], *item);
            assert_eq!(events[1]["type"], "response.incomplete");
            assert_eq!(events[1]["response"], r);
            assert!(
                !events
                    .iter()
                    .any(|v| v["type"] == "response.function_call_arguments.done")
            );
        }
        assert!(!events.iter().any(|v| v["type"] == "content_block_start"));
        let mut e = response::Encoder::new(wire, "j".into(), 1, 13);
        let mut meta = complete("stop");
        meta["stop_sequence"] = json!("END");
        let (r, _) = e.finish(&meta).unwrap();
        assert_eq!(r["usage"]["output_tokens"], 5);
        if kind.messages() {
            assert_eq!(r["stop_reason"], "stop_sequence");
            assert_eq!(r["stop_sequence"], "END");
        }
    }
}
#[tokio::test]
async fn cancelling_stream_drops_receiver_and_failures_are_native_terminal_events() {
    for kind in [Kind::Responses, Kind::Messages] {
        let (_, wire) = translate(kind, source(kind, true)).unwrap();
        let (tx, rx) = mpsc::channel(4);
        let r = deliver(wire.clone(), "id".into(), 0, 13, rx).await;
        drop(r);
        assert!(tx.is_closed());
        let (tx, rx) = mpsc::channel(4);
        tx.send(ModelEvent::Failed("fixture".into())).await.unwrap();
        drop(tx);
        let r = deliver(wire, "id".into(), 0, 13, rx).await;
        let bytes = axum::body::to_bytes(r.into_body(), 32768).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains(if kind.messages() {
            "event: error"
        } else {
            "event: response.failed"
        }));
        assert!(!text.contains("[DONE]"));
    }
}
#[test]
fn unsupported_state_and_duplicate_replies_fail_before_generation() {
    for (key, value) in [
        ("previous_response_id", json!("r")),
        ("store", json!(true)),
        ("background", json!(true)),
        ("tools", json!([{"type":"web_search"}])),
    ] {
        let mut v = source(Kind::Responses, false);
        v[key] = value;
        assert!(translate(Kind::Responses, v).is_err())
    }
    let mut v = source(Kind::Messages, false);
    v["messages"] = json!([{"role":"user","content":[{"type":"tool_result","tool_use_id":"absent","content":"no"}]}]);
    assert!(translate(Kind::Messages, v).is_err());
}

#[test]
fn emitted_ordered_content_replays_with_reversed_parallel_results() {
    for kind in [Kind::Responses, Kind::Messages] {
        let mut v = source(kind, false);
        let (_, wire) = translate(kind, v.clone()).unwrap();
        let mut e = response::Encoder::new(wire, "turn".into(), 0, 13);
        let mut events = Vec::new();
        for (i, (id, name, text)) in [("a", "read", "before"), ("b", "read", "between")]
            .into_iter()
            .enumerate()
        {
            events.extend(e.chunk(&delta(json!({"content":text}))).unwrap());
            events.extend(e.chunk(&delta(json!({"tool_calls":[{"index":i,"id":id,"function":{"name":name,"arguments":json!({"file":id}).to_string()}}]}))).unwrap());
            events.extend(e.chunk(&delta(json!({"validated_tool_index":i}))).unwrap());
        }
        events.extend(e.chunk(&delta(json!({"content":"after"}))).unwrap());
        let (r, closing) = e.finish(&complete("tool_calls")).unwrap();
        events.extend(closing);
        if kind.messages() {
            v["messages"].as_array_mut().unwrap().extend([
                json!({"role":"assistant","content":r["content"]}),
                json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"Y"},{"type":"tool_result","tool_use_id":"a","content":"X"},{"type":"text","text":"compare"}]})
            ]);
            let indices: Vec<_> = events
                .iter()
                .filter(|e| e["type"] == "content_block_stop")
                .map(|e| e["index"].as_u64().unwrap())
                .collect();
            assert_eq!(indices, vec![0, 1, 2, 3, 4]);
        } else {
            let mut items = vec![json!({"role":"user","content":"hello"})];
            items.extend(r["output"].as_array().unwrap().clone());
            items.extend([
                json!({"type":"function_call_output","call_id":"b","output":"Y"}),
                json!({"type":"function_call_output","call_id":"a","output":"X"}),
                json!({"role":"user","content":"compare"}),
            ]);
            v["input"] = json!(items);
        }
        let (chat, _) = translate(kind, v).unwrap();
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m.len(), 7);
        assert_eq!(
            (
                text_of(&m[1]["content"]),
                m[1]["tool_calls"][0]["id"].as_str()
            ),
            ("before", Some("a"))
        );
        assert_eq!(
            (
                text_of(&m[2]["content"]),
                m[2]["tool_calls"][0]["id"].as_str()
            ),
            ("between", Some("b"))
        );
        assert_eq!(text_of(&m[3]["content"]), "after");
        assert_eq!(m[4]["content"], "X");
        assert_eq!(m[5]["content"], "Y");
        assert_eq!(text_of(&m[6]["content"]), "compare");
    }
}

#[test]
fn responses_echo_protocol_tools_and_effective_choice() {
    let tools = json!([{"type":"namespace","name":"files","tools":[{"type":"custom","name":"patch","format":{"type":"text"}}]}]);
    let forced = json!({"type":"custom","name":"patch","namespace":"files"});
    for (declared, choice, expected) in [
        (Some(tools.clone()), Some(forced.clone()), forced),
        (Some(tools), None, json!("auto")),
        (None, None, json!("none")),
    ] {
        let mut input = json!({"model":"m","input":"hello"});
        if let Some(t) = &declared {
            input["tools"] = t.clone();
        }
        if let Some(c) = choice {
            input["tool_choice"] = c;
        }
        let (chat, mut wire) = translate(Kind::Responses, input).unwrap();
        wire.resolve_response_tools(&serde_json::from_value(chat).unwrap());
        let mut encoder = response::Encoder::new(wire, "r".into(), 1, 13);
        let start = encoder.start();
        let (body, end) = encoder.finish(&complete("stop")).unwrap();
        for obj in [
            &body,
            &start[0]["response"],
            &start[1]["response"],
            &end.last().unwrap()["response"],
        ] {
            assert_eq!(obj["tools"], declared.clone().unwrap_or(json!([])));
            assert_eq!(obj["tool_choice"], expected);
        }
    }
}

#[test]
fn messages_explicit_custom_is_an_input_schema_tool() {
    let mut body = source(Kind::Messages, false);
    let (implicit, _) = translate(Kind::Messages, body.clone()).unwrap();
    body["tools"][0]["type"] = json!("custom");
    let (explicit, wire) = translate(Kind::Messages, body).unwrap();
    assert_eq!(implicit["tools"], explicit["tools"]);
    assert!(!wire.tools["read"].custom);
}

fn qwen_frontend() -> (ChatCodec, thorustinfer_engine::vision::VisionSpec) {
    let dir = std::path::PathBuf::from(
        std::env::var_os("THORUSTINFER_PROTOCOL_TEST_MODEL")
            .expect("set the Qwen3.5 vision model directory"),
    );
    let model: Value =
        serde_json::from_slice(&std::fs::read(dir.join("cache/model.json")).unwrap()).unwrap();
    (
        ChatCodec::load(&dir).unwrap(),
        serde_json::from_value(model["metadata"]["vision"].clone()).unwrap(),
    )
}

#[test]
#[ignore = "requires Qwen3.5 frontend assets; CPU only"]
fn qwen_responses_inline_instructions_render_as_ordered_system_turns() {
    let (codec, _) = qwen_frontend();
    let input = json!({"model":"m","instructions":"initial","input":[{"role":"user","content":"old question"},{"role":"assistant","content":"old answer"},{"role":"developer","content":"new rule"},{"role":"system","content":"next rule"},{"role":"user","content":"new question"}]});
    let (chat, wire) = translate(Kind::Responses, input).unwrap();
    let mut ctx = preparation::Context::unbounded();
    assert!(
        codec
            .prepare(
                serde_json::from_value(chat.clone()).unwrap(),
                "m",
                32768,
                None,
                &mut ctx
            )
            .err()
            .unwrap()
            .contains("must precede")
    );
    ctx.wire_kind = Some(wire.kind);
    let result = codec
        .prepare(
            serde_json::from_value(chat).unwrap(),
            "m",
            32768,
            None,
            &mut ctx,
        )
        .unwrap();
    let prompt = codec.tokenizer.decode(&result.input, false).unwrap();
    assert!(prompt.contains("<|im_start|>system\ninitial<|im_end|>\n<|im_start|>user\nold question<|im_end|>\n<|im_start|>assistant\nold answer<|im_end|>\n<|im_start|>system\nnew rule<|im_end|>\n<|im_start|>system\nnext rule<|im_end|>\n<|im_start|>user\nnew question<|im_end|>"),"{prompt}");
}

#[test]
#[ignore = "requires Qwen3.5 vision frontend assets; CPU only"]
fn tool_result_images_reach_their_qwen_tool_turns_and_pixels() {
    use base64::{Engine, engine::general_purpose::STANDARD};
    let (codec, vision) = qwen_frontend();
    let png = |rgb| {
        let img = ::image::RgbImage::from_pixel(112, 112, ::image::Rgb(rgb));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, ::image::ImageFormat::Png).unwrap();
        STANDARD.encode(out.into_inner())
    };
    let red = png([255, 0, 0]);
    let blue = png([0, 0, 255]);
    for kind in [Kind::Messages, Kind::Responses] {
        let parts = |label: &str, data: &str| {
            if kind.messages() {
                json!([{"type":"text","text":label},{"type":"image","source":{"type":"base64","media_type":"image/png","data":data}},{"type":"text","text":" suffix"}])
            } else {
                json!([{"type":"input_text","text":label},{"type":"input_image","image_url":format!("data:image/png;base64,{data}")},{"type":"input_text","text":" suffix"}])
            }
        };
        let mut v = source(kind, false);
        if kind.messages() {
            v["messages"] = json!([{"role":"user","content":"read both"},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"read","input":{}},{"type":"tool_use","id":"b","name":"read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":parts("second",&blue)},{"type":"tool_result","tool_use_id":"a","is_error":true,"content":parts("first",&red)},{"type":"text","text":"follow up"}]}]);
        } else {
            v["input"] = json!([{"role":"user","content":"read both"},{"type":"function_call","call_id":"a","name":"read","arguments":"{}"},{"type":"custom_tool_call","call_id":"b","name":"patch","input":"text"},{"type":"custom_tool_call_output","call_id":"b","output":parts("second",&blue)},{"type":"function_call_output","call_id":"a","output":parts("first",&red)},{"role":"user","content":"follow up"}]);
        }
        let (chat, wire) = translate(kind, v).unwrap();
        assert_eq!(chat["messages"][2]["tool_call_id"], "a");
        assert_eq!(chat["messages"][3]["tool_call_id"], "b");
        let mut ctx = preparation::Context::unbounded();
        assert!(
            codec
                .prepare(
                    serde_json::from_value(chat.clone()).unwrap(),
                    "m",
                    32768,
                    Some(&vision),
                    &mut ctx
                )
                .err()
                .unwrap()
                .contains("Image parts")
        );
        ctx.wire_kind = Some(wire.kind);
        let result = codec
            .prepare(
                serde_json::from_value(chat).unwrap(),
                "m",
                32768,
                Some(&vision),
                &mut ctx,
            )
            .unwrap();
        assert_eq!(result.images.len(), 2);
        let mut blocks = Vec::new();
        for (image, (label, rgb)) in result
            .images
            .iter()
            .zip([("first", [255, 0, 0]), ("second", [0, 0, 255])])
        {
            assert_eq!(image.rgb[..3], rgb);
            let n = vision.feature_count(image).unwrap();
            assert!(n > 1);
            let prefix = if kind.messages() && label == "first" {
                "Tool error: "
            } else {
                ""
            };
            blocks.push(format!("<tool_response>\n{prefix}{label}<|vision_start|>{}<|vision_end|> suffix\n</tool_response>","<|image_pad|>".repeat(n)));
        }
        let prompt = codec.tokenizer.decode(&result.input, false).unwrap();
        assert!(
            prompt.contains(&format!(
                "{}\n{}<|im_end|>\n<|im_start|>user\nfollow up",
                blocks[0], blocks[1]
            )),
            "{prompt}"
        );
    }
}
