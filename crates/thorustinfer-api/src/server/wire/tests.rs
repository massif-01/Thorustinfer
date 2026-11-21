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
        e.chunk(&delta(json!({"tool_calls":[{"index":0,"id":"a","function":{"name":"read","arguments":"{\"file\":"}}]}))).unwrap();
        let (r, events) = e.finish(&complete("length")).unwrap();
        if kind.messages() {
            assert_eq!(r["stop_reason"], "max_tokens");
            assert_eq!(r["content"], json!([]));
        } else {
            assert_eq!(r["status"], "incomplete");
            assert_eq!(r["output"], json!([]))
        }
        assert!(!events.iter().any(
            |v| v["type"] == "response.output_item.done" || v["type"] == "content_block_start"
        ));
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
