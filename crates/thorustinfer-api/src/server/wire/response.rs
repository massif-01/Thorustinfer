use super::*;
enum Part {
    Text(String),
    Call {
        slot: usize,
        id: String,
        name: String,
        args: String,
    },
}
pub(super) struct Encoder {
    wire: Wire,
    id: String,
    created: u64,
    prompt: usize,
    active: Option<Part>,
    items: Vec<Value>,
    seq: u64,
    reason: Option<String>,
    usage: Option<Value>,
}
impl Encoder {
    pub fn new(wire: Wire, id: String, created: u64, prompt: usize) -> Self {
        Self {
            wire,
            id,
            created,
            prompt,
            active: None,
            items: vec![],
            seq: 0,
            reason: None,
            usage: None,
        }
    }
    fn event(&mut self, typ: &str, mut v: Value) -> Value {
        v["type"] = json!(typ);
        if !self.wire.kind.messages() {
            v["sequence_number"] = json!(self.seq);
            self.seq += 1;
        }
        v
    }
    fn usage(&self) -> Value {
        let u = self.usage.as_ref().unwrap_or(&Value::Null);
        let i = u["prompt_tokens"].as_u64().unwrap_or(self.prompt as u64);
        let o = u["completion_tokens"].as_u64().unwrap_or(0);
        let cache = u["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0);
        if self.wire.kind.messages() {
            json!({"input_tokens":i-cache.min(i),"output_tokens":o,"cache_creation_input_tokens":0,"cache_read_input_tokens":cache})
        } else {
            json!({"input_tokens":i,"output_tokens":o,"total_tokens":i+o,"input_tokens_details":{"cached_tokens":cache},"output_tokens_details":{"reasoning_tokens":u["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap_or(0)}})
        }
    }
    fn response(&self, status: &str, out: Vec<Value>, stop: &Value) -> Value {
        if self.wire.kind.messages() {
            json!({"id":format!("msg_{}",self.id),"type":"message","role":"assistant","model":self.wire.model,"content":out,"stop_reason":if status=="in_progress"{Value::Null}else if status=="incomplete"{json!("max_tokens")}else if !stop.is_null(){json!("stop_sequence")}else if self.reason.as_deref()==Some("tool_calls"){json!("tool_use")}else{json!("end_turn")},"stop_sequence":stop,"usage":self.usage()})
        } else {
            json!({"id":format!("resp_{}",self.id),"object":"response","created_at":self.created,"status":status,"error":null,"incomplete_details":if status=="incomplete"{json!({"reason":"max_output_tokens"})}else{Value::Null},"model":self.wire.model,"output":out,"tools":self.wire.response_tools,"tool_choice":self.wire.response_choice,"parallel_tool_calls":self.wire.parallel,"store":false,"usage":if status=="in_progress"{Value::Null}else{self.usage()}})
        }
    }
    pub fn start(&mut self) -> Vec<Value> {
        let v = self.response("in_progress", vec![], &Value::Null);
        if self.wire.kind.messages() {
            vec![self.event("message_start", json!({"message":v}))]
        } else {
            vec![
                self.event("response.created", json!({"response":v})),
                self.event("response.in_progress", json!({"response":v})),
            ]
        }
    }
    fn item_id(&self, i: usize) -> String {
        format!("item_{}_{i}", self.id)
    }
    fn text_item(&self, i: usize, s: &str, status: &str) -> Value {
        if self.wire.kind.messages() {
            json!({"type":"text","text":s})
        } else {
            json!({"id":self.item_id(i),"type":"message","role":"assistant","status":status,"content":[{"type":"output_text","text":s,"annotations":[]}]})
        }
    }
    fn text_delta(&mut self, i: usize, text: &str, start: bool) -> Vec<Value> {
        let mut v = Vec::new();
        let id = self.item_id(i);
        if self.wire.kind.messages() {
            if start {
                v.push(self.event(
                    "content_block_start",
                    json!({"index":i,"content_block":{"type":"text","text":""}}),
                ))
            }
            v.push(self.event(
                "content_block_delta",
                json!({"index":i,"delta":{"type":"text_delta","text":text}}),
            ));
        } else {
            if start {
                let mut item = self.text_item(i, "", "in_progress");
                item["content"] = json!([]);
                v.push(self.event(
                    "response.output_item.added",
                    json!({"output_index":i,"item":item}),
                ));
                v.push(self.event("response.content_part.added",json!({"item_id":id,"output_index":i,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}})))
            }
            v.push(self.event(
                "response.output_text.delta",
                json!({"item_id":id,"output_index":i,"content_index":0,"delta":text,"logprobs":[]}),
            ));
        }
        v
    }
    fn close_text(&mut self, status: &str, events: &mut Vec<Value>) {
        if !matches!(self.active, Some(Part::Text(_))) {
            return;
        }
        let Some(Part::Text(s)) = self.active.take() else {
            unreachable!()
        };
        let i = self.items.len();
        let id = self.item_id(i);
        let item = self.text_item(i, &s, status);
        if self.wire.kind.messages() {
            events.push(self.event("content_block_stop", json!({"index":i})));
        } else {
            events.push(self.event(
                "response.output_text.done",
                json!({"item_id":id,"output_index":i,"content_index":0,"text":s,"logprobs":[]}),
            ));
            events.push(self.event(
                "response.content_part.done",
                json!({"item_id":id,"output_index":i,"content_index":0,"part":item["content"][0]}),
            ));
            events.push(self.event(
                "response.output_item.done",
                json!({"output_index":i,"item":item}),
            ));
        }
        self.items.push(item);
    }
    fn tool_item(&self, i: usize, id: &str, t: &Tool) -> Value {
        let mut item = json!({"id":self.item_id(i),"type":if t.custom{"custom_tool_call"}else{"function_call"},"name":t.name,"call_id":id,"status":"in_progress"});
        if let Some(ns) = &t.namespace {
            item["namespace"] = json!(ns);
        }
        item[if t.custom { "input" } else { "arguments" }] = json!("");
        item
    }
    fn seal_tool(&mut self, index: u64, events: &mut Vec<Value>) -> Result<()> {
        let Some(Part::Call {
            slot,
            id,
            name,
            args,
        }) = self.active.take()
        else {
            return Err("Tool close without an open call".into());
        };
        if slot as u64 != index {
            return Err("Mismatched tool index".into());
        }
        let t = self
            .wire
            .tools
            .get(&name)
            .ok_or("Generated undeclared tool")?
            .clone();
        let v: Value =
            serde_json::from_str(&args).map_err(|_| "Malformed generated tool arguments")?;
        if !v.is_object() {
            return Err("Tool arguments must be an object".into());
        }
        let i = self.items.len();
        let item_id = self.item_id(i);
        let item = if self.wire.kind.messages() {
            let item = json!({"type":"tool_use","id":id,"name":t.name,"input":v});
            events.push(self.event("content_block_start",json!({"index":i,"content_block":{"type":"tool_use","id":id,"name":t.name,"input":{}}})));
            events.push(self.event(
                "content_block_delta",
                json!({"index":i,"delta":{"type":"input_json_delta","partial_json":args}}),
            ));
            events.push(self.event("content_block_stop", json!({"index":i})));
            item
        } else {
            let mut item = self.tool_item(i, &id, &t);
            let (field, payload, event) = if t.custom {
                let raw = v["input"]
                    .as_str()
                    .ok_or("Custom tool input must be a string")?;
                events.push(self.event(
                    "response.output_item.added",
                    json!({"output_index":i,"item":item}),
                ));
                events.push(self.event(
                    "response.custom_tool_call_input.delta",
                    json!({"output_index":i,"item_id":item_id,"delta":raw}),
                ));
                ("input", raw, "response.custom_tool_call_input.done")
            } else {
                (
                    "arguments",
                    args.as_str(),
                    "response.function_call_arguments.done",
                )
            };
            let mut done = json!({"output_index":i,"item_id":item_id,"name":t.name});
            done[field] = json!(payload);
            events.push(self.event(event, done));
            item[field] = json!(payload);
            item["status"] = json!("completed");
            events.push(self.event(
                "response.output_item.done",
                json!({"output_index":i,"item":item}),
            ));
            item
        };
        self.items.push(item);
        Ok(())
    }
    pub fn chunk(&mut self, v: &Value) -> Result<Vec<Value>> {
        let delta = &v["choices"][0]["delta"];
        let mut events = vec![];
        if let Some(s) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            let first = self.active.is_none();
            if first {
                self.active = Some(Part::Text(String::new()));
            }
            let Some(Part::Text(acc)) = &mut self.active else {
                return Err("Text before tool closure".into());
            };
            acc.push_str(s);
            events.extend(self.text_delta(self.items.len(), s, first));
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for c in calls {
                let slot = c["index"].as_u64().ok_or("Missing tool delta index")? as usize;
                if let Some(id) = c["id"].as_str() {
                    self.close_text("completed", &mut events);
                    if self.active.is_some() {
                        return Err("Overlapping tool deltas".into());
                    }
                    let name = c["function"]["name"].as_str().ok_or("Missing tool name")?;
                    let t = self
                        .wire
                        .tools
                        .get(name)
                        .ok_or("Generated an undeclared tool")?;
                    if !self.wire.kind.messages() && !t.custom {
                        let i = self.items.len();
                        let item = self.tool_item(i, id, t);
                        events.push(self.event(
                            "response.output_item.added",
                            json!({"output_index":i,"item":item}),
                        ));
                    }
                    self.active = Some(Part::Call {
                        slot,
                        id: id.into(),
                        name: name.into(),
                        args: String::new(),
                    });
                }
                let Some(Part::Call {
                    slot: active,
                    name,
                    args,
                    ..
                }) = &mut self.active
                else {
                    return Err("Tool delta before tool start".into());
                };
                if *active != slot {
                    return Err("Unexpected tool delta index".into());
                }
                if let Some(s) = c["function"]["arguments"]
                    .as_str()
                    .filter(|v| !v.is_empty())
                {
                    args.push_str(s);
                    if !self.wire.kind.messages() && !self.wire.tools[name].custom {
                        let i = self.items.len();
                        events.push(self.event(
                            "response.function_call_arguments.delta",
                            json!({"output_index":i,"item_id":self.item_id(i),"delta":s}),
                        ));
                    }
                }
            }
        }
        if let Some(i) = delta["validated_tool_index"].as_u64() {
            self.seal_tool(i, &mut events)?;
        }
        Ok(events)
    }
    pub fn finish(&mut self, meta: &Value) -> Result<(Value, Vec<Value>)> {
        self.usage = meta.get("usage").cloned();
        self.reason = meta["finish_reason"].as_str().map(str::to_owned);
        if self.usage.is_none() || self.reason.is_none() {
            return Err("Missing terminal generation metadata".into());
        }
        if (meta["incomplete_call"] == true || matches!(self.active, Some(Part::Call { .. })))
            && meta["exhausted"] != true
        {
            return Err("Generation ended with an incomplete tool call".into());
        }
        let status = if self.reason.as_deref() == Some("length") {
            "incomplete"
        } else {
            "completed"
        };
        let mut events = Vec::new();
        self.close_text(status, &mut events);
        if status == "incomplete"
            && !self.wire.kind.messages()
            && let Some(Part::Call { id, name, args, .. }) = self.active.take()
            && !self.wire.tools[&name].custom
        {
            let i = self.items.len();
            let mut item = self.tool_item(i, &id, &self.wire.tools[&name]);
            item["arguments"] = json!(args);
            item["status"] = json!("incomplete");
            events.push(self.event(
                "response.output_item.done",
                json!({"output_index":i,"item":item}),
            ));
            self.items.push(item);
        }
        let items = std::mem::take(&mut self.items);
        let response = self.response(status, items, &meta["stop_sequence"]);
        if self.wire.kind.messages() {
            events.push(self.event("message_delta",json!({"delta":{"stop_reason":response["stop_reason"],"stop_sequence":response["stop_sequence"]},"usage":response["usage"]})));
            events.push(self.event("message_stop", json!({})))
        } else {
            events.push(self.event(&format!("response.{status}"), json!({"response":response})))
        }
        Ok((response, events))
    }
    pub fn failure(&mut self, message: String) -> Value {
        if self.wire.kind.messages() {
            self.event(
                "error",
                json!({"error":{"type":"api_error","message":message}}),
            )
        } else {
            let mut r = self.response("failed", vec![], &Value::Null);
            r["error"] = json!({"code":"server_error","message":message});
            self.event("response.failed", json!({"response":r}))
        }
    }
}
