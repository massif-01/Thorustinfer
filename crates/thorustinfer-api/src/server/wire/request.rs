use super::{Kind, Tool, Wire};
use serde_json::{Value, json};
use std::collections::BTreeMap;

type Result<T> = std::result::Result<T, String>;
fn string<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
    v[k].as_str().ok_or_else(|| format!("{k} must be a string"))
}
fn list<'a>(v: &'a Value, k: &str) -> Result<&'a [Value]> {
    v[k].as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{k} must be an array"))
}
fn text(v: &Value) -> Result<String> {
    if let Some(s) = v.as_str() {
        return Ok(s.to_owned());
    }
    let mut out = String::new();
    for b in v.as_array().ok_or("Expected text or text blocks")? {
        match b["type"].as_str() {
            Some("input_text" | "output_text" | "text") => out.push_str(string(b, "text")?),
            _ => return Err("Unsupported content block; expected text".into()),
        }
    }
    Ok(out)
}
fn content(v: &Value) -> Result<Value> {
    if v.is_string() {
        return Ok(v.clone());
    }
    let mut out = Vec::new();
    for part in v.as_array().ok_or("Expected content blocks")? {
        let url = match part["type"].as_str() {
            Some("text" | "input_text" | "output_text") => {
                out.push(json!({"type":"text","text":string(part,"text")?}));
                continue;
            }
            Some("input_image") => string(part, "image_url")?.to_owned(),
            Some("image") => match part["source"]["type"].as_str() {
                Some("base64") => format!(
                    "data:{};base64,{}",
                    string(&part["source"], "media_type")?,
                    string(&part["source"], "data")?
                ),
                Some("url") => string(&part["source"], "url")?.to_owned(),
                _ => return Err("Unsupported image source".into()),
            },
            _ => return Err("Unsupported input content block".into()),
        };
        out.push(json!({"type":"image_url","image_url":{"url":url}}));
    }
    Ok(Value::Array(out))
}
fn add_tool(w: &mut Wire, out: &mut Vec<Value>, t: &Value, namespace: Option<&str>) -> Result<()> {
    let kind = t["type"]
        .as_str()
        .unwrap_or(if w.kind.messages() { "function" } else { "" });
    if kind == "namespace" && !w.kind.messages() && namespace.is_none() {
        for child in list(t, "tools")? {
            add_tool(w, out, child, Some(string(t, "name")?))?;
        }
        return Ok(());
    }
    let custom = kind == "custom" && !w.kind.messages();
    if kind != "function" && !custom {
        return Err(format!("Unsupported tool type {kind}"));
    }
    let name = string(t, "name")?;
    let local = Wire::local(name, namespace);
    let description = t["description"].as_str().unwrap_or("");
    let (schema, description) = if custom {
        let grammar = match t.get("format") {
            None => "",
            Some(f) if f["type"] == "text" => "",
            Some(f) if f["type"] == "grammar" => string(f, "definition")?,
            _ => return Err("Unsupported custom tool format".into()),
        };
        (
            json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false}),
            format!("{description}\nPut the exact tool input in the input string.\n{grammar}"),
        )
    } else {
        let schema = t
            .get(if w.kind.messages() {
                "input_schema"
            } else {
                "parameters"
            })
            .cloned()
            .unwrap_or_else(|| json!({"type":"object","properties":{}}));
        if !schema.is_object() {
            return Err("Tool schema must be an object".into());
        }
        (schema, description.to_owned())
    };
    if w.tools
        .insert(
            local.clone(),
            Tool {
                name: name.to_owned(),
                namespace: namespace.map(str::to_owned),
                custom,
            },
        )
        .is_some()
    {
        return Err(format!("Duplicate tool {name}"));
    }
    let mut f = json!({"name":local,"description":description,"parameters":schema});
    if let Some(strict) = t.get("strict") {
        f["strict"] = strict.clone();
    }
    out.push(json!({"type":"function","function":f}));
    Ok(())
}
#[derive(Default)]
struct History {
    messages: Vec<Value>,
    system: Vec<String>,
    pending: Vec<String>,
    results: BTreeMap<String, Value>,
}
impl History {
    // Some templates omit call IDs from tool responses. Render results in the
    // declaration order, even when parallel clients deliver them in reverse.
    fn flush_results(&mut self) -> Result<()> {
        if self.pending.len() != self.results.len() {
            return Err("Missing result for a preceding tool call".into());
        }
        for id in self.pending.drain(..) {
            self.messages
                .push(self.results.remove(&id).ok_or("Missing tool result")?);
        }
        Ok(())
    }
    fn message(&mut self, role: &str, value: Value) -> Result<()> {
        if role != "assistant" || !self.results.is_empty() {
            self.flush_results()?;
        }
        if matches!(role, "system" | "developer") && self.messages.is_empty() {
            self.system.push(text(&value)?);
            return Ok(());
        }
        if !matches!(role, "assistant" | "user" | "system" | "developer") {
            return Err(format!("Unsupported message role {role}"));
        }
        self.messages.push(json!({"role":role,"content":value}));
        Ok(())
    }
    fn call(&mut self, id: &str, name: &str, args: String) -> Result<()> {
        if !self.results.is_empty() {
            self.flush_results()?;
        }
        if id.is_empty() || self.pending.iter().any(|x| x == id) {
            return Err("Empty or duplicate tool call ID".into());
        }
        if !self
            .messages
            .last()
            .is_some_and(|m| m["role"] == "assistant")
        {
            self.messages
                .push(json!({"role":"assistant","content":null}));
        }
        let m = self.messages.last_mut().unwrap();
        if m.get("tool_calls").is_none() {
            m["tool_calls"] = json!([]);
        }
        m["tool_calls"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":id,"type":"function","function":{"name":name,"arguments":args}}));
        self.pending.push(id.into());
        Ok(())
    }
    fn result(&mut self, id: &str, content: String) -> Result<()> {
        if !self.pending.iter().any(|n| n == id)
            || self
                .results
                .insert(
                    id.into(),
                    json!({"role":"tool","tool_call_id":id,"content":content}),
                )
                .is_some()
        {
            return Err("Unknown or duplicate tool result ID".into());
        }
        Ok(())
    }
    fn finish(mut self) -> Result<Vec<Value>> {
        self.flush_results()?;
        if !self.system.is_empty() {
            self.messages.insert(
                0,
                json!({"role":"system","content":self.system.join("\n\n")}),
            );
        }
        Ok(self.messages)
    }
}
fn response_input(v: &Value) -> Result<Vec<Value>> {
    let mut h = History::default();
    if let Some(s) = v.get("instructions").filter(|s| !s.is_null()) {
        h.system
            .push(s.as_str().ok_or("instructions must be text")?.into())
    }
    if let Some(s) = v["input"].as_str() {
        h.message("user", s.into())?;
    } else {
        for x in list(v, "input")? {
            match x["type"].as_str().unwrap_or("message") {
                "message" => h.message(string(x, "role")?, content(&x["content"])?)?,
                kind @ ("function_call" | "custom_tool_call") => {
                    let n = string(x, "name")?;
                    let local = Wire::local(n, x["namespace"].as_str());
                    let args = if kind == "custom_tool_call" {
                        json!({"input":string(x,"input")?}).to_string()
                    } else {
                        string(x, "arguments")?.into()
                    };
                    h.call(string(x, "call_id")?, &local, args)?
                }
                "function_call_output" | "custom_tool_call_output" => {
                    h.result(string(x, "call_id")?, text(&x["output"])?)?
                }
                "reasoning" if x["encrypted_content"].is_null() => {}
                kind => return Err(format!("Unsupported Responses input item {kind}")),
            }
        }
    }
    h.finish()
}
fn messages_input(v: &Value) -> Result<Vec<Value>> {
    let mut h = History::default();
    if let Some(s) = v.get("system") {
        h.system.push(text(s)?);
    }
    for m in list(v, "messages")? {
        let role = string(m, "role")?;
        if !matches!(role, "user" | "assistant") {
            return Err("Invalid Messages role".into());
        }
        if let Some(s) = m["content"].as_str() {
            h.message(role, s.into())?;
            continue;
        }
        let mut body = Vec::new();
        for b in list(m, "content")? {
            match b["type"].as_str() {
                Some("text" | "image") => body.push(b.clone()),
                Some("tool_use") if role == "assistant" => {
                    if !body.is_empty() {
                        h.message(role, content(&Value::Array(std::mem::take(&mut body)))?)?;
                    }
                    if !b["input"].is_object() {
                        return Err("tool_use input must be an object".into());
                    }
                    h.call(string(b, "id")?, string(b, "name")?, b["input"].to_string())?;
                }
                Some("tool_result") if role == "user" && body.is_empty() => {
                    let mut result = if b["content"].is_null() {
                        String::new()
                    } else {
                        text(&b["content"])?
                    };
                    if b["is_error"] == true {
                        result = format!("Tool error: {result}");
                    }
                    h.result(string(b, "tool_use_id")?, result)?;
                }
                _ => return Err("Unsupported or misplaced Messages content block".into()),
            }
        }
        if !body.is_empty() {
            h.message(role, content(&Value::Array(body))?)?;
        }
    }
    h.finish()
}
pub(crate) fn translate(kind: Kind, v: Value) -> Result<(Value, Wire)> {
    let model = string(&v, "model")?.to_owned();
    let stream = v
        .get("stream")
        .map(|x| x.as_bool().ok_or("stream must be boolean"))
        .transpose()?
        .unwrap_or(false);
    let mut w = Wire {
        kind,
        model: model.clone(),
        stream,
        tools: BTreeMap::new(),
        parallel: true,
    };
    let mut chat = json!({"model":model});
    let mut tools = Vec::new();
    if v.get("tools").is_some() {
        for t in list(&v, "tools")? {
            add_tool(&mut w, &mut tools, t, None)?
        }
    }
    if !tools.is_empty() {
        chat["tools"] = json!(tools);
    }
    let allow = if kind.messages() {
        &[
            "model",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stream",
            "stop_sequences",
            "metadata",
            "thinking",
            "output_config",
        ][..]
    } else {
        &[
            "model",
            "input",
            "instructions",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "max_output_tokens",
            "temperature",
            "top_p",
            "stream",
            "store",
            "previous_response_id",
            "background",
            "include",
            "reasoning",
            "text",
            "metadata",
            "client_metadata",
            "prompt_cache_key",
            "truncation",
            "service_tier",
        ][..]
    };
    for (key, val) in v.as_object().unwrap() {
        if !allow.contains(&key.as_str()) && !val.is_null() {
            return Err(format!("Unsupported request field: {key}"));
        }
    }
    for k in [
        "temperature",
        "top_p",
        "top_k",
        "parallel_tool_calls",
        "prompt_cache_key",
        "service_tier",
    ] {
        if let Some(x) = v.get(k) {
            chat[k] = x.clone();
        }
    }
    if kind.messages() {
        if kind != Kind::Count && v["max_tokens"].as_u64().unwrap_or(0) == 0 {
            return Err("max_tokens must be positive".into());
        }
        if kind != Kind::Count {
            chat["max_tokens"] = v["max_tokens"].clone();
        }
        if let Some(x) = v.get("stop_sequences") {
            chat["stop"] = x.clone();
        }
        if let Some(th) = v.get("thinking") {
            match th["type"].as_str() {
                Some("enabled") => {
                    chat["enable_thinking"] = json!(true);
                    chat["thinking_token_budget"] = th["budget_tokens"].clone();
                }
                Some("disabled") => chat["enable_thinking"] = json!(false),
                Some("adaptive") => {}
                _ => return Err("Unsupported thinking configuration".into()),
            }
        }
        if let Some(effort) = v["output_config"]["effort"].as_str() {
            if !matches!(effort, "low" | "medium" | "high" | "xhigh" | "max") {
                return Err("Unsupported output_config.effort".into());
            }
            if v["thinking"]["type"] == "enabled" {
                chat["reasoning_effort"] = json!(effort);
            }
        }
        if let Some(cfg) = v.get("output_config")
            && cfg
                .as_object()
                .is_none_or(|o| o.keys().any(|k| k != "effort"))
        {
            return Err("Only output_config.effort is supported".into());
        }
        chat["messages"] = json!(messages_input(&v)?);
    } else {
        if !v["previous_response_id"].is_null()
            || !matches!(v["store"], Value::Null | Value::Bool(false))
            || !matches!(v["background"], Value::Null | Value::Bool(false))
        {
            return Err("Responses supports foreground requests with store=false and client-supplied history".into());
        }
        if !v["truncation"].is_null() && v["truncation"] != "disabled" {
            return Err("Only disabled truncation is supported".into());
        }
        if !v["text"]["format"].is_null() && v["text"]["format"]["type"] != "text" {
            return Err("Only plain text response format is supported".into());
        }
        if !v["include"].is_null()
            && list(&v, "include")?
                .iter()
                .any(|x| x != "reasoning.encrypted_content")
        {
            return Err("Unsupported include value".into());
        }
        if let Some(x) = v.get("max_output_tokens") {
            chat["max_tokens"] = x.clone();
        }
        if let Some(eff) = v["reasoning"]["effort"].as_str() {
            if eff == "none" {
                chat["enable_thinking"] = json!(false)
            } else {
                chat["reasoning_effort"] = json!(eff)
            }
        }
        chat["messages"] = json!(response_input(&v)?);
    }
    if let Some(choice) = v.get("tool_choice").filter(|x| !x.is_null()) {
        let typ = choice
            .as_str()
            .or_else(|| choice["type"].as_str())
            .ok_or("Invalid tool_choice")?;
        chat["tool_choice"] = match typ {
            "auto" | "none" | "required" => json!(typ),
            "any" => json!("required"),
            "tool" | "function" | "custom" => {
                json!({"type":"function","function":{"name":Wire::local(string(choice,"name")?,choice["namespace"].as_str())}})
            }
            _ => return Err("Unsupported tool_choice".into()),
        };
        if let Some(x) = choice.get("disable_parallel_tool_use") {
            chat["parallel_tool_calls"] =
                json!(!x.as_bool().ok_or("Invalid disable_parallel_tool_use")?);
        }
    }
    if kind != Kind::Count {
        chat["stream"] = json!(true);
    }
    Ok((chat, w))
}
