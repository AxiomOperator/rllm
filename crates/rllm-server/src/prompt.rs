//! Building model prompts from API requests: OpenAI chat messages become
//! the Hugging Face message shape the GGUF chat templates expect.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::error::{ApiError, ApiResult};
use crate::json_constraint::{JsonState, Schema};
use crate::output::Format;
use crate::state::AppState;

/// `tool_choice`, normalized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Named(String),
}

impl ToolChoice {
    /// Parse a Chat Completions or Responses `tool_choice` value.
    pub fn parse(v: Option<&Value>, has_tools: bool) -> ApiResult<Self> {
        let default = if has_tools { ToolChoice::Auto } else { ToolChoice::None };
        let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(default) };
        match v {
            Value::String(s) => match s.as_str() {
                "none" => Ok(ToolChoice::None),
                "auto" => Ok(default),
                "required" => Ok(ToolChoice::Required),
                other => Err(ApiError::invalid(format!("Invalid value for 'tool_choice': '{other}'."), Some("tool_choice"))),
            },
            Value::Object(o) => {
                // Chat: {"type":"function","function":{"name":…}}; Responses: {"type":"function","name":…}.
                let name = o.get("function").and_then(|f| f.get("name")).or_else(|| o.get("name")).and_then(Value::as_str);
                match (o.get("type").and_then(Value::as_str), name) {
                    (Some("function"), Some(n)) => Ok(ToolChoice::Named(n.to_owned())),
                    (Some("allowed_tools"), _) => {
                        let mode = o.get("mode").or_else(|| o.get("allowed_tools").and_then(|a| a.get("mode")));
                        Ok(if mode.and_then(Value::as_str) == Some("required") { ToolChoice::Required } else { default })
                    }
                    (Some(t), _) => Err(ApiError::invalid(format!("tool_choice type '{t}' is not supported by this server."), Some("tool_choice"))),
                    _ => Err(ApiError::invalid("Invalid 'tool_choice' object.", Some("tool_choice"))),
                }
            }
            _ => Err(ApiError::invalid("Invalid type for 'tool_choice'.", Some("tool_choice"))),
        }
    }
}

/// Requested structured output.
#[derive(Debug, Clone)]
pub enum JsonFormat {
    Object,
    Schema { name: String, schema: Value },
}

impl JsonFormat {
    /// The instruction added to the system prompt (decoding is constrained
    /// to valid JSON separately).
    fn instruction(&self) -> String {
        match self {
            JsonFormat::Object => "Respond only with a single valid JSON object. Do not add any text before or after it.".into(),
            JsonFormat::Schema { name, schema } => format!(
                "Respond only with a single JSON value that conforms to the JSON schema `{name}` below. Do not add any text before or after it.\n{}",
                schema
            ),
        }
    }

    /// The decoding constraint: any object, or the schema.
    pub fn constraint(&self) -> JsonState {
        match self {
            JsonFormat::Object => JsonState::with_schema(Arc::new(Schema::any_object())),
            JsonFormat::Schema { schema, .. } => JsonState::with_schema(Arc::new(Schema::compile(schema))),
        }
    }

    /// Parse `response_format` (Chat) or `text.format` (Responses; flat fields).
    pub fn parse(v: Option<&Value>, param: &str) -> ApiResult<Option<Self>> {
        let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(None) };
        match v.get("type").and_then(Value::as_str) {
            Some("text") | None => Ok(None),
            Some("json_object") => Ok(Some(JsonFormat::Object)),
            Some("json_schema") => {
                let spec = v.get("json_schema").unwrap_or(v);
                let schema = spec.get("schema").cloned().unwrap_or(json!({}));
                let name = spec.get("name").and_then(Value::as_str).unwrap_or("response").to_owned();
                Ok(Some(JsonFormat::Schema { name, schema }))
            }
            Some(t) => Err(ApiError::invalid(format!("Invalid value for '{param}.type': '{t}'."), Some(param))),
        }
    }
}

/// A rendered prompt plus how to parse what the model writes after it.
pub struct ChatPrompt {
    pub ids: Vec<u32>,
    /// Text appended to the prompt that the output parser must see first
    /// (a forced tool-call opening).
    pub forced: String,
    pub tools: bool,
    pub reasoning: bool,
    /// Decoding constraint for the output (JSON mode, or a forced tool
    /// call's JSON), and text that closes the output once it completes.
    pub constraint: Option<JsonState>,
    pub suffix: String,
}

/// Text of a message `content`: a string or an array of text parts.
pub fn content_text(content: Option<&Value>, param: &str) -> ApiResult<String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for (i, p) in parts.iter().enumerate() {
                let ty = p.get("type").and_then(Value::as_str).unwrap_or("");
                match ty {
                    "text" | "input_text" | "output_text" => {
                        out.push_str(p.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                    "refusal" => out.push_str(p.get("refusal").and_then(Value::as_str).unwrap_or("")),
                    "image_url" | "input_image" | "input_audio" | "file" | "input_file" => {
                        return Err(ApiError::invalid(
                            format!("Content part type '{ty}' is not supported: this model accepts text only."),
                            Some(&format!("{param}[{i}].type")),
                        ));
                    }
                    other => {
                        return Err(ApiError::invalid(format!("Invalid content part type '{other}'."), Some(&format!("{param}[{i}].type"))));
                    }
                }
            }
            Ok(out)
        }
        Some(_) => Err(ApiError::invalid("Invalid type for message content.", Some(param))),
    }
}

/// `function.arguments` as the templates want it: a JSON object.
fn arguments_object(args: Option<&Value>, param: &str) -> ApiResult<Value> {
    match args {
        None | Some(Value::Null) => Ok(json!({})),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(json!({})),
        Some(Value::String(s)) => serde_json::from_str(s)
            .map_err(|_| ApiError::invalid("Tool call arguments must be a JSON object encoded as a string.", Some(param))),
        Some(v @ Value::Object(_)) => Ok(v.clone()),
        Some(_) => Err(ApiError::invalid("Invalid type for tool call arguments.", Some(param))),
    }
}

/// OpenAI Chat Completions messages to template messages.
pub fn chat_messages(messages: &[Value]) -> ApiResult<Vec<Value>> {
    if messages.is_empty() {
        return Err(ApiError::invalid("[] is too short - 'messages'", Some("messages")));
    }
    let mut out = Vec::with_capacity(messages.len());
    for (i, m) in messages.iter().enumerate() {
        let p = format!("messages[{i}]");
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::invalid(format!("Missing required parameter: '{p}.role'."), Some(&format!("{p}.role"))))?;
        let content = content_text(m.get("content"), &format!("{p}.content"))?;
        let msg = match role {
            "system" | "developer" => json!({ "role": "system", "content": content }),
            "user" => {
                let mut v = json!({ "role": "user", "content": content });
                if let Some(n) = m.get("name") {
                    v["name"] = n.clone();
                }
                v
            }
            "assistant" => {
                let mut v = json!({ "role": "assistant", "content": content });
                if let Some(calls) = m.get("tool_calls").and_then(Value::as_array).filter(|c| !c.is_empty()) {
                    let mut tc = Vec::with_capacity(calls.len());
                    for (j, c) in calls.iter().enumerate() {
                        let f = c.get("function").unwrap_or(&Value::Null);
                        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = arguments_object(f.get("arguments"), &format!("{p}.tool_calls[{j}].function.arguments"))?;
                        tc.push(json!({ "id": c.get("id"), "type": "function", "function": { "name": name, "arguments": args } }));
                    }
                    v["tool_calls"] = Value::Array(tc);
                } else if let Some(fc) = m.get("function_call").filter(|f| !f.is_null()) {
                    let name = fc.get("name").and_then(Value::as_str).unwrap_or("");
                    let args = arguments_object(fc.get("arguments"), &format!("{p}.function_call.arguments"))?;
                    v["tool_calls"] = json!([{ "id": "call_0", "type": "function", "function": { "name": name, "arguments": args } }]);
                }
                for key in ["reasoning_content", "reasoning"] {
                    if let Some(r) = m.get(key).and_then(Value::as_str) {
                        v["reasoning_content"] = json!(r);
                    }
                }
                v
            }
            "tool" => json!({ "role": "tool", "content": content, "tool_call_id": m.get("tool_call_id") }),
            "function" => json!({ "role": "tool", "content": content, "name": m.get("name") }),
            other => {
                return Err(ApiError::invalid(
                    format!("Invalid value: '{other}'. Supported values are: 'system', 'assistant', 'user', 'tool', 'function', and 'developer'."),
                    Some(&format!("{p}.role")),
                ));
            }
        };
        out.push(msg);
    }
    Ok(out)
}

/// Chat Completions `tools` to the template shape; only function tools.
pub fn chat_tools(tools: Option<&Value>) -> ApiResult<Vec<Value>> {
    let Some(arr) = tools.and_then(Value::as_array) else { return Ok(Vec::new()) };
    arr.iter()
        .enumerate()
        .map(|(i, t)| match t.get("type").and_then(Value::as_str) {
            Some("function") => {
                let f = t.get("function").ok_or_else(|| ApiError::invalid("Missing 'function'.", Some(&format!("tools[{i}].function"))))?;
                if f.get("name").and_then(Value::as_str).is_none() {
                    return Err(ApiError::invalid("Missing required parameter: 'name'.", Some(&format!("tools[{i}].function.name"))));
                }
                Ok(json!({ "type": "function", "function": f }))
            }
            Some(other) => Err(ApiError::invalid(format!("Tool type '{other}' is not supported by this server."), Some(&format!("tools[{i}].type")))),
            None => Err(ApiError::invalid("Missing required parameter: 'type'.", Some(&format!("tools[{i}].type")))),
        })
        .collect()
}

/// Render and tokenize a chat prompt.
pub fn build(
    state: &AppState,
    mut messages: Vec<Value>,
    tools: &[Value],
    choice: &ToolChoice,
    reasoning_effort: Option<&str>,
    json_format: Option<&JsonFormat>,
) -> ApiResult<ChatPrompt> {
    if let Some(f) = json_format {
        let instruction = f.instruction();
        match messages.first_mut().filter(|m| m["role"] == "system") {
            Some(sys) => {
                let text = sys["content"].as_str().unwrap_or("").to_owned();
                sys["content"] = json!(format!("{text}\n\n{instruction}"));
            }
            None => messages.insert(0, json!({ "role": "system", "content": instruction })),
        }
    }
    if let ToolChoice::Named(n) = choice
        && !tools.iter().any(|t| t["function"]["name"] == n.as_str())
    {
        return Err(ApiError::invalid(format!("Tool choice '{n}' not found in 'tools' parameter."), Some("tool_choice")));
    }
    let use_tools = !tools.is_empty() && *choice != ToolChoice::None;
    let forced = match choice {
        _ if !use_tools => String::new(),
        ToolChoice::Required => state.format.forced_tool_prefix(None),
        ToolChoice::Named(n) => state.format.forced_tool_prefix(Some(n)),
        _ => String::new(),
    };
    // Thinking: off for effort "none"/"minimal", JSON output and forced tool
    // calls; on for other explicit efforts; otherwise the template default.
    let thinking = if json_format.is_some() || !forced.is_empty() {
        Some(false)
    } else {
        match reasoning_effort {
            Some("none" | "minimal") => Some(false),
            Some(_) => Some(true),
            None => None,
        }
    };
    let text = state
        .tok
        .apply_chat_template_json(&messages, use_tools.then_some(tools), true, thinking)
        .map_err(|e| ApiError::invalid(format!("Could not apply the chat template: {e:#}"), Some("messages")))?;
    let ids = state.tok.encode(&format!("{text}{forced}")).map_err(ApiError::from)?;
    check_context(state, ids.len())?;
    // Forced tool calls are constrained to the tools' names and parameter
    // schemas (JSON for Hermes, Gemma's own syntax for Gemma), then closed.
    let (constraint, suffix) = match choice {
        _ if forced.is_empty() => (json_format.map(JsonFormat::constraint), String::new()),
        ToolChoice::Named(n) if state.format == Format::Gemma => {
            let params = tools.iter().find(|t| t["function"]["name"] == n.as_str()).map(tool_params).unwrap_or_default();
            (Some(JsonState::with_schema(Arc::new(Schema::gemma(&params)))), "<tool_call|>".to_owned())
        }
        _ if state.format == Format::Gemma => {
            let calls: Vec<(String, Value)> =
                tools.iter().filter_map(|t| Some((t["function"]["name"].as_str()?.to_owned(), tool_params(t)))).collect();
            (Some(JsonState::with_schema(Arc::new(Schema::gemma_calls(&calls)))), "<tool_call|>".to_owned())
        }
        ToolChoice::Named(n) => {
            let params = tools.iter().find(|t| t["function"]["name"] == n.as_str()).map(tool_params).unwrap_or_default();
            (Some(JsonState::with_schema(Arc::new(Schema::compile(&params)))), "}\n</tool_call>".to_owned())
        }
        _ => {
            let calls: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "object",
                        "properties": { "name": { "const": t["function"]["name"] }, "arguments": tool_params(t) },
                        "required": ["name", "arguments"],
                        "additionalProperties": false,
                    })
                })
                .collect();
            (Some(JsonState::with_schema(Arc::new(Schema::any_of(&calls)))), "\n</tool_call>".to_owned())
        }
    };
    Ok(ChatPrompt { ids, forced, tools: use_tools, reasoning: thinking != Some(false) && json_format.is_none(), constraint, suffix })
}

/// A function tool's parameter schema (an object schema by default).
fn tool_params(tool: &Value) -> Value {
    match tool["function"].get("parameters") {
        Some(p) if p.is_object() => p.clone(),
        _ => json!({ "type": "object", "properties": {} }),
    }
}

/// 400 `context_length_exceeded` unless the prompt leaves room to generate.
pub fn check_context(state: &AppState, prompt_tokens: usize) -> ApiResult<()> {
    if prompt_tokens >= state.ctx {
        return Err(ApiError::invalid_code(
            format!(
                "This model's maximum context length is {} tokens. However, your messages resulted in {prompt_tokens} tokens. Please reduce the length of the messages.",
                state.ctx
            ),
            Some("messages"),
            "context_length_exceeded",
        ));
    }
    Ok(())
}
