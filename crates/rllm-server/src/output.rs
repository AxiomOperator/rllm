//! Turning generated text into API output: stop sequences, reasoning
//! ("thinking") blocks and tool calls, in each supported chat format.
//!
//! Text arrives in streaming pieces. Anything that might be the start of a
//! marker (`<tool_call>`, `<think>`, a stop sequence) is held back until it
//! is resolved, so clients never see half a marker.

use serde_json::Value;

/// How a model family writes reasoning and tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Qwen / Hermes: `<think>…</think>`, `<tool_call>{"name":…, "arguments":…}</tool_call>`.
    Hermes,
    /// Gemma 4: `<|channel>thought…<channel|>`, `<|tool_call>call:name{…}<tool_call|>`.
    Gemma,
}

impl Format {
    /// Pick the format from the model's chat template.
    pub fn detect(template: &str) -> Self {
        if template.contains("<|tool_call>") || template.contains("<|turn>") { Format::Gemma } else { Format::Hermes }
    }

    fn think(self) -> (&'static str, &'static str) {
        match self {
            Format::Hermes => ("<think>", "</think>"),
            Format::Gemma => ("<|channel>thought", "<channel|>"),
        }
    }

    fn tool(self) -> (&'static str, &'static str) {
        match self {
            Format::Hermes => ("<tool_call>", "</tool_call>"),
            Format::Gemma => ("<|tool_call>", "<tool_call|>"),
        }
    }

    /// Text that makes the model emit a tool call (optionally a specific
    /// one) when appended to the prompt: used for `tool_choice` `required`
    /// or a named function. The parser is fed the same text first.
    pub fn forced_tool_prefix(self, name: Option<&str>) -> String {
        match (self, name) {
            (Format::Hermes, None) => "<tool_call>\n".into(),
            (Format::Hermes, Some(n)) => format!("<tool_call>\n{{\"name\": \"{n}\", \"arguments\": "),
            (Format::Gemma, None) => "<|tool_call>call:".into(),
            (Format::Gemma, Some(n)) => format!("<|tool_call>call:{n}"),
        }
    }

    /// Text the model writes between reasoning and the answer; stripped.
    fn after_think(self, s: &str) -> &str {
        s.trim_start_matches(['\n', ' '])
    }
}

/// A parsed tool call. `arguments` is a JSON string (as in the API).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Content(String),
    Reasoning(String),
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Content,
    Reasoning,
    Tool,
}

pub struct OutputParser {
    format: Format,
    tools: bool,
    reasoning: bool,
    mode: Mode,
    buf: String,
    /// Nothing has been emitted as content yet (leading whitespace is trimmed).
    at_start: bool,
    /// Just left a reasoning block (whitespace after it is trimmed).
    after_reasoning: bool,
}

impl OutputParser {
    /// `tools`: recognize tool calls. `reasoning`: recognize reasoning blocks.
    pub fn new(format: Format, tools: bool, reasoning: bool) -> Self {
        Self { format, tools, reasoning, mode: Mode::Content, buf: String::new(), at_start: true, after_reasoning: false }
    }

    /// Plain text output (legacy completions): no markers recognized and
    /// the text is passed on verbatim (leading whitespace included).
    pub fn plain() -> Self {
        Self { at_start: false, ..Self::new(Format::Hermes, false, false) }
    }

    /// Currently inside a reasoning block (for counting reasoning tokens).
    pub fn in_reasoning(&self) -> bool {
        self.mode == Mode::Reasoning
    }

    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.mode {
                Mode::Content => {
                    let mut markers: Vec<(&str, Mode)> = Vec::new();
                    if self.reasoning {
                        markers.push((self.format.think().0, Mode::Reasoning));
                    }
                    if self.tools {
                        markers.push((self.format.tool().0, Mode::Tool));
                    }
                    let found = markers.iter().filter_map(|&(m, mode)| self.buf.find(m).map(|p| (p, m, mode))).min_by_key(|x| x.0);
                    if let Some((pos, marker, mode)) = found {
                        let before = self.buf[..pos].to_owned();
                        self.emit_content(&before, &mut out);
                        self.buf.drain(..pos + marker.len());
                        if mode == Mode::Reasoning && self.format == Format::Gemma {
                            // `<|channel>thought` is followed by a newline.
                            if self.buf.starts_with('\n') {
                                self.buf.remove(0);
                            }
                        }
                        self.mode = mode;
                        continue;
                    }
                    let keep = markers.iter().map(|&(m, _)| partial_suffix(&self.buf, m)).max().unwrap_or(0);
                    let emit = self.buf[..self.buf.len() - keep].to_owned();
                    self.emit_content(&emit, &mut out);
                    self.buf.drain(..self.buf.len() - keep);
                    break;
                }
                Mode::Reasoning => {
                    let end = self.format.think().1;
                    if let Some(pos) = self.buf.find(end) {
                        let text = self.buf[..pos].to_owned();
                        if !text.is_empty() {
                            out.push(Piece::Reasoning(text));
                        }
                        self.buf.drain(..pos + end.len());
                        self.mode = Mode::Content;
                        self.after_reasoning = true;
                        continue;
                    }
                    let keep = partial_suffix(&self.buf, end);
                    let text = self.buf[..self.buf.len() - keep].to_owned();
                    if !text.is_empty() {
                        out.push(Piece::Reasoning(text));
                    }
                    self.buf.drain(..self.buf.len() - keep);
                    break;
                }
                Mode::Tool => {
                    let end = self.format.tool().1;
                    if let Some(pos) = self.buf.find(end) {
                        let body = self.buf[..pos].to_owned();
                        self.buf.drain(..pos + end.len());
                        self.mode = Mode::Content;
                        match parse_tool_call(self.format, &body) {
                            Some(call) => out.push(Piece::ToolCall(call)),
                            None => {
                                let (s, e) = self.format.tool();
                                self.emit_content(&format!("{s}{body}{e}"), &mut out);
                            }
                        }
                        continue;
                    }
                    break;
                }
            }
        }
        out
    }

    /// End of output: flush whatever is buffered. An unterminated tool call
    /// is parsed if it is complete enough, else returned as text.
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        match self.mode {
            Mode::Content => self.emit_content(&rest, &mut out),
            Mode::Reasoning => {
                if !rest.is_empty() {
                    out.push(Piece::Reasoning(rest));
                }
            }
            Mode::Tool => match parse_tool_call(self.format, rest.trim_end()) {
                Some(call) => out.push(Piece::ToolCall(call)),
                None => {
                    let s = self.format.tool().0;
                    self.emit_content(&format!("{s}{rest}"), &mut out);
                }
            },
        }
        self.mode = Mode::Content;
        out
    }

    fn emit_content(&mut self, text: &str, out: &mut Vec<Piece>) {
        let mut t = text;
        if self.after_reasoning {
            t = self.format.after_think(t);
            if !t.is_empty() {
                self.after_reasoning = false;
            }
        }
        if self.at_start {
            t = t.trim_start();
        }
        if t.is_empty() {
            return;
        }
        self.at_start = false;
        match out.last_mut() {
            Some(Piece::Content(prev)) => prev.push_str(t),
            _ => out.push(Piece::Content(t.to_owned())),
        }
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `marker`
/// (text to hold back because the marker may be starting).
fn partial_suffix(s: &str, marker: &str) -> usize {
    (1..marker.len().min(s.len() + 1))
        .rev()
        .find(|&n| s.is_char_boundary(s.len() - n) && marker.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}

fn parse_tool_call(format: Format, body: &str) -> Option<ToolCall> {
    match format {
        Format::Hermes => {
            let v: Value = serde_json::from_str(body.trim()).ok()?;
            let name = v.get("name")?.as_str()?.to_owned();
            let arguments = match v.get("arguments") {
                Some(Value::String(s)) => s.clone(),
                Some(a) => a.to_string(),
                None => "{}".into(),
            };
            Some(ToolCall { name, arguments })
        }
        Format::Gemma => {
            let body = body.trim();
            let rest = body.strip_prefix("call:")?;
            let brace = rest.find('{')?;
            let name = rest[..brace].trim().to_owned();
            let mut p = GemmaArgs { s: &rest[brace..] };
            let args = p.value()?;
            if !args.is_object() || !p.s.trim().is_empty() {
                return None;
            }
            Some(ToolCall { name, arguments: args.to_string() })
        }
    }
}

/// Parser for Gemma 4's argument syntax: JSON-like, with bare or
/// `<|"|>`-quoted keys and `<|"|>`-quoted strings.
struct GemmaArgs<'a> {
    s: &'a str,
}

const GQ: &str = "<|\"|>";

impl GemmaArgs<'_> {
    fn ws(&mut self) {
        self.s = self.s.trim_start();
    }

    fn eat(&mut self, t: &str) -> bool {
        self.ws();
        if let Some(r) = self.s.strip_prefix(t) {
            self.s = r;
            true
        } else {
            false
        }
    }

    fn quoted(&mut self) -> Option<String> {
        let r = self.s.strip_prefix(GQ)?;
        let end = r.find(GQ)?;
        let v = r[..end].to_owned();
        self.s = &r[end + GQ.len()..];
        Some(v)
    }

    fn value(&mut self) -> Option<Value> {
        self.ws();
        if self.s.starts_with(GQ) {
            return self.quoted().map(Value::String);
        }
        if self.eat("{") {
            let mut map = serde_json::Map::new();
            if self.eat("}") {
                return Some(Value::Object(map));
            }
            loop {
                self.ws();
                let key = if self.s.starts_with(GQ) {
                    self.quoted()?
                } else {
                    let end = self.s.find(':')?;
                    let k = self.s[..end].trim().to_owned();
                    self.s = &self.s[end..];
                    k
                };
                if !self.eat(":") {
                    return None;
                }
                map.insert(key, self.value()?);
                if self.eat(",") {
                    continue;
                }
                if self.eat("}") {
                    return Some(Value::Object(map));
                }
                return None;
            }
        }
        if self.eat("[") {
            let mut items = Vec::new();
            if self.eat("]") {
                return Some(Value::Array(items));
            }
            loop {
                items.push(self.value()?);
                if self.eat(",") {
                    continue;
                }
                if self.eat("]") {
                    return Some(Value::Array(items));
                }
                return None;
            }
        }
        // Scalar: up to the next delimiter.
        let end = self.s.find([',', '}', ']']).unwrap_or(self.s.len());
        let tok = self.s[..end].trim();
        self.s = &self.s[end..];
        match tok {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            "null" => Some(Value::Null),
            t => serde_json::from_str::<serde_json::Number>(t).ok().map(Value::Number).or_else(|| (!t.is_empty()).then(|| Value::String(t.to_owned()))),
        }
    }
}

/// Stop sequences: emits text up to (not including) the first stop string,
/// holding back any suffix that could still become one.
pub struct StopMatcher {
    stops: Vec<String>,
    buf: String,
    stopped: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), buf: String::new(), stopped: false }
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Returns the text that is safe to pass on.
    pub fn push(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.buf.push_str(text);
        if let Some(pos) = self.stops.iter().filter_map(|s| self.buf.find(s.as_str())).min() {
            self.stopped = true;
            let out = self.buf[..pos].to_owned();
            self.buf.clear();
            return out;
        }
        let keep = self.stops.iter().map(|s| partial_suffix(&self.buf, s)).max().unwrap_or(0);
        let out = self.buf[..self.buf.len() - keep].to_owned();
        self.buf.drain(..self.buf.len() - keep);
        out
    }

    /// End of output: anything held back was not a stop sequence after all.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(p: &mut OutputParser, chunks: &[&str]) -> Vec<Piece> {
        let mut out: Vec<Piece> = Vec::new();
        for c in chunks {
            out.extend(p.push(c));
        }
        out.extend(p.finish());
        // Merge adjacent pieces of the same kind for easy comparison.
        let mut merged: Vec<Piece> = Vec::new();
        for piece in out {
            match (merged.last_mut(), &piece) {
                (Some(Piece::Content(a)), Piece::Content(b)) => a.push_str(b),
                (Some(Piece::Reasoning(a)), Piece::Reasoning(b)) => a.push_str(b),
                _ => merged.push(piece),
            }
        }
        merged
    }

    #[test]
    fn hermes_reasoning_and_tool_call_split_anywhere() {
        let text = "<think>\nLet me check.\n</think>\n\nSure.<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        for split in 1..text.len() {
            if !text.is_char_boundary(split) {
                continue;
            }
            let mut p = OutputParser::new(Format::Hermes, true, true);
            let out = run(&mut p, &[&text[..split], &text[split..]]);
            assert_eq!(
                out,
                vec![
                    Piece::Reasoning("\nLet me check.\n".into()),
                    Piece::Content("Sure.".into()),
                    Piece::ToolCall(ToolCall { name: "get_weather".into(), arguments: "{\"city\":\"Paris\"}".into() }),
                ],
                "split at {split}"
            );
        }
    }

    #[test]
    fn gemma_tool_call_arguments() {
        let mut p = OutputParser::new(Format::Gemma, true, true);
        let out = run(
            &mut p,
            &["<|tool_call>call:get_weather{location:<|\"|>Boston, MA<|\"|>,", "days:3,opts:{<|\"|>metric<|\"|>:true},tags:[<|\"|>a<|\"|>]}<tool_call|>"],
        );
        let Piece::ToolCall(call) = &out[0] else { panic!("{out:?}") };
        assert_eq!(call.name, "get_weather");
        let v: Value = serde_json::from_str(&call.arguments).unwrap();
        assert_eq!(v, serde_json::json!({"location": "Boston, MA", "days": 3, "opts": {"metric": true}, "tags": ["a"]}));
    }

    #[test]
    fn markers_disabled_pass_through() {
        let mut p = OutputParser::plain();
        assert_eq!(run(&mut p, &[" a <tool_call> b"]), vec![Piece::Content(" a <tool_call> b".into())]);
    }

    #[test]
    fn stop_sequences_across_chunks() {
        let mut s = StopMatcher::new(vec!["END".into(), "\n\n".into()]);
        let mut out = String::new();
        for c in ["Hello E", "N", "x then EN", "D more"] {
            out.push_str(&s.push(c));
        }
        assert!(s.stopped());
        assert_eq!(out, "Hello ENx then ");
    }
}
