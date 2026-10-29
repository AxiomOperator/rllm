//! Incremental JSON validation for constrained decoding (`response_format`
//! `json_object` / `json_schema`, Responses `text.format`, forced tool calls).
//!
//! [`JsonState`] accepts text one character at a time and rejects anything
//! that can't be the prefix of a JSON document matching a [`Schema`]. The
//! engine uses it to skip candidate tokens that would break the output, and
//! stops once the top-level value is complete.
//!
//! Schemas cover the structural part of JSON Schema: `type` (and type
//! lists), `properties` / `required` / `additionalProperties`, `items` /
//! `prefixItems` / `minItems` / `maxItems`, `enum`, `const`, `anyOf` /
//! `oneOf`, `allOf` (merged), `nullable` and local `$ref`s (`#/$defs/…`,
//! recursive ones too). Value constraints (`pattern`, `minimum`,
//! `minLength`, `format`, …) are not enforced. As in OpenAI's structured
//! outputs, an object with `properties` and no `additionalProperties` allows
//! no other keys. Since the parse can be ambiguous (`anyOf`), the state is a
//! set of alternative parser stacks.
//!
//! A second dialect covers Gemma 4 tool calls, `name{key:<|"|>text<|"|>,n:1}`:
//! bare keys, strings between `<|"|>` tokens (no escapes), and the function
//! name before the arguments.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

type NodeId = usize;

const ANY: NodeId = 0;
const ANY_OBJECT: NodeId = 1;
const ANY_ARRAY: NodeId = 2;
const ANY_LITERAL: NodeId = 3;
const NEVER: NodeId = 4;

#[derive(Debug)]
enum Node {
    Any,
    Object { props: Vec<(String, NodeId)>, required: Vec<String>, additional: Option<NodeId> },
    Array { prefix: Vec<NodeId>, items: NodeId, min: usize, max: usize },
    /// A string; `choices` limits it to these (decoded) values.
    String { choices: Option<Arc<Vec<String>>> },
    Number { integer: bool },
    /// One of these exact JSON texts (`true`, `null`, `42`, …).
    Literals(Vec<String>),
    AnyOf(Vec<NodeId>),
    /// Exactly `text`, then a value of `then` (a Gemma call's function name).
    Prefixed { text: String, then: NodeId },
}

/// Gemma 4's string delimiter (one special token).
pub const GEMMA_QUOTE: &str = "<|\"|>";
/// What a Gemma quote becomes inside the validator.
const QUOTE: char = '\u{1}';

/// A compiled JSON schema.
#[derive(Debug)]
pub struct Schema {
    nodes: Vec<Node>,
    root: NodeId,
    /// Gemma 4 argument syntax instead of JSON.
    gemma: bool,
}

impl Schema {
    fn builtin() -> Vec<Node> {
        vec![
            Node::Any,
            Node::Object { props: Vec::new(), required: Vec::new(), additional: Some(ANY) },
            Node::Array { prefix: Vec::new(), items: ANY, min: 0, max: usize::MAX },
            Node::Literals(vec!["true".into(), "false".into(), "null".into()]),
            Node::AnyOf(Vec::new()),
        ]
    }

    /// Any JSON object (`json_object` mode).
    pub fn any_object() -> Self {
        Self { nodes: Self::builtin(), root: ANY_OBJECT, gemma: false }
    }

    /// Any JSON object or array.
    pub fn any_container() -> Self {
        let mut nodes = Self::builtin();
        nodes.push(Node::AnyOf(vec![ANY_OBJECT, ANY_ARRAY]));
        let root = nodes.len() - 1;
        Self { nodes, root, gemma: false }
    }

    pub fn compile(schema: &Value) -> Self {
        let mut c = Compiler { doc: schema, nodes: Self::builtin(), refs: HashMap::new(), depth: 0 };
        let root = c.compile(schema);
        Self { nodes: c.nodes, root, gemma: false }
    }

    /// A schema in Gemma 4 argument syntax (a named tool call's arguments).
    pub fn gemma(schema: &Value) -> Self {
        Self { gemma: true, ..Self::compile(schema) }
    }

    /// Any of these Gemma 4 tool calls: `name{arguments}` for each
    /// (name, parameter schema).
    pub fn gemma_calls(tools: &[(String, Value)]) -> Self {
        let mut c = Compiler { doc: &Value::Null, nodes: Self::builtin(), refs: HashMap::new(), depth: 0 };
        let mut alts = Vec::new();
        for (name, params) in tools {
            c.doc = params;
            c.refs.clear();
            let then = c.compile(params);
            alts.push(c.push(Node::Prefixed { text: name.clone(), then }));
        }
        let root = c.push(Node::AnyOf(alts));
        Self { nodes: c.nodes, root, gemma: true }
    }

    /// One of several schemas.
    pub fn any_of(schemas: &[Value]) -> Self {
        let mut c = Compiler { doc: &Value::Null, nodes: Self::builtin(), refs: HashMap::new(), depth: 0 };
        let mut alts = Vec::new();
        for s in schemas {
            c.doc = s;
            c.refs.clear();
            alts.push(c.compile(s));
        }
        let root = c.push(Node::AnyOf(alts));
        Self { nodes: c.nodes, root, gemma: false }
    }
}

struct Compiler<'a> {
    doc: &'a Value,
    nodes: Vec<Node>,
    refs: HashMap<String, NodeId>,
    depth: usize,
}

impl Compiler<'_> {
    fn push(&mut self, n: Node) -> NodeId {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    fn literal(&mut self, values: &[Value]) -> NodeId {
        let strings: Vec<String> = values.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect();
        let others: Vec<String> = values.iter().filter(|v| !v.is_string()).map(|v| v.to_string()).collect();
        let s = (!strings.is_empty()).then(|| self.push(Node::String { choices: Some(Arc::new(strings)) }));
        let o = (!others.is_empty()).then(|| self.push(Node::Literals(others)));
        match (s, o) {
            (Some(a), Some(b)) => self.push(Node::AnyOf(vec![a, b])),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => NEVER,
        }
    }

    fn resolve(&self, pointer: &str) -> Option<&Value> {
        let path = pointer.strip_prefix('#')?;
        if path.is_empty() {
            return Some(self.doc);
        }
        self.doc.pointer(path)
    }

    fn compile(&mut self, s: &Value) -> NodeId {
        // Guard against pathological nesting.
        if self.depth > 64 {
            return ANY;
        }
        self.depth += 1;
        let id = self.compile_inner(s);
        self.depth -= 1;
        id
    }

    fn compile_inner(&mut self, s: &Value) -> NodeId {
        let o = match s {
            Value::Bool(true) => return ANY,
            Value::Bool(false) => return NEVER,
            Value::Object(o) => o,
            _ => return ANY,
        };
        if let Some(r) = o.get("$ref").and_then(Value::as_str) {
            if let Some(&id) = self.refs.get(r) {
                return id;
            }
            let Some(target) = self.resolve(r).cloned() else { return ANY };
            // Placeholder first, so recursive references resolve to it.
            let id = self.push(Node::AnyOf(Vec::new()));
            self.refs.insert(r.to_owned(), id);
            let t = self.compile(&target);
            self.nodes[id] = Node::AnyOf(vec![t]);
            return id;
        }
        let base = if let Some(v) = o.get("const") {
            self.literal(std::slice::from_ref(v))
        } else if let Some(Value::Array(vals)) = o.get("enum") {
            self.literal(vals)
        } else if let Some(Value::Array(alts)) = o.get("anyOf").or_else(|| o.get("oneOf")) {
            let ids = alts.iter().map(|a| self.compile(a)).collect();
            self.push(Node::AnyOf(ids))
        } else if let Some(Value::Array(parts)) = o.get("allOf") {
            let mut merged = serde_json::Map::new();
            for p in parts {
                let p = match p.get("$ref").and_then(Value::as_str) {
                    Some(r) => self.resolve(r).cloned().unwrap_or(Value::Null),
                    None => p.clone(),
                };
                let Value::Object(p) = p else { continue };
                for (k, v) in p {
                    match (k.as_str(), merged.get_mut(&k), &v) {
                        ("properties", Some(Value::Object(m)), Value::Object(add)) => m.extend(add.clone()),
                        ("required", Some(Value::Array(m)), Value::Array(add)) => m.extend(add.iter().cloned()),
                        _ => {
                            merged.insert(k, v);
                        }
                    }
                }
            }
            for (k, v) in o {
                if k != "allOf" {
                    merged.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
            self.compile(&Value::Object(merged))
        } else {
            match o.get("type") {
                Some(Value::String(t)) => self.typed(t, o),
                Some(Value::Array(ts)) => {
                    let ids = ts.iter().filter_map(Value::as_str).map(|t| self.typed(t, o)).collect();
                    self.push(Node::AnyOf(ids))
                }
                _ if o.contains_key("properties") => self.typed("object", o),
                _ if o.contains_key("items") || o.contains_key("prefixItems") => self.typed("array", o),
                _ => ANY,
            }
        };
        if o.get("nullable").and_then(Value::as_bool) == Some(true) {
            let null = self.push(Node::Literals(vec!["null".into()]));
            return self.push(Node::AnyOf(vec![base, null]));
        }
        base
    }

    fn typed(&mut self, t: &str, o: &serde_json::Map<String, Value>) -> NodeId {
        let count = |k: &str| o.get(k).and_then(Value::as_u64).map(|v| v as usize);
        match t {
            "object" => {
                let props: Vec<(String, NodeId)> = match o.get("properties") {
                    Some(Value::Object(p)) => p.iter().map(|(k, v)| (k.clone(), self.compile(v))).collect(),
                    _ => Vec::new(),
                };
                let required = match o.get("required") {
                    Some(Value::Array(r)) => r.iter().filter_map(Value::as_str).map(str::to_owned).collect(),
                    _ => Vec::new(),
                };
                let additional = match o.get("additionalProperties") {
                    None => (!o.contains_key("properties")).then_some(ANY),
                    Some(Value::Bool(false)) => None,
                    Some(Value::Bool(true)) => Some(ANY),
                    Some(s) => Some(self.compile(s)),
                };
                self.push(Node::Object { props, required, additional })
            }
            "array" => {
                let prefix = match o.get("prefixItems") {
                    Some(Value::Array(p)) => p.iter().map(|s| self.compile(s)).collect(),
                    _ => Vec::new(),
                };
                let items = match o.get("items") {
                    Some(Value::Array(p)) if prefix.is_empty() => {
                        // Draft-4 tuple form.
                        let ids: Vec<NodeId> = p.iter().map(|s| self.compile(s)).collect();
                        let n = ids.len();
                        return self.push(Node::Array { prefix: ids, items: NEVER, min: count("minItems").unwrap_or(0), max: n });
                    }
                    Some(s) => self.compile(s),
                    None => ANY,
                };
                self.push(Node::Array { prefix, items, min: count("minItems").unwrap_or(0), max: count("maxItems").unwrap_or(usize::MAX) })
            }
            "string" => self.push(Node::String { choices: None }),
            "integer" => self.push(Node::Number { integer: true }),
            "number" => self.push(Node::Number { integer: false }),
            "boolean" => self.push(Node::Literals(vec!["true".into(), "false".into()])),
            "null" => self.push(Node::Literals(vec!["null".into()])),
            _ => ANY,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Num {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    E,
    ESign,
    Exp,
}

impl Num {
    fn can_end(self) -> bool {
        matches!(self, Num::Zero | Num::Int | Num::Frac | Num::Exp)
    }

    fn step(self, c: char, integer: bool) -> Option<Num> {
        use Num::*;
        if integer && matches!(c, '.' | 'e' | 'E') {
            return None;
        }
        Some(match (self, c) {
            (Minus, '0') => Zero,
            (Minus, '1'..='9') => Int,
            (Zero, '.') | (Int, '.') => Dot,
            (Zero, 'e' | 'E') | (Int, 'e' | 'E') | (Frac, 'e' | 'E') => E,
            (Int, '0'..='9') => Int,
            (Dot, '0'..='9') | (Frac, '0'..='9') => Frac,
            (E, '+' | '-') => ESign,
            (E, '0'..='9') | (ESign, '0'..='9') | (Exp, '0'..='9') => Exp,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ObjSt {
    /// After `{`: a key or `}`.
    First,
    /// After `,`: a key.
    Key,
    /// After a key: `:`.
    Colon(String),
    /// After a value: `,` or `}`.
    After,
}

#[derive(Debug, Clone)]
enum Frame {
    /// Expecting a value of this node.
    Value(NodeId),
    Obj { node: NodeId, seen: Vec<String>, st: ObjSt },
    /// `count` elements started so far; `open`: right after `[`.
    Arr { node: NodeId, count: usize, open: bool },
    /// Inside a string. `buf` holds the decoded text when it matters (keys,
    /// choices). `esc`: 0 normal, 1 after `\`, 2..=5 in `\uXXXX`. `bare`: a
    /// Gemma key, ended by the `:` after it.
    Str { key: bool, bare: bool, choices: Option<Arc<Vec<String>>>, buf: String, esc: u8, hex: u32 },
    Num { st: Num, integer: bool },
    Lit { node: NodeId, text: String },
    /// Inside a [`Node::Prefixed`] text; `pos` bytes matched.
    Prefix { node: NodeId, pos: usize },
}

#[derive(Debug, Clone)]
struct Stack {
    frames: Vec<Frame>,
    done: bool,
    /// Consecutive whitespace outside strings (bounded to stop models from
    /// padding with newlines forever).
    ws_run: usize,
}

const MAX_WS_RUN: usize = 16;
/// Upper bound on parallel parses (ambiguous `anyOf`s).
const MAX_ALTS: usize = 32;

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// Validator state: a schema and the possible parses of the text so far.
#[derive(Debug, Clone)]
pub struct JsonState {
    schema: Arc<Schema>,
    alts: Vec<Stack>,
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonState {
    /// Any JSON object or array.
    pub fn new() -> Self {
        Self::with_schema(Arc::new(Schema::any_container()))
    }

    pub fn with_schema(schema: Arc<Schema>) -> Self {
        let root = schema.root;
        Self { schema, alts: vec![Stack { frames: vec![Frame::Value(root)], done: false, ws_run: 0 }] }
    }

    pub fn is_done(&self) -> bool {
        self.alts.iter().any(|s| s.done)
    }

    /// Whether a special token with this text may appear (Gemma's quote).
    pub fn allows_special(&self, text: &str) -> bool {
        self.schema.gemma && text == GEMMA_QUOTE
    }

    /// Feed `text`; false (and a partly updated state) if it can't continue
    /// a valid document. Callers try candidates on a clone.
    pub fn feed(&mut self, text: &str) -> bool {
        let mapped;
        let text = if self.schema.gemma && text.contains(GEMMA_QUOTE) {
            mapped = text.replace(GEMMA_QUOTE, &QUOTE.to_string());
            mapped.as_str()
        } else {
            text
        };
        for c in text.chars() {
            let mut next = Vec::new();
            for s in self.alts.drain(..) {
                push(&self.schema, s, c, &mut next);
            }
            next.truncate(MAX_ALTS);
            self.alts = next;
            if self.alts.is_empty() {
                return false;
            }
        }
        true
    }
}

fn value_done(s: &mut Stack) {
    s.frames.pop();
    if s.frames.is_empty() {
        s.done = true;
    }
}

fn push(schema: &Schema, mut s: Stack, c: char, out: &mut Vec<Stack>) {
    // Whitespace is content here, not padding.
    let in_string = matches!(s.frames.last(), Some(Frame::Str { .. } | Frame::Prefix { .. }));
    // Numbers and literals end at the first character that can't extend
    // them; that character then goes to the enclosing value.
    match s.frames.last_mut() {
        Some(Frame::Num { st, integer }) => {
            if let Some(n) = st.step(c, *integer) {
                *st = n;
                out.push(s);
                return;
            }
            if !st.can_end() {
                return;
            }
            value_done(&mut s);
        }
        Some(Frame::Lit { node, text }) => {
            let Node::Literals(texts) = &schema.nodes[*node] else { return };
            let complete = texts.iter().any(|t| t == text);
            text.push(c);
            if texts.iter().any(|t| t.starts_with(text.as_str())) {
                // Nothing longer possible (`true`, `null`): done right away.
                if !texts.iter().any(|t| t.len() > text.len() && t.starts_with(text.as_str())) {
                    value_done(&mut s);
                }
                out.push(s);
                return;
            }
            if !complete {
                return;
            }
            value_done(&mut s);
        }
        _ => {}
    }
    if is_ws(c) && !in_string {
        // Gemma's format has no padding (its template writes none).
        if schema.gemma {
            return;
        }
        if s.done && !s.frames.is_empty() {
            return;
        }
        if s.done {
            // Whitespace that ended a top-level number: accept, nothing more.
            if s.ws_run == 0 {
                s.ws_run = MAX_WS_RUN;
                out.push(s);
            }
            return;
        }
        s.ws_run += 1;
        if s.ws_run <= MAX_WS_RUN {
            out.push(s);
        }
        return;
    }
    if !in_string {
        s.ws_run = 0;
    }
    let Some(top) = s.frames.last_mut() else { return };
    match top {
        Frame::Value(node) => {
            let node = *node;
            s.frames.pop();
            start_value(schema, s, node, c, out, 0);
        }
        Frame::Obj { node, seen, st } => {
            let Node::Object { props, required, additional } = &schema.nodes[*node] else { return };
            match (st.clone(), c) {
                (ObjSt::First, '}') | (ObjSt::After, '}') => {
                    if required.iter().all(|r| seen.contains(r)) {
                        value_done(&mut s);
                        out.push(s);
                    }
                }
                (ObjSt::First | ObjSt::Key, c) if c == '"' || (schema.gemma && is_key_char(c)) => {
                    let choices = if additional.is_some() {
                        None
                    } else {
                        let left: Vec<String> = props.iter().map(|p| p.0.clone()).filter(|k| !seen.contains(k)).collect();
                        if left.is_empty() {
                            return;
                        }
                        Some(Arc::new(left))
                    };
                    let bare = schema.gemma;
                    s.frames.push(Frame::Str { key: true, bare, choices, buf: String::new(), esc: 0, hex: 0 });
                    if bare {
                        // The character is the key's first.
                        push(schema, s, c, out);
                    } else {
                        out.push(s);
                    }
                }
                (ObjSt::Colon(key), ':') => {
                    start_member(schema, &mut s, key);
                    out.push(s);
                }
                (ObjSt::After, ',') => {
                    let more = additional.is_some() || props.iter().any(|p| !seen.contains(&p.0));
                    if more {
                        *st = ObjSt::Key;
                        out.push(s);
                    }
                }
                _ => {}
            }
        }
        Frame::Arr { node, count, open } => {
            let Node::Array { prefix, items, min, max } = &schema.nodes[*node] else { return };
            let elem = |i: usize| prefix.get(i).copied().unwrap_or(*items);
            match c {
                ']' if *count >= *min => {
                    value_done(&mut s);
                    out.push(s);
                }
                ',' if !*open && *count < *max => {
                    *count += 1;
                    let e = elem(*count - 1);
                    s.frames.push(Frame::Value(e));
                    out.push(s);
                }
                _ if *open && *max > 0 => {
                    *open = false;
                    *count = 1;
                    let e = elem(0);
                    s.frames.push(Frame::Value(e));
                    push(schema, s, c, out);
                }
                _ => {}
            }
        }
        Frame::Str { key, bare: true, choices, buf, .. } => {
            debug_assert!(*key);
            if c == ':' {
                if buf.is_empty() || choices.as_ref().is_some_and(|ch| !ch.iter().any(|x| x == buf)) {
                    return;
                }
                let key = std::mem::take(buf);
                s.frames.pop();
                start_member(schema, &mut s, key);
                out.push(s);
                return;
            }
            if !is_key_char(c) {
                return;
            }
            buf.push(c);
            if let Some(ch) = choices
                && !ch.iter().any(|x| x.starts_with(buf.as_str()))
            {
                return;
            }
            out.push(s);
        }
        Frame::Str { key, choices, buf, .. } if schema.gemma => {
            // Gemma strings: no escapes, anything up to the closing quote.
            let track = *key || choices.is_some();
            if c == QUOTE {
                if choices.as_ref().is_some_and(|ch| !ch.iter().any(|x| x == buf)) {
                    return;
                }
                s.frames.pop();
                if s.frames.is_empty() {
                    s.done = true;
                }
                out.push(s);
                return;
            }
            if track {
                buf.push(c);
            }
            if let Some(ch) = choices
                && !ch.iter().any(|x| x.starts_with(buf.as_str()))
            {
                return;
            }
            out.push(s);
        }
        Frame::Prefix { node, pos } => {
            let Node::Prefixed { text, then } = &schema.nodes[*node] else { return };
            if !text[*pos..].starts_with(c) {
                return;
            }
            *pos += c.len_utf8();
            if *pos == text.len() {
                let then = *then;
                s.frames.pop();
                s.frames.push(Frame::Value(then));
            }
            out.push(s);
        }
        Frame::Str { key, choices, buf, esc, hex, .. } => {
            let track = *key || choices.is_some();
            let add = |buf: &mut String, ch: char| {
                if track {
                    buf.push(ch);
                }
            };
            match *esc {
                0 => match c {
                    '"' => {
                        if choices.as_ref().is_some_and(|ch| !ch.iter().any(|x| x == buf)) {
                            return;
                        }
                        let (is_key, text) = (*key, std::mem::take(buf));
                        s.frames.pop();
                        if is_key {
                            if let Some(Frame::Obj { st, .. }) = s.frames.last_mut() {
                                *st = ObjSt::Colon(text);
                            }
                        } else if s.frames.is_empty() {
                            s.done = true;
                        }
                        out.push(s);
                        return;
                    }
                    '\\' => *esc = 1,
                    c if (c as u32) < 0x20 => return,
                    c => add(buf, c),
                },
                1 => {
                    let ch = match c {
                        '"' => '"',
                        '\\' => '\\',
                        '/' => '/',
                        'b' => '\u{8}',
                        'f' => '\u{c}',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'u' => {
                            *esc = 2;
                            *hex = 0;
                            out.push(s);
                            return;
                        }
                        _ => return,
                    };
                    *esc = 0;
                    add(buf, ch);
                }
                _ => {
                    let Some(d) = c.to_digit(16) else { return };
                    *hex = *hex * 16 + d;
                    if *esc == 5 {
                        *esc = 0;
                        // Surrogate halves don't decode alone; good enough for matching.
                        add(buf, char::from_u32(*hex).unwrap_or('\u{fffd}'));
                    } else {
                        *esc += 1;
                    }
                }
            }
            if let Some(ch) = choices
                && !ch.iter().any(|x| x.starts_with(buf.as_str()))
            {
                return;
            }
            out.push(s);
        }
        Frame::Num { .. } | Frame::Lit { .. } => unreachable!("handled above"),
    }
}

/// Characters of a bare Gemma key.
fn is_key_char(c: char) -> bool {
    !is_ws(c) && !matches!(c, ':' | ',' | '{' | '}' | '[' | ']' | '"' | QUOTE)
}

/// After an object key and its `:`: record the key, expect its value.
fn start_member(schema: &Schema, s: &mut Stack, key: String) {
    let Some(Frame::Obj { node, seen, st }) = s.frames.last_mut() else { return };
    let Node::Object { props, additional, .. } = &schema.nodes[*node] else { return };
    let value = props.iter().find(|p| p.0 == key).map(|p| p.1).or(*additional).unwrap_or(NEVER);
    seen.push(key);
    *st = ObjSt::After;
    s.frames.push(Frame::Value(value));
}

/// Start a value of `node` with character `c` on top of stack `s`.
fn start_value(schema: &Schema, mut s: Stack, node: NodeId, c: char, out: &mut Vec<Stack>, depth: usize) {
    if depth > 32 {
        return;
    }
    let quote = if schema.gemma { QUOTE } else { '"' };
    let string = |choices: Option<Arc<Vec<String>>>| Frame::Str { key: false, bare: false, choices, buf: String::new(), esc: 0, hex: 0 };
    let frame = match &schema.nodes[node] {
        Node::AnyOf(alts) => {
            for &a in alts {
                start_value(schema, s.clone(), a, c, out, depth + 1);
            }
            return;
        }
        Node::Any => match c {
            '{' => Frame::Obj { node: ANY_OBJECT, seen: Vec::new(), st: ObjSt::First },
            '[' => Frame::Arr { node: ANY_ARRAY, count: 0, open: true },
            c if c == quote => string(None),
            '-' | '0'..='9' => Frame::Num { st: Num::Minus.step(c, false).unwrap_or(Num::Minus), integer: false },
            _ => return start_value(schema, s, ANY_LITERAL, c, out, depth + 1),
        },
        Node::Object { .. } if c == '{' => Frame::Obj { node, seen: Vec::new(), st: ObjSt::First },
        Node::Array { .. } if c == '[' => Frame::Arr { node, count: 0, open: true },
        Node::String { choices } if c == quote => string(choices.clone()),
        Node::Number { integer } if c == '-' || c.is_ascii_digit() => {
            Frame::Num { st: if c == '-' { Num::Minus } else { Num::Minus.step(c, false).unwrap() }, integer: *integer }
        }
        Node::Literals(texts) if texts.iter().any(|t| t.starts_with(c)) => Frame::Lit { node, text: c.to_string() },
        Node::Prefixed { text, then } if text.starts_with(c) => {
            if text.len() == c.len_utf8() {
                Frame::Value(*then)
            } else {
                Frame::Prefix { node, pos: c.len_utf8() }
            }
        }
        _ => return,
    };
    s.frames.push(frame);
    out.push(s);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn accepts(s: &str) -> bool {
        JsonState::new().feed(s)
    }

    fn complete(s: &str) -> bool {
        let mut j = JsonState::new();
        j.feed(s) && j.is_done()
    }

    fn schema_state(schema: Value) -> JsonState {
        JsonState::with_schema(Arc::new(Schema::compile(&schema)))
    }

    fn matches(schema: &Value, text: &str) -> bool {
        let mut j = schema_state(schema.clone());
        j.feed(text) && j.is_done()
    }

    fn prefix_ok(schema: &Value, text: &str) -> bool {
        schema_state(schema.clone()).feed(text)
    }

    #[test]
    fn valid_documents() {
        assert!(complete(r#"{"a": 1, "b": [true, false, null, -0.5e+3, "x\né"], "c": {}}"#));
        assert!(complete("[]"));
        assert!(complete("{ }"));
        assert!(complete(r#"[1,2,{"k":"v"}]"#));
    }

    #[test]
    fn prefixes_and_rejections() {
        assert!(accepts(r#"{"a": [1, 2"#));
        assert!(!complete(r#"{"a": [1, 2"#));
        assert!(!accepts("hello"));
        assert!(!accepts("```json"));
        assert!(!accepts(r#"{"a" 1}"#));
        assert!(!accepts(r#"{"a": 01}"#));
        assert!(!accepts(r#"{"a": tru e}"#));
        assert!(!accepts(r#"{"a": "\q"}"#));
        assert!(!accepts("{}x"));
        assert!(!accepts(&format!("{{{}", " ".repeat(40))));
    }

    #[test]
    fn object_schema() {
        let s = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                "kind": {"enum": ["cat", "dog", 3]}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        });
        assert!(matches(&s, r#"{"name": "Rex", "age": 3}"#));
        assert!(matches(&s, r#"{"age": 3, "name": "R\"x", "tags": ["a", "b"], "kind": "dog"}"#));
        assert!(matches(&s, r#"{"name": "a", "age": -1, "kind": 3}"#));
        // Missing required key, unknown key, wrong types, enum, too many items.
        assert!(!matches(&s, r#"{"name": "Rex"}"#));
        assert!(!prefix_ok(&s, r#"{"nam": 1"#));
        assert!(!prefix_ok(&s, r#"{"color""#));
        assert!(!prefix_ok(&s, r#"{"age": 3.5"#));
        assert!(!prefix_ok(&s, r#"{"age": "3""#));
        assert!(!prefix_ok(&s, r#"{"kind": "cow"#));
        assert!(!prefix_ok(&s, r#"{"kind": 4"#));
        assert!(!prefix_ok(&s, r#"{"tags": ["a", "b", "#));
        // Duplicate keys.
        assert!(!prefix_ok(&s, r#"{"age": 1, "age""#));
        // Top level must follow the schema too.
        assert!(!prefix_ok(&s, "["));
    }

    #[test]
    fn any_of_refs_and_nullable() {
        let s = json!({
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {"v": {"type": "number"}, "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}},
                    "required": ["v", "next"],
                    "additionalProperties": false
                }
            },
            "$ref": "#/$defs/node"
        });
        assert!(matches(&s, r#"{"v": 1, "next": {"v": 2.5, "next": null}}"#));
        assert!(!prefix_ok(&s, r#"{"v": 1, "next": 3"#));

        let n = json!({"type": ["string", "null"]});
        assert!(matches(&n, "null"));
        assert!(matches(&n, r#""x""#));
        assert!(!prefix_ok(&n, "1"));

        let u = json!({"anyOf": [
            {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "boolean"}}, "required": ["b"]}
        ]});
        assert!(matches(&u, r#"{"a": 1}"#));
        assert!(matches(&u, r#"{"b": true}"#));
        assert!(!prefix_ok(&u, r#"{"b": 1"#));
    }

    #[test]
    fn gemma_dialect() {
        let q = GEMMA_QUOTE;
        let params = json!({
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}, "days": {"type": "integer"}},
            "required": ["city", "unit"],
            "additionalProperties": false
        });
        let named = Arc::new(Schema::gemma(&params));
        let ok = |s: &str| {
            let mut j = JsonState::with_schema(named.clone());
            j.feed(s) && j.is_done()
        };
        assert!(ok(&format!("{{city:{q}Oslo, \"Norway\"{q},unit:{q}c{q},days:3}}")));
        assert!(!ok(&format!("{{city:{q}Oslo{q}}}")));
        assert!(!JsonState::with_schema(named.clone()).feed(&format!("{{unit:{q}celsius")));
        assert!(!JsonState::with_schema(named.clone()).feed("{\"city\""));
        assert!(!JsonState::with_schema(named.clone()).feed("{town:"));
        // The quote arrives as its own special token.
        let mut j = JsonState::with_schema(named.clone());
        assert!(j.allows_special(q) && !JsonState::new().allows_special(q));
        for t in ["{", "city", ":", q, "Paris", q, ",unit:", q, "f", q, "}"] {
            assert!(j.feed(t), "{t}");
        }
        assert!(j.is_done());

        let calls = Arc::new(Schema::gemma_calls(&[("get_weather".into(), params), ("get_time".into(), json!({"type": "object", "properties": {}}))]));
        let ok = |s: &str| {
            let mut j = JsonState::with_schema(calls.clone());
            j.feed(s) && j.is_done()
        };
        assert!(ok(&format!("get_weather{{city:{q}Rome{q},unit:{q}c{q}}}")));
        assert!(ok("get_time{}"));
        assert!(!ok("get_time{tz:1}"));
        assert!(!JsonState::with_schema(calls.clone()).feed("get_date"));
        assert!(!JsonState::with_schema(calls).feed(" get_time"));
    }

    #[test]
    fn json_object_mode_and_choices() {
        let mut j = JsonState::with_schema(Arc::new(Schema::any_object()));
        assert!(!j.clone().feed("["));
        assert!(j.feed(r#"{"x": [1, {"y": null}]}"#) && j.is_done());

        let tools = Schema::any_of(&[
            json!({"type": "object", "properties": {"name": {"const": "get_weather"}, "arguments": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}, "required": ["name", "arguments"]}),
            json!({"type": "object", "properties": {"name": {"const": "get_time"}, "arguments": {"type": "object", "properties": {}}}, "required": ["name", "arguments"]}),
        ]);
        let tools = Arc::new(tools);
        let ok = |s: &str| {
            let mut j = JsonState::with_schema(tools.clone());
            j.feed(s) && j.is_done()
        };
        assert!(ok(r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#));
        assert!(ok(r#"{"name": "get_time", "arguments": {}}"#));
        assert!(!ok(r#"{"name": "get_time", "arguments": {"city": "Paris"}}"#));
        assert!(!ok(r#"{"name": "get_weather", "arguments": {}}"#));
        assert!(!JsonState::with_schema(tools.clone()).feed(r#"{"name": "get_w0"#));
    }
}
