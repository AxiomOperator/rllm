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

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
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
/// What a Gemma quote becomes inside the validator: a noncharacter that
/// model output doesn't contain. Text that does contain it is rejected in
/// Gemma mode, so no token (a byte token, say) can forge a quote.
const QUOTE: char = '\u{10FFFF}';

/// Schema keywords that don't constrain the value (ignored next to `$ref`).
const ANNOTATIONS: &[&str] =
    &["$defs", "definitions", "$schema", "$id", "$anchor", "$comment", "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly"];

/// A compiled JSON schema.
#[derive(Debug)]
pub struct Schema {
    nodes: Vec<Node>,
    root: NodeId,
    /// Gemma 4 argument syntax instead of JSON.
    gemma: bool,
    /// Compiling hit the work budget (see `Compiler::budget`), or the schema
    /// refers to itself without consuming input (`a = anyOf[a, …]`): it is
    /// rejected rather than enforced partially.
    too_complex: bool,
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

    fn finish(c: Compiler, root: NodeId, gemma: bool) -> Self {
        let too_complex = c.over_budget() || left_recursive(&c.nodes);
        Self { too_complex, nodes: c.nodes, root, gemma }
    }

    /// Any JSON object (`json_object` mode).
    pub fn any_object() -> Self {
        Self { nodes: Self::builtin(), root: ANY_OBJECT, gemma: false, too_complex: false }
    }

    /// Any JSON object or array.
    pub fn any_container() -> Self {
        let mut nodes = Self::builtin();
        nodes.push(Node::AnyOf(vec![ANY_OBJECT, ANY_ARRAY]));
        let root = nodes.len() - 1;
        Self { nodes, root, gemma: false, too_complex: false }
    }

    pub fn compile(schema: &Value) -> Self {
        let mut c = Compiler::new(schema);
        let root = c.compile(schema);
        Self::finish(c, root, false)
    }

    /// True if the schema was too large or too self-referencing to enforce.
    pub fn too_complex(&self) -> bool {
        self.too_complex
    }

    /// A schema in Gemma 4 argument syntax (a named tool call's arguments).
    pub fn gemma(schema: &Value) -> Self {
        Self { gemma: true, ..Self::compile(schema) }
    }

    /// Any of these Gemma 4 tool calls: `name{arguments}` for each
    /// (name, parameter schema).
    pub fn gemma_calls(tools: &[(String, Value)]) -> Self {
        let mut c = Compiler::new(&Value::Null);
        let mut alts = Vec::new();
        for (name, params) in tools {
            c.doc = params;
            c.refs.clear();
            let then = c.compile(params);
            alts.push(c.push(Node::Prefixed { text: name.clone(), then }));
        }
        let root = c.push(Node::AnyOf(alts));
        Self::finish(c, root, true)
    }

    /// Any of these JSON tool calls, `{"name": <name>, "arguments": <args>}`
    /// for each (name, parameter schema). Each tool's `$ref`s resolve
    /// against its own parameter schema.
    pub fn tool_calls(tools: &[(String, Value)]) -> Self {
        let mut c = Compiler::new(&Value::Null);
        let mut alts = Vec::new();
        for (name, params) in tools {
            c.doc = params;
            c.refs.clear();
            let args = c.compile(params);
            let name_id = c.literal(&[Value::String(name.clone())]);
            alts.push(c.push(Node::Object {
                props: vec![("name".into(), name_id), ("arguments".into(), args)],
                required: vec!["name".into(), "arguments".into()],
                additional: None,
            }));
        }
        let root = c.push(Node::AnyOf(alts));
        Self::finish(c, root, false)
    }
}

/// Whether some `anyOf` reaches itself through `anyOf`s alone, without
/// consuming input (`a = anyOf[a, …]`: left recursion through `$ref`s).
/// Expanding such a value would never end.
fn left_recursive(nodes: &[Node]) -> bool {
    // 0: unvisited, 1: on the DFS path, 2: done.
    let mut color = vec![0u8; nodes.len()];
    for start in 0..nodes.len() {
        if color[start] != 0 {
            continue;
        }
        color[start] = 1;
        let mut path: Vec<(NodeId, usize)> = vec![(start, 0)];
        while let Some((n, next)) = path.last_mut() {
            let n = *n;
            let kids: &[NodeId] = match &nodes[n] {
                Node::AnyOf(alts) => alts,
                _ => &[],
            };
            if let Some(&k) = kids.get(*next) {
                *next += 1;
                match color[k] {
                    0 => {
                        color[k] = 1;
                        path.push((k, 0));
                    }
                    1 => return true,
                    _ => {}
                }
            } else {
                color[n] = 2;
                path.pop();
            }
        }
    }
    false
}

/// Merge schema object `add` into `merged` (for `allOf`): `properties` and
/// `required` are unions; other keywords are replaced if `overwrite`.
fn merge_schema(merged: &mut serde_json::Map<String, Value>, add: Value, overwrite: bool) {
    let Value::Object(add) = add else { return };
    for (k, v) in add {
        let Some(cur) = merged.get_mut(&k) else {
            merged.insert(k, v);
            continue;
        };
        match (k.as_str(), cur, v) {
            ("properties", Value::Object(m), Value::Object(add)) => {
                for (pk, pv) in add {
                    if overwrite || !m.contains_key(&pk) {
                        m.insert(pk, pv);
                    }
                }
            }
            ("required", Value::Array(m), Value::Array(add)) => {
                for r in add {
                    if !m.contains(&r) {
                        m.push(r);
                    }
                }
            }
            (_, cur, v) => {
                if overwrite {
                    *cur = v;
                }
            }
        }
    }
}

/// Whether a `type` keyword allows null.
fn type_allows_null(t: Option<&Value>) -> bool {
    match t {
        Some(Value::String(t)) => t == "null",
        Some(Value::Array(ts)) => ts.iter().any(|t| t == "null"),
        _ => false,
    }
}

struct Compiler<'a> {
    doc: &'a Value,
    nodes: Vec<Node>,
    refs: HashMap<String, NodeId>,
    depth: usize,
    /// Schema values compiled so far, across the whole schema.
    work: usize,
}

impl<'a> Compiler<'a> {
    /// Most schema values one request may compile. `$ref`s inlined by
    /// `allOf` are compiled again at each use, so a small self-referencing
    /// schema could otherwise grow exponentially (a 250-byte one took 3.5 GB).
    const BUDGET: usize = 20_000;

    fn new(doc: &'a Value) -> Self {
        Compiler { doc, nodes: Schema::builtin(), refs: HashMap::new(), depth: 0, work: 0 }
    }

    fn over_budget(&self) -> bool {
        self.work > Self::BUDGET
    }

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

    /// `id` or null.
    fn nullable(&mut self, id: NodeId) -> NodeId {
        let null = self.push(Node::Literals(vec!["null".into()]));
        self.push(Node::AnyOf(vec![id, null]))
    }

    fn resolve(&self, pointer: &str) -> Option<&Value> {
        let path = pointer.strip_prefix('#')?;
        if path.is_empty() {
            return Some(self.doc);
        }
        self.doc.pointer(path)
    }

    fn compile(&mut self, s: &Value) -> NodeId {
        // Guard against pathological nesting and size.
        self.work += 1;
        if self.depth > 64 || self.over_budget() {
            return ANY;
        }
        self.depth += 1;
        let id = self.compile_inner(s);
        self.depth -= 1;
        id
    }

    /// A plain `$ref`: compiled once, shared by every use.
    fn compile_ref(&mut self, r: &str) -> NodeId {
        if let Some(&id) = self.refs.get(r) {
            return id;
        }
        let Some(target) = self.resolve(r).cloned() else { return ANY };
        // Placeholder first, so recursive references resolve to it.
        let id = self.push(Node::AnyOf(Vec::new()));
        self.refs.insert(r.to_owned(), id);
        let t = self.compile(&target);
        self.nodes[id] = Node::AnyOf(vec![t]);
        id
    }

    fn compile_inner(&mut self, s: &Value) -> NodeId {
        let o = match s {
            Value::Bool(true) => return ANY,
            Value::Bool(false) => return NEVER,
            Value::Object(o) => o,
            _ => return ANY,
        };
        let null_ok = o.get("nullable").and_then(Value::as_bool) == Some(true);
        if let Some(r) = o.get("$ref").and_then(Value::as_str) {
            let null_ok = null_ok || type_allows_null(o.get("type"));
            // Keywords next to a `$ref` apply too (as in JSON Schema 2019+):
            // compiled as an `allOf` of the target and them. A lone `type`
            // is taken as restating the target's.
            let siblings = o.keys().any(|k| !matches!(k.as_str(), "$ref" | "nullable" | "type") && !ANNOTATIONS.contains(&k.as_str()));
            let id = if siblings {
                let mut rest = o.clone();
                rest.remove("$ref");
                rest.remove("nullable");
                let mut parts = vec![serde_json::json!({ "$ref": r })];
                if let Some(Value::Array(more)) = rest.remove("allOf") {
                    parts.extend(more);
                }
                rest.insert("allOf".into(), Value::Array(parts));
                self.compile(&Value::Object(rest))
            } else {
                self.compile_ref(r)
            };
            return if null_ok { self.nullable(id) } else { id };
        }
        let base = if let Some(v) = o.get("const") {
            self.literal(std::slice::from_ref(v))
        } else if let Some(Value::Array(vals)) = o.get("enum") {
            self.literal(vals)
        } else if let Some(Value::Array(alts)) = o.get("anyOf").or_else(|| o.get("oneOf")) {
            let ids = alts.iter().map(|a| self.compile(a)).collect();
            self.push(Node::AnyOf(ids))
        } else if let Some(Value::Array(parts)) = o.get("allOf") {
            // Merged into one schema: the parts (later ones win conflicting
            // keywords), then the keywords next to `allOf`; `properties` and
            // `required` are unions.
            let mut merged = serde_json::Map::new();
            for p in parts {
                let mut p = p.clone();
                if let Some(r) = p.get("$ref").and_then(Value::as_str) {
                    let target = self.resolve(r).cloned().unwrap_or(Value::Null);
                    merge_schema(&mut merged, target, true);
                    if let Value::Object(m) = &mut p {
                        m.remove("$ref");
                    }
                }
                merge_schema(&mut merged, p, true);
            }
            let mut outer = o.clone();
            outer.remove("allOf");
            merge_schema(&mut merged, Value::Object(outer), false);
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
        if null_ok {
            return self.nullable(base);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Stack {
    frames: Vec<Frame>,
    done: bool,
    /// Consecutive whitespace outside strings (bounded to stop models from
    /// padding with newlines forever).
    ws_run: usize,
}

const MAX_WS_RUN: usize = 16;
/// Upper bound on parallel parses (ambiguous `anyOf`s) after merging
/// identical ones. Beyond it the schema is too ambiguous to follow: the
/// input fails as [`Fed::TooComplex`] rather than losing valid parses.
const MAX_ALTS: usize = 1024;
/// Parser steps one [`JsonState::feed`] may take (tests).
#[cfg(test)]
const FEED_WORK: usize = 1_000_000;

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// Result of feeding input to a [`JsonState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fed {
    /// Still a valid prefix.
    Ok,
    /// Can't continue a valid document.
    Rejected,
    /// Following the schema took more than the work budget (or too many
    /// parallel parses): the schema is too ambiguous to enforce.
    TooComplex,
}

/// Return early unless `Fed::Ok`.
macro_rules! try_fed {
    ($e:expr) => {
        match $e {
            Fed::Ok => {}
            other => return other,
        }
    };
}

/// Validator state: a schema and the possible parses of the text so far.
#[derive(Debug, Clone)]
pub struct JsonState {
    schema: Arc<Schema>,
    alts: Vec<Stack>,
    /// The start of a UTF-8 character split across tokens, waiting for the
    /// rest of its bytes.
    pending: Vec<u8>,
}

impl JsonState {
    /// True if the schema was rejected as too large (see `Schema::too_complex`).
    pub fn too_complex(&self) -> bool {
        self.schema.too_complex()
    }
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
        Self { schema, alts: vec![Stack { frames: vec![Frame::Value(root)], done: false, ws_run: 0 }], pending: Vec::new() }
    }

    pub fn is_done(&self) -> bool {
        self.pending.is_empty() && self.alts.iter().any(|s| s.done)
    }

    /// Whether a special token with this text may appear (Gemma's quote).
    pub fn allows_special(&self, text: &str) -> bool {
        self.schema.gemma && text == GEMMA_QUOTE
    }

    /// Feed `text` (tests); false (and a partly updated state) if it can't continue
    /// a valid document. Callers try candidates on a clone.
    #[cfg(test)]
    pub fn feed(&mut self, text: &str) -> bool {
        let mut work = FEED_WORK;
        self.feed_text(text, &mut work) == Fed::Ok
    }

    /// Feed whole-character text: a token's text, including a Gemma quote
    /// token. Spends parser steps from `work`.
    pub fn feed_text(&mut self, text: &str, work: &mut usize) -> Fed {
        if !self.pending.is_empty() {
            // A split character must be completed by its continuation bytes.
            return Fed::Rejected;
        }
        for (i, part) in text.split(GEMMA_QUOTE).enumerate() {
            if i > 0 {
                if !self.schema.gemma {
                    // Plain text in JSON mode.
                    for c in GEMMA_QUOTE.chars() {
                        try_fed!(self.feed_char(c, work));
                    }
                } else {
                    try_fed!(self.feed_char(QUOTE, work));
                }
            }
            try_fed!(self.feed_str(part, work));
        }
        Fed::Ok
    }

    /// Feed a token's bytes, which may start or finish a UTF-8 character
    /// split across tokens (byte tokens). Incomplete characters wait in
    /// `pending` and are checked once complete; meanwhile they must be able
    /// to continue some parse (inside a string, a key or a name).
    pub fn feed_bytes(&mut self, bytes: &[u8], work: &mut usize) -> Fed {
        let mut buf = std::mem::take(&mut self.pending);
        buf.extend_from_slice(bytes);
        let (text, rest) = match std::str::from_utf8(&buf) {
            Ok(t) => (t, &[][..]),
            Err(e) if e.error_len().is_none() => {
                let (a, b) = buf.split_at(e.valid_up_to());
                (std::str::from_utf8(a).expect("valid up to here"), b)
            }
            Err(_) => return Fed::Rejected,
        };
        try_fed!(self.feed_str(text, work));
        if !rest.is_empty() {
            if !self.partial_ok(rest) {
                return Fed::Rejected;
            }
            self.pending = rest.to_vec();
        }
        Fed::Ok
    }

    /// Whether some parse can take a character starting with these bytes.
    fn partial_ok(&self, start: &[u8]) -> bool {
        let gemma = self.schema.gemma;
        self.alts.iter().any(|s| match s.frames.last() {
            Some(Frame::Str { esc: 0, choices, buf, .. }) => match choices {
                None => true,
                Some(ch) => ch.iter().any(|x| x.starts_with(buf.as_str()) && x.as_bytes()[buf.len()..].starts_with(start)),
            },
            Some(Frame::Prefix { node, pos }) => {
                matches!(&self.schema.nodes[*node], Node::Prefixed { text, .. } if text.as_bytes()[*pos..].starts_with(start))
            }
            // A bare Gemma key or a call's name can start with it: checked
            // once the character is complete.
            Some(Frame::Obj { st: ObjSt::First | ObjSt::Key, .. } | Frame::Value(_)) => gemma,
            _ => false,
        })
    }

    /// Feed text from the model (no Gemma quotes in it).
    fn feed_str(&mut self, text: &str, work: &mut usize) -> Fed {
        for c in text.chars() {
            if self.schema.gemma && c == QUOTE {
                return Fed::Rejected;
            }
            try_fed!(self.feed_char(c, work));
        }
        Fed::Ok
    }

    /// Feed one character to every parse.
    fn feed_char(&mut self, c: char, work: &mut usize) -> Fed {
        let mut next = Vec::new();
        for s in self.alts.drain(..) {
            push(&self.schema, s, c, &mut next, work);
        }
        if *work == 0 {
            return Fed::TooComplex;
        }
        let next = dedupe(next);
        if next.len() > MAX_ALTS {
            return Fed::TooComplex;
        }
        self.alts = next;
        if self.alts.is_empty() {
            return Fed::Rejected;
        }
        Fed::Ok
    }
}

/// Drop parses identical to an earlier one: they would go on identically.
fn dedupe(v: Vec<Stack>) -> Vec<Stack> {
    if v.len() < 2 {
        return v;
    }
    let mut seen: HashMap<u64, Vec<usize>> = HashMap::with_capacity(v.len());
    let mut out: Vec<Stack> = Vec::with_capacity(v.len());
    for s in v {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut h);
        let same = seen.entry(h.finish()).or_default();
        if same.iter().any(|&i| out[i] == s) {
            continue;
        }
        same.push(out.len());
        out.push(s);
    }
    out
}

fn value_done(s: &mut Stack) {
    s.frames.pop();
    if s.frames.is_empty() {
        s.done = true;
    }
}

fn push(schema: &Schema, mut s: Stack, c: char, out: &mut Vec<Stack>, work: &mut usize) {
    if *work == 0 {
        return;
    }
    *work -= 1;
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
            start_value(schema, s, node, c, out, work);
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
                        push(schema, s, c, out, work);
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
                    push(schema, s, c, out, work);
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
fn start_value(schema: &Schema, s: Stack, node: NodeId, c: char, out: &mut Vec<Stack>, work: &mut usize) {
    let mut seen = Expanded::default();
    expand(schema, &s, node, c, out, work, &mut seen, 0);
}

/// `anyOf` nodes already expanded for one character on one stack: reaching
/// one again (shared `$ref`s, cycles) gives the same parses, so it's skipped.
/// This keeps expansion linear in the schema size.
#[derive(Default)]
struct Expanded {
    few: Vec<NodeId>,
    many: HashSet<NodeId>,
}

impl Expanded {
    fn insert(&mut self, n: NodeId) -> bool {
        if self.few.contains(&n) {
            return false;
        }
        if self.few.len() < 16 {
            self.few.push(n);
            return true;
        }
        self.many.insert(n)
    }
}

#[allow(clippy::too_many_arguments)]
fn expand(schema: &Schema, s: &Stack, node: NodeId, c: char, out: &mut Vec<Stack>, work: &mut usize, seen: &mut Expanded, depth: usize) {
    // Left-recursive schemas are rejected when compiled; the depth bound only
    // guards the stack.
    if *work == 0 || depth > 256 {
        return;
    }
    *work -= 1;
    let quote = if schema.gemma { QUOTE } else { '"' };
    let string = |choices: Option<Arc<Vec<String>>>| Frame::Str { key: false, bare: false, choices, buf: String::new(), esc: 0, hex: 0 };
    let frame = match &schema.nodes[node] {
        Node::AnyOf(alts) => {
            if seen.insert(node) {
                for &a in alts {
                    expand(schema, s, a, c, out, work, seen, depth + 1);
                }
            }
            return;
        }
        Node::Any => match c {
            '{' => Frame::Obj { node: ANY_OBJECT, seen: Vec::new(), st: ObjSt::First },
            '[' => Frame::Arr { node: ANY_ARRAY, count: 0, open: true },
            c if c == quote => string(None),
            '-' | '0'..='9' => Frame::Num { st: Num::Minus.step(c, false).unwrap_or(Num::Minus), integer: false },
            _ => return expand(schema, s, ANY_LITERAL, c, out, work, seen, depth + 1),
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
    let mut s = s.clone();
    s.frames.push(frame);
    out.push(s);
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use serde_json::json;

    /// A self-referencing allOf doubles its work at each level: it must be
    /// rejected quickly, not compiled exponentially.
    #[test]
    fn exploding_ref_schema_is_rejected() {
        let schema = json!({
            "$defs": { "a": { "properties": {
                "l": { "allOf": [ { "$ref": "#/$defs/a" } ] },
                "r": { "allOf": [ { "$ref": "#/$defs/a" } ] }
            } } },
            "$ref": "#/$defs/a"
        });
        let t = std::time::Instant::now();
        let s = Schema::compile(&schema);
        assert!(s.too_complex());
        assert!(t.elapsed().as_secs() < 2, "{:?}", t.elapsed());
        assert!(!Schema::compile(&json!({"type": "object", "properties": {"a": {"type": "string"}}})).too_complex());
    }
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

        let tools = Schema::tool_calls(&[
            ("get_weather".into(), json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})),
            ("get_time".into(), json!({"type": "object", "properties": {}})),
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

#[cfg(test)]
mod review_tests {
    use super::*;
    use serde_json::json;
    use std::time::{Duration, Instant};

    fn state(schema: &Value) -> JsonState {
        let s = Schema::compile(schema);
        assert!(!s.too_complex(), "{schema}");
        JsonState::with_schema(Arc::new(s))
    }

    fn matches(schema: &Value, text: &str) -> bool {
        let mut j = state(schema);
        j.feed(text) && j.is_done()
    }

    /// `a = anyOf[a, a, a]` expands without consuming input: rejected when
    /// compiled instead of costing exponential time per character.
    #[test]
    fn left_recursive_any_of_is_rejected() {
        let t = Instant::now();
        let s = json!({"$defs": {"a": {"anyOf": [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/a"}, {"$ref": "#/$defs/a"}]}}, "$ref": "#/$defs/a"});
        assert!(Schema::compile(&s).too_complex());
        let s = json!({"anyOf": [{"$ref": "#"}, {"type": "string"}]});
        assert!(Schema::compile(&s).too_complex());
        // Recursion through a container consumes input: fine.
        let ok = json!({"anyOf": [{"type": "array", "items": {"$ref": "#"}}, {"type": "integer"}]});
        assert!(matches(&ok, "[1, [2, []]]"));
        assert!(t.elapsed() < Duration::from_secs(1));
    }

    /// A chain of shared `anyOf`s has exponentially many paths but few
    /// nodes: each is expanded once per character.
    #[test]
    fn shared_any_of_paths_expand_once() {
        let mut defs = serde_json::Map::new();
        for i in 0..40 {
            let next = json!({"$ref": format!("#/$defs/l{}", i + 1)});
            defs.insert(format!("l{i}"), json!({"anyOf": [next.clone(), next]}));
        }
        defs.insert("l40".into(), json!({"type": "array", "items": {"$ref": "#/$defs/l0"}}));
        let s = json!({"$defs": defs, "$ref": "#/$defs/l0"});
        let t = Instant::now();
        assert!(matches(&s, "[[[], [[]]], []]"));
        assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    }

    #[test]
    fn too_many_parses_fail_instead_of_dropping_some() {
        let alts: Vec<Value> = (0..2000).map(|i| json!({"type": "object", "properties": {format!("k{i}"): {"type": "integer"}}})).collect();
        let mut j = state(&json!({"anyOf": alts}));
        let mut work = FEED_WORK;
        assert_eq!(j.feed_text("{", &mut work), Fed::TooComplex);
        // Running out of work is reported as such too.
        let mut j = state(&json!({"type": "string"}));
        let mut work = 3;
        assert_eq!(j.feed_text("\"abcdef\"", &mut work), Fed::TooComplex);
    }

    #[test]
    fn forty_tools_are_all_callable() {
        let tools: Vec<(String, Value)> = (0..40)
            .map(|i| (format!("tool_{i}"), json!({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})))
            .collect();
        let schema = Arc::new(Schema::tool_calls(&tools));
        for i in [0, 31, 32, 39] {
            let mut j = JsonState::with_schema(schema.clone());
            assert!(j.feed(&format!(r#"{{"name": "tool_{i}", "arguments": {{"x": 1}}}}"#)) && j.is_done(), "tool_{i}");
            // Arguments first works too.
            let mut j = JsonState::with_schema(schema.clone());
            assert!(j.feed(&format!(r#"{{"arguments": {{"x": 1}}, "name": "tool_{i}"}}"#)) && j.is_done(), "tool_{i}");
        }
        assert!(!JsonState::with_schema(schema).feed(r#"{"name": "tool_40""#));
    }

    #[test]
    fn tool_refs_resolve_in_their_own_schema() {
        let params = json!({"type": "object", "properties": {"x": {"$ref": "#/$defs/X"}}, "required": ["x"], "$defs": {"X": {"type": "integer"}}});
        let schema = Arc::new(Schema::tool_calls(&[("f".into(), params.clone()), ("g".into(), json!({"type": "object"}))]));
        let ok = |s: &str| {
            let mut j = JsonState::with_schema(schema.clone());
            j.feed(s) && j.is_done()
        };
        assert!(ok(r#"{"name": "f", "arguments": {"x": 3}}"#));
        assert!(!ok(r#"{"name":"f","arguments":{"x":"str"}}"#));
        let gemma = Arc::new(Schema::gemma_calls(&[("f".into(), params)]));
        let mut j = JsonState::with_schema(gemma);
        assert!(!j.feed(&format!("f{{x:{GEMMA_QUOTE}")));
    }

    #[test]
    fn all_of_merges_sibling_properties_and_required() {
        let s = json!({
            "$defs": {"B": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false}},
            "allOf": [{"$ref": "#/$defs/B"}],
            "properties": {"b": {"type": "string"}},
            "required": ["b"]
        });
        assert!(matches(&s, r#"{"a":1,"b":"x"}"#));
        assert!(!matches(&s, r#"{"a":1}"#));
        assert!(!matches(&s, r#"{"b":"x"}"#));
        assert!(!matches(&s, r#"{"a":1,"b":"x","c":2}"#));
    }

    #[test]
    fn ref_siblings_apply() {
        let defs = json!({"P": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}});
        let n = json!({"$defs": defs, "type": "object", "properties": {"p": {"$ref": "#/$defs/P", "nullable": true}}, "required": ["p"]});
        assert!(matches(&n, r#"{"p": null}"#));
        assert!(matches(&n, r#"{"p": {"a": 1}}"#));
        assert!(!matches(&n, r#"{"p": 1}"#));
        let t = json!({"$defs": defs, "type": "object", "properties": {"p": {"$ref": "#/$defs/P", "type": ["object", "null"]}}, "required": ["p"]});
        assert!(matches(&t, r#"{"p": null}"#));
        let more = json!({"$defs": defs, "$ref": "#/$defs/P", "properties": {"b": {"type": "boolean"}}, "required": ["b"]});
        assert!(matches(&more, r#"{"a": 1, "b": true}"#));
        assert!(!matches(&more, r#"{"a": 1}"#));
        assert!(!matches(&more, r#"{"a": 1, "b": 2}"#));
    }

    #[test]
    fn characters_split_across_tokens() {
        let s = json!({"type": "object", "properties": {"k": {"enum": ["é", "ü"]}}, "required": ["k"], "additionalProperties": false});
        let mut j = state(&s);
        let mut w = FEED_WORK;
        // "é" is C3 A9: its bytes arrive in two tokens.
        assert_eq!(j.feed_bytes(br#"{"k": ""#, &mut w), Fed::Ok);
        assert_eq!(j.clone().feed_bytes(&[0xC3], &mut w), Fed::Ok);
        // "ü" is C3 BC; no choice starts with E2.
        assert_eq!(j.clone().feed_bytes(&[0xE2], &mut w), Fed::Rejected);
        assert_eq!(j.feed_bytes(&[0xC3], &mut w), Fed::Ok);
        assert!(!j.is_done());
        // An incomplete character can't be followed by a new one.
        assert_eq!(j.clone().feed_bytes(b"\"", &mut w), Fed::Rejected);
        assert_eq!(j.clone().feed_bytes(&[0xA8], &mut w), Fed::Rejected);
        assert_eq!(j.feed_bytes(&[0xA9], &mut w), Fed::Ok);
        assert_eq!(j.feed_bytes(b"\"}", &mut w), Fed::Ok);
        assert!(j.is_done());
        // Outside strings a partial character is invalid; keys may have them.
        let mut j = JsonState::new();
        assert_eq!(j.feed_bytes(&[b'{', 0xC3], &mut w), Fed::Rejected);
        let mut j = JsonState::new();
        assert_eq!(j.feed_bytes(&[b'{', b'"', 0xC3], &mut w), Fed::Ok);
        assert_eq!(j.feed_bytes(&[0xA9, b'"', b':', b'1', b'}'], &mut w), Fed::Ok);
        assert!(j.is_done());
        // Invalid UTF-8 is rejected outright.
        assert_eq!(JsonState::new().feed_bytes(&[b'[', b'"', 0xFF], &mut w), Fed::Rejected);
    }

    #[test]
    fn gemma_quote_is_not_forgeable() {
        let params = json!({"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]});
        let named = Arc::new(Schema::gemma(&params));
        let mut w = FEED_WORK;
        let mut j = JsonState::with_schema(named.clone());
        assert_eq!(j.feed_text(&format!("{{s:{GEMMA_QUOTE}a"), &mut w), Fed::Ok);
        // A control character (byte token <0x01>) is string content, not a quote.
        assert_eq!(j.feed_bytes(&[0x01], &mut w), Fed::Ok);
        assert_eq!(j.clone().feed_bytes(b"}", &mut w), Fed::Ok);
        assert!(!j.is_done());
        // Nor can the internal placeholder be fed as text.
        assert_eq!(j.clone().feed_text(&QUOTE.to_string(), &mut w), Fed::Rejected);
        assert_eq!(j.clone().feed_bytes(QUOTE.to_string().as_bytes(), &mut w), Fed::Rejected);
        assert_eq!(j.feed_text(&format!("{GEMMA_QUOTE}}}"), &mut w), Fed::Ok);
        assert!(j.is_done());
    }
}
