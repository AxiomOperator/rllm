//! `/v1/systemone`: typed decisions from a decision model (Clef).
//!
//! The request has a `state` (text, or any JSON) and `questions`, an object
//! of question id → `{type, instructions, criteria}`:
//!
//! - `choice`: `criteria` maps option keys to descriptions (or null);
//!   answers `choice`, `probabilities`, `confidence`.
//! - `score`: `criteria` is a list of 2 to 10 level descriptions, lowest
//!   first; answers `score` (the expected level), `legend`,
//!   `probabilities`, `confidence`.
//! - `noul` (yes/no): optional `criteria` describing `"true"` and
//!   `"false"`; answers `noul`, the probability of true.
//!
//! All questions go in one prompt (the model's `systemone` template) and are
//! answered in one forward pass: the backbone's hidden states of every
//! prompt token, then the decision head ([`rllm_model::clef`]). The request
//! handling and answers follow llama.cpp's server (`server-decision.cpp`),
//! so clients of either see the same API. Text only: images are refused.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use rllm_gguf::Gguf;
use rllm_model::clef::{ClefHead, Order};
use rllm_tokenizer::Tokenizer;
use serde::de::{Deserializer, IgnoredAny, MapAccess, Visitor};
use serde_json::{Map, Value, json};
use tokio::sync::oneshot;

use crate::engine::Job;
use crate::error::{ApiError, ApiResult};
use crate::params;
use crate::state::AppState;

/// Text the template gets for the separators between prompt pieces and the
/// starts of question and option spans (llama.cpp's values). Request text
/// containing the marker prefix is changed so it can't forge one.
const MARKER: &str = "<<clef:";
const SEP: &str = "<<clef:sep>>";
const MARK_QUESTION: &str = "<<clef:question>>";
const MARK_OPTION: &str = "<<clef:option>>";

/// Most options one question may have.
const MAX_OPTIONS: usize = 255;
/// JSON nesting limit (the recursive walks below).
const MAX_DEPTH: usize = 128;

/// A loaded decision model: its head, prompt template and calibration.
pub struct Decider {
    head: Arc<ClefHead>,
    template: String,
    /// `"<type>"` or `"<type>.<bucket>"` → softmax temperature
    /// (`clef.decision.temperature.*`; 1 when absent).
    temperatures: HashMap<String, f32>,
    /// Number of levels → offsets added to a score question's logits
    /// (`clef.decision.score_bias.<n>.<level>`).
    score_bias: HashMap<usize, Vec<f32>>,
}

impl Decider {
    /// The decision head of `g`, if it is a decision model.
    pub fn load(g: &Gguf) -> Result<Option<Self>> {
        if !rllm_model::clef::is_clef(g) {
            return Ok(None);
        }
        let head = Arc::new(ClefHead::load(g).context("loading the decision head")?);
        let template = g.get_str("tokenizer.chat_template.systemone").context("decision model has no \"systemone\" template")?.to_string();
        let (mut temperatures, mut score_bias) = (HashMap::new(), HashMap::<usize, Vec<f32>>::new());
        for (k, v) in &g.metadata {
            let num = || v.as_f64().map(|x| x as f32).with_context(|| format!("{k}: expected a number"));
            if let Some(name) = k.strip_prefix("clef.decision.temperature.") {
                let t = num()?;
                if t.is_nan() || t <= 0.0 {
                    bail!("invalid decision temperature: {k} = {t}");
                }
                temperatures.insert(name.to_string(), t);
            } else if let Some(rest) = k.strip_prefix("clef.decision.score_bias.") {
                let (n, level) = rest.split_once('.').and_then(|(n, l)| Some((n.parse::<usize>().ok()?, l.parse::<usize>().ok()?))).unwrap_or((0, 0));
                if !(2..=255).contains(&n) || level >= n {
                    bail!("invalid decision score bias: {k}");
                }
                let b = score_bias.entry(n).or_insert_with(|| vec![0.0; n]);
                b[level] = num()?;
            }
        }
        Ok(Some(Self { head, template, temperatures, score_bias }))
    }

    fn temperature(&self, q: &Question) -> f32 {
        let n = q.options.len();
        let bucket = if n <= 2 {
            "2"
        } else if n <= 5 {
            "3_5"
        } else if n <= 10 {
            "6_10"
        } else {
            "11"
        };
        let name = q.kind.name();
        self.temperatures.get(&format!("{name}.{bucket}")).or_else(|| self.temperatures.get(name)).copied().unwrap_or(1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Choice,
    Score,
    Noul,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Score => "score",
            Kind::Noul => "noul",
        }
    }

    fn order(self) -> Order {
        match self {
            Kind::Choice => Order::QuestionChoice,
            Kind::Score => Order::QuestionScore,
            Kind::Noul => Order::QuestionNoul,
        }
    }
}

#[derive(Debug)]
struct Question {
    id: String,
    kind: Kind,
    instructions: Value,
    /// Key and description (null if none), in the order the model sees them.
    options: Vec<(String, Value)>,
}

/// The keys of the request's `questions` object, in request order
/// (`serde_json::Map` sorts them, but the order is part of the prompt).
fn question_order(body: &[u8]) -> Vec<String> {
    struct Keys(Vec<String>);
    impl<'de> serde::Deserialize<'de> for Keys {
        fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = Keys;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("an object")
                }
                fn visit_map<A: MapAccess<'de>>(self, mut m: A) -> std::result::Result<Keys, A::Error> {
                    let mut keys = Vec::new();
                    while let Some((k, IgnoredAny)) = m.next_entry::<String, IgnoredAny>()? {
                        keys.push(k);
                    }
                    Ok(Keys(keys))
                }
            }
            d.deserialize_map(V)
        }
    }
    #[derive(serde::Deserialize)]
    struct Body {
        questions: Option<Keys>,
    }
    let mut keys = serde_json::from_slice::<Body>(body).ok().and_then(|b| b.questions).map(|k| k.0).unwrap_or_default();
    // A repeated key counts once, where it first appears.
    let mut seen = std::collections::HashSet::new();
    keys.retain(|k| seen.insert(k.clone()));
    keys
}

fn parse_questions(m: &Map<String, Value>, order: &[String]) -> ApiResult<Vec<Question>> {
    match m.get("state") {
        None | Some(Value::Null) => return Err(ApiError::invalid("\"state\" must be provided", Some("state"))),
        _ => {}
    }
    let qs = match m.get("questions") {
        Some(Value::Object(q)) if !q.is_empty() => q,
        _ => return Err(ApiError::invalid("\"questions\" must be a non-empty object", Some("questions"))),
    };
    let mut out = Vec::with_capacity(qs.len());
    for id in order.iter().filter(|id| qs.contains_key(*id)) {
        let q = &qs[id];
        let err = |msg: &str| ApiError::invalid(format!("questions.{id}: {msg}"), Some("questions"));
        let Value::Object(q) = q else { return Err(err("must be an object")) };
        let instructions = q.get("instructions").cloned().unwrap_or(Value::Null);
        if instructions.is_null() {
            return Err(err("\"instructions\" must be provided"));
        }
        let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
        let (kind, options) = match q.get("type").and_then(Value::as_str).unwrap_or("") {
            "choice" => {
                let Value::Object(c) = criteria else { return Err(err("\"criteria\" must be a non-empty object")) };
                if c.is_empty() {
                    return Err(err("\"criteria\" must be a non-empty object"));
                }
                // Sorted by key (the map already is), as the model was trained.
                (Kind::Choice, c.into_iter().collect::<Vec<_>>())
            }
            "score" => {
                let Value::Array(levels) = criteria else { return Err(err("\"criteria\" must be an array of 2 to 10 levels")) };
                if !(2..=10).contains(&levels.len()) {
                    return Err(err("\"criteria\" must be an array of 2 to 10 levels"));
                }
                (Kind::Score, levels.into_iter().enumerate().map(|(i, d)| (i.to_string(), d)).collect())
            }
            "noul" => {
                let c = match criteria {
                    Value::Null => Map::new(),
                    Value::Object(c) => c,
                    _ => return Err(err("\"criteria\" must be an object")),
                };
                // True first, as the model was trained.
                (Kind::Noul, ["true", "false"].iter().map(|k| (k.to_string(), c.get(*k).cloned().unwrap_or(Value::Null))).collect())
            }
            _ => return Err(err("\"type\" must be one of: choice, score, noul")),
        };
        if options.len() > MAX_OPTIONS {
            return Err(err(&format!("too many options ({}), this model supports at most {MAX_OPTIONS}", options.len())));
        }
        out.push(Question { id: id.clone(), kind, instructions, options });
    }
    Ok(out)
}

/// Refuse images: there's no vision encoder.
fn check_text_only(m: &Map<String, Value>) -> ApiResult<()> {
    let unsupported = || ApiError::invalid("This server does not support image input for decisions (text only).", Some("images"));
    for key in ["files", "images", "videos"] {
        match m.get(key) {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) if a.is_empty() => {}
            Some(Value::Array(_)) => return Err(unsupported()),
            Some(_) => return Err(ApiError::invalid(format!("\"{key}\" must be an array"), Some(key))),
        }
    }
    // A state made of chat messages can carry images in its content parts.
    let state = &m["state"];
    let messages = match state {
        Value::Object(o) => o.get("messages"),
        _ => Some(state),
    };
    if let Some(Value::Array(msgs)) = messages {
        for msg in msgs {
            if let Some(Value::Array(parts)) = msg.get("content")
                && parts.iter().any(|p| matches!(p.get("type").and_then(Value::as_str), Some("image_url" | "input_audio" | "video_url")))
            {
                return Err(unsupported());
            }
        }
    }
    Ok(())
}

/// `v` with object keys sorted at every level and `from` replaced by `to`
/// in every string (keys too).
fn sanitize(v: &Value, depth: usize) -> ApiResult<Value> {
    if depth >= MAX_DEPTH {
        return Err(ApiError::invalid(format!("JSON is nested more than {MAX_DEPTH} levels deep"), None));
    }
    Ok(match v {
        Value::String(s) => Value::String(s.replace(MARKER, "<<clef ")),
        Value::Array(a) => Value::Array(a.iter().map(|x| sanitize(x, depth + 1)).collect::<ApiResult<_>>()?),
        // serde_json's map keeps its keys sorted.
        Value::Object(o) => Value::Object(o.iter().map(|(k, x)| Ok((k.replace(MARKER, "<<clef "), sanitize(x, depth + 1)?))).collect::<ApiResult<_>>()?),
        v => v.clone(),
    })
}

/// The prompt's tokens and their tags. Each piece between separators is
/// tokenized on its own (as in training); pieces starting with a span mark
/// are a question's instructions or an option.
fn build_prompt(d: &Decider, tok: &Tokenizer, state: &Value, qs: &[Question]) -> ApiResult<(Vec<u32>, Vec<Order>)> {
    let questions: Vec<Value> = qs
        .iter()
        .map(|q| {
            json!({
                "id": q.id,
                "type": q.kind.name(),
                "instructions": q.instructions,
                "options": q.options.iter().map(|(k, d)| json!({"key": k, "description": d})).collect::<Vec<_>>(),
            })
        })
        .collect();
    let mut ctx = sanitize(&json!({ "state": state, "questions": questions }), 0)?;
    ctx["images"] = json!([]);
    ctx["sep"] = json!(SEP);
    ctx["mark_question"] = json!(MARK_QUESTION);
    ctx["mark_option"] = json!(MARK_OPTION);
    let text = rllm_tokenizer::render_template(&d.template, &ctx).map_err(|e| ApiError::server(format!("rendering the decision prompt: {e:#}")))?;

    let layout = || ApiError::server("unexpected layout of the decision prompt");
    let (mut tokens, mut order) = (Vec::new(), Vec::new());
    let (mut n_questions, mut n_options) = (0, 0);
    for piece in text.split(SEP) {
        let (piece, tag) = if let Some(p) = piece.strip_prefix(MARK_QUESTION) {
            let q = qs.get(n_questions).ok_or_else(layout)?;
            n_questions += 1;
            (p, q.kind.order())
        } else if let Some(p) = piece.strip_prefix(MARK_OPTION) {
            n_options += 1;
            (p, Order::Option)
        } else {
            (piece, Order::None)
        };
        let ids = tok.encode_with(piece, false).map_err(|e| ApiError::server(format!("{e:#}")))?;
        if tag != Order::None && ids.is_empty() {
            return Err(ApiError::invalid("the instructions and the options of a question must not be empty", Some("questions")));
        }
        order.resize(order.len() + ids.len(), tag);
        tokens.extend(ids);
    }
    if n_questions != qs.len() || n_options != qs.iter().map(|q| q.options.len()).sum::<usize>() {
        return Err(layout());
    }
    Ok((tokens, order))
}

/// Confidence of a choice: how far the top probability is above uniform.
fn confidence_choice(p: &[f64]) -> f64 {
    if p.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / p.len() as f64;
    let top = p.iter().cloned().fold(f64::MIN, f64::max);
    ((top - uniform) / (1.0 - uniform)).max(0.0)
}

/// Confidence of a score: 1 minus the mean distance to the mode, relative
/// to that of a uniform distribution around the middle.
fn confidence_score(p: &[f64]) -> f64 {
    let n = p.len();
    if n < 2 {
        return 1.0;
    }
    let mode = argmax(p) as f64;
    let mut dist = 0.0;
    let mut uniform = 0.0;
    for (i, &x) in p.iter().enumerate() {
        dist += x * (i as f64 - mode).abs();
        uniform += (i as f64 - (n - 1) as f64 / 2.0).abs() / n as f64;
    }
    (1.0 - dist / uniform).max(0.0)
}

fn argmax(p: &[f64]) -> usize {
    // The first of equal maxima.
    p.iter().enumerate().fold(0, |b, (i, &x)| if x > p[b] { i } else { b })
}

fn answer(d: &Decider, q: &Question, scores: &[f32]) -> Value {
    let n = scores.len();
    let mut logits: Vec<f64> = scores.iter().map(|&s| s as f64).collect();
    if q.kind == Kind::Score
        && let Some(b) = d.score_bias.get(&n)
    {
        logits.iter_mut().zip(b).for_each(|(l, &b)| *l += b as f64);
    }
    let t = d.temperature(q) as f64;
    let max = logits.iter().cloned().fold(f64::MIN, f64::max);
    let e: Vec<f64> = logits.iter().map(|l| ((l - max) / t).exp()).collect();
    let sum: f64 = e.iter().sum();
    let probs: Vec<f64> = e.iter().map(|x| x / sum).collect();
    let mut a = Map::new();
    a.insert("type".into(), json!(q.kind.name()));
    match q.kind {
        Kind::Noul => {
            let i = q.options.iter().position(|(k, _)| k == "true").unwrap_or(0);
            a.insert("noul".into(), json!(probs[i]));
        }
        Kind::Choice => {
            a.insert("choice".into(), json!(q.options[argmax(&probs)].0));
            a.insert("probabilities".into(), Value::Object(q.options.iter().zip(&probs).map(|((k, _), p)| (k.clone(), json!(p))).collect()));
            a.insert("confidence".into(), json!(confidence_choice(&probs)));
        }
        Kind::Score => {
            let expected: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
            a.insert("score".into(), json!(expected));
            a.insert("legend".into(), Value::Object(q.options.iter().map(|(k, d)| (k.clone(), d.clone())).collect()));
            a.insert("probabilities".into(), Value::Object(q.options.iter().zip(&probs).map(|((k, _), p)| (k.clone(), json!(p))).collect()));
            a.insert("confidence".into(), json!(confidence_score(&probs)));
        }
    }
    Value::Object(a)
}

pub async fn create(State(state): State<Arc<AppState>>, body: Bytes) -> ApiResult<Json<Value>> {
    let Some(decider) = state.decision.clone() else {
        return Err(ApiError::invalid("This model is not a decision model", None));
    };
    let m = params::object(&body)?;
    let model = match m.get("model") {
        None | Some(Value::Null) => state.model_id.clone(),
        v => state.resolve_model(v)?,
    };
    let qs = parse_questions(&m, &question_order(&body))?;
    check_text_only(&m)?;
    let (tokens, order) = build_prompt(&decider, &state.tok, &m["state"], &qs)?;
    if tokens.len() > state.ctx {
        return Err(ApiError::invalid_code(
            format!("This model's maximum context length is {} tokens, however the decision prompt has {} tokens.", state.ctx, tokens.len()),
            Some("state"),
            "context_length_exceeded",
        ));
    }
    let (reply, rx) = oneshot::channel();
    state.engine.submit(Job::Embed { inputs: vec![tokens.clone()], rows: true, reply }).map_err(|e| ApiError::unavailable(format!("{e:#}")))?;
    let mut rows = rx.await.map_err(|_| ApiError::server("the engine stopped unexpectedly"))?.map_err(ApiError::server)?;
    let hidden = rows.pop().ok_or_else(|| ApiError::server("the engine returned no hidden states"))?;
    let n_tokens = tokens.len();
    let head = decider.head.clone();
    let scores = tokio::task::spawn_blocking(move || head.scores(&hidden, &tokens, &order))
        .await
        .map_err(|e| ApiError::server(format!("decision head: {e}")))?
        .map_err(|e| ApiError::server(format!("{e:#}")))?;
    let mut answers = Map::new();
    let mut at = 0;
    for q in &qs {
        let n = q.options.len();
        answers.insert(q.id.clone(), answer(&decider, q, &scores[at..at + n]));
        at += n;
    }
    Ok(Json(json!({
        "model": model,
        "answers": answers,
        "usage": { "input_tokens": n_tokens, "output_tokens": 0 },
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_order_follows_the_request() {
        let body = br#"{"state":"x","questions":{"zeta":{},"alpha":{},"mid":{"nested":{"b":1,"a":2}},"alpha":{}}}"#;
        assert_eq!(question_order(body), vec!["zeta", "alpha", "mid"]);
    }

    #[test]
    fn questions_parse_like_llama_cpp() {
        let body = br#"{"state":"s","questions":{
            "r":{"type":"choice","instructions":"route","criteria":{"sales":null,"billing":"money"}},
            "u":{"type":"noul","instructions":"urgent?"},
            "s":{"type":"score","instructions":"rate","criteria":["low","mid","high"]}}}"#;
        let m = params::object(body).unwrap();
        let qs = parse_questions(&m, &question_order(body)).unwrap();
        assert_eq!(qs.iter().map(|q| q.id.as_str()).collect::<Vec<_>>(), ["r", "u", "s"]);
        // Choice options sorted by key, noul true first, score levels by index.
        assert_eq!(qs[0].options.iter().map(|o| o.0.as_str()).collect::<Vec<_>>(), ["billing", "sales"]);
        assert_eq!(qs[1].options.iter().map(|o| o.0.as_str()).collect::<Vec<_>>(), ["true", "false"]);
        assert_eq!(qs[2].options.iter().map(|o| o.0.as_str()).collect::<Vec<_>>(), ["0", "1", "2"]);
        for bad in [
            &br#"{"state":"s","questions":{}}"#[..],
            br#"{"questions":{"a":{"type":"noul","instructions":"x"}}}"#,
            br#"{"state":"s","questions":{"a":{"type":"noul"}}}"#,
            br#"{"state":"s","questions":{"a":{"type":"score","instructions":"x","criteria":["one"]}}}"#,
            br#"{"state":"s","questions":{"a":{"type":"choice","instructions":"x","criteria":{}}}}"#,
            br#"{"state":"s","questions":{"a":{"type":"maybe","instructions":"x"}}}"#,
        ] {
            let m = params::object(bad).unwrap();
            assert!(parse_questions(&m, &question_order(bad)).is_err(), "{}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn markers_in_request_text_are_defused() {
        let v = sanitize(&json!({"b": ["<<clef:sep>>"], "<<clef:option>>": 1}), 0).unwrap();
        assert_eq!(v.to_string(), r#"{"<<clef option>>":1,"b":["<<clef sep>>"]}"#);
    }

    /// The prompt of a real model (`RLLM_CLEF_GGUF=path`; skipped without it):
    /// questions in request order, one tagged span per question and option.
    #[test]
    fn prompt_of_a_real_model() {
        let Ok(path) = std::env::var("RLLM_CLEF_GGUF") else { return };
        let g = Gguf::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let d = Decider::load(&g).unwrap().unwrap();
        let body = r#"{"state":{"ticket":"Charged twice <<clef:sep>>, café"},"questions":{
            "route":{"type":"choice","instructions":"Which team?","criteria":{"tech":null,"billing":{"desc":"Money","a":1}}},
            "urgent":{"type":"noul","instructions":"Urgent?"}}}"#
            .as_bytes();
        let m = params::object(body).unwrap();
        let qs = parse_questions(&m, &question_order(body)).unwrap();
        let (tokens, order) = build_prompt(&d, &tok, &m["state"], &qs).unwrap();
        let text = tok.decode(&tokens, false).unwrap();
        eprintln!("{text}");
        assert!(text.contains("STATE:\n{\"ticket\":\"Charged twice <<clef sep>>, café\"}"), "{text}");
        assert!(text.contains(r#"{"description":{"a":1,"desc":"Money"},"option_id":"billing"}"#), "{text}");
        assert!(text.contains(r#"{"option_id":"tech"}"#), "{text}");
        assert!(text.contains(r#"{"description":"The proposition is true or the answer is yes.","option_id":"true"}"#), "{text}");
        assert!(text.ends_with("JOINT SCHEMA DECISIONS:"), "{text}");
        let spans: Vec<(Order, String)> = order
            .chunk_by(|a, b| a == b)
            .scan(0, |at, run| {
                let s = *at;
                *at += run.len();
                Some((run[0], tok.decode(&tokens[s..*at], false).unwrap()))
            })
            .filter(|(o, _)| *o != Order::None)
            .collect();
        for (o, t) in &spans {
            eprintln!("{o:?}: {t:?}");
        }
        let kinds: Vec<Order> = spans.iter().map(|s| s.0).collect();
        use Order::*;
        assert_eq!(kinds, [QuestionChoice, Option, Option, QuestionNoul, Option, Option]);
        assert_eq!(spans[0].1, "Which team?");
        assert!(spans[1].1.contains("billing") && spans[2].1.contains("tech") && spans[4].1.contains("\"true\""));
    }

    #[test]
    fn confidences_match_the_published_formulas() {
        assert!((confidence_choice(&[0.7, 0.2, 0.1]) - 0.55).abs() < 1e-12);
        assert_eq!(confidence_choice(&[0.5, 0.5]), 0.0);
        assert!((confidence_score(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-12);
        // Uniform over 3 levels: mean distance to mode 0 is 1, uniform's is 2/3.
        assert_eq!(confidence_score(&[1.0 / 3.0; 3]), 0.0);
    }
}
