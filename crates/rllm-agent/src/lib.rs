//! `rllm agent`: runs on each machine of a fleet. It manages rllm servers
//! (systemd user units), lists and fetches models (from a URL or
//! another agent over the LAN, SHA-256 checked), deploys binaries (gated on
//! the GPU parity tests), shows power and memory, runs test suites (also on
//! a schedule), and keeps llama-server from running. It serves a dashboard
//! that controls every agent in its peer list through `/fleet/<name>/...`.
//!
//! Every API call needs `Authorization: Bearer <token>`; agents share one
//! token. Plain HTTP: for a trusted LAN only.

mod deploy;
mod http;
mod jobs;
mod models;
mod power;
mod servers;
mod sys;
mod tests;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use axum::Router;
use axum::body::Body;
use axum::extract::{Path as UrlPath, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use jobs::{JobCtx, Jobs};

const DASHBOARD: &str = include_str!("dashboard.html");
const STALE_SH: &str = include_str!("../../../tools/stale.sh");

#[derive(Deserialize, Serialize, Clone)]
pub struct Peer {
    pub name: String,
    /// host:port of its agent, reachable from every other machine.
    pub addr: String,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Config {
    pub token: String,
    #[serde(default = "default_listen")]
    pub listen: String,
    /// This machine's name in the peer list (default: hostname).
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub peers: Vec<Peer>,
    #[serde(default = "default_bin_dir")]
    pub bin_dir: PathBuf,
    /// Where models live (default: `<bin_dir>/models`). Fetches go to the first.
    #[serde(default)]
    pub model_dirs: Vec<PathBuf>,
    /// API key for servers the agent starts (and to read their metrics).
    #[serde(default)]
    pub serve_api_key: Option<String>,
    #[serde(default)]
    pub schedules: Vec<tests::Schedule>,
    /// Environment for the servers and tests the agent starts (systemd
    /// units don't inherit the agent's), e.g. LD_LIBRARY_PATH.
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
    /// Job history and server records (default ~/.local/share/rllm-agent).
    #[serde(default)]
    pub state_dir: Option<PathBuf>,
}

fn default_listen() -> String {
    "0.0.0.0:8090".into()
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| "/tmp".into())
}

fn default_bin_dir() -> PathBuf {
    home().join("rllm-bin")
}

fn expand(p: &Path) -> PathBuf {
    match p.to_str().and_then(|s| s.strip_prefix("~/")) {
        Some(rest) => home().join(rest),
        None => p.to_path_buf(),
    }
}

pub fn default_config_path() -> PathBuf {
    home().join(".config/rllm/agent.json")
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {} (create it with `rllm agent init`)", path.display()))?;
        let mut c: Config = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(c.token.len() >= 16, "token must be at least 16 characters");
        c.bin_dir = expand(&c.bin_dir);
        if c.model_dirs.is_empty() {
            c.model_dirs.push(c.bin_dir.join("models"));
        }
        c.model_dirs = c.model_dirs.iter().map(|d| expand(d)).collect();
        Ok(c)
    }

    fn port(&self) -> u16 {
        self.listen.rsplit(':').next().and_then(|p| p.parse().ok()).unwrap_or(8090)
    }
}

struct AppState {
    cfg: Config,
    name: String,
    jobs: Jobs,
    state_dir: PathBuf,
    /// Running as the rllm-agent systemd unit (deploys restart it).
    under_systemd: bool,
}

type S = State<Arc<AppState>>;

impl AppState {
    fn peers(&self) -> Vec<Peer> {
        let mut p = self.cfg.peers.clone();
        if !p.iter().any(|x| x.name == self.name) {
            p.insert(0, Peer { name: self.name.clone(), addr: format!("127.0.0.1:{}", self.cfg.port()) });
        }
        p
    }

    fn peer(&self, name: &str) -> Result<Peer> {
        self.peers().into_iter().find(|p| p.name == name).with_context(|| format!("no peer named {name:?}"))
    }

    fn servers_ctx(&self) -> servers::Ctx<'_> {
        servers::Ctx {
            bin_dir: &self.cfg.bin_dir,
            model_dirs: &self.cfg.model_dirs,
            state_dir: &self.state_dir,
            agent_port: self.cfg.port(),
            default_key: self.cfg.serve_api_key.as_deref(),
            env: &self.cfg.env,
        }
    }

    fn test_env(&self) -> tests::Env {
        tests::Env { bin_dir: self.cfg.bin_dir.clone(), model_dirs: self.cfg.model_dirs.clone(), env: self.cfg.env.clone() }
    }
}

/// Errors as `{"error": ...}` with status 400.
struct ApiErr(anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for ApiErr {
    fn from(e: E) -> Self {
        ApiErr(e.into())
    }
}

impl IntoResponse for ApiErr {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{:#}", self.0)}))).into_response()
    }
}

type R<T> = std::result::Result<T, ApiErr>;

/// `ensure!`/`bail!` for handlers (which return [`ApiErr`]).
macro_rules! check {
    ($c:expr, $($t:tt)*) => {
        if !$c {
            return Err(ApiErr(anyhow::anyhow!($($t)*)));
        }
    };
}
macro_rules! fail {
    ($($t:tt)*) => {
        return Err(ApiErr(anyhow::anyhow!($($t)*)))
    };
}

fn token_ok(expected: &str, got: &str) -> bool {
    let (a, b) = (expected.as_bytes(), got.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn auth(State(st): S, req: Request, next: Next) -> Response {
    let got = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).unwrap_or("");
    if !token_ok(&st.cfg.token, got) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "missing or wrong agent token"}))).into_response();
    }
    next.run(req).await
}

// ---------------------------------------------------------------------------
// This machine

async fn status(State(st): S) -> R<Json<Value>> {
    let host = sys::host(&st.cfg.model_dirs[0]);
    let usable = sys::usable_bytes(&host);
    Ok(Json(json!({
        "name": st.name,
        "host": host,
        "usable_bytes": usable,
        "servers": servers::list(&st.state_dir, st.cfg.serve_api_key.as_deref()).await,
        "llama": sys::llama().await,
        "jobs": st.jobs.list().into_iter().take(10).collect::<Vec<_>>(),
        "binaries": binaries_brief(&st.cfg.bin_dir),
        "agent_version": env!("CARGO_PKG_VERSION"),
    })))
}

fn binaries_brief(bin_dir: &Path) -> Value {
    let rllm = bin_dir.join("rllm");
    json!({
        "rllm_sha256": sys::cached_sha256(&rllm),
        "rllm_modified": std::fs::metadata(&rllm).ok().map(|m| { use std::os::unix::fs::MetadataExt; m.mtime() }),
    })
}

async fn list_models(State(st): S) -> Json<Value> {
    let dirs = st.cfg.model_dirs.clone();
    let list = tokio::task::spawn_blocking(move || models::list(&dirs)).await.unwrap_or_default();
    Json(json!(list))
}

async fn model_sha(State(st): S, UrlPath(name): UrlPath<String>) -> R<Json<Value>> {
    let p = models::find(&st.cfg.model_dirs, &name)?;
    let size = std::fs::metadata(&p)?.len();
    Ok(Json(json!({"name": name, "size": size, "sha256": sys::sha256(&p).await?})))
}

#[derive(Deserialize)]
struct FetchReq {
    name: String,
    /// A peer's name, or ...
    peer: Option<String>,
    /// ... an http(s) URL (Hugging Face `.../resolve/...` URLs bring their
    /// own SHA-256).
    url: Option<String>,
    sha256: Option<String>,
}

async fn fetch_model(State(st): S, Json(r): Json<FetchReq>) -> R<Json<Value>> {
    sys::safe_name(&r.name)?;
    check!(r.name.ends_with(".gguf"), "model files end in .gguf");
    let src = match (&r.peer, &r.url) {
        (Some(p), None) => {
            let peer = st.peer(p)?;
            check!(peer.name != st.name, "{} is this machine", peer.name);
            models::Source::Peer(peer.addr, st.cfg.token.clone())
        }
        (None, Some(u)) => models::Source::Url(u.clone()),
        _ => fail!("give either peer or url"),
    };
    let dir = st.cfg.model_dirs[0].clone();
    let title = format!("fetch {}", r.name);
    let id = st.jobs.spawn("fetch", title, move |ctx| models::fetch(ctx, dir, r.name, src, r.sha256));
    Ok(Json(json!({"job": id})))
}

#[derive(Deserialize)]
struct NameReq {
    name: String,
}

async fn delete_model(State(st): S, Json(r): Json<NameReq>) -> R<Json<Value>> {
    let p = models::find(&st.cfg.model_dirs, &r.name)?;
    let servers = servers::list(&st.state_dir, None).await;
    check!(!servers.iter().any(|s| s.args.contains(&p.display().to_string())), "{} is being served; stop that server first", r.name);
    std::fs::remove_file(&p)?;
    Ok(Json(json!({"ok": format!("deleted {}", p.display())})))
}

async fn serve_file(dir: PathBuf, name: String, req: Request) -> R<Response> {
    use tower::ServiceExt;
    sys::safe_name(&name)?;
    let p = dir.join(&name);
    check!(p.is_file(), "no file {name}");
    let r = tower_http::services::ServeFile::new(p).oneshot(req).await?;
    Ok(r.map(Body::new))
}

async fn file_model(State(st): S, UrlPath(name): UrlPath<String>, req: Request) -> R<Response> {
    let p = models::find(&st.cfg.model_dirs, &name)?;
    serve_file(p.parent().unwrap().to_path_buf(), name, req).await
}

async fn file_bin(State(st): S, UrlPath(name): UrlPath<String>, req: Request) -> R<Response> {
    check!(deploy::BINARIES.contains(&name.as_str()), "not a deployable binary: {name}");
    serve_file(st.cfg.bin_dir.clone(), name, req).await
}

async fn list_servers(State(st): S) -> Json<Value> {
    Json(json!(servers::list(&st.state_dir, st.cfg.serve_api_key.as_deref()).await))
}

async fn start_server(State(st): S, Json(r): Json<servers::StartReq>) -> R<Json<Value>> {
    let l = sys::llama().await;
    check!(l.processes.is_empty(), "a llama-server is running here; stop it first");
    let port = r.port;
    let msg = servers::start(&st.servers_ctx(), r).await?;
    // Report readiness as a job (loading takes up to a minute).
    let id = st.jobs.spawn("serve", format!("start server on :{port}"), move |ctx: JobCtx| async move {
        ctx.log(msg.clone());
        servers::wait_ready(port, Duration::from_secs(600)).await?;
        Ok(format!("{msg}; ready"))
    });
    Ok(Json(json!({"job": id})))
}

#[derive(Deserialize)]
struct PortReq {
    port: u16,
}

async fn stop_server(State(st): S, Json(r): Json<PortReq>) -> R<Json<Value>> {
    Ok(Json(json!({"ok": servers::stop(&st.state_dir, r.port).await?})))
}

async fn server_log(UrlPath(port): UrlPath<u16>) -> R<Json<Value>> {
    Ok(Json(json!({"log": servers::log(port, 200).await?})))
}

async fn plan(State(st): S, Json(r): Json<servers::StartReq>) -> R<Json<Value>> {
    let info = models::list(&st.cfg.model_dirs).into_iter().find(|m| m.name == r.model).context("no such model")?;
    let kv = if r.kv_type == "f16" { rllm_core::KvType::F16 } else { rllm_core::KvType::Q8 };
    let host = sys::host(&st.cfg.model_dirs[0]);
    Ok(Json(json!(models::plan(&info, r.ctx, r.parallel, kv, r.kv_tokens, sys::usable_bytes(&host)))))
}

async fn stop_llama() -> R<Json<Value>> {
    Ok(Json(json!({"ok": sys::stop_llama().await?})))
}

async fn run_stale(args: &[&str]) -> Result<String> {
    use tokio::io::AsyncWriteExt;
    let mut c = tokio::process::Command::new("bash");
    c.arg("-s").arg("--").args(args).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
    let mut child = c.spawn()?;
    child.stdin.take().unwrap().write_all(STALE_SH.as_bytes()).await?;
    let out = child.wait_with_output().await?;
    Ok(format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

async fn stale() -> R<Json<Value>> {
    Ok(Json(json!({"output": run_stale(&[]).await?})))
}

#[derive(Deserialize)]
struct StaleKill {
    #[serde(default)]
    services: bool,
}

async fn stale_kill(Json(r): Json<StaleKill>) -> R<Json<Value>> {
    let mut a = vec!["--kill"];
    if r.services {
        a.push("--services");
    }
    Ok(Json(json!({"output": run_stale(&a).await?})))
}

async fn power_status() -> Json<Value> {
    Json(json!(power::status().await))
}

async fn power_apply(Json(r): Json<power::Apply>) -> R<Json<Value>> {
    Ok(Json(json!({"ok": power::apply(r).await?})))
}

async fn bin_manifest(State(st): S) -> R<Json<Value>> {
    Ok(Json(json!(deploy::manifest(&st.cfg.bin_dir).await?)))
}

#[derive(Deserialize)]
struct DeployReq {
    /// The peer to pull binaries from.
    source: String,
}

async fn deploy_from(State(st): S, Json(r): Json<DeployReq>) -> R<Json<Value>> {
    let src = st.peer(&r.source)?;
    check!(src.name != st.name, "{} is this machine", src.name);
    check!(!st.jobs.running("deploy"), "a deploy is already running here");
    let (bin, token, sd, env) = (st.cfg.bin_dir.clone(), st.cfg.token.clone(), st.under_systemd, st.cfg.env.clone());
    let id = st.jobs.spawn("deploy", format!("deploy from {}", src.name), move |ctx| deploy::deploy(ctx, bin, src.addr, token, sd, env));
    Ok(Json(json!({"job": id})))
}

async fn rollback(State(st): S) -> R<Json<Value>> {
    Ok(Json(json!({"ok": deploy::rollback(&st.cfg.bin_dir)?})))
}

#[derive(Deserialize)]
struct TestReq {
    suite: String,
    model: Option<String>,
}

fn start_test(st: &AppState, suite: String, model: Option<String>) -> Result<u64> {
    ensure!(tests::SUITES.contains(&suite.as_str()), "unknown suite {suite:?}");
    ensure!(!st.jobs.running("test"), "a test suite is already running here");
    let env = st.test_env();
    let title = format!("{suite}{}", model.as_ref().map(|m| format!(" on {m}")).unwrap_or_default());
    Ok(st.jobs.spawn("test", title, move |ctx| tests::run(ctx, env, suite, model)))
}

async fn run_test(State(st): S, Json(r): Json<TestReq>) -> R<Json<Value>> {
    Ok(Json(json!({"job": start_test(&st, r.suite, r.model)?})))
}

async fn history(State(st): S) -> Json<Value> {
    Json(json!(st.jobs.history(50)))
}

async fn jobs(State(st): S) -> Json<Value> {
    Json(json!(st.jobs.list()))
}

async fn job(State(st): S, UrlPath(id): UrlPath<u64>) -> R<Json<Value>> {
    Ok(Json(json!(st.jobs.get(id).context("no such job")?)))
}

async fn cancel_job(State(st): S, UrlPath(id): UrlPath<u64>) -> R<Json<Value>> {
    check!(st.jobs.cancel(id), "job {id} isn't running");
    Ok(Json(json!({"ok": format!("cancelled job {id}")})))
}

/// Cancels a peer's job if the local job waiting on it is dropped
/// (cancelled) before the peer's finished.
struct PeerJob {
    addr: String,
    token: String,
    id: u64,
    live: bool,
}

impl Drop for PeerJob {
    fn drop(&mut self) {
        if self.live {
            let (addr, token, id) = (self.addr.clone(), self.token.clone(), self.id);
            tokio::spawn(async move {
                let _ = http::request(&addr, "POST", &format!("/api/jobs/{id}/cancel"), Some(&token), Some(b"{}"), Duration::from_secs(10)).await;
            });
        }
    }
}

// ---------------------------------------------------------------------------
// The fleet

async fn fleet(State(st): S) -> Json<Value> {
    Json(json!({"self": st.name, "peers": st.peers(), "suites": tests::SUITES}))
}

/// Forward `/fleet/<name>/<rest>` to that peer's agent.
async fn proxy(State(st): S, UrlPath((name, rest)): UrlPath<(String, String)>, req: Request) -> R<Response> {
    let peer = st.peer(&name)?;
    let method = req.method().as_str().to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    let body = axum::body::to_bytes(req.into_body(), 1 << 20).await?;
    let timeout = if rest.ends_with("/sha256") { 900 } else { 60 };
    let r = http::request(&peer.addr, &method, &format!("/{rest}{query}"), Some(&st.cfg.token), (!body.is_empty()).then_some(&body[..]), Duration::from_secs(timeout))
        .await
        .map_err(|e| anyhow::anyhow!("{name}: {e:#}"))?;
    let status = StatusCode::from_u16(r.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = if r.content_type.is_empty() { "application/json".to_string() } else { r.content_type };
    Ok((status, [(header::CONTENT_TYPE, ct)], r.body).into_response())
}

/// POST to a peer's agent and wait for the job it starts, mirroring it.
/// Progress of several remote jobs, combined into the local job's.
type Shares = Option<(Arc<std::sync::Mutex<Vec<f64>>>, usize)>;

async fn remote_job(ctx: &JobCtx, peer: &Peer, token: &str, path: &str, body: Value, share: Shares) -> Result<String> {
    let r = http::request(&peer.addr, "POST", path, Some(token), Some(body.to_string().as_bytes()), Duration::from_secs(60)).await?;
    let v: Value = serde_json::from_slice(&r.body).unwrap_or(Value::Null);
    ensure!(r.status == 200, "{}: {}", peer.name, v["error"].as_str().unwrap_or("request failed"));
    let id = v["job"].as_u64().context("no job id")?;
    let mut guard = PeerJob { addr: peer.addr.clone(), token: token.to_string(), id, live: true };
    let mut seen = 0usize;
    loop {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let j = match http::get_json(&peer.addr, &format!("/api/jobs/{id}"), token).await {
            Ok(j) => j,
            // An agent restarting onto a new binary drops its jobs: a
            // deploy that got that far has installed.
            Err(e) if path == "/api/deploy" => {
                guard.live = false;
                ctx.log(format!("{}: agent restarting ({e:#})", peer.name));
                tokio::time::sleep(Duration::from_secs(8)).await;
                return Ok("installed; agent restarted".into());
            }
            Err(e) => return Err(e),
        };
        let log = j["log"].as_array().cloned().unwrap_or_default();
        for l in log.iter().skip(seen) {
            ctx.log(format!("{}: {}", peer.name, l.as_str().unwrap_or("")));
        }
        seen = log.len();
        if let (Some((all, i)), Some(p)) = (&share, j["progress"].as_f64()) {
            let mut v = all.lock().unwrap();
            v[*i] = p;
            ctx.progress(v.iter().sum::<f64>() / v.len().max(1) as f64);
        }
        match j["state"].as_str() {
            Some("ok") => {
                guard.live = false;
                return Ok(j["summary"].as_str().unwrap_or("").to_string());
            }
            Some(s @ ("failed" | "cancelled")) => {
                guard.live = false;
                bail!("{}: {}", peer.name, if s == "cancelled" { "cancelled" } else { j["summary"].as_str().unwrap_or("failed") });
            }
            _ => {}
        }
    }
}

#[derive(Deserialize)]
struct FleetDeploy {
    source: String,
    targets: Vec<String>,
}

/// Rolling deploy: one machine at a time, stopping at the first failure.
async fn fleet_deploy(State(st): S, Json(r): Json<FleetDeploy>) -> R<Json<Value>> {
    st.peer(&r.source)?;
    let targets: Vec<Peer> = r.targets.iter().filter(|t| **t != r.source).map(|t| st.peer(t)).collect::<Result<_>>()?;
    check!(!targets.is_empty(), "no targets");
    let token = st.cfg.token.clone();
    let title = format!("rolling deploy from {} to {}", r.source, targets.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "));
    let id = st.jobs.spawn("fleet", title, move |ctx: JobCtx| async move {
        let mut done = Vec::new();
        for (i, t) in targets.iter().enumerate() {
            ctx.log(format!("== {} ({}/{})", t.name, i + 1, targets.len()));
            match remote_job(&ctx, t, &token, "/api/deploy", json!({"source": r.source}), None).await {
                Ok(s) => done.push(format!("{}: {s}", t.name)),
                Err(e) => bail!("stopped at {}: {e:#} (done: {})", t.name, if done.is_empty() { "none".into() } else { done.join("; ") }),
            }
            ctx.progress((i + 1) as f64 / targets.len() as f64);
        }
        Ok(done.join(" | "))
    });
    Ok(Json(json!({"job": id})))
}

#[derive(Deserialize)]
struct FleetDistribute {
    model: String,
    source: String,
    targets: Vec<String>,
}

/// Copy a model from one machine to others over the LAN, in parallel.
async fn fleet_distribute(State(st): S, Json(r): Json<FleetDistribute>) -> R<Json<Value>> {
    st.peer(&r.source)?;
    let targets: Vec<Peer> = r.targets.iter().filter(|t| **t != r.source).map(|t| st.peer(t)).collect::<Result<_>>()?;
    check!(!targets.is_empty(), "no targets");
    let token = st.cfg.token.clone();
    let title = format!("copy {} from {} to {} machine(s)", r.model, r.source, targets.len());
    let id = st.jobs.spawn("fleet", title, move |ctx: JobCtx| async move {
        let mut tasks = Vec::new();
        let mut guard = jobs::Tasks::default();
        let all = Arc::new(std::sync::Mutex::new(vec![0.0; targets.len()]));
        for (i, t) in targets.into_iter().enumerate() {
            let (ctx, token, body, share) = (ctx.clone(), token.clone(), json!({"name": r.model, "peer": r.source}), Some((all.clone(), i)));
            let h = tokio::spawn(async move { (t.name.clone(), remote_job(&ctx, &t, &token, "/api/models/fetch", body, share).await) });
            guard.add(&h);
            tasks.push(h);
        }
        let mut lines = Vec::new();
        let mut failed = false;
        for t in tasks {
            let (name, res) = t.await?;
            match res {
                Ok(s) => lines.push(format!("{name}: {s}")),
                Err(e) => {
                    failed = true;
                    lines.push(format!("{name}: FAILED {e:#}"));
                }
            }
        }
        let s = lines.join(" | ");
        if failed { bail!(s) } else { Ok(s) }
    });
    Ok(Json(json!({"job": id})))
}

#[derive(Deserialize)]
struct FleetTests {
    suite: String,
    model: Option<String>,
    targets: Vec<String>,
}

/// Run a test suite on several machines at once.
async fn fleet_tests(State(st): S, Json(r): Json<FleetTests>) -> R<Json<Value>> {
    let targets: Vec<Peer> = r.targets.iter().map(|t| st.peer(t)).collect::<Result<_>>()?;
    check!(!targets.is_empty(), "no targets");
    let token = st.cfg.token.clone();
    let title = format!("{} on {} machine(s)", r.suite, targets.len());
    let id = st.jobs.spawn("fleet", title, move |ctx: JobCtx| async move {
        let mut tasks = Vec::new();
        let mut guard = jobs::Tasks::default();
        for t in targets {
            let (ctx, token, body) = (ctx.clone(), token.clone(), json!({"suite": r.suite, "model": r.model}));
            let h = tokio::spawn(async move { (t.name.clone(), remote_job(&ctx, &t, &token, "/api/tests/run", body, None).await) });
            guard.add(&h);
            tasks.push(h);
        }
        let (mut lines, mut failed) = (Vec::new(), false);
        for t in tasks {
            let (name, res) = t.await?;
            match res {
                Ok(s) => lines.push(format!("{name}: {s}")),
                Err(e) => {
                    failed = true;
                    lines.push(format!("{name}: FAILED {e:#}"));
                }
            }
        }
        let s = lines.join(" | ");
        if failed { bail!(s) } else { Ok(s) }
    });
    Ok(Json(json!({"job": id})))
}

// ---------------------------------------------------------------------------

/// Run due test schedules (checked every minute; last runs kept on disk).
async fn scheduler(st: Arc<AppState>) {
    let file = st.state_dir.join("schedule.json");
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        let mut last: std::collections::HashMap<String, u64> = std::fs::read_to_string(&file).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        for s in &st.cfg.schedules {
            let key = format!("{}:{}", s.suite, s.model.clone().unwrap_or_default());
            let due = last.get(&key).is_none_or(|&t| jobs::now().saturating_sub(t) as f64 >= s.every_hours * 3600.0);
            if due && !st.jobs.running("test") {
                if start_test(&st, s.suite.clone(), s.model.clone()).is_ok() {
                    last.insert(key, jobs::now());
                    let _ = std::fs::write(&file, serde_json::to_string(&last).unwrap_or_default());
                }
                break;
            }
        }
    }
}

fn router(st: Arc<AppState>) -> Router {
    let api = Router::new()
        .route("/status", get(status))
        .route("/models", get(list_models))
        .route("/models/{name}/sha256", get(model_sha))
        .route("/models/fetch", post(fetch_model))
        .route("/models/delete", post(delete_model))
        .route("/files/models/{name}", get(file_model))
        .route("/files/bin/{name}", get(file_bin))
        .route("/servers", get(list_servers))
        .route("/servers/plan", post(plan))
        .route("/servers/start", post(start_server))
        .route("/servers/stop", post(stop_server))
        .route("/servers/{port}/log", get(server_log))
        .route("/llama/stop", post(stop_llama))
        .route("/stale", get(stale))
        .route("/stale/kill", post(stale_kill))
        .route("/power", get(power_status))
        .route("/power/apply", post(power_apply))
        .route("/bin/manifest", get(bin_manifest))
        .route("/deploy", post(deploy_from))
        .route("/deploy/rollback", post(rollback))
        .route("/tests/run", post(run_test))
        .route("/tests/history", get(history))
        .route("/jobs", get(jobs))
        .route("/jobs/{id}", get(job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .route("/fleet", get(fleet))
        .route("/fleet/deploy", post(fleet_deploy))
        .route("/fleet/distribute", post(fleet_distribute))
        .route("/fleet/tests", post(fleet_tests))
        .layer(middleware::from_fn_with_state(st.clone(), auth));
    let fleet = Router::new().route("/{name}/{*rest}", any(proxy)).layer(middleware::from_fn_with_state(st.clone(), auth));
    Router::new()
        .route("/", get(|| async { Html(DASHBOARD) }))
        .route("/healthz", get(|| async { "ok" }))
        .nest("/api", api)
        .nest("/fleet", fleet)
        .with_state(st)
}

/// `rllm agent run`.
pub fn run(config: &Path) -> Result<()> {
    let cfg = Config::load(config)?;
    let name = cfg.name.clone().unwrap_or_else(|| std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string()).unwrap_or_else(|_| "localhost".into()));
    let state_dir = cfg.state_dir.as_deref().map(expand).unwrap_or_else(|| home().join(".local/share/rllm-agent"));
    std::fs::create_dir_all(&state_dir)?;
    let st = Arc::new(AppState {
        jobs: Jobs::new(state_dir.join("history.jsonl")),
        // Running as rllm-agent.service itself (not merely under some
        // systemd-launched shell): deploys may restart it.
        under_systemd: std::fs::read_to_string("/proc/self/cgroup").is_ok_and(|c| c.contains("/rllm-agent.service")),
        name,
        state_dir,
        cfg,
    });
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&st.cfg.listen).await.with_context(|| format!("listening on {}", st.cfg.listen))?;
        eprintln!("[rllm-agent] {} on http://{} ({} peers, binaries in {})", st.name, st.cfg.listen, st.peers().len(), st.cfg.bin_dir.display());
        tokio::spawn(scheduler(st.clone()));
        // Keep the deployed binary's hash known (the dashboard compares builds).
        let bin = st.cfg.bin_dir.join("rllm");
        tokio::spawn(async move {
            loop {
                let _ = sys::sha256(&bin).await;
                tokio::time::sleep(Duration::from_secs(300)).await;
            }
        });
        axum::serve(listener, router(st)).await?;
        Ok(())
    })
}

fn random_token() -> Result<String> {
    use std::io::Read;
    let mut b = [0u8; 24];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

/// `rllm agent init`: write a config (a fresh token unless given).
pub fn init(config: &Path, token: Option<String>, peers: &[String], serve_api_key: Option<String>, force: bool) -> Result<()> {
    ensure!(force || !config.exists(), "{} exists (--force to overwrite)", config.display());
    let peers = peers
        .iter()
        .map(|p| {
            let (n, a) = p.split_once('=').with_context(|| format!("peer {p:?}: expected name=host:port"))?;
            Ok(Peer { name: n.into(), addr: a.into() })
        })
        .collect::<Result<Vec<_>>>()?;
    let c = Config {
        token: match token {
            Some(t) => t,
            None => random_token()?,
        },
        listen: default_listen(),
        name: None,
        peers,
        bin_dir: default_bin_dir(),
        model_dirs: Vec::new(),
        serve_api_key,
        schedules: Vec::new(),
        env: Default::default(),
        state_dir: None,
    };
    ensure!(c.token.len() >= 16, "token must be at least 16 characters");
    if let Some(d) = config.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(config, serde_json::to_string_pretty(&c)?)?;
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(config, std::fs::Permissions::from_mode(0o600))?;
    println!("wrote {} (token {}...)", config.display(), &c.token[..8]);
    Ok(())
}

/// `rllm agent install`: a systemd user unit running the agent (restarted
/// if it crashes, started at boot when the user lingers).
pub fn install(config: &Path) -> Result<()> {
    Config::load(config)?;
    let exe = std::env::current_exe()?;
    let unit_dir = home().join(".config/systemd/user");
    std::fs::create_dir_all(&unit_dir)?;
    let unit = format!(
        "[Unit]\nDescription=rllm fleet agent\nAfter=network-online.target\n\n[Service]\nExecStart={} agent run --config {}\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
        exe.display(),
        config.display()
    );
    std::fs::write(unit_dir.join("rllm-agent.service"), unit)?;
    let sh = |args: &[&str]| -> Result<String> {
        let o = std::process::Command::new("systemctl").arg("--user").args(args).output()?;
        if !o.status.success() {
            bail!("systemctl --user {}: {}", args.join(" "), String::from_utf8_lossy(&o.stderr));
        }
        Ok(String::from_utf8_lossy(&o.stdout).to_string())
    };
    sh(&["daemon-reload"])?;
    sh(&["enable", "rllm-agent.service"])?;
    sh(&["restart", "rllm-agent.service"])?;
    let user = std::env::var("USER").unwrap_or_default();
    let linger = std::process::Command::new("loginctl").args(["show-user", &user, "-p", "Linger"]).output().map(|o| String::from_utf8_lossy(&o.stdout).contains("yes")).unwrap_or(false);
    println!("rllm-agent.service installed and started");
    if !linger {
        println!("note: user {user} doesn't linger, so the agent stops at logout and won't start at boot: sudo loginctl enable-linger {user}");
    }
    Ok(())
}
