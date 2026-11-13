//! rllm servers on this machine. Started ones run as transient systemd user
//! units (`rllm-serve-<port>`, no restart: a crashed server stays down and
//! shows as failed), so they outlive the SSH session or dashboard that
//! started them and stop exactly. rllm servers started by hand show up too.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use rllm_core::KvType;
use serde::{Deserialize, Serialize};

use crate::models;
use crate::sys;

#[derive(Deserialize, Serialize, Clone)]
pub struct StartReq {
    pub model: String,
    pub port: u16,
    #[serde(default = "default_ctx")]
    pub ctx: u64,
    #[serde(default = "default_parallel")]
    pub parallel: u64,
    #[serde(default = "default_kv_type")]
    pub kv_type: String,
    pub kv_tokens: Option<u64>,
    #[serde(default = "yes")]
    pub mtp: bool,
    /// Defaults to the agent's `serve_api_key`.
    pub api_key: Option<String>,
    #[serde(default = "default_host")]
    pub host: String,
    /// Start even if the memory estimate says it won't fit.
    #[serde(default)]
    pub force: bool,
}

fn default_ctx() -> u64 {
    32768
}
fn default_parallel() -> u64 {
    8
}
fn default_kv_type() -> String {
    "q8".into()
}
fn yes() -> bool {
    true
}
fn default_host() -> String {
    "0.0.0.0".into()
}

#[derive(Serialize, Clone)]
pub struct Server {
    pub port: u16,
    pub managed: bool,
    pub unit: Option<String>,
    pub state: String,
    pub pid: Option<i32>,
    pub model: String,
    pub args: String,
    pub healthy: bool,
    pub uptime_s: u64,
    pub requests_finished: Option<f64>,
    pub tokens_generated: Option<f64>,
    pub requests_running: Option<f64>,
    /// Generated tokens per second since the previous poll.
    pub tok_per_s: Option<f64>,
    pub draft_accept: Option<f64>,
}

/// Parameters of servers this agent started, by port (for the dashboard
/// and for the API key to read their metrics).
fn state_file(dir: &Path) -> PathBuf {
    dir.join("servers.json")
}

fn load_state(dir: &Path) -> HashMap<u16, StartReq> {
    std::fs::read_to_string(state_file(dir)).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

fn save_state(dir: &Path, m: &HashMap<u16, StartReq>) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(state_file(dir), serde_json::to_string_pretty(m).unwrap_or_default());
}

fn unit(port: u16) -> String {
    format!("rllm-serve-{port}")
}

async fn unit_props(name: &str) -> HashMap<String, String> {
    let (_, out) = sys::run("systemctl", &["--user", "show", &format!("{name}.service"), "-p", "ActiveState,SubState,MainPID,Result,LoadState"]).await.unwrap_or_default();
    out.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn port_of(args: &str) -> Option<u16> {
    let mut it = args.split_whitespace();
    while let Some(a) = it.next() {
        if a == "--port" {
            return it.next().and_then(|p| p.parse().ok());
        }
        if let Some(p) = a.strip_prefix("--port=") {
            return p.parse().ok();
        }
    }
    None
}

static LAST: Mutex<Option<HashMap<u16, (Instant, f64)>>> = Mutex::new(None);

async fn poll(port: u16, key: Option<&str>, s: &mut Server) {
    let addr = format!("127.0.0.1:{port}");
    s.healthy = crate::http::request(&addr, "GET", "/health", None, None, Duration::from_secs(2)).await.is_ok_and(|r| r.status == 200);
    if !s.healthy {
        return;
    }
    let Ok(r) = crate::http::request(&addr, "GET", "/metrics", key, None, Duration::from_secs(2)).await else { return };
    if r.status != 200 {
        return;
    }
    let text = String::from_utf8_lossy(&r.body);
    let metric = |name: &str| -> Option<f64> {
        text.lines().filter(|l| l.starts_with(name) && l[name.len()..].starts_with(['{', ' '])).filter_map(|l| l.rsplit(' ').next()?.parse::<f64>().ok()).reduce(|a, b| a + b)
    };
    s.requests_finished = metric("rllm_requests_finished_total");
    s.tokens_generated = metric("rllm_generation_tokens_total");
    s.requests_running = metric("rllm_requests_running");
    if let (Some(d), Some(a)) = (metric("rllm_draft_tokens_total"), metric("rllm_draft_tokens_accepted_total"))
        && d > 0.0
    {
        s.draft_accept = Some(a / d);
    }
    if let Some(t) = s.tokens_generated {
        let now = Instant::now();
        let mut g = LAST.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        if let Some(&(then, prev)) = m.get(&port) {
            let dt = now.duration_since(then).as_secs_f64();
            if dt > 0.5 && t >= prev {
                s.tok_per_s = Some((t - prev) / dt);
            }
        }
        m.insert(port, (now, t));
    }
}

pub async fn list(state_dir: &Path, default_key: Option<&str>) -> Vec<Server> {
    let known = load_state(state_dir);
    let mut out: Vec<Server> = Vec::new();
    let (_, units) = sys::run("systemctl", &["--user", "list-units", "--all", "--plain", "--no-legend", "rllm-serve-*"]).await.unwrap_or_default();
    let mut managed_pids = Vec::new();
    for l in units.lines() {
        let Some(name) = l.split_whitespace().next().and_then(|u| u.strip_suffix(".service")) else { continue };
        let Some(port) = name.strip_prefix("rllm-serve-").and_then(|p| p.parse::<u16>().ok()) else { continue };
        let p = unit_props(name).await;
        let pid = p.get("MainPID").and_then(|x| x.parse::<i32>().ok()).filter(|&x| x > 0);
        managed_pids.extend(pid);
        let req = known.get(&port);
        let args = pid.and_then(|pid| std::fs::read(format!("/proc/{pid}/cmdline")).ok()).map(|b| String::from_utf8_lossy(&b).replace('\0', " ")).unwrap_or_default();
        let state = match (p.get("ActiveState").map(String::as_str), p.get("Result").map(String::as_str)) {
            (Some("active"), _) => "running".to_string(),
            (Some("failed"), r) => format!("failed ({})", r.unwrap_or("?")),
            (Some(s), _) => s.to_string(),
            _ => "unknown".into(),
        };
        let mut s = Server {
            port,
            managed: true,
            unit: Some(name.to_string()),
            state,
            pid,
            model: req.map(|r| r.model.clone()).unwrap_or_default(),
            args,
            healthy: false,
            uptime_s: pid.map(|p| sys::procs_named("rllm").into_iter().find(|x| x.pid == p).map_or(0, |x| x.age_s)).unwrap_or(0),
            requests_finished: None,
            tokens_generated: None,
            requests_running: None,
            tok_per_s: None,
            draft_accept: None,
        };
        let key = req.and_then(|r| r.api_key.clone()).or(default_key.map(String::from));
        if pid.is_some() {
            poll(port, key.as_deref(), &mut s).await;
        }
        out.push(s);
    }
    // rllm servers started some other way.
    for p in sys::procs_named("rllm") {
        if managed_pids.contains(&p.pid) || !p.args.split_whitespace().any(|a| a == "serve") {
            continue;
        }
        let port = port_of(&p.args).unwrap_or(8080);
        let model = p.args.split_whitespace().skip_while(|a| *a != "serve").nth(1).map(|m| m.rsplit('/').next().unwrap_or(m).to_string()).unwrap_or_default();
        let mut s = Server {
            port,
            managed: false,
            unit: None,
            state: "running".into(),
            pid: Some(p.pid),
            model,
            args: p.args.clone(),
            healthy: false,
            uptime_s: p.age_s,
            requests_finished: None,
            tokens_generated: None,
            requests_running: None,
            tok_per_s: None,
            draft_accept: None,
        };
        let key = p.args.split_whitespace().skip_while(|a| *a != "--api-key").nth(1).map(String::from).or(default_key.map(String::from));
        poll(port, key.as_deref(), &mut s).await;
        out.push(s);
    }
    out.sort_by_key(|s| s.port);
    out
}

fn port_in_use(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], port)), Duration::from_millis(300)).is_ok()
}

pub struct Ctx<'a> {
    pub bin_dir: &'a Path,
    pub model_dirs: &'a [PathBuf],
    pub state_dir: &'a Path,
    pub agent_port: u16,
    pub default_key: Option<&'a str>,
    pub env: &'a std::collections::BTreeMap<String, String>,
}

/// Start a server; returns the plan's description.
pub async fn start(c: &Ctx<'_>, mut req: StartReq) -> Result<String> {
    ensure!(req.port >= 1024 && req.port != c.agent_port, "port {} not allowed (1024+, not the agent's {})", req.port, c.agent_port);
    ensure!(!port_in_use(req.port), "port {} is already in use", req.port);
    ensure!((1..=8).contains(&req.parallel), "parallel must be 1..8");
    ensure!((256..=1 << 20).contains(&req.ctx), "ctx must be 256..1048576");
    let kv = match req.kv_type.as_str() {
        "q8" => KvType::Q8,
        "f16" => KvType::F16,
        other => bail!("kv_type {other:?} (q8 or f16)"),
    };
    let props = unit_props(&unit(req.port)).await;
    if props.get("ActiveState").is_some_and(|s| s == "active" || s == "activating") {
        bail!("{} is already running", unit(req.port));
    }
    // A failed unit of the same name must be cleared before reuse.
    let _ = sys::run("systemctl", &["--user", "reset-failed", &format!("{}.service", unit(req.port))]).await;
    let path = models::find(c.model_dirs, &req.model)?;
    let info = models::list(c.model_dirs).into_iter().find(|m| m.name == req.model).context("model vanished")?;
    if let Some(e) = &info.error {
        bail!("{}: {e}", req.model);
    }
    let host = sys::host(&c.model_dirs[0]);
    let plan = models::plan(&info, req.ctx, req.parallel, kv, req.kv_tokens, sys::usable_bytes(&host));
    if !plan.fits && !req.force {
        bail!("won't fit: {} (stop something, lower ctx/parallel, or force)", plan.detail);
    }
    if req.api_key.is_none() {
        req.api_key = c.default_key.map(String::from);
    }
    let bin = c.bin_dir.join("rllm");
    ensure!(bin.is_file(), "{} not found", bin.display());
    let mut args: Vec<String> = vec![
        "--user".into(),
        format!("--unit={}", unit(req.port)),
        "--collect".into(),
        "--property=Restart=no".into(),
        format!("--working-directory={}", c.bin_dir.display()),
    ];
    if let Some(k) = &req.api_key {
        args.push(format!("--setenv=RLLM_API_KEY={k}"));
    }
    for (k, v) in c.env {
        args.push(format!("--setenv={k}={v}"));
    }
    args.extend([
        bin.display().to_string(),
        "serve".into(),
        path.display().to_string(),
        "-d".into(),
        host.device.clone(),
        "--host".into(),
        req.host.clone(),
        "--port".into(),
        req.port.to_string(),
        "--ctx".into(),
        req.ctx.to_string(),
        "--parallel".into(),
        req.parallel.to_string(),
        "--kv-type".into(),
        req.kv_type.clone(),
        "--kv-tokens".into(),
        plan.kv_tokens.to_string(),
    ]);
    if !req.mtp {
        args.push("--no-mtp".into());
    }
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    let (ok, out) = sys::run("systemd-run", &a).await?;
    ensure!(ok, "systemd-run: {out}");
    let mut known = load_state(c.state_dir);
    known.insert(req.port, req.clone());
    save_state(c.state_dir, &known);
    Ok(format!("{} started: {}", unit(req.port), plan.detail))
}

/// Wait until a started server answers /health (or its unit stops).
pub async fn wait_ready(port: u16, timeout: Duration) -> Result<()> {
    let t0 = Instant::now();
    while t0.elapsed() < timeout {
        if crate::http::request(&format!("127.0.0.1:{port}"), "GET", "/health", None, None, Duration::from_secs(2)).await.is_ok_and(|r| r.status == 200) {
            return Ok(());
        }
        let p = unit_props(&unit(port)).await;
        if p.get("ActiveState").is_some_and(|s| s == "failed" || s == "inactive") {
            bail!("{} exited: {}", unit(port), log(port, 15).await.unwrap_or_default());
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    bail!("not ready after {}s", timeout.as_secs())
}

/// Stop the server on `port`: its unit, or a hand-started rllm serve
/// process with that port (matched by pid, never by command-line search).
pub async fn stop(state_dir: &Path, port: u16) -> Result<String> {
    let name = unit(port);
    let p = unit_props(&name).await;
    if p.get("LoadState").is_some_and(|s| s == "loaded") {
        let _ = sys::run("systemctl", &["--user", "stop", &format!("{name}.service")]).await;
        let _ = sys::run("systemctl", &["--user", "reset-failed", &format!("{name}.service")]).await;
        let mut known = load_state(state_dir);
        known.remove(&port);
        save_state(state_dir, &known);
        return Ok(format!("{name} stopped"));
    }
    let hits: Vec<_> = sys::procs_named("rllm").into_iter().filter(|x| x.args.split_whitespace().any(|a| a == "serve") && port_of(&x.args) == Some(port)).collect();
    ensure!(!hits.is_empty(), "no rllm server on port {port}");
    for h in &hits {
        // SAFETY: plain signal to a pid read from /proc just now.
        unsafe { libc::kill(h.pid, libc::SIGTERM) };
    }
    tokio::time::sleep(Duration::from_secs(3)).await;
    for h in sys::procs_named("rllm").into_iter().filter(|x| hits.iter().any(|h| h.pid == x.pid)) {
        // SAFETY: as above.
        unsafe { libc::kill(h.pid, libc::SIGKILL) };
    }
    Ok(format!("stopped rllm serve pid {}", hits.iter().map(|h| h.pid.to_string()).collect::<Vec<_>>().join(",")))
}

pub async fn log(port: u16, lines: usize) -> Result<String> {
    let (_, out) = sys::run("journalctl", &["--user", "-u", &format!("{}.service", unit(port)), "-n", &lines.to_string(), "-o", "cat", "--no-pager"]).await?;
    Ok(out)
}
