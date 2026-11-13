//! Test suites run on this machine as jobs: GPU kernel parity, the model
//! tests, a benchmark, perplexity, and MTP-vs-no-MTP output identity. Plus a
//! scheduler for running them periodically.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::jobs::JobCtx;
use crate::{models, sys};

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Schedule {
    pub suite: String,
    #[serde(default)]
    pub model: Option<String>,
    pub every_hours: f64,
}

pub const SUITES: [&str; 6] = ["parity", "model", "bench", "perplexity", "mtp", "full"];

/// Run a command, streaming its output into the job log; returns all output.
async fn stream(ctx: &JobCtx, mut cmd: Command) -> Result<(bool, String)> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let mut child = cmd.spawn().context("starting test process")?;
    let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let collect = |r: Box<dyn tokio::io::AsyncRead + Unpin + Send>, ctx: JobCtx| {
        tokio::spawn(async move {
            let mut all = String::new();
            let mut lines = BufReader::new(r).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                ctx.log(l.clone());
                all.push_str(&l);
                all.push('\n');
            }
            all
        })
    };
    let (a, b) = (collect(Box::new(out), ctx.clone()), collect(Box::new(err), ctx.clone()));
    let status = child.wait().await?;
    let mut s = a.await.unwrap_or_default();
    s.push_str(&b.await.unwrap_or_default());
    Ok((status.success(), s))
}

fn test_result(out: &str) -> Option<String> {
    out.lines().rev().find(|l| l.starts_with("test result:")).map(|l| l.trim().to_string())
}

pub struct Env {
    pub bin_dir: PathBuf,
    pub model_dirs: Vec<PathBuf>,
    pub env: std::collections::BTreeMap<String, String>,
}

impl Env {
    /// A command in `bin_dir` with the configured environment.
    fn cmd(&self, prog: impl AsRef<std::ffi::OsStr>) -> Command {
        let mut c = Command::new(prog);
        c.current_dir(&self.bin_dir).envs(&self.env);
        c
    }
}

fn need_memory(env: &Env, model: &str) -> Result<PathBuf> {
    let path = models::find(&env.model_dirs, model)?;
    let info = models::list(&env.model_dirs).into_iter().find(|m| m.name == model).context("model vanished")?;
    let host = sys::host(&env.model_dirs[0]);
    let usable = sys::usable_bytes(&host);
    let need = info.weights_bytes + (3 << 30);
    ensure!(need <= usable, "not enough free GPU memory for {model}: needs about {} GiB, {} GiB usable (stop a server first)", need >> 30, usable >> 30);
    Ok(path)
}

async fn parity(ctx: &JobCtx, env: &Env) -> Result<String> {
    let bin = env.bin_dir.join("parity");
    ensure!(bin.is_file(), "{} not found", bin.display());
    let c = env.cmd(&bin);
    let (ok, out) = stream(ctx, c).await?;
    let r = test_result(&out).unwrap_or_else(|| "no test result line".into());
    ensure!(ok, "parity failed: {r}");
    Ok(format!("parity: {r}"))
}

async fn model_tests(ctx: &JobCtx, env: &Env, model: &str) -> Result<String> {
    let path = need_memory(env, model)?;
    let bin = env.bin_dir.join("gpu_tests");
    ensure!(bin.is_file(), "{} not found", bin.display());
    let mut c = env.cmd(&bin);
    c.env("RLLM_TEST_MODEL", &path).arg("--test-threads=1");
    let (ok, out) = stream(ctx, c).await?;
    let r = test_result(&out).unwrap_or_else(|| "no test result line".into());
    ensure!(ok, "model tests failed: {r}");
    Ok(format!("model tests: {r}"))
}

async fn bench(ctx: &JobCtx, env: &Env, model: &str) -> Result<String> {
    let path = need_memory(env, model)?;
    let mut c = env.cmd(env.bin_dir.join("rllm"));
    c.arg("bench").arg(&path).args(["-d", sys::device_kind(), "-p", "512", "-n", "128"]);
    let (ok, out) = stream(ctx, c).await?;
    ensure!(ok, "bench failed");
    let nums: Vec<String> = out
        .lines()
        .filter_map(|l| {
            let cols: Vec<&str> = l.split('|').map(str::trim).collect();
            let test = cols.iter().find(|c| c.starts_with("pp") || c.starts_with("tg"))?;
            let val = cols.iter().rev().find(|c| c.chars().next().is_some_and(|ch| ch.is_ascii_digit()))?;
            Some(format!("{test} {}", val.split_whitespace().next().unwrap_or(val)))
        })
        .collect();
    Ok(format!("bench: {} tok/s", nums.join(", ")))
}

async fn perplexity(ctx: &JobCtx, env: &Env, model: &str) -> Result<String> {
    let path = need_memory(env, model)?;
    let text = env.bin_dir.join("wiki.test.raw");
    ensure!(text.is_file(), "{} not found (perplexity needs wikitext-2's test split)", text.display());
    let mut c = env.cmd(env.bin_dir.join("rllm"));
    c.arg("perplexity").arg(&path).arg(&text).args(["-d", sys::device_kind(), "--chunks", "32"]);
    let (ok, out) = stream(ctx, c).await?;
    ensure!(ok, "perplexity failed");
    let line = out.lines().rev().find(|l| l.contains("PPL")).unwrap_or("no PPL line").trim().to_string();
    Ok(format!("perplexity: {line}"))
}

const MTP_PROMPTS: [(&str, u32); 4] = [
    ("Write a detailed essay about the history of computing.", 600),
    ("Write a Python function that parses a CSV file and computes per-column statistics, with tests.", 400),
    ("List the numbers from 1 to 30, separated by commas.", 200),
    ("Explain what a hash table is in three sentences.", 200),
];

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// Greedy outputs of a temporary server (with or without MTP).
async fn greedy_outputs(ctx: &JobCtx, env: &Env, path: &Path, mtp: bool) -> Result<(Vec<String>, f64)> {
    let port = free_port()?;
    let mut c = env.cmd(env.bin_dir.join("rllm"));
    c.arg("serve")
        .arg(path)
        .args(["-d", sys::device_kind(), "--host", "127.0.0.1", "--port", &port.to_string(), "--ctx", "8192", "--parallel", "4"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    if !mtp {
        c.arg("--no-mtp");
    }
    let mut child = c.spawn().context("starting a test server")?;
    let addr = format!("127.0.0.1:{port}");
    let ready = async {
        for _ in 0..300 {
            if crate::http::request(&addr, "GET", "/health", None, None, Duration::from_secs(2)).await.is_ok_and(|r| r.status == 200) {
                return true;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        false
    };
    if !ready.await {
        let _ = child.kill().await;
        bail!("test server didn't come up");
    }
    let t0 = std::time::Instant::now();
    let mut outs = Vec::new();
    for (p, n) in MTP_PROMPTS {
        let body = serde_json::json!({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0});
        let r = crate::http::request(&addr, "POST", "/v1/chat/completions", None, Some(body.to_string().as_bytes()), Duration::from_secs(600)).await?;
        ensure!(r.status == 200, "HTTP {}: {}", r.status, String::from_utf8_lossy(&r.body));
        let v: serde_json::Value = serde_json::from_slice(&r.body)?;
        let m = &v["choices"][0]["message"];
        outs.push(format!("{}|{}", m["reasoning_content"].as_str().unwrap_or(""), m["content"].as_str().unwrap_or("")));
    }
    let secs = t0.elapsed().as_secs_f64();
    let _ = child.kill().await;
    ctx.log(format!("{}: {:.1} s", if mtp { "with MTP" } else { "without MTP" }, secs));
    Ok((outs, secs))
}

async fn mtp(ctx: &JobCtx, env: &Env, model: &str) -> Result<String> {
    let path = need_memory(env, model)?;
    let (a, ta) = greedy_outputs(ctx, env, &path, true).await?;
    let (b, tb) = greedy_outputs(ctx, env, &path, false).await?;
    let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    ensure!(same == a.len(), "MTP changed the output of {} of {} prompts", a.len() - same, a.len());
    Ok(format!("mtp: outputs identical ({} prompts), {ta:.1} s with MTP vs {tb:.1} s without", a.len()))
}

/// Run `suite` (on `model` where it needs one).
pub async fn run(ctx: JobCtx, env: Env, suite: String, model: Option<String>) -> Result<String> {
    let need = || model.clone().context("this suite needs a model");
    match suite.as_str() {
        "parity" => parity(&ctx, &env).await,
        "model" => model_tests(&ctx, &env, &need()?).await,
        "bench" => bench(&ctx, &env, &need()?).await,
        "perplexity" => perplexity(&ctx, &env, &need()?).await,
        "mtp" => mtp(&ctx, &env, &need()?).await,
        "full" => {
            let m = need()?;
            let mut parts = Vec::new();
            let mut failed = false;
            let steps: [&str; 5] = ["parity", "model", "bench", "perplexity", "mtp"];
            for (i, s) in steps.iter().enumerate() {
                ctx.log(format!("== {s}"));
                let r = match *s {
                    "parity" => parity(&ctx, &env).await,
                    "model" => model_tests(&ctx, &env, &m).await,
                    "bench" => bench(&ctx, &env, &m).await,
                    "perplexity" => perplexity(&ctx, &env, &m).await,
                    _ => mtp(&ctx, &env, &m).await,
                };
                match r {
                    Ok(s) => parts.push(s),
                    Err(e) => {
                        failed = true;
                        parts.push(format!("{s} FAILED: {e:#}"));
                    }
                }
                ctx.progress((i + 1) as f64 / steps.len() as f64);
            }
            let s = parts.join(" | ");
            if failed { bail!(s) } else { Ok(s) }
        }
        other => bail!("unknown suite {other:?} (one of {})", SUITES.join(", ")),
    }
}
