//! Binary deploys: a machine pulls `rllm`, `parity` and `gpu_tests` from a
//! peer, checks their SHA-256, runs the new `parity` on its own GPU, and
//! only then swaps them in (keeping the previous ones for rollback).

use std::path::Path;
use std::process::Stdio;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use tokio::process::Command;

use crate::jobs::JobCtx;
use crate::sys;

pub const BINARIES: [&str; 3] = ["rllm", "parity", "gpu_tests"];

#[derive(Serialize)]
pub struct Entry {
    pub name: String,
    pub size: u64,
    pub sha256: String,
    pub modified: u64,
}

pub async fn manifest(bin_dir: &Path) -> Result<Vec<Entry>> {
    let mut out = Vec::new();
    for n in BINARIES {
        let p = bin_dir.join(n);
        let Ok(m) = std::fs::metadata(&p) else { continue };
        use std::os::unix::fs::MetadataExt;
        out.push(Entry { name: n.into(), size: m.len(), sha256: sys::sha256(&p).await?, modified: m.mtime().max(0) as u64 });
    }
    Ok(out)
}

/// Pull the binaries from `addr`, gate on parity, install.
pub async fn deploy(ctx: JobCtx, bin_dir: std::path::PathBuf, addr: String, token: String, under_systemd: bool, env: std::collections::BTreeMap<String, String>) -> Result<String> {
    let v = crate::http::get_json(&addr, "/api/bin/manifest", &token).await?;
    let entries = v.as_array().context("bad manifest")?;
    ensure!(entries.iter().any(|e| e["name"] == "rllm") && entries.iter().any(|e| e["name"] == "parity"), "{addr} has no rllm/parity to deploy");
    let stage = bin_dir.join(".deploy");
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage)?;
    let mut same = 0;
    for (i, e) in entries.iter().enumerate() {
        let name = e["name"].as_str().context("name")?;
        ensure!(BINARIES.contains(&name), "{addr} offered {name:?}, which isn't a deployable binary");
        let sha = e["sha256"].as_str().context("sha256")?;
        if sys::sha256(&bin_dir.join(name)).await.is_ok_and(|h| h == sha) {
            same += 1;
        }
        let dest = stage.join(name);
        ctx.log(format!("fetching {name} ({} bytes)", e["size"]));
        let (ok, out) = sys::curl(&["-sSf", "--max-time", "600", "-o", &dest.to_string_lossy(), &format!("http://{addr}/api/files/bin/{name}")], Some(&token)).await?;
        ensure!(ok, "fetching {name}: {out}");
        let got = sys::sha256(&dest).await?;
        ensure!(got == sha, "{name}: SHA-256 mismatch ({got} vs {sha})");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))?;
        ctx.progress((i + 1) as f64 / entries.len() as f64 * 0.4);
    }
    if same == entries.len() {
        let _ = std::fs::remove_dir_all(&stage);
        return Ok("already up to date".into());
    }
    ctx.log("running the new parity tests on this GPU");
    let mut c = Command::new(stage.join("parity"));
    c.current_dir(&bin_dir).envs(&env).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let out = c.output().await.context("running parity")?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let result = text.lines().rev().find(|l| l.starts_with("test result:")).unwrap_or("no test result line").to_string();
    ctx.log(format!("parity: {result}"));
    if !out.status.success() {
        for l in text.lines().filter(|l| l.contains("FAILED") || l.contains("panicked")).take(20) {
            ctx.log(l);
        }
        bail!("new build failed parity here ({result}); nothing installed");
    }
    ctx.progress(0.9);
    // Install: keep the old binaries in .prev, then rename each in.
    let prev = bin_dir.join(".prev");
    let _ = std::fs::remove_dir_all(&prev);
    std::fs::create_dir_all(&prev)?;
    for e in entries {
        let name = e["name"].as_str().unwrap_or_default();
        let cur = bin_dir.join(name);
        if cur.exists() {
            std::fs::copy(&cur, prev.join(name))?;
        }
        std::fs::rename(stage.join(name), &cur)?;
        if let Some(h) = e["sha256"].as_str() {
            sys::remember_sha256(&cur, h);
        }
    }
    let _ = std::fs::remove_dir_all(&stage);
    let restart = if under_systemd {
        // Restart the agent onto the new binary in a moment (after this job
        // has reported).
        let _ = sys::run("systemd-run", &["--user", "--on-active=3", "--collect", "systemctl", "--user", "restart", "rllm-agent.service"]).await;
        "; agent restarting onto it"
    } else {
        "; restart the agent to run it"
    };
    Ok(format!("installed ({result}); servers keep their binary until restarted{restart}"))
}

/// Put back the binaries a deploy replaced.
pub fn rollback(bin_dir: &Path) -> Result<String> {
    let prev = bin_dir.join(".prev");
    ensure!(prev.is_dir(), "nothing to roll back to");
    let mut n = Vec::new();
    for e in std::fs::read_dir(&prev)?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        std::fs::rename(e.path(), bin_dir.join(&name))?;
        n.push(name);
    }
    Ok(format!("restored {}", n.join(", ")))
}
