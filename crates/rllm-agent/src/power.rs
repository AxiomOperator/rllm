//! APU power limits (ryzenadj): what's configured, what's measured, and
//! applying a profile. ryzenadj needs root: the agent uses `sudo -n` and
//! says which sudoers line to add when that isn't allowed.

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::sys;

const RYZENADJ: &str = "/usr/local/bin/ryzenadj";

#[derive(Serialize)]
pub struct Power {
    pub ryzenadj_installed: bool,
    /// `sudo -n ryzenadj` works for the agent's user.
    pub can_apply: bool,
    /// ryzenadj.service (applies limits at boot): its state and command.
    pub boot_service: Option<String>,
    pub boot_limits: Option<String>,
    /// From `ryzenadj --info` when sudo allows it.
    pub current: Option<String>,
    /// The live (sustained, slow, fast) limits in W, and which profile they
    /// are ("balanced", "performance" or "custom").
    pub limits_w: Option<[f64; 3]>,
    pub active_profile: Option<String>,
    pub sudoers_hint: String,
}

#[derive(Deserialize)]
pub struct Apply {
    /// "balanced" (85/85/100 W), "performance" (120/120/140 W) or "custom".
    pub profile: String,
    pub stapm_w: Option<u32>,
    pub slow_w: Option<u32>,
    pub fast_w: Option<u32>,
}

fn user() -> String {
    std::env::var("USER").unwrap_or_else(|_| "administrator".into())
}

pub async fn status() -> Power {
    let installed = std::path::Path::new(RYZENADJ).is_file();
    let can_apply = installed && sys::run("sudo", &["-n", "-l", RYZENADJ]).await.is_ok_and(|(ok, _)| ok);
    let (_, svc) = sys::run("systemctl", &["show", "ryzenadj.service", "-p", "ActiveState,UnitFileState,ExecStart"]).await.unwrap_or_default();
    let get = |k: &str| svc.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).unwrap_or("").to_string();
    let boot_service = (!get("UnitFileState").is_empty()).then(|| format!("{} ({})", get("UnitFileState"), get("ActiveState")));
    let boot_limits = {
        let e = get("ExecStart");
        let lim: Vec<String> = e
            .split_whitespace()
            .filter(|a| a.starts_with("--stapm-limit") || a.starts_with("--slow-limit") || a.starts_with("--fast-limit"))
            .map(|a| a.trim_end_matches(';').to_string())
            .collect();
        (!lim.is_empty()).then(|| lim.join(" "))
    };
    let info = if can_apply { sys::run("sudo", &["-n", RYZENADJ, "--info"]).await.ok().map(|(_, out)| out) } else { None };
    // `| STAPM LIMIT | 120.000 | stapm-limit |` rows.
    let value = |key: &str| -> Option<f64> {
        info.as_deref()?.lines().find(|l| l.split('|').nth(1).is_some_and(|k| k.trim() == key))?.split('|').nth(2)?.trim().parse().ok()
    };
    let current = info.as_ref().map(|out| {
        out.lines()
            .filter(|l| ["STAPM LIMIT", "PPT LIMIT FAST", "PPT LIMIT SLOW", "THM LIMIT CORE"].iter().any(|k| l.contains(k)))
            .map(|l| l.split('|').map(str::trim).filter(|s| !s.is_empty()).take(2).collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>()
            .join(", ")
    });
    let limits_w = match (value("STAPM LIMIT"), value("PPT LIMIT SLOW"), value("PPT LIMIT FAST")) {
        (Some(a), Some(b), Some(c)) => Some([a, b, c]),
        _ => None,
    };
    let active_profile = limits_w.map(|l| {
        let near = |p: [f64; 3]| l.iter().zip(p).all(|(a, b)| (a - b).abs() < 1.0);
        if near([85.0, 85.0, 100.0]) {
            "balanced".to_string()
        } else if near([120.0, 120.0, 140.0]) {
            "performance".to_string()
        } else {
            "custom".to_string()
        }
    });
    Power {
        ryzenadj_installed: installed,
        can_apply,
        boot_service,
        boot_limits,
        current,
        limits_w,
        active_profile,
        sudoers_hint: format!("echo '{} ALL=(root) NOPASSWD: {RYZENADJ}' | sudo tee /etc/sudoers.d/rllm-ryzenadj && sudo chmod 440 /etc/sudoers.d/rllm-ryzenadj", user()),
    }
}

/// Apply a power profile now (until reboot; ryzenadj.service sets the boot one).
pub async fn apply(a: Apply) -> Result<String> {
    let (stapm, slow, fast) = match a.profile.as_str() {
        "balanced" => (85, 85, 100),
        "performance" => (120, 120, 140),
        "custom" => (a.stapm_w.unwrap_or(0), a.slow_w.unwrap_or(0), a.fast_w.unwrap_or(0)),
        other => bail!("profile {other:?} (balanced, performance or custom)"),
    };
    for (n, w) in [("stapm", stapm), ("slow", slow), ("fast", fast)] {
        ensure!((30..=160).contains(&w), "{n} limit {w} W outside 30..160 W");
    }
    ensure!(stapm <= slow && slow <= fast, "limits must satisfy stapm <= slow <= fast");
    let p = status().await;
    ensure!(p.ryzenadj_installed, "ryzenadj isn't installed at {RYZENADJ}");
    ensure!(p.can_apply, "the agent may not run ryzenadj; allow it with: {}", p.sudoers_hint);
    let args = [format!("--stapm-limit={}", stapm * 1000), format!("--slow-limit={}", slow * 1000), format!("--fast-limit={}", fast * 1000)];
    let mut v = vec!["-n", RYZENADJ];
    v.extend(args.iter().map(String::as_str));
    let (ok, out) = sys::run("sudo", &v).await?;
    ensure!(ok, "ryzenadj: {out}");
    Ok(format!("applied {stapm}/{slow}/{fast} W (sustained/slow/fast) until reboot"))
}
