//! What the machine looks like: identity, GPU memory and telemetry, RAM,
//! disk, and anything else running LLMs on it (llama-server is never left
//! running).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use tokio::process::Command;

fn read_trim(p: impl AsRef<Path>) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

#[derive(Serialize, Default)]
pub struct Gpu {
    pub name: String,
    pub integrated: bool,
    pub mem_used: u64,
    pub mem_total: u64,
    /// From the SMU's metrics table (AMD APUs), when available.
    pub gfx_mhz: Option<u32>,
    pub gfx_max_mhz: Option<u32>,
    pub busy_pct: Option<u32>,
    pub socket_watts: Option<f64>,
    pub temp_c: Option<f64>,
    pub mem_read_mbs: Option<u32>,
}

#[derive(Serialize)]
pub struct Host {
    pub hostname: String,
    pub vendor: String,
    pub product: String,
    pub kernel: String,
    pub uptime_s: u64,
    pub ram_total: u64,
    pub ram_available: u64,
    pub disk_free: u64,
    pub disk_total: u64,
    pub gpu: Option<Gpu>,
    /// `hip` or `cuda`: the `-d` value for this machine.
    pub device: String,
}

pub fn device_kind() -> &'static str {
    if Path::new("/dev/kfd").exists() { "hip" } else { "cuda" }
}

fn meminfo() -> HashMap<String, u64> {
    let mut m = HashMap::new();
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for l in s.lines() {
            let mut it = l.split_whitespace();
            if let (Some(k), Some(v)) = (it.next(), it.next())
                && let Ok(v) = v.parse::<u64>()
            {
                m.insert(k.trim_end_matches(':').to_string(), v * 1024);
            }
        }
    }
    m
}

/// (free, total) bytes of the filesystem holding `p`.
pub fn disk_space(p: &Path) -> (u64, u64) {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = CString::new(p.as_os_str().as_bytes()) else { return (0, 0) };
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a valid C string and `st` a properly sized out-parameter.
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return (0, 0);
    }
    (st.f_bavail as u64 * st.f_frsize as u64, st.f_blocks as u64 * st.f_frsize as u64)
}

/// The first AMD GPU's sysfs directory (card0, card1, ...).
fn amd_card() -> Option<PathBuf> {
    let mut cards: Vec<PathBuf> = std::fs::read_dir("/sys/class/drm")
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit())))
        .map(|p| p.join("device"))
        .filter(|d| d.join("mem_info_gtt_total").exists())
        .collect();
    cards.sort();
    cards.into_iter().next()
}

/// Decode `gpu_metrics` v3.0 (APUs: Strix Point/Halo). Layout from the
/// kernel's `struct gpu_metrics_v3_0` (264 bytes).
fn gpu_metrics_v3(b: &[u8], g: &mut Gpu) {
    if b.len() < 264 || u16::from_le_bytes([b[0], b[1]]) != 264 || b[2] != 3 || b[3] != 0 {
        return;
    }
    let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    // Offsets (natural alignment): temperature_gfx 4; average_gfx_activity
    // 42; average_dram_reads 94; system_clock_counter (u64) 104;
    // average_socket_power 112; core_power[16] 136..168; then sys_power,
    // stapm limits, average_gfxclk_frequency 174; current_coreclk[16]
    // 190..222, current_core_maxfreq 222, current_gfx_maxfreq 224.
    g.temp_c = Some(u16_at(4) as f64 / 100.0);
    g.busy_pct = Some(u16_at(42) as u32);
    g.mem_read_mbs = Some(u16_at(94) as u32);
    g.socket_watts = Some(u32_at(112) as f64 / 1000.0);
    g.gfx_mhz = Some(u16_at(174) as u32);
    g.gfx_max_mhz = Some(u16_at(224) as u32);
}

/// An NVIDIA GPU, through nvidia-smi (memory in MiB, power in W, clocks in MHz).
fn nvidia_gpu() -> Option<Gpu> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.used,memory.total,power.draw,clocks.gr,clocks.max.gr,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let f: Vec<&str> = text.lines().next()?.split(',').map(str::trim).collect();
    if f.len() < 8 {
        return None;
    }
    let num = |i: usize| f[i].parse::<f64>().ok();
    Some(Gpu {
        name: f[0].to_string(),
        integrated: false,
        mem_used: (num(1)? * 1048576.0) as u64,
        mem_total: (num(2)? * 1048576.0) as u64,
        gfx_mhz: num(4).map(|v| v as u32),
        gfx_max_mhz: num(5).map(|v| v as u32),
        busy_pct: num(7).map(|v| v as u32),
        socket_watts: num(3),
        temp_c: num(6),
        mem_read_mbs: None,
    })
}

pub fn gpu() -> Option<Gpu> {
    let Some(d) = amd_card() else { return nvidia_gpu() };
    let num = |f: &str| read_trim(d.join(f)).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    let vram_total = num("mem_info_vram_total");
    let mut g = Gpu {
        name: "AMD GPU".into(),
        // Integrated: a token VRAM carve-out, memory is GTT (system RAM).
        integrated: vram_total < (4 << 30),
        mem_used: num("mem_info_gtt_used") + num("mem_info_vram_used"),
        mem_total: num("mem_info_gtt_total") + vram_total,
        ..Default::default()
    };
    if let Ok(b) = std::fs::read(d.join("gpu_metrics")) {
        gpu_metrics_v3(&b, &mut g);
    }
    if let Some(n) = read_trim("/sys/class/dmi/id/product_name") {
        g.name = format!("AMD Radeon (in {n})");
    }
    Some(g)
}

pub fn host(model_dir: &Path) -> Host {
    let mi = meminfo();
    let (disk_free, disk_total) = disk_space(model_dir);
    let uptime_s = read_trim("/proc/uptime").and_then(|s| s.split('.').next().and_then(|x| x.parse().ok())).unwrap_or(0);
    Host {
        hostname: read_trim("/proc/sys/kernel/hostname").unwrap_or_default(),
        vendor: read_trim("/sys/class/dmi/id/sys_vendor").unwrap_or_default(),
        product: read_trim("/sys/class/dmi/id/product_name").unwrap_or_default(),
        kernel: read_trim("/proc/sys/kernel/osrelease").unwrap_or_default(),
        uptime_s,
        ram_total: *mi.get("MemTotal").unwrap_or(&0),
        ram_available: *mi.get("MemAvailable").unwrap_or(&0),
        disk_free,
        disk_total,
        gpu: gpu(),
        device: device_kind().into(),
    }
}

/// Memory a new server may plan to use: on an integrated GPU the smaller of
/// free GPU memory (less the last 4 GiB, which can't be allocated) and
/// available RAM (less 8 GiB for everything else), as `rllm tune` budgets.
pub fn usable_bytes(h: &Host) -> u64 {
    match &h.gpu {
        Some(g) if g.integrated => {
            let gpu_free = g.mem_total.saturating_sub(g.mem_used).saturating_sub(4 << 30);
            gpu_free.min(h.ram_available.saturating_sub(8 << 30))
        }
        Some(g) => g.mem_total.saturating_sub(g.mem_used),
        None => h.ram_available.saturating_sub(8 << 30),
    }
}

/// A process: pid, its command line, seconds running.
#[derive(Serialize, Clone)]
pub struct Proc {
    pub pid: i32,
    pub args: String,
    pub age_s: u64,
}

fn proc_args(pid: i32) -> Option<String> {
    let b = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    Some(b.split(|&c| c == 0).filter(|s| !s.is_empty()).map(|s| String::from_utf8_lossy(s).to_string()).collect::<Vec<_>>().join(" "))
}

fn proc_comm(pid: i32) -> Option<String> {
    read_trim(format!("/proc/{pid}/comm"))
}

fn proc_age(pid: i32) -> u64 {
    // Field 22 of /proc/pid/stat: start time in clock ticks since boot.
    let Some(stat) = read_trim(format!("/proc/{pid}/stat")) else { return 0 };
    let after = stat.rsplit_once(')').map(|(_, r)| r).unwrap_or("");
    let start: u64 = after.split_whitespace().nth(19).and_then(|s| s.parse().ok()).unwrap_or(0);
    // SAFETY: sysconf has no preconditions.
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as u64;
    let up = read_trim("/proc/uptime").and_then(|s| s.split('.').next().and_then(|x| x.parse::<u64>().ok())).unwrap_or(0);
    up.saturating_sub(start / hz)
}

/// Processes whose `comm` is exactly `name` (never a command-line match,
/// which would also catch shells mentioning it).
pub fn procs_named(name: &str) -> Vec<Proc> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/proc") {
        for e in rd.flatten() {
            let Ok(pid) = e.file_name().to_string_lossy().parse::<i32>() else { continue };
            if proc_comm(pid).as_deref() == Some(name)
                && let Some(args) = proc_args(pid)
            {
                out.push(Proc { pid, args, age_s: proc_age(pid) });
            }
        }
    }
    out.sort_by_key(|p| p.pid);
    out
}

/// Run a command, capturing stdout+stderr. Err on spawn failure only.
pub async fn run(cmd: &str, args: &[&str]) -> Result<(bool, String)> {
    let out = Command::new(cmd).args(args).kill_on_drop(true).output().await.with_context(|| format!("running {cmd}"))?;
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok((out.status.success(), s))
}

/// llama-server must never be left running: what's there now.
#[derive(Serialize)]
pub struct Llama {
    pub processes: Vec<Proc>,
    pub service_active: bool,
    pub service_enabled: bool,
}

pub async fn llama() -> Llama {
    let st = |a: &'static str| async move { run("systemctl", &["--user", a, "llama-rocm.service"]).await.map(|(_, s)| s.trim().to_string()).unwrap_or_default() };
    Llama { processes: procs_named("llama-server"), service_active: st("is-active").await == "active", service_enabled: st("is-enabled").await == "enabled" }
}

/// Stop the llama-rocm service (and disable it, so systemd doesn't restart
/// it), then kill any llama-server left.
pub async fn stop_llama() -> Result<String> {
    let mut msg = Vec::new();
    let l = llama().await;
    if l.service_active || l.service_enabled {
        let _ = run("systemctl", &["--user", "stop", "llama-rocm.service"]).await;
        let _ = run("systemctl", &["--user", "disable", "llama-rocm.service"]).await;
        msg.push("llama-rocm.service stopped and disabled".to_string());
    }
    for p in procs_named("llama-server") {
        // SAFETY: plain signal to a pid read from /proc just now.
        unsafe { libc::kill(p.pid, libc::SIGTERM) };
        msg.push(format!("sent SIGTERM to llama-server {}", p.pid));
    }
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    for p in procs_named("llama-server") {
        // SAFETY: as above.
        unsafe { libc::kill(p.pid, libc::SIGKILL) };
        msg.push(format!("sent SIGKILL to llama-server {}", p.pid));
    }
    if msg.is_empty() {
        msg.push("no llama-server running".into());
    }
    Ok(msg.join("; "))
}

/// sha256 of a file, cached by (path, size, mtime).
/// (path, size, mtime): identifies a file's contents for the hash cache.
type FileKey = (PathBuf, u64, i64);
static SHA_CACHE: Mutex<Option<HashMap<FileKey, String>>> = Mutex::new(None);

pub fn cached_sha256(p: &Path) -> Option<String> {
    let key = file_key(p)?;
    SHA_CACHE.lock().unwrap().as_ref().and_then(|m| m.get(&key).cloned())
}

fn file_key(p: &Path) -> Option<FileKey> {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::metadata(p).ok()?;
    Some((p.to_path_buf(), m.len(), m.mtime()))
}

/// Record a hash already verified (e.g. after renaming a checked file).
pub fn remember_sha256(p: &Path, h: &str) {
    if let Some(key) = file_key(p) {
        SHA_CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, h.to_string());
    }
}

pub async fn sha256(p: &Path) -> Result<String> {
    if let Some(h) = cached_sha256(p) {
        return Ok(h);
    }
    let key = file_key(p).with_context(|| format!("{} not found", p.display()))?;
    let (ok, out) = run("sha256sum", &[&p.to_string_lossy()]).await?;
    if !ok {
        bail!("sha256sum {}: {out}", p.display());
    }
    let h = out.split_whitespace().next().unwrap_or("").to_string();
    if h.len() != 64 {
        bail!("sha256sum {}: unexpected output {out:?}", p.display());
    }
    SHA_CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, h.clone());
    Ok(h)
}

/// A file name that stays inside its directory and is inert anywhere it is
/// shown or passed on (HTML, URLs, command lines): ASCII letters, digits
/// and `._+-`, not starting with a dot.
pub fn safe_name(n: &str) -> Result<&str> {
    let ok = !n.is_empty() && n.len() <= 255 && !n.starts_with('.') && n.bytes().all(|b| b.is_ascii_alphanumeric() || b"._+-".contains(&b));
    if !ok {
        bail!("bad file name {n:?} (use letters, digits and . _ + -)");
    }
    Ok(n)
}

/// curl flags for every transfer: http(s) only, also after redirects, and
/// no URL globbing.
pub const CURL_SAFE: [&str; 5] = ["--proto", "=http,https", "--proto-redir", "=http,https", "--globoff"];

/// curl reading an `Authorization: Bearer` header from stdin (`-K -`), so
/// the token never appears on a command line (`ps`, /proc/*/cmdline).
pub fn curl_command(auth: Option<&str>) -> Command {
    let mut c = Command::new("curl");
    c.args(CURL_SAFE).kill_on_drop(true);
    if auth.is_some() {
        c.args(["-K", "-"]).stdin(std::process::Stdio::piped());
    } else {
        c.stdin(std::process::Stdio::null());
    }
    c
}

/// Run a command from [`curl_command`], feeding it the token, and collect
/// its output.
pub async fn curl_output(mut c: Command, auth: Option<&str>) -> Result<std::process::Output> {
    use tokio::io::AsyncWriteExt;
    c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
    let mut child = c.spawn().context("running curl")?;
    if let Some(t) = auth {
        let mut stdin = child.stdin.take().context("curl stdin")?;
        let t = t.replace('\\', "\\\\").replace('"', "\\\"");
        stdin.write_all(format!("header = \"Authorization: Bearer {t}\"\n").as_bytes()).await?;
        drop(stdin);
    }
    Ok(child.wait_with_output().await?)
}

/// [`run`] for curl, with [`CURL_SAFE`] and the token passed on stdin.
pub async fn curl(args: &[&str], auth: Option<&str>) -> Result<(bool, String)> {
    let mut c = curl_command(auth);
    c.args(args);
    let out = curl_output(c, auth).await?;
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok((out.status.success(), s))
}

/// Refuse URLs pointing at this machine or link-local addresses (cloud
/// metadata): `/api/models/fetch` must not become a way to read local
/// services. LAN addresses stay allowed (other machines' file servers).
pub async fn check_fetch_target(url: &str) -> Result<()> {
    let rest = url.split_once("://").map(|(_, r)| r).context("not a URL")?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let hostport = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let (host, port) = if let Some(r) = hostport.strip_prefix('[') {
        let (h, p) = r.split_once(']').context("bad IPv6 host")?;
        (h.to_string(), p.strip_prefix(':').and_then(|p| p.parse().ok()).unwrap_or(443u16))
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse().context("bad port")?),
            None => (hostport.to_string(), 443),
        }
    };
    ensure!(!host.is_empty(), "no host in {url}");
    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host.as_str(), port)).await.with_context(|| format!("can't resolve {host}"))?.collect();
    for a in &addrs {
        let ip = match a.ip() {
            std::net::IpAddr::V6(v6) => v6.to_ipv4_mapped().map(std::net::IpAddr::V4).unwrap_or(std::net::IpAddr::V6(v6)),
            ip => ip,
        };
        let bad = match ip {
            std::net::IpAddr::V4(v) => v.is_loopback() || v.is_link_local() || v.is_unspecified() || v.is_multicast() || v.is_broadcast(),
            std::net::IpAddr::V6(v) => v.is_loopback() || v.is_unspecified() || v.is_multicast() || (v.segments()[0] & 0xffc0) == 0xfe80,
        };
        ensure!(!bad, "{host} resolves to {ip}: fetching from this machine or link-local addresses isn't allowed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_names() {
        assert!(safe_name("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf").is_ok());
        for bad in ["", ".hidden", "a/b", "x');alert(1);('.gguf", "a b.gguf", "a\"b", "a<b", "é.gguf"] {
            assert!(safe_name(bad).is_err(), "{bad:?}");
        }
    }

    #[tokio::test]
    async fn fetch_targets() {
        for bad in ["http://127.0.0.1:8090/api/status", "http://localhost/x", "http://169.254.169.254/latest", "http://[::1]:80/", "http://0.0.0.0/", "http://user@127.0.0.1/"] {
            assert!(check_fetch_target(bad).await.is_err(), "{bad}");
        }
        for ok in ["http://172.99.99.31:8000/m.gguf", "https://10.0.0.5/x"] {
            assert!(check_fetch_target(ok).await.is_ok(), "{ok}");
        }
    }
}
