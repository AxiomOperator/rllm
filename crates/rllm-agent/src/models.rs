//! Model files on this machine: listing, sizing a server for them, and
//! fetching them (from Hugging Face in parallel ranges, or from a peer agent
//! over the LAN), always checked against a SHA-256 before use.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use rllm_core::KvType;
use rllm_gguf::Gguf;
use rllm_model::ModelProfile;
use serde::Serialize;
use tokio::process::Command;

use crate::jobs::JobCtx;
use crate::sys;

#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
    pub arch: String,
    pub title: String,
    /// Device memory the weights take (file size plus repacking).
    pub weights_bytes: u64,
    pub kv_bytes_per_token_q8: u64,
    pub kv_bytes_per_token_f16: u64,
    pub state_bytes_per_seq: u64,
    pub recurrent: bool,
    pub mtp: bool,
    pub max_context: u64,
    pub error: Option<String>,
}

static PROFILES: Mutex<Option<HashMap<(PathBuf, u64), ModelInfo>>> = Mutex::new(None);

fn describe(path: &Path, size: u64) -> ModelInfo {
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let mut m = ModelInfo {
        name,
        path: path.display().to_string(),
        size,
        sha256: None,
        arch: String::new(),
        title: String::new(),
        weights_bytes: 0,
        kv_bytes_per_token_q8: 0,
        kv_bytes_per_token_f16: 0,
        state_bytes_per_seq: 0,
        recurrent: false,
        mtp: false,
        max_context: 0,
        error: None,
    };
    match Gguf::open(path).map_err(anyhow::Error::from).and_then(|g| ModelProfile::read(&g).map(|p| (g, p))) {
        Ok((g, p)) => {
            m.arch = p.arch.clone();
            m.title = g.get_str("general.name").unwrap_or_default().to_string();
            m.weights_bytes = (p.weight_bytes + p.repack_bytes) as u64;
            m.kv_bytes_per_token_q8 = p.kv_bytes_per_token(KvType::Q8) as u64;
            m.kv_bytes_per_token_f16 = p.kv_bytes_per_token(KvType::F16) as u64;
            m.state_bytes_per_seq = p.state_bytes_per_seq as u64;
            m.recurrent = rllm_model::is_recurrent(&g);
            m.mtp = p.mtp;
            m.max_context = p.max_context as u64;
        }
        Err(e) => m.error = Some(format!("{e:#}")),
    }
    m
}

/// The `.gguf` files in `dirs` (multi-part files' later parts skipped).
pub fn list(dirs: &[PathBuf]) -> Vec<ModelInfo> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            let n = e.file_name().to_string_lossy().to_string();
            if !n.ends_with(".gguf") || n.starts_with('.') || !seen.insert(n.clone()) {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&p) else { continue };
            let key = (p.clone(), meta.len());
            let cached = PROFILES.lock().unwrap().as_ref().and_then(|m| m.get(&key).cloned());
            let mut info = cached.unwrap_or_else(|| {
                let i = describe(&p, meta.len());
                PROFILES.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, i.clone());
                i
            });
            info.sha256 = sys::cached_sha256(&p);
            out.push(info);
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

pub fn find(dirs: &[PathBuf], name: &str) -> Result<PathBuf> {
    sys::safe_name(name)?;
    dirs.iter().map(|d| d.join(name)).find(|p| p.is_file()).with_context(|| format!("no model {name:?} on this machine"))
}

/// A server's memory needs.
#[derive(Serialize)]
pub struct Plan {
    pub kv_tokens: u64,
    pub need_bytes: u64,
    pub usable_bytes: u64,
    pub fits: bool,
    pub detail: String,
}

const RUNTIME_BYTES: u64 = 3 << 29; // 1.5 GiB: activations, logits, graphs (rllm tune measures 0.2-1.3)

pub fn plan(m: &ModelInfo, ctx: u64, parallel: u64, kv: KvType, kv_tokens: Option<u64>, usable: u64) -> Plan {
    let kv_tokens = kv_tokens.unwrap_or(ctx * parallel);
    let per_tok = if kv == KvType::Q8 { m.kv_bytes_per_token_q8 } else { m.kv_bytes_per_token_f16 };
    // State slots: parallel plus spares, and as many prefix-cache
    // checkpoints as parallel (serve's default); MTP snapshots about as
    // many again as the sequences'.
    let state = m.state_bytes_per_seq * ((parallel + 2) * if m.mtp { 2 } else { 1 } + if m.recurrent { parallel } else { 0 });
    let need = m.weights_bytes + kv_tokens * per_tok + state + RUNTIME_BYTES;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    Plan {
        kv_tokens,
        need_bytes: need,
        usable_bytes: usable,
        fits: need <= usable,
        detail: format!(
            "weights {:.1} + KV {:.1} ({} tokens) + state {:.1} + runtime {:.1} = {:.1} GiB, {:.1} GiB usable",
            gib(m.weights_bytes),
            gib(kv_tokens * per_tok),
            kv_tokens,
            gib(state),
            gib(RUNTIME_BYTES),
            gib(need),
            gib(usable)
        ),
    }
}

/// Where a model comes from.
pub enum Source {
    /// A peer agent: (addr, token).
    Peer(String, String),
    /// An http(s) URL. Hugging Face `.../resolve/<rev>/<file>` URLs get
    /// their SHA-256 from the Hub's API.
    Url(String),
}

/// curl's failure, said plainly: which host couldn't be reached and why.
fn curl_error(url: &str, out: &str) -> anyhow::Error {
    let host = url.split('/').nth(2).unwrap_or(url);
    let msg = out.lines().find(|l| l.starts_with("curl:")).unwrap_or(out).trim();
    let code: Option<u32> = msg.strip_prefix("curl: (").and_then(|r| r.split(')').next()).and_then(|c| c.parse().ok());
    match code {
        Some(6) => anyhow::anyhow!("can't resolve {host} from this machine ({msg})"),
        Some(7) => anyhow::anyhow!("{host} refused the connection from this machine ({msg})"),
        Some(28) if msg.contains("Connection timed out") => anyhow::anyhow!(
            "can't reach {host} from this machine: no answer in {CONNECT_TIMEOUT} s. Is it reachable from here, not only over a VPN on your own computer? ({msg})"
        ),
        _ => anyhow::anyhow!("{url}: {msg}"),
    }
}

/// Seconds to wait for a TCP connection.
const CONNECT_TIMEOUT: u32 = 15;

fn is_hf(url: &str) -> bool {
    url.starts_with("https://huggingface.co/") && url.contains("/resolve/")
}

/// Size of the file at `url` and whether its server serves byte ranges:
/// a GET for byte 0 that answers 206 with a Content-Range total, or a 200
/// with a Content-Length (one stream then).
async fn url_meta(url: &str) -> Result<(u64, bool)> {
    let ct = CONNECT_TIMEOUT.to_string();
    let (ok, out) = sys::curl(&["-sSL", "--connect-timeout", &ct, "--max-time", "60", "-r", "0-0", "-D", "-", "-o", "/dev/null", url], None).await?;
    // A server that ignores the range sends the whole file: the headers
    // are all that's needed, so running out of time after them is fine.
    if !ok && !(out.contains("curl: (28) Operation timed out") && out.contains("HTTP/")) {
        return Err(curl_error(url, &out));
    }
    // The headers of every hop; the last response is the file's.
    let last = out.split("\r\n\r\n").map(str::trim_start).filter(|b| b.starts_with("HTTP/")).last().unwrap_or("");
    let code: u16 = last.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let header = |name: &str| {
        last.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case(name).then(|| v.trim().to_string())
        })
    };
    match code {
        206 => {
            let total = header("content-range").and_then(|r| r.rsplit('/').next().and_then(|t| t.parse().ok()));
            Ok((total.context("206 without a Content-Range total")?, true))
        }
        200 => Ok((header("content-length").and_then(|l| l.parse().ok()).context("the server didn't say how big the file is")?, false)),
        c => bail!("{url}: HTTP {c}"),
    }
}

/// Expected (size, sha256) of a Hugging Face file, from the repo's tree API.
async fn hf_meta(url: &str) -> Result<(u64, String)> {
    // https://huggingface.co/<org>/<repo>/resolve/<rev>/<path>
    let rest = url.strip_prefix("https://huggingface.co/").context("not a huggingface.co URL")?;
    let (repo_rev, file) = rest.split_once("/resolve/").context("expected a .../resolve/<rev>/<file> URL")?;
    let (rev, path) = file.split_once('/').context("no file in the URL")?;
    let (dir, fname) = match path.rsplit_once('/') {
        Some((d, f)) => (format!("/{d}"), f),
        None => (String::new(), path),
    };
    let api = format!("https://huggingface.co/api/models/{repo_rev}/tree/{rev}{dir}");
    let ct = CONNECT_TIMEOUT.to_string();
    let (ok, body) = sys::curl(&["-sSfL", "--connect-timeout", &ct, "--max-time", "30", &api], None).await?;
    if !ok {
        return Err(curl_error(&api, &body)).context("asking the Hugging Face API");
    }
    let v: serde_json::Value = serde_json::from_str(&body).context("Hugging Face API answer")?;
    for e in v.as_array().into_iter().flatten() {
        if e["path"].as_str().is_some_and(|p| p.rsplit('/').next() == Some(fname)) {
            let size = e["size"].as_u64().context("no size")?;
            let sha = e["lfs"]["oid"].as_str().context("no LFS sha256 (not an LFS file?)")?.to_string();
            return Ok((size, sha));
        }
    }
    bail!("{fname} not listed by the Hugging Face API")
}

/// A range's scratch file: `<part>.try` (appended, not `with_extension`,
/// which would give every `.partN` the same name).
fn try_path(part: &Path) -> PathBuf {
    let mut s = part.as_os_str().to_owned();
    s.push(".try");
    PathBuf::from(s)
}

fn file_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// curl gives up (and the attempt is retried) on no connection in 15 s or
/// under 64 KB/s for a minute: a hung transfer doesn't hang the job.
const STALL: [&str; 6] = ["--connect-timeout", "15", "--speed-limit", "65536", "--speed-time", "60"];

/// Whether a `Content-Range` header value starts at byte `from`
/// (`bytes <from>-<to>/<total>`).
fn range_starts_at(content_range: &str, from: u64) -> bool {
    content_range.trim().strip_prefix("bytes ").and_then(|r| r.split('-').next()).and_then(|s| s.trim().parse::<u64>().ok()) == Some(from)
}

/// Fetch byte range [start, end] of `url` into `part`, resuming; an attempt
/// is kept only if the server answered 206 for exactly the requested offset.
async fn fetch_range(url: &str, auth: Option<&str>, part: &Path, start: u64, end: u64, ranged: bool) -> Result<()> {
    let len = end - start + 1;
    let tryf = try_path(part);
    let mut failures = 0;
    while file_len(part) < len {
        let got = file_len(part);
        let mut c = sys::curl_command(auth);
        if !ranged {
            // No ranges: the whole file in one go, from the start each try.
            // curl stops a body bigger than announced (a disk-filling server).
            c.args(STALL).args(["-sSL", "--max-time", "21600", "--max-filesize", &len.to_string(), "-o"]).arg(&tryf).args(["-w", "%{http_code}", url]);
            let out = sys::curl_output(c, auth).await?;
            let code = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if code == "200" && file_len(&tryf) == len {
                tokio::fs::rename(&tryf, part).await?;
            } else {
                let _ = tokio::fs::remove_file(&tryf).await;
                failures += 1;
                ensure!(failures < 5, "giving up after 5 failed attempts (last HTTP {code}, {} of {len} bytes)", file_len(&tryf));
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
            continue;
        }
        let want = len - got;
        c.args(STALL)
            .args(["-sSL", "--max-time", "1800", "--max-filesize", &want.to_string(), "-o"])
            .arg(&tryf)
            .args(["-w", "%{http_code} %header{content-range}", "-r", &format!("{}-{}", start + got, end)])
            .arg(url);
        let out = sys::curl_output(c, auth).await?;
        let w = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let (code, content_range) = w.split_once(' ').unwrap_or((w.as_str(), ""));
        let code = if code == "206" && !range_starts_at(content_range, start + got) { format!("206 for the wrong range ({content_range:?})") } else { code.to_string() };
        let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
        let n = file_len(&tryf);
        if code == "206" && got + n <= len && n > 0 && got == 0 {
            // The first attempt: it is the part.
            tokio::fs::rename(&tryf, part).await?;
            failures = 0;
        } else if code == "206" && got + n <= len && n > 0 {
            let mut from = tokio::fs::File::open(&tryf).await?;
            let mut to = tokio::fs::OpenOptions::new().create(true).append(true).open(part).await?;
            tokio::io::copy(&mut from, &mut to).await?;
            failures = 0;
        } else {
            failures += 1;
            ensure!(failures < 50, "range {start}-{end}: giving up after 50 failed attempts (last: HTTP {code} {why})");
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
        let _ = tokio::fs::remove_file(&tryf).await;
    }
    Ok(())
}

/// Bytes per second over the last few seconds, from (time, bytes) samples.
struct Meter(std::collections::VecDeque<(std::time::Instant, u64)>);

impl Meter {
    fn new() -> Self {
        Self(std::collections::VecDeque::new())
    }
    fn rate(&mut self, done: u64) -> f64 {
        let now = std::time::Instant::now();
        self.0.push_back((now, done));
        while self.0.len() > 2 && now.duration_since(self.0[0].0).as_secs_f64() > 8.0 {
            self.0.pop_front();
        }
        let (t0, d0) = self.0[0];
        let dt = now.duration_since(t0).as_secs_f64();
        if dt < 0.5 { 0.0 } else { done.saturating_sub(d0) as f64 / dt }
    }
}

/// How far process `pid` has read into `path` (its file offset), from /proc.
fn read_offset(pid: u32, path: &Path) -> Option<u64> {
    for e in std::fs::read_dir(format!("/proc/{pid}/fd")).ok()?.flatten() {
        if std::fs::read_link(e.path()).ok().as_deref() == Some(path) {
            let info = std::fs::read_to_string(format!("/proc/{pid}/fdinfo/{}", e.file_name().to_string_lossy())).ok()?;
            return info.lines().find_map(|l| l.strip_prefix("pos:")).and_then(|v| v.trim().parse().ok());
        }
    }
    None
}

/// SHA-256 of `path` with sha256sum, reporting how far it has read.
async fn sha256_watched(ctx: &JobCtx, path: &Path, size: u64, p0: f64, p1: f64) -> Result<String> {
    let path = std::fs::canonicalize(path)?;
    let mut child = Command::new("sha256sum").arg(&path).stdout(std::process::Stdio::piped()).kill_on_drop(true).spawn().context("running sha256sum")?;
    let pid = child.id().unwrap_or(0);
    let mut stdout = child.stdout.take().context("sha256sum stdout")?;
    let mut meter = Meter::new();
    let status = loop {
        tokio::select! {
            st = child.wait() => break st?,
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                if let Some(pos) = read_offset(pid, &path) {
                    let rate = meter.rate(pos);
                    ctx.transfer(|t| { t.done = pos; t.rate = rate; });
                    ctx.progress(p0 + (p1 - p0) * pos as f64 / size.max(1) as f64);
                }
            }
        }
    };
    let mut out = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut stdout, &mut out).await?;
    ensure!(status.success(), "sha256sum {} failed", path.display());
    let h = out.split_whitespace().next().unwrap_or("").to_string();
    ensure!(h.len() == 64, "sha256sum: unexpected output {out:?}");
    Ok(h)
}

/// Removes a fetch's scratch files if it's dropped unfinished (cancelled).
/// A failed fetch keeps its parts, so a retry resumes them.
struct Scratch {
    dir: PathBuf,
    name: String,
    done: bool,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let (dir, prefix) = (self.dir.clone(), format!(".{}.", self.name));
        let clean = move || {
            for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                if e.file_name().to_string_lossy().starts_with(&prefix) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        };
        clean();
        // Again once the aborted range tasks have stopped writing.
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            clean();
        });
    }
}

/// Fetches in progress: destination path -> disk bytes reserved for it
/// (one fetch per destination; the free-space check counts the others).
static ACTIVE: Mutex<Option<HashMap<PathBuf, u64>>> = Mutex::new(None);

/// A destination claimed in [`ACTIVE`], released when dropped.
struct Claim(PathBuf);

impl Claim {
    fn new(dest: PathBuf) -> Result<Self> {
        let mut g = ACTIVE.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        ensure!(!m.contains_key(&dest), "{} is already being downloaded", dest.display());
        m.insert(dest.clone(), 0);
        Ok(Claim(dest))
    }
    /// Reserve `bytes` for this fetch; returns what the others reserve.
    fn reserve(&self, bytes: u64) -> u64 {
        let mut g = ACTIVE.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        m.insert(self.0.clone(), bytes);
        m.iter().filter(|(k, _)| **k != self.0).map(|(_, v)| *v).sum()
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        if let Some(m) = ACTIVE.lock().unwrap().as_mut() {
            m.remove(&self.0);
        }
    }
}

/// Download `name` into `dir` from `src`, verify, and move it into place.
pub async fn fetch(ctx: JobCtx, dir: PathBuf, name: String, src: Source, want_sha: Option<String>) -> Result<String> {
    sys::safe_name(&name)?;
    // Claimed before the scratch guard, so a refused duplicate never
    // deletes the running fetch's parts.
    let claim = Claim::new(dir.join(&name))?;
    let mut scratch = Scratch { dir: dir.clone(), name: name.clone(), done: false };
    let r = fetch_inner(ctx, dir, name, src, want_sha, &claim).await;
    scratch.done = true;
    r
}

async fn fetch_inner(ctx: JobCtx, dir: PathBuf, name: String, src: Source, want_sha: Option<String>, claim: &Claim) -> Result<String> {
    let dest = dir.join(&name);
    ensure!(!dest.exists(), "{} already exists", dest.display());
    std::fs::create_dir_all(&dir)?;
    ctx.transfer(|t| {
        t.model = name.clone();
        t.source = match &src {
            Source::Url(u) if is_hf(u) => "Hugging Face".into(),
            Source::Url(u) => u.split('/').nth(2).unwrap_or(u).to_string(),
            Source::Peer(addr, _) => addr.clone(),
        };
        t.phase = "preparing".into();
    });
    let (url, auth, size, sha, ranged) = match &src {
        Source::Url(u) if is_hf(u) => {
            ctx.log(format!("asking Hugging Face for the size and SHA-256 of {u}"));
            let (size, sha) = hf_meta(u).await?;
            (u.clone(), None, size, Some(want_sha.unwrap_or(sha)), true)
        }
        Source::Url(u) => {
            ensure!(u.starts_with("http://") || u.starts_with("https://"), "only http(s) URLs");
            sys::check_fetch_target(u).await?;
            ctx.log(format!("asking {u} for the file's size"));
            let (size, ranged) = url_meta(u).await?;
            if !ranged {
                ctx.log("the server doesn't serve byte ranges: one stream");
            }
            if want_sha.is_none() {
                ctx.log("no SHA-256 given: the download is checked by size only");
            }
            (u.clone(), None, size, want_sha, ranged)
        }
        Source::Peer(addr, token) => {
            ctx.log(format!("asking {addr} for the size and SHA-256 of {name} (hashing may take a minute)"));
            let v = crate::http::request(addr, "GET", &format!("/api/models/{name}/sha256"), Some(token), None, std::time::Duration::from_secs(900)).await?;
            ensure!(v.status == 200, "{addr}: HTTP {} {}", v.status, String::from_utf8_lossy(&v.body));
            let j: serde_json::Value = serde_json::from_slice(&v.body)?;
            let size = j["size"].as_u64().context("no size")?;
            let sha = j["sha256"].as_str().context("no sha256")?.to_string();
            (format!("http://{addr}/api/files/models/{name}"), Some(token.clone()), size, Some(want_sha.unwrap_or(sha)), true)
        }
    };
    let sha = sha.map(|h| h.trim().to_ascii_lowercase());
    if let Some(h) = &sha {
        ensure!(h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()), "not a SHA-256: {h:?}");
    }
    ctx.log(format!("{name}: {size} bytes, sha256 {}", sha.as_deref().unwrap_or("unknown")));
    ctx.transfer(|t| {
        t.phase = "downloading".into();
        t.total = size;
    });
    // Peers on the LAN: 2 streams; the internet: 8 (one stream gets a
    // fraction of the link).
    let n: u64 = if !ranged { 1 } else if matches!(src, Source::Peer(..)) { 2 } else { 8 };
    let chunk = size.div_ceil(n);
    let parts: Vec<PathBuf> = (0..n).map(|i| dir.join(format!(".{name}.part{i}"))).collect();
    // Parts left by a failed attempt are resumed only if they were cut the
    // same way from the same file; otherwise their bytes would be joined at
    // the wrong offsets.
    let origin = dir.join(format!(".{name}.origin"));
    let this = format!("{url}\n{size}\n{n}\n");
    if std::fs::read_to_string(&origin).ok().as_deref() != Some(this.as_str()) {
        for p in &parts {
            let _ = std::fs::remove_file(p);
            let _ = std::fs::remove_file(try_path(p));
        }
        std::fs::write(&origin, &this)?;
    }
    // Disk: the file plus one part (parts are deleted as they are joined),
    // less what earlier attempts already wrote, plus 2 GiB to spare, after
    // what other running fetches have reserved.
    let have: u64 = parts.iter().map(|p| file_len(p)).sum();
    let need = (size + chunk).saturating_sub(have) + (2 << 30);
    let others = claim.reserve(need);
    let (free, _) = sys::disk_space(&dir);
    ensure!(
        free > need + others,
        "not enough disk: {} GiB free, {} GiB needed{}",
        free >> 30,
        need >> 30,
        if others > 0 { format!(" plus {} GiB reserved by other downloads", others >> 30) } else { String::new() }
    );
    let started = std::time::Instant::now();
    let mut tasks = Vec::new();
    let mut guard = crate::jobs::Tasks::default();
    for (i, part) in parts.iter().enumerate() {
        let start = i as u64 * chunk;
        if start >= size {
            continue;
        }
        let end = (start + chunk).min(size) - 1;
        let (url, auth, part) = (url.clone(), auth.clone(), part.clone());
        let t = tokio::spawn(async move { fetch_range(&url, auth.as_deref(), &part, start, end, ranged).await });
        guard.add(&t);
        tasks.push(t);
    }
    // Progress while the ranges download.
    let watch = {
        let (ctx, parts) = (ctx.clone(), parts.clone());
        tokio::spawn(async move {
            let mut meter = Meter::new();
            loop {
                let have: u64 = parts.iter().map(|p| file_len(p) + file_len(&try_path(p))).sum::<u64>().min(size);
                let rate = meter.rate(have);
                ctx.transfer(|t| {
                    t.done = have;
                    t.rate = rate;
                });
                ctx.progress(have as f64 / size.max(1) as f64 * 0.9);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        })
    };
    guard.add(&watch);
    let mut err = None;
    for t in tasks {
        if let Err(e) = t.await.map_err(anyhow::Error::from).and_then(|r| r) {
            err.get_or_insert(e);
        }
    }
    watch.abort();
    if let Some(e) = err {
        return Err(e);
    }
    let secs = started.elapsed().as_secs_f64().max(0.001);
    ctx.log(format!("downloaded in {secs:.0} s ({:.0} MB/s); joining parts", size as f64 / secs / 1e6));
    // The other parts are appended to the first, which becomes the file.
    let tmp = dir.join(format!(".{name}.tmp"));
    std::fs::rename(&parts[0], &tmp)?;
    ctx.transfer(|t| {
        t.phase = "joining".into();
        t.done = file_len(&tmp);
        t.rate = 0.0;
    });
    {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut out = tokio::fs::OpenOptions::new().append(true).open(&tmp).await?;
        let mut buf = vec![0u8; 8 << 20];
        let (mut done, mut meter, mut last) = (file_len(&tmp), Meter::new(), std::time::Instant::now());
        meter.rate(done);
        for p in &parts[1..] {
            if !p.exists() {
                continue;
            }
            let mut f = tokio::fs::File::open(p).await?;
            // Unlinked now (the open handle keeps it readable): its space is
            // freed as soon as it is copied, so the join never holds the
            // whole file twice.
            out.flush().await?;
            std::fs::remove_file(p)?;
            loop {
                let n = f.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n]).await?;
                done += n as u64;
                if last.elapsed().as_secs_f64() >= 1.0 {
                    last = std::time::Instant::now();
                    let rate = meter.rate(done);
                    ctx.transfer(|t| {
                        t.done = done;
                        t.rate = rate;
                    });
                    ctx.progress(0.9 + 0.03 * done as f64 / size.max(1) as f64);
                }
            }
        }
        out.flush().await?;
    }
    let _ = std::fs::remove_file(&origin);
    ensure!(file_len(&tmp) == size, "joined file is {} bytes, expected {size}", file_len(&tmp));
    ctx.log(if sha.is_some() { "verifying SHA-256" } else { "computing SHA-256" });
    ctx.transfer(|t| {
        t.phase = "verifying".into();
        t.done = 0;
        t.rate = 0.0;
    });
    let got = sha256_watched(&ctx, &tmp, size, 0.93, 1.0).await?;
    if let Some(sha) = &sha {
        if got != *sha {
            let _ = std::fs::remove_file(&tmp);
            bail!("SHA-256 mismatch: got {got}, expected {sha} (file removed)");
        }
    }
    std::fs::rename(&tmp, &dest)?;
    sys::remember_sha256(&dest, &got);
    let check = if sha.is_some() { "SHA-256 verified".to_string() } else { format!("SHA-256 {got} (not checked: none given)") };
    Ok(format!("{name}: {:.1} GB in {secs:.0} s, {check}", size as f64 / 1e9))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_range_offsets() {
        assert!(range_starts_at("bytes 100-199/1000", 100));
        assert!(!range_starts_at("bytes 0-199/1000", 100));
        assert!(!range_starts_at("", 0));
    }

    #[test]
    fn one_fetch_per_destination() {
        let a = Claim::new("/tmp/x/claim-test.gguf".into()).unwrap();
        assert!(Claim::new("/tmp/x/claim-test.gguf".into()).is_err());
        let b = Claim::new("/tmp/x/claim-test2.gguf".into()).unwrap();
        a.reserve(10);
        assert_eq!(b.reserve(5), 10);
        drop(a);
        assert_eq!(b.reserve(5), 0);
        assert!(Claim::new("/tmp/x/claim-test.gguf".into()).is_ok());
    }
}
