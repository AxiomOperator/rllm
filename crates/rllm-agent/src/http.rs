//! A small HTTP/1.1 client for agent-to-agent calls and local server polls
//! (plain HTTP on the LAN; large files go through curl, which resumes).

use std::time::Duration;

use anyhow::{Context, Result, bail};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

/// `method` `http://addr/path` with an optional bearer token and JSON body.
pub async fn request(addr: &str, method: &str, path: &str, token: Option<&str>, body: Option<&[u8]>, timeout: Duration) -> Result<Response> {
    tokio::time::timeout(timeout, request_inner(addr, method, path, token, body))
        .await
        .with_context(|| format!("{addr}: no answer within {}s", timeout.as_secs()))?
}

async fn request_inner(addr: &str, method: &str, path: &str, token: Option<&str>, body: Option<&[u8]>) -> Result<Response> {
    let mut s = TcpStream::connect(addr).await.with_context(|| format!("connecting to {addr}"))?;
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nAccept: */*\r\n");
    if let Some(t) = token {
        req += &format!("Authorization: Bearer {t}\r\n");
    }
    if let Some(b) = body {
        req += &format!("Content-Type: application/json\r\nContent-Length: {}\r\n", b.len());
    }
    req += "\r\n";
    s.write_all(req.as_bytes()).await?;
    if let Some(b) = body {
        s.write_all(b).await?;
    }
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).await?;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").context("malformed HTTP response")?;
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let mut rest = raw[split + 4..].to_vec();
    let mut lines = head.lines();
    let status: u16 = lines.next().and_then(|l| l.split_whitespace().nth(1)).and_then(|c| c.parse().ok()).context("no status line")?;
    let mut content_type = String::new();
    let mut chunked = false;
    for l in lines {
        let lower = l.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-type:") {
            content_type = v.trim().to_string();
        }
        if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
    }
    if chunked {
        rest = dechunk(&rest)?;
    }
    Ok(Response { status, content_type, body: rest })
}

fn dechunk(mut b: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let eol = b.windows(2).position(|w| w == b"\r\n").context("bad chunk header")?;
        let size_str = String::from_utf8_lossy(&b[..eol]);
        let size = usize::from_str_radix(size_str.split(';').next().unwrap_or("").trim(), 16).context("bad chunk size")?;
        b = &b[eol + 2..];
        if size == 0 {
            return Ok(out);
        }
        if b.len() < size + 2 {
            bail!("truncated chunk");
        }
        out.extend_from_slice(&b[..size]);
        b = &b[size + 2..];
    }
}

/// GET JSON from a peer agent.
pub async fn get_json(addr: &str, path: &str, token: &str) -> Result<serde_json::Value> {
    let r = request(addr, "GET", path, Some(token), None, Duration::from_secs(10)).await?;
    if r.status != 200 {
        bail!("{addr}{path}: HTTP {} {}", r.status, String::from_utf8_lossy(&r.body).chars().take(200).collect::<String>());
    }
    Ok(serde_json::from_slice(&r.body)?)
}
