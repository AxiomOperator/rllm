//! A small HTTP/1.1 client for agent-to-agent calls and local server polls
//! (plain HTTP on the LAN; large files go through curl, which resumes).

use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

/// `method` `http://addr/path` with an optional bearer token and JSON body.
pub async fn request(addr: &str, method: &str, path: &str, token: Option<&str>, body: Option<&[u8]>, timeout: Duration) -> Result<Response> {
    tokio::time::timeout(timeout, request_inner(addr, method, path, token, body))
        .await
        .with_context(|| format!("{addr}: no answer within {}s", timeout.as_secs()))?
}

async fn request_inner(addr: &str, method: &str, path: &str, token: Option<&str>, body: Option<&[u8]>) -> Result<Response> {
    ensure!(!path.bytes().any(|b| b.is_ascii_control() || b == b' '), "bad request path");
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
    let mut chunked = false;
    for l in lines {
        let lower = l.to_ascii_lowercase();
        if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
    }
    if chunked {
        rest = dechunk(&rest)?;
    }
    Ok(Response { status, body: rest })
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

/// A response whose body is still streaming in (see [`stream`]).
pub struct Streaming {
    pub status: u16,
    /// Headers worth passing on (content type, length, range).
    pub headers: Vec<(String, String)>,
    pub body: axum::body::Body,
}

/// `method` `http://addr/path`, returning once the headers are in; the body
/// streams through a small bounded channel, so a large file is never held
/// in memory. Sent as HTTP/1.0, so the answer is not chunked and the body
/// simply runs to the end of the connection.
pub async fn stream(addr: &str, method: &str, path: &str, token: Option<&str>, body: Option<&[u8]>, timeout: Duration) -> Result<Streaming> {
    ensure!(!path.bytes().any(|b| b.is_ascii_control() || b == b' '), "bad request path");
    let mut s = tokio::time::timeout(Duration::from_secs(10), TcpStream::connect(addr)).await.with_context(|| format!("{addr}: no connection"))?.with_context(|| format!("connecting to {addr}"))?;
    let mut req = format!("{method} {path} HTTP/1.0\r\nHost: {addr}\r\nAccept: */*\r\n");
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
    // Headers (at most 64 KiB), within the timeout.
    let mut raw = Vec::new();
    let mut buf = vec![0u8; 16384];
    let split = tokio::time::timeout(timeout, async {
        loop {
            let n = s.read(&mut buf).await?;
            ensure!(n > 0, "connection closed before the headers");
            raw.extend_from_slice(&buf[..n]);
            if let Some(p) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                return Ok(p);
            }
            ensure!(raw.len() < 65536, "headers too large");
        }
    })
    .await
    .with_context(|| format!("{addr}: no answer within {}s", timeout.as_secs()))??;
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let first = raw[split + 4..].to_vec();
    let mut lines = head.lines();
    let status: u16 = lines.next().and_then(|l| l.split_whitespace().nth(1)).and_then(|c| c.parse().ok()).context("no status line")?;
    let mut headers = Vec::new();
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let k = k.trim().to_ascii_lowercase();
            if ["content-type", "content-length", "content-range", "accept-ranges", "last-modified", "etag"].contains(&k.as_str()) {
                headers.push((k, v.trim().to_string()));
            }
        }
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<std::io::Result<axum::body::Bytes>>(8);
    tokio::spawn(async move {
        if !first.is_empty() && tx.send(Ok(first.into())).await.is_err() {
            return;
        }
        let mut buf = vec![0u8; 1 << 16];
        loop {
            match s.read(&mut buf).await {
                Ok(0) => return,
                Ok(n) => {
                    if tx.send(Ok(axum::body::Bytes::copy_from_slice(&buf[..n]))).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            }
        }
    });
    Ok(Streaming { status, headers, body: axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)) })
}

/// GET JSON from a peer agent.
pub async fn get_json(addr: &str, path: &str, token: &str) -> Result<serde_json::Value> {
    let r = request(addr, "GET", path, Some(token), None, Duration::from_secs(10)).await?;
    if r.status != 200 {
        bail!("{addr}{path}: HTTP {} {}", r.status, String::from_utf8_lossy(&r.body).chars().take(200).collect::<String>());
    }
    Ok(serde_json::from_slice(&r.body)?)
}
