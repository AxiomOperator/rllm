//! Background jobs (downloads, deploys, test runs): state, progress and a
//! bounded log, visible over the API while they run. Finished jobs are
//! appended to a history file so test results survive agent restarts.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

const MAX_LOG_LINES: usize = 400;
const MAX_JOBS: usize = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Running,
    Ok,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
pub struct Job {
    pub id: u64,
    pub kind: String,
    pub title: String,
    pub state: JobState,
    /// 0..1 when known.
    pub progress: Option<f64>,
    /// One-line outcome (results of a test suite, why it failed).
    pub summary: String,
    pub started: u64,
    pub finished: Option<u64>,
    pub log: VecDeque<String>,
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[derive(Default)]
struct Inner {
    next: u64,
    jobs: VecDeque<Job>,
}

/// The agent's job registry.
#[derive(Clone)]
pub struct Jobs {
    inner: Arc<Mutex<Inner>>,
    history: PathBuf,
}

/// A running job's handle: everything it reports goes through here.
#[derive(Clone)]
pub struct JobCtx {
    jobs: Jobs,
    pub id: u64,
}

impl Jobs {
    pub fn new(history: PathBuf) -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { next: 1, jobs: VecDeque::new() })), history }
    }

    /// Start `f` as a job; returns its id at once.
    pub fn spawn<F, Fut>(&self, kind: &str, title: String, f: F) -> u64
    where
        F: FnOnce(JobCtx) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = anyhow::Result<String>> + Send + 'static,
    {
        let id = {
            let mut g = self.inner.lock().unwrap();
            let id = g.next;
            g.next += 1;
            g.jobs.push_back(Job {
                id,
                kind: kind.into(),
                title,
                state: JobState::Running,
                progress: None,
                summary: String::new(),
                started: now(),
                finished: None,
                log: VecDeque::new(),
            });
            while g.jobs.len() > MAX_JOBS {
                // Drop the oldest finished job (running ones stay).
                match g.jobs.iter().position(|j| j.state != JobState::Running) {
                    Some(i) => {
                        g.jobs.remove(i);
                    }
                    None => break,
                }
            }
            id
        };
        let ctx = JobCtx { jobs: self.clone(), id };
        tokio::spawn(async move {
            let r = f(ctx.clone()).await;
            ctx.finish(r);
        });
        id
    }

    pub fn list(&self) -> Vec<Job> {
        let g = self.inner.lock().unwrap();
        g.jobs.iter().rev().map(|j| Job { log: j.log.iter().rev().take(8).rev().cloned().collect(), ..j.clone() }).collect()
    }

    pub fn get(&self, id: u64) -> Option<Job> {
        self.inner.lock().unwrap().jobs.iter().find(|j| j.id == id).cloned()
    }

    pub fn running(&self, kind: &str) -> bool {
        self.inner.lock().unwrap().jobs.iter().any(|j| j.kind == kind && j.state == JobState::Running)
    }

    /// Finished jobs from the history file, newest first.
    pub fn history(&self, limit: usize) -> Vec<serde_json::Value> {
        let Ok(text) = std::fs::read_to_string(&self.history) else { return Vec::new() };
        text.lines().rev().take(limit).filter_map(|l| serde_json::from_str(l).ok()).collect()
    }

    fn update(&self, id: u64, f: impl FnOnce(&mut Job)) {
        if let Some(j) = self.inner.lock().unwrap().jobs.iter_mut().find(|j| j.id == id) {
            f(j);
        }
    }
}

impl JobCtx {
    pub fn log(&self, line: impl Into<String>) {
        let line = line.into();
        self.jobs.update(self.id, |j| {
            j.log.push_back(line);
            while j.log.len() > MAX_LOG_LINES {
                j.log.pop_front();
            }
        });
    }

    pub fn progress(&self, p: f64) {
        self.jobs.update(self.id, |j| j.progress = Some(p.clamp(0.0, 1.0)));
    }

    fn finish(&self, r: anyhow::Result<String>) {
        let mut done = None;
        self.jobs.update(self.id, |j| {
            j.finished = Some(now());
            match r {
                Ok(s) => {
                    j.state = JobState::Ok;
                    j.progress = Some(1.0);
                    j.summary = s;
                }
                Err(e) => {
                    j.state = JobState::Failed;
                    j.summary = format!("{e:#}");
                    j.log.push_back(format!("error: {e:#}"));
                }
            }
            done = Some(serde_json::json!({
                "id": j.id, "kind": j.kind, "title": j.title, "state": j.state,
                "summary": j.summary, "started": j.started, "finished": j.finished,
            }));
        });
        if let Some(rec) = done {
            if let Some(dir) = self.jobs.history.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.jobs.history) {
                let _ = writeln!(f, "{rec}");
            }
        }
    }
}
