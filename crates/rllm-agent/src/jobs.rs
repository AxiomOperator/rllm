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
    Cancelled,
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
    /// A file being moved (model fetches): what, how far, how fast.
    pub transfer: Option<Transfer>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Transfer {
    pub model: String,
    /// "Hugging Face" or the peer's address.
    pub source: String,
    /// preparing, downloading, joining, verifying
    pub phase: String,
    pub done: u64,
    pub total: u64,
    /// Bytes per second over the last few seconds (current phase).
    pub rate: f64,
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[derive(Default)]
struct Inner {
    next: u64,
    jobs: VecDeque<Job>,
    /// Running jobs' cancel switches.
    cancels: std::collections::HashMap<u64, tokio::sync::oneshot::Sender<()>>,
}

/// The agent's job registry.
#[derive(Clone)]
pub struct Jobs {
    inner: Arc<Mutex<Inner>>,
    history: PathBuf,
}

/// Aborts spawned tasks when dropped, so cancelling a job stops the tasks
/// it spawned (a dropped JoinHandle would leave them running).
#[derive(Default)]
pub struct Tasks(Vec<tokio::task::AbortHandle>);

impl Tasks {
    pub fn add<T>(&mut self, h: &tokio::task::JoinHandle<T>) {
        self.0.push(h.abort_handle());
    }
}

impl Drop for Tasks {
    fn drop(&mut self) {
        for h in &self.0 {
            h.abort();
        }
    }
}

/// A running job's handle: everything it reports goes through here.
#[derive(Clone)]
pub struct JobCtx {
    jobs: Jobs,
    pub id: u64,
}

impl Jobs {
    pub fn new(history: PathBuf) -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { next: 1, ..Default::default() })), history }
    }

    /// Start `f` as a job; returns its id at once.
    pub fn spawn<F, Fut>(&self, kind: &str, title: String, f: F) -> u64
    where
        F: FnOnce(JobCtx) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = anyhow::Result<String>> + Send + 'static,
    {
        let (cancel, cancelled) = tokio::sync::oneshot::channel::<()>();
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
                transfer: None,
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
            g.cancels.insert(id, cancel);
            id
        };
        let ctx = JobCtx { jobs: self.clone(), id };
        tokio::spawn(async move {
            // Cancelling drops the job's future: its child processes are
            // killed (kill_on_drop) and its guards clean up.
            let r = tokio::select! {
                r = f(ctx.clone()) => Some(r),
                Ok(()) = cancelled => None,
            };
            ctx.finish(r);
        });
        id
    }

    /// Cancel a running job; false if it isn't running.
    pub fn cancel(&self, id: u64) -> bool {
        let tx = self.inner.lock().unwrap().cancels.remove(&id);
        tx.is_some_and(|tx| tx.send(()).is_ok())
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

    /// Set or change the job's transfer.
    pub fn transfer(&self, f: impl FnOnce(&mut Transfer)) {
        self.jobs.update(self.id, |j| f(j.transfer.get_or_insert_with(Transfer::default)));
    }

    /// `None`: cancelled.
    fn finish(&self, r: Option<anyhow::Result<String>>) {
        self.jobs.inner.lock().unwrap().cancels.remove(&self.id);
        let mut done = None;
        self.jobs.update(self.id, |j| {
            j.finished = Some(now());
            match r {
                None => {
                    j.state = JobState::Cancelled;
                    j.summary = "cancelled".into();
                    j.log.push_back("cancelled".into());
                }
                Some(Ok(s)) => {
                    j.state = JobState::Ok;
                    j.progress = Some(1.0);
                    j.summary = s;
                }
                Some(Err(e)) => {
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
