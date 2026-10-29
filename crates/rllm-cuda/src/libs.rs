//! Locate NVRTC and cuBLAS when they aren't on the system library path.
//!
//! cudarc dlopens libraries by soname (e.g. `libnvrtc.so.13`). If we load
//! them first by full path with RTLD_GLOBAL, the dynamic linker hands back
//! these already-loaded copies, so no LD_LIBRARY_PATH is needed.
//!
//! Search order: `$RLLM_CUDA_LIB_DIR`, then `.cuda/lib` in the current
//! directory or any ancestor of the executable (the repo-local copy).

use std::path::{Path, PathBuf};
use std::sync::Once;

// Dependency order: builtins before nvrtc, cublasLt before cublas.
const LIBS: &[&str] = &["libnvrtc-builtins.so", "libnvrtc.so", "libcublasLt.so", "libcublas.so"];

fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("RLLM_CUDA_LIB_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join(".cuda/lib"));
    }
    if let Ok(exe) = std::env::current_exe() {
        dirs.extend(exe.ancestors().skip(1).map(|a| a.join(".cuda/lib")));
    }
    dirs
}

/// Find `prefix` or a versioned variant (`prefix.13`, `prefix.13.4`) in `dir`.
fn find(dir: &Path, prefix: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == prefix || n.strip_prefix(prefix).is_some_and(|rest| rest.starts_with('.')))
    })
}

pub fn preload() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let Some(dir) = candidate_dirs().into_iter().find(|d| d.join("libnvrtc.so.13").exists() || find(d, "libnvrtc.so").is_some())
        else {
            return; // fall back to the system search path
        };
        for lib in LIBS {
            let Some(path) = find(&dir, lib) else { continue };
            // SAFETY: loading NVIDIA's libraries runs their initializers, same as cudarc would.
            match unsafe { libloading::os::unix::Library::open(Some(&path), libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL) } {
                // Keep loaded for the life of the process.
                Ok(l) => std::mem::forget(l),
                Err(e) => eprintln!("[rllm] warning: failed to load {}: {e}", path.display()),
            }
        }
    });
}
