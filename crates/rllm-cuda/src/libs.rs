//! Locate NVRTC and cuBLAS when they aren't on the system library path.
//!
//! cudarc dlopens libraries by soname (e.g. `libnvrtc.so.13`). If we load
//! them first by full path with RTLD_GLOBAL, the dynamic linker hands back
//! these already-loaded copies, so no LD_LIBRARY_PATH is needed.
//!
//! Search order: `$RLLM_CUDA_LIB_DIR`, then `.cuda/lib` in any ancestor of
//! the executable (the repo-local copy: `target/release/rllm` finds
//! `<repo>/.cuda/lib`). The current directory is deliberately not searched:
//! running rllm inside an untrusted directory must not load its libraries.

use std::path::{Path, PathBuf};
use std::sync::Once;

/// CUDA major version cudarc is built against (its `cuda-13020` feature).
const CUDA_MAJOR: &str = "13";

/// Libraries in dependency order: builtins before nvrtc, cublasLt before
/// cublas. Each soname is `<library>.<major>`; NVRTC's builtins file also
/// carries the minor version (`libnvrtc-builtins.so.13.4`).
const LIBS: &[&str] = &["libnvrtc-builtins.so", "libnvrtc.so", "libcublasLt.so", "libcublas.so"];

fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("RLLM_CUDA_LIB_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Ok(exe) = std::env::current_exe() {
        dirs.extend(exe.ancestors().skip(1).map(|a| a.join(".cuda/lib")));
    }
    dirs
}

/// The version components after `<lib>.<major>` in a file name (`[]` for the
/// exact soname, `[4]` for `<lib>.13.4`), or None if `name` isn't that library
/// at that major version.
fn version_suffix(name: &str, lib: &str) -> Option<Vec<u32>> {
    let rest = name.strip_prefix(lib)?.strip_prefix('.')?.strip_prefix(CUDA_MAJOR)?;
    if rest.is_empty() {
        return Some(Vec::new());
    }
    rest.strip_prefix('.')?.split('.').map(|p| p.parse().ok()).collect()
}

/// Find `lib` at the expected major version in `dir`: the exact soname
/// (`libnvrtc.so.13`) if present, else the highest `<soname>.<minor>...`.
/// Deterministic, unlike taking the first match in directory order.
fn find(dir: &Path, lib: &str) -> Option<PathBuf> {
    let exact = dir.join(format!("{lib}.{CUDA_MAJOR}"));
    if exact.is_file() {
        return Some(exact);
    }
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let v = version_suffix(e.file_name().to_str()?, lib)?;
            let p = e.path();
            p.is_file().then_some((v, p))
        })
        .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, p)| p)
}

pub fn preload() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let Some(dir) = candidate_dirs().into_iter().find(|d| find(d, "libnvrtc.so").is_some()) else {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soname_matching() {
        assert_eq!(version_suffix("libnvrtc.so.13", "libnvrtc.so"), Some(vec![]));
        assert_eq!(version_suffix("libnvrtc-builtins.so.13.4", "libnvrtc-builtins.so"), Some(vec![4]));
        // Other libraries sharing the prefix, other majors, junk suffixes.
        assert_eq!(version_suffix("libnvrtc-builtins.so.13.4", "libnvrtc.so"), None);
        assert_eq!(version_suffix("libcublasLt.so.13", "libcublas.so"), None);
        assert_eq!(version_suffix("libnvrtc.so.12", "libnvrtc.so"), None);
        assert_eq!(version_suffix("libnvrtc.so.130", "libnvrtc.so"), None);
        assert_eq!(version_suffix("libnvrtc.so.13.bak", "libnvrtc.so"), None);
        assert_eq!(version_suffix("libnvrtc.so", "libnvrtc.so"), None);
    }

    #[test]
    fn find_prefers_exact_then_highest() {
        let dir = std::env::temp_dir().join(format!("rllm-libs-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["libnvrtc-builtins.so.13.2", "libnvrtc-builtins.so.13.10", "libnvrtc.so.13.1", "libnvrtc.so.12"] {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        assert_eq!(find(&dir, "libnvrtc-builtins.so"), Some(dir.join("libnvrtc-builtins.so.13.10")));
        assert_eq!(find(&dir, "libnvrtc.so"), Some(dir.join("libnvrtc.so.13.1")));
        std::fs::write(dir.join("libnvrtc.so.13"), b"").unwrap();
        assert_eq!(find(&dir, "libnvrtc.so"), Some(dir.join("libnvrtc.so.13")));
        assert_eq!(find(&dir, "libcublas.so"), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
