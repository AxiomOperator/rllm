//! Hugging Face safetensors checkpoints.
//!
//! A `.safetensors` file is an 8-byte little-endian header length, a JSON
//! header (`{"name": {"dtype": "BF16", "shape": [..], "data_offsets": [b, e]},
//! "__metadata__": {..}}`) and the tensor bytes, row-major, offsets relative
//! to the end of the header. Large checkpoints are split into shards listed by
//! `model.safetensors.index.json` (`{"weight_map": {"name": "file"}}`).
//!
//! This crate parses and validates that (every tensor in bounds, sized as its
//! shape and dtype say, no overlaps, no duplicates across shards) and maps the
//! files read-only: tensor bytes are borrowed, never copied. Turning a
//! checkpoint into the GGUF view rllm's loaders read (metadata keys, tensor
//! names, transforms) is the job of the converter built on it; see
//! `docs/safetensors.md`.

pub mod convert;

pub use convert::{Options, open_model, open_model_with, parse_quantize, to_gguf, to_gguf_with};

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use rllm_gguf::Backing;

/// Largest JSON header accepted (real headers are a few MB at most).
const MAX_HEADER: u64 = 100 << 20;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}: {1}")]
    Malformed(String, String),
    #[error("no safetensors checkpoint at {0} (expected a .safetensors file, or a directory with model.safetensors or model.safetensors.index.json)")]
    NotFound(PathBuf),
    #[error("tensor {0:?} not in the checkpoint")]
    MissingTensor(String),
    #[error(transparent)]
    Gguf(#[from] rllm_gguf::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Element types safetensors files use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "F8_E4M3" => Dtype::F8E4M3,
            "F8_E5M2" => Dtype::F8E5M2,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }

    /// The rllm element type with the same bytes, if there is one (the
    /// others need converting at load: FP8 with its block scales, GPTQ/AWQ
    /// packed integers).
    pub fn rllm(self) -> Option<rllm_core::DType> {
        Some(match self {
            Dtype::F32 => rllm_core::DType::F32,
            Dtype::F16 => rllm_core::DType::F16,
            Dtype::BF16 => rllm_core::DType::BF16,
            _ => return None,
        })
    }
}

/// One tensor of a checkpoint.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: Dtype,
    /// Row-major, outermost first (GGUF lists dims the other way round).
    pub shape: Vec<usize>,
    /// Index into [`Checkpoint::files`].
    pub file: usize,
    /// Byte range in that file.
    pub offset: usize,
    pub size: usize,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A checkpoint's tensors over its mapped shard files.
pub struct Checkpoint {
    /// The directory holding config.json, tokenizer.json, ... (the file's
    /// directory for a single file).
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
    pub tensors: Vec<Tensor>,
    /// Each file's `__metadata__` entries, merged.
    pub metadata: BTreeMap<String, String>,
    index: HashMap<String, usize>,
    maps: Vec<Backing>,
}

impl Checkpoint {
    /// Open a `.safetensors` file, or a directory holding `model.safetensors`
    /// or a sharded `model.safetensors.index.json`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (dir, files) = if path.is_file() {
            (path.parent().map(Path::to_path_buf).unwrap_or_default(), vec![path.to_path_buf()])
        } else if path.join("model.safetensors.index.json").is_file() {
            (path.to_path_buf(), shard_files(&path.join("model.safetensors.index.json"))?)
        } else if path.join("model.safetensors").is_file() {
            (path.to_path_buf(), vec![path.join("model.safetensors")])
        } else {
            return Err(Error::NotFound(path.to_path_buf()));
        };
        // Clef's decision head is a separate file, outside the shard index:
        // its tensors are named `joint_head.*` (as llama.cpp's converter does).
        let mut files = files;
        let head = dir.join("joint_head.safetensors");
        let head_file = (dir.join("joint_head_config.json").is_file() && head.is_file() && !files.contains(&head)).then(|| {
            files.push(head);
            files.len() - 1
        });
        let mut tensors = Vec::new();
        let mut metadata = BTreeMap::new();
        let mut maps = Vec::with_capacity(files.len());
        for (fi, f) in files.iter().enumerate() {
            let map = Backing::map_file(f)?;
            let (mut ts, meta) = parse_header(map.bytes(), &f.display().to_string(), fi)?;
            if head_file == Some(fi) {
                ts.iter_mut().for_each(|t| t.name = format!("joint_head.{}", t.name));
            }
            tensors.extend(ts);
            metadata.extend(meta);
            maps.push(map);
        }
        let mut index = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            if index.insert(t.name.clone(), i).is_some() {
                return Err(Error::Malformed(dir.display().to_string(), format!("tensor {:?} is in more than one shard", t.name)));
            }
        }
        Ok(Self { dir, files, tensors, metadata, index, maps })
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn require(&self, name: &str) -> Result<&Tensor> {
        self.tensor(name).ok_or_else(|| Error::MissingTensor(name.into()))
    }

    /// A tensor's bytes, borrowed from its mapped file.
    pub fn data(&self, t: &Tensor) -> &[u8] {
        &self.maps[t.file].bytes()[t.offset..t.offset + t.size]
    }

    /// A JSON file next to the weights (config.json, tokenizer_config.json,
    /// generation_config.json), if present.
    pub fn json(&self, name: &str) -> Result<Option<serde_json::Value>> {
        let p = self.dir.join(name);
        if !p.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&p)?;
        serde_json::from_str(&text).map(Some).map_err(|e| Error::Malformed(p.display().to_string(), e.to_string()))
    }

    /// The mapped files, in [`Self::files`] order (the buffers of a GGUF view
    /// built with `rllm_gguf::Gguf::from_parts`).
    pub fn into_backings(self) -> Vec<Backing> {
        self.maps
    }
}

/// The shard files a `model.safetensors.index.json` lists, in first-use order.
fn shard_files(index: &Path) -> Result<Vec<PathBuf>> {
    let what = index.display().to_string();
    let text = std::fs::read_to_string(index)?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error::Malformed(what.clone(), e.to_string()))?;
    let map = v.get("weight_map").and_then(|m| m.as_object()).ok_or_else(|| Error::Malformed(what.clone(), "no weight_map".into()))?;
    let dir = index.parent().unwrap_or(Path::new("."));
    let mut files: Vec<PathBuf> = Vec::new();
    for f in map.values() {
        let f = f.as_str().ok_or_else(|| Error::Malformed(what.clone(), "weight_map values must be file names".into()))?;
        // A plain file name in the same directory.
        if f.is_empty() || f.contains('/') || f.contains('\\') || f.starts_with('.') {
            return Err(Error::Malformed(what.clone(), format!("shard name {f:?}")));
        }
        let p = dir.join(f);
        if !files.contains(&p) {
            files.push(p);
        }
    }
    if files.is_empty() {
        return Err(Error::Malformed(what, "empty weight_map".into()));
    }
    Ok(files)
}

type Header = (Vec<Tensor>, BTreeMap<String, String>);

/// Parse and check a safetensors file's header; tensors refer to file `fi`.
fn parse_header(buf: &[u8], what: &str, fi: usize) -> Result<Header> {
    let bad = |m: String| Error::Malformed(what.to_string(), m);
    let len = buf.get(..8).map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes"))).ok_or_else(|| bad("shorter than its header length".into()))?;
    if len > MAX_HEADER || len > (buf.len() - 8) as u64 {
        return Err(bad(format!("header of {len} bytes")));
    }
    let data_start = 8 + len as usize;
    let data_len = buf.len() - data_start;
    let header: serde_json::Value = serde_json::from_slice(&buf[8..data_start]).map_err(|e| bad(format!("header JSON: {e}")))?;
    let obj = header.as_object().ok_or_else(|| bad("header is not a JSON object".into()))?;
    let mut tensors = Vec::with_capacity(obj.len());
    let mut metadata = BTreeMap::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            for (k, mv) in v.as_object().into_iter().flatten() {
                metadata.insert(k.clone(), mv.as_str().unwrap_or_default().to_string());
            }
            continue;
        }
        let dtype = v.get("dtype").and_then(|d| d.as_str()).and_then(Dtype::parse).ok_or_else(|| bad(format!("tensor {name}: dtype")))?;
        let shape: Vec<usize> = v
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| bad(format!("tensor {name}: shape")))?
            .iter()
            .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
            .collect::<Option<_>>()
            .ok_or_else(|| bad(format!("tensor {name}: shape")))?;
        let offs = v.get("data_offsets").and_then(|o| o.as_array()).filter(|o| o.len() == 2).ok_or_else(|| bad(format!("tensor {name}: data_offsets")))?;
        let (b, e) = match (offs[0].as_u64(), offs[1].as_u64()) {
            (Some(b), Some(e)) if b <= e && e <= data_len as u64 => (b as usize, e as usize),
            _ => return Err(bad(format!("tensor {name}: data_offsets out of bounds"))),
        };
        let bytes = shape.iter().try_fold(dtype.size(), |a, &d| a.checked_mul(d));
        if bytes != Some(e - b) {
            return Err(bad(format!("tensor {name}: {} bytes for shape {shape:?} of {dtype:?}", e - b)));
        }
        tensors.push(Tensor { name: name.clone(), dtype, shape, file: fi, offset: data_start + b, size: e - b });
    }
    let mut by_offset: Vec<&Tensor> = tensors.iter().filter(|t| t.size > 0).collect();
    by_offset.sort_by_key(|t| t.offset);
    if let Some(w) = by_offset.windows(2).find(|w| w[0].offset + w[0].size > w[1].offset) {
        return Err(bad(format!("tensors {:?} and {:?} overlap", w[0].name, w[1].name)));
    }
    Ok((tensors, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors file's bytes from (name, dtype, shape, data).
    fn file(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let b = data.len();
            data.extend_from_slice(bytes);
            header.insert(name.to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [b, data.len()]}));
        }
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let h = serde_json::to_vec(&header).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(data);
        out
    }

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rllm-safetensors-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn reads_single_and_sharded_checkpoints() {
        let d = scratch("ok");
        let w: Vec<u8> = (0..12u8).collect();
        std::fs::write(d.join("model.safetensors"), file(&[("w", "BF16", vec![2, 3], w.clone()), ("b", "F32", vec![1], vec![0, 0, 128, 63])])).unwrap();
        std::fs::write(d.join("config.json"), r#"{"architectures": ["Qwen3ForCausalLM"]}"#).unwrap();
        let c = Checkpoint::open(&d).unwrap();
        let t = c.require("w").unwrap();
        assert_eq!((t.dtype, t.shape.clone(), c.data(t).to_vec()), (Dtype::BF16, vec![2, 3], w));
        assert_eq!(c.metadata.get("format").map(String::as_str), Some("pt"));
        assert_eq!(c.json("config.json").unwrap().unwrap()["architectures"][0], "Qwen3ForCausalLM");
        assert!(c.json("missing.json").unwrap().is_none());

        let s = scratch("shards");
        std::fs::write(s.join("a.safetensors"), file(&[("x", "F16", vec![2], vec![1, 2, 3, 4])])).unwrap();
        std::fs::write(s.join("b.safetensors"), file(&[("y", "I64", vec![1], vec![9; 8])])).unwrap();
        std::fs::write(s.join("model.safetensors.index.json"), r#"{"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}"#).unwrap();
        let c = Checkpoint::open(&s).unwrap();
        assert_eq!(c.files.len(), 2);
        assert_eq!(c.data(c.require("y").unwrap()), &[9; 8]);
        assert_eq!(c.require("x").unwrap().dtype.rllm(), Some(rllm_core::DType::F16));
        let _ = std::fs::remove_dir_all(d);
        let _ = std::fs::remove_dir_all(s);
    }

    #[test]
    fn malformed_files_are_errors() {
        let d = scratch("bad");
        let good = file(&[("w", "F32", vec![2], vec![0; 8])]);
        let mut cases: Vec<(&str, Vec<u8>)> = vec![
            ("short", vec![1, 2, 3]),
            ("huge header", u64::MAX.to_le_bytes().to_vec()),
            ("wrong size", file(&[("w", "F32", vec![3], vec![0; 8])])),
            ("unknown dtype", file(&[("w", "Q4", vec![2], vec![0; 8])])),
        ];
        let mut past_end = good.clone();
        past_end.truncate(good.len() - 1);
        cases.push(("past end", past_end));
        // Two tensors over the same bytes.
        let h = br#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#;
        let mut overlap = (h.len() as u64).to_le_bytes().to_vec();
        overlap.extend_from_slice(h);
        overlap.extend([0; 4]);
        cases.push(("overlap", overlap));
        for (what, bytes) in cases {
            let p = d.join("m.safetensors");
            std::fs::write(&p, bytes).unwrap();
            assert!(Checkpoint::open(&p).is_err(), "{what}");
        }
        // Shard names stay in the directory; a tensor in two shards is refused.
        std::fs::write(d.join("model.safetensors.index.json"), r#"{"weight_map": {"w": "../escape.safetensors"}}"#).unwrap();
        assert!(Checkpoint::open(&d).is_err());
        std::fs::write(d.join("a.safetensors"), &good).unwrap();
        std::fs::write(d.join("b.safetensors"), &good).unwrap();
        std::fs::write(d.join("model.safetensors.index.json"), r#"{"weight_map": {"w": "a.safetensors", "v": "b.safetensors"}}"#).unwrap();
        assert!(Checkpoint::open(&d).is_err());
        let _ = std::fs::remove_dir_all(d);
    }
}
