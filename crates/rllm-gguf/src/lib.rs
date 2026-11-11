//! GGUF reader.
//!
//! Parses the header, metadata key/values and tensor index, and memory-maps the
//! file so tensor data is borrowed directly from the page cache with no copy.
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use rllm_core::DType;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;
/// GGML tensors have at most 4 dims; anything far beyond is corrupt.
const MAX_DIMS: u32 = 8;
/// Arrays of arrays: real files nest one level.
const MAX_NESTING: usize = 8;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0}")]
    Version(u32),
    #[error("unexpected end of file while reading {0}")]
    Eof(&'static str),
    #[error("invalid metadata value type {0}")]
    ValueType(u32),
    #[error("invalid utf-8 in string")]
    Utf8,
    #[error("unknown ggml tensor type {1} for tensor {0}")]
    TensorType(String, u32),
    #[error("tensor {0} extends past end of file")]
    TensorBounds(String),
    #[error("missing metadata key {0}")]
    MissingKey(String),
    #[error("metadata key {0} has unexpected type")]
    KeyType(String),
    #[error("missing tensor {0}")]
    MissingTensor(String),
    #[error("malformed GGUF: {0}")]
    Malformed(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Short human-readable form; long arrays are summarized.
    pub fn summary(&self) -> String {
        match self {
            Value::String(s) if s.len() > 80 => format!("{:?}… ({} bytes)", &s[..s.floor_char_boundary(80)], s.len()),
            Value::String(s) => format!("{s:?}"),
            Value::Array(a) if a.len() > 8 => {
                let head: Vec<String> = a[..4].iter().map(Value::summary).collect();
                format!("[{}, …] ({} items)", head.join(", "), a.len())
            }
            Value::Array(a) => {
                let items: Vec<String> = a.iter().map(Value::summary).collect();
                format!("[{}]", items.join(", "))
            }
            Value::U8(v) => v.to_string(),
            Value::I8(v) => v.to_string(),
            Value::U16(v) => v.to_string(),
            Value::I16(v) => v.to_string(),
            Value::U32(v) => v.to_string(),
            Value::I32(v) => v.to_string(),
            Value::U64(v) => v.to_string(),
            Value::I64(v) => v.to_string(),
            Value::F32(v) => v.to_string(),
            Value::F64(v) => v.to_string(),
            Value::Bool(v) => v.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGUF order: `dims[0]` is the innermost (contiguous) axis.
    pub dims: Vec<u64>,
    pub dtype: DType,
    /// Absolute byte offset of the tensor data in the file.
    pub offset: usize,
    pub size: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    meta_index: HashMap<String, usize>,
    tensor_index: HashMap<String, usize>,
    mmap: Arc<Mmap>,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the file is opened read-only. Modifying it while mapped is
        // undefined behaviour, same as for every other mmap-based model loader.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(Arc::new(mmap))
    }

    fn parse(mmap: Arc<Mmap>) -> Result<Self> {
        let mut r = Reader { buf: &mmap, pos: 0 };
        if r.u32("magic")? != MAGIC {
            return Err(Error::BadMagic);
        }
        let version = r.u32("version")?;
        if !(2..=3).contains(&version) {
            return Err(Error::Version(version));
        }
        let n_tensors = r.u64("tensor count")?;
        let n_kv = r.u64("kv count")?;

        // Corrupt counts must not size allocations: every entry takes at
        // least a dozen bytes of the file.
        let mut metadata = Vec::with_capacity((n_kv as usize).min(mmap.len() / 12));
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32("value type")?;
            let value = r.value(ty, 0)?;
            metadata.push((key, value));
        }
        let mut meta_index: HashMap<String, usize> = HashMap::with_capacity(metadata.len());
        for (i, (k, _)) in metadata.iter().enumerate() {
            if meta_index.insert(k.clone(), i).is_some() {
                return Err(Error::Malformed(format!("duplicate metadata key {k:?}")));
            }
        }

        let alignment = meta_index
            .get("general.alignment")
            .and_then(|&i| metadata[i].1.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        if !alignment.is_power_of_two() {
            return Err(Error::Malformed(format!("alignment {alignment}")));
        }

        let mut raw = Vec::with_capacity((n_tensors as usize).min(mmap.len() / 24));
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32("n_dims")?;
            if n_dims > MAX_DIMS {
                return Err(Error::Malformed(format!("tensor {name} has {n_dims} dims")));
            }
            let dims = (0..n_dims).map(|_| r.u64("dim")).collect::<Result<Vec<_>>>()?;
            let ty = r.u32("tensor type")?;
            let offset = r.u64("tensor offset")?;
            raw.push((name, dims, ty, offset));
        }

        let data_start = (r.pos as u64).next_multiple_of(alignment);
        let mut tensors = Vec::with_capacity(raw.len());
        for (name, dims, ty, offset) in raw {
            let Some(dtype) = DType::from_ggml(ty) else {
                return Err(Error::TensorType(name, ty));
            };
            // Checked: a tensor's elements can't outnumber the file's bits.
            let numel = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).filter(|&n| n <= mmap.len() as u64 * 8);
            let Some(numel) = numel else {
                return Err(Error::TensorBounds(name));
            };
            // Rows (the innermost axis) are stored as whole blocks.
            let row = dims.first().copied().unwrap_or(1);
            if row % dtype.block_size() as u64 != 0 {
                return Err(Error::Malformed(format!("tensor {name}: rows of {row} elements are not whole {dtype} blocks")));
            }
            if offset % alignment != 0 {
                return Err(Error::Malformed(format!("tensor {name}: offset {offset} is not a multiple of the alignment {alignment}")));
            }
            let size = (numel / dtype.block_size() as u64).checked_mul(dtype.block_bytes() as u64);
            let end = size.and_then(|size| data_start.checked_add(offset)?.checked_add(size));
            let (Some(size), Some(end)) = (size, end) else {
                return Err(Error::TensorBounds(name));
            };
            if end > mmap.len() as u64 {
                return Err(Error::TensorBounds(name));
            }
            let offset = (data_start + offset) as usize;
            tensors.push(TensorInfo { name, dims, dtype, offset, size: size as usize });
        }
        let mut tensor_index = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            if tensor_index.insert(t.name.clone(), i).is_some() {
                return Err(Error::Malformed(format!("duplicate tensor {:?}", t.name)));
            }
        }
        // Tensors' data must not overlap (empty tensors take no space).
        let mut by_offset: Vec<&TensorInfo> = tensors.iter().filter(|t| t.size > 0).collect();
        by_offset.sort_by_key(|t| t.offset);
        if let Some(w) = by_offset.windows(2).find(|w| w[0].offset + w[0].size > w[1].offset) {
            return Err(Error::Malformed(format!("tensors {:?} and {:?} overlap", w[0].name, w[1].name)));
        }

        Ok(Self { version, metadata, tensors, meta_index, tensor_index, mmap })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.meta_index.get(key).map(|&i| &self.metadata[i].1)
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        let v = self.get(key).ok_or_else(|| Error::MissingKey(key.into()))?;
        v.as_u64().ok_or_else(|| Error::KeyType(key.into()))
    }

    pub fn get_f64(&self, key: &str) -> Result<f64> {
        let v = self.get(key).ok_or_else(|| Error::MissingKey(key.into()))?;
        v.as_f64().ok_or_else(|| Error::KeyType(key.into()))
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        let v = self.get(key).ok_or_else(|| Error::MissingKey(key.into()))?;
        v.as_str().ok_or_else(|| Error::KeyType(key.into()))
    }

    pub fn get_array(&self, key: &str) -> Result<&[Value]> {
        let v = self.get(key).ok_or_else(|| Error::MissingKey(key.into()))?;
        v.as_array().ok_or_else(|| Error::KeyType(key.into()))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn require_tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor(name).ok_or_else(|| Error::MissingTensor(name.into()))
    }

    /// Raw bytes of a tensor, borrowed from the mapping.
    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        &self.mmap[info.offset..info.offset + info.size]
    }

    /// Shared handle to the underlying mapping, for zero-copy weight views.
    pub fn mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }
}

/// A few metadata values read cheaply with [`peek`].
#[derive(Debug, Default)]
pub struct Peek {
    /// The values of the requested keys that the file has.
    pub values: HashMap<String, Value>,
    /// Length of every top-level array value (e.g. `tokenizer.ggml.tokens`).
    pub array_lens: HashMap<String, u64>,
}

/// Wanted values larger than this aren't worth a peek.
const MAX_PEEK_VALUE: u64 = 1 << 20;

/// Read only the metadata keys in `wanted` (plus every array's length)
/// without mapping the file or decoding the rest: other values, such as a
/// tokenizer's vocabulary, are skipped over. For screening many files.
pub fn peek(path: impl AsRef<Path>, wanted: &[&str]) -> Result<Peek> {
    use std::io::{BufReader, Read};

    struct Stream {
        r: BufReader<File>,
        pos: u64,
        len: u64,
    }
    impl Stream {
        fn bytes(&mut self, n: u64, what: &'static str) -> Result<Vec<u8>> {
            self.skip(n, what)?;
            let mut b = vec![0u8; n as usize];
            self.r.seek_relative(-(n as i64))?;
            self.r.read_exact(&mut b)?;
            Ok(b)
        }
        fn skip(&mut self, n: u64, what: &'static str) -> Result<()> {
            let end = self.pos.checked_add(n).filter(|&e| e <= self.len).ok_or(Error::Eof(what))?;
            self.r.seek_relative((end - self.pos) as i64)?;
            self.pos = end;
            Ok(())
        }
        fn u32(&mut self, what: &'static str) -> Result<u32> {
            Ok(u32::from_le_bytes(self.bytes(4, what)?.try_into().unwrap()))
        }
        fn u64(&mut self, what: &'static str) -> Result<u64> {
            Ok(u64::from_le_bytes(self.bytes(8, what)?.try_into().unwrap()))
        }
        /// Skip one value of type `ty`; for an array, return its length.
        fn skip_value(&mut self, ty: u32, depth: usize) -> Result<Option<u64>> {
            let fixed = |ty| match ty {
                0 | 1 | 7 => Some(1),
                2 | 3 => Some(2),
                4..=6 => Some(4),
                10..=12 => Some(8),
                _ => None,
            };
            match ty {
                8 => {
                    let n = self.u64("string length")?;
                    self.skip(n, "string")?;
                }
                9 => {
                    let item = self.u32("array type")?;
                    if depth >= MAX_NESTING {
                        return Err(Error::Malformed("arrays nested too deeply".into()));
                    }
                    let n = self.u64("array length")?;
                    match fixed(item) {
                        Some(size) => self.skip(n.checked_mul(size).ok_or(Error::Eof("array"))?, "array")?,
                        None => {
                            for _ in 0..n {
                                self.skip_value(item, depth + 1)?;
                            }
                        }
                    }
                    return Ok(Some(n));
                }
                t => self.skip(fixed(t).ok_or(Error::ValueType(t))?, "value")?,
            }
            Ok(None)
        }
    }

    let file = File::open(path)?;
    let len = file.metadata()?.len();
    let mut s = Stream { r: BufReader::new(file), pos: 0, len };
    if s.u32("magic")? != MAGIC {
        return Err(Error::BadMagic);
    }
    let version = s.u32("version")?;
    if !(2..=3).contains(&version) {
        return Err(Error::Version(version));
    }
    let _n_tensors = s.u64("tensor count")?;
    let n_kv = s.u64("kv count")?;
    let mut out = Peek::default();
    for _ in 0..n_kv {
        let n = s.u64("string length")?;
        if n > MAX_PEEK_VALUE {
            return Err(Error::Malformed("metadata key too long".into()));
        }
        let key = String::from_utf8(s.bytes(n, "string")?).map_err(|_| Error::Utf8)?;
        let ty = s.u32("value type")?;
        let start = s.pos;
        let array_len = s.skip_value(ty, 0)?;
        if let Some(n) = array_len {
            out.array_lens.insert(key.clone(), n);
        }
        if wanted.contains(&key.as_str()) && s.pos - start <= MAX_PEEK_VALUE {
            let n = s.pos - start;
            s.r.seek_relative(-(n as i64))?;
            s.pos = start;
            let buf = s.bytes(n, "value")?;
            out.values.insert(key, Reader { buf: &buf, pos: 0 }.value(ty, 0)?);
        }
    }
    Ok(out)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&[u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len()).ok_or(Error::Eof(what))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N]> {
        Ok(self.take(N, what)?.try_into().unwrap())
    }

    fn u32(&mut self, what: &'static str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64("string length")? as usize;
        let bytes = self.take(len, "string")?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::Utf8)
    }

    fn value(&mut self, ty: u32, depth: usize) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.array::<1>("u8")?[0]),
            1 => Value::I8(self.array::<1>("i8")?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array("u16")?)),
            3 => Value::I16(i16::from_le_bytes(self.array("i16")?)),
            4 => Value::U32(self.u32("u32")?),
            5 => Value::I32(i32::from_le_bytes(self.array("i32")?)),
            6 => Value::F32(f32::from_le_bytes(self.array("f32")?)),
            7 => Value::Bool(self.array::<1>("bool")?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let item_ty = self.u32("array type")?;
                if item_ty == 9 && depth >= MAX_NESTING {
                    return Err(Error::Malformed("arrays nested too deeply".into()));
                }
                let len = self.u64("array length")? as usize;
                // Cap the preallocation so a corrupt length can't OOM us before
                // the EOF check fires.
                let mut items = Vec::with_capacity(len.min(1 << 20));
                for _ in 0..len {
                    items.push(self.value(item_ty, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64("u64")?),
            11 => Value::I64(i64::from_le_bytes(self.array("i64")?)),
            12 => Value::F64(f64::from_le_bytes(self.array("f64")?)),
            t => return Err(Error::ValueType(t)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header with `n_tensors` and the given metadata/tensor bytes.
    fn file(n_tensors: u64, n_kv: u64, body: &[u8]) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend(MAGIC.to_le_bytes());
        f.extend(3u32.to_le_bytes());
        f.extend(n_tensors.to_le_bytes());
        f.extend(n_kv.to_le_bytes());
        f.extend(body);
        f
    }

    fn string(s: &str) -> Vec<u8> {
        let mut b = (s.len() as u64).to_le_bytes().to_vec();
        b.extend(s.as_bytes());
        b
    }

    /// Run `f` on `bytes` written to a temporary file.
    fn with_file<T>(bytes: &[u8], f: impl FnOnce(&Path) -> T) -> T {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("rllm-gguf-test-{}-{n}.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let r = f(&path);
        std::fs::remove_file(&path).unwrap();
        r
    }

    fn parse(bytes: &[u8]) -> Result<Gguf> {
        with_file(bytes, |p| Gguf::open(p))
    }

    fn kv_u32(key: &str, v: u32) -> Vec<u8> {
        let mut b = string(key);
        b.extend(4u32.to_le_bytes());
        b.extend(v.to_le_bytes());
        b
    }

    /// A file with the given f32 tensors (`(name, elements, offset)`) and enough data.
    fn with_tensors(ts: &[(&str, u64, u64)]) -> Result<Gguf> {
        let mut body: Vec<u8> = ts.iter().flat_map(|&(n, e, o)| tensor(n, &[e], 0, o)).collect();
        body.resize(body.len() + 512, 0);
        parse(&file(ts.len() as u64, 0, &body))
    }

    #[test]
    fn tensor_index_is_checked() {
        // Back to back, and with a gap: fine.
        assert!(with_tensors(&[("a", 8, 0), ("b", 8, 32), ("c", 8, 96)]).is_ok());
        // The same name twice.
        assert!(matches!(with_tensors(&[("a", 8, 0), ("a", 8, 32)]), Err(Error::Malformed(e)) if e.contains("duplicate")));
        // Overlapping data, in either listing order.
        assert!(matches!(with_tensors(&[("a", 16, 0), ("b", 8, 32)]), Err(Error::Malformed(e)) if e.contains("overlap")));
        assert!(matches!(with_tensors(&[("b", 8, 32), ("a", 16, 0)]), Err(Error::Malformed(e)) if e.contains("overlap")));
        assert!(matches!(with_tensors(&[("a", 8, 64), ("b", 8, 64)]), Err(Error::Malformed(e)) if e.contains("overlap")));
        // Not a multiple of the (default 32-byte) alignment.
        assert!(matches!(with_tensors(&[("a", 8, 4)]), Err(Error::Malformed(e)) if e.contains("alignment")));
        // A custom alignment applies too.
        let mut body = kv_u32("general.alignment", 64);
        body.extend(tensor("a", &[8], 0, 32));
        body.resize(body.len() + 512, 0);
        assert!(matches!(parse(&file(1, 1, &body)), Err(Error::Malformed(e)) if e.contains("alignment")));
        // Quantized rows must be whole blocks even if the total is (2 x 16 of Q8_0).
        let mut body = tensor("q", &[16, 2], 8, 0);
        body.resize(body.len() + 512, 0);
        assert!(matches!(parse(&file(1, 0, &body)), Err(Error::Malformed(_))));
    }

    #[test]
    fn duplicate_metadata_keys_are_errors() {
        let mut body = kv_u32("k", 1);
        body.extend(kv_u32("k", 2));
        assert!(matches!(parse(&file(0, 2, &body)), Err(Error::Malformed(e)) if e.contains("duplicate")));
    }

    #[test]
    fn peek_reads_wanted_keys_and_array_lengths() {
        let mut body = string("general.architecture");
        body.extend(8u32.to_le_bytes());
        body.extend(string("qwen3"));
        body.extend(string("tokenizer.ggml.tokens"));
        body.extend(9u32.to_le_bytes());
        body.extend(8u32.to_le_bytes());
        body.extend(3u64.to_le_bytes());
        for t in ["a", "bc", "def"] {
            body.extend(string(t));
        }
        body.extend(string("scores"));
        body.extend(9u32.to_le_bytes());
        body.extend(6u32.to_le_bytes());
        body.extend(2u64.to_le_bytes());
        body.extend([0u8; 8]);
        body.extend(kv_u32("n", 7));
        let bytes = file(0, 4, &body);
        let p = with_file(&bytes, |p| peek(p, &["general.architecture", "n", "missing"])).unwrap();
        assert_eq!(p.values.get("general.architecture"), Some(&Value::String("qwen3".into())));
        assert_eq!(p.values.get("n"), Some(&Value::U32(7)));
        assert_eq!(p.values.len(), 2);
        assert_eq!(p.array_lens.get("tokenizer.ggml.tokens"), Some(&3));
        assert_eq!(p.array_lens.get("scores"), Some(&2));
        // Truncated: an error, not a panic or a huge allocation.
        assert!(with_file(&bytes[..bytes.len() - 6], |p| peek(p, &["n"])).is_err());
        assert!(with_file(&file(0, u64::MAX, &[]), |p| peek(p, &[])).is_err());
    }

    fn tensor(name: &str, dims: &[u64], ty: u32, offset: u64) -> Vec<u8> {
        let mut b = string(name);
        b.extend((dims.len() as u32).to_le_bytes());
        dims.iter().for_each(|d| b.extend(d.to_le_bytes()));
        b.extend(ty.to_le_bytes());
        b.extend(offset.to_le_bytes());
        b
    }

    #[test]
    fn valid_file_parses() {
        let mut body = tensor("w", &[4, 2], 0, 0);
        body.resize(body.len() + 64, 0);
        let g = parse(&file(1, 0, &body)).unwrap();
        assert_eq!(g.tensors[0].size, 32);
    }

    #[test]
    fn malformed_files_are_errors() {
        // Huge counts: no giant reservation, just EOF.
        assert!(matches!(parse(&file(u64::MAX / 2, u64::MAX / 2, &[])), Err(Error::Eof(_))));
        // Alignment 0.
        let mut body = string("general.alignment");
        body.extend(4u32.to_le_bytes());
        body.extend(0u32.to_le_bytes());
        assert!(matches!(parse(&file(0, 1, &body)), Err(Error::Malformed(_))));
        // Arrays nested a million deep.
        let mut body = string("k");
        body.extend(9u32.to_le_bytes());
        for _ in 0..1_000_000 {
            body.extend(9u32.to_le_bytes());
            body.extend(1u64.to_le_bytes());
        }
        assert!(matches!(parse(&file(0, 1, &body)), Err(Error::Malformed(_))));
        // Offsets and sizes that wrap around.
        for t in [tensor("w", &[4], 0, u64::MAX - 8), tensor("w", &[1 << 32, 1 << 32, 16], 0, 0), tensor("w", &[3], 8, 0)] {
            let mut body = t;
            body.resize(body.len() + 64, 0);
            assert!(matches!(parse(&file(1, 0, &body)), Err(Error::TensorBounds(_) | Error::Malformed(_))));
        }
        // Too many dims.
        assert!(matches!(parse(&file(1, 0, &tensor("w", &[1; 9], 0, 0))), Err(Error::Malformed(_))));
    }
}
