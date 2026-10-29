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

        let mut metadata = Vec::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32("value type")?;
            let value = r.value(ty)?;
            metadata.push((key, value));
        }
        let meta_index: HashMap<String, usize> =
            metadata.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect();

        let alignment = meta_index
            .get("general.alignment")
            .and_then(|&i| metadata[i].1.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);

        let mut raw = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32("n_dims")?;
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
            let numel: u64 = dims.iter().product();
            let size = dtype.row_bytes(numel as usize);
            let offset = (data_start + offset) as usize;
            if offset + size > mmap.len() {
                return Err(Error::TensorBounds(name));
            }
            tensors.push(TensorInfo { name, dims, dtype, offset, size });
        }
        let tensor_index = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();

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

    fn value(&mut self, ty: u32) -> Result<Value> {
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
                let len = self.u64("array length")? as usize;
                // Cap the preallocation so a corrupt length can't OOM us before
                // the EOF check fires.
                let mut items = Vec::with_capacity(len.min(1 << 20));
                for _ in 0..len {
                    items.push(self.value(item_ty)?);
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
