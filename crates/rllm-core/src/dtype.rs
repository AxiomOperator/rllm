/// Element types for weights, using ggml's numbering so GGUF type ids map 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4K,
    Q6K,
}

impl DType {
    pub fn from_ggml(id: u32) -> Option<Self> {
        Some(match id {
            0 => DType::F32,
            1 => DType::F16,
            8 => DType::Q8_0,
            12 => DType::Q4K,
            14 => DType::Q6K,
            30 => DType::BF16,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::Q8_0 => "Q8_0",
            DType::Q4K => "Q4_K",
            DType::Q6K => "Q6_K",
        }
    }

    /// Number of elements per quantization block (1 for plain float types).
    pub fn block_size(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q8_0 => 32,
            DType::Q4K | DType::Q6K => 256,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q8_0 => 34,
            DType::Q4K => 144,
            DType::Q6K => 210,
        }
    }

    /// Bytes needed for `n` contiguous elements. `n` must be a multiple of the block size.
    pub fn row_bytes(self, n: usize) -> usize {
        debug_assert_eq!(n % self.block_size(), 0, "{n} not a multiple of {} block", self.name());
        n / self.block_size() * self.block_bytes()
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
