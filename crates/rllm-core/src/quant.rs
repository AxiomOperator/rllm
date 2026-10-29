//! Reference dequantization for the supported ggml block formats.
//!
//! These are scalar, straightforward ports of ggml's `dequantize_row_*`
//! functions (ggml-quants.c, MIT). They define what "correct" means for every
//! backend: GPU kernels are tested against these.

use half::{bf16, f16};

use crate::DType;

/// Dequantize `out.len()` elements from `src` (which must hold exactly that many).
pub fn dequantize_row(dtype: DType, src: &[u8], out: &mut [f32]) {
    assert_eq!(src.len(), dtype.row_bytes(out.len()), "{dtype}: byte length mismatch");
    match dtype {
        DType::F32 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes(b.try_into().unwrap());
            }
        }
        DType::F16 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        DType::BF16 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = bf16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        DType::Q8_0 => {
            for (y, b) in out.chunks_exact_mut(32).zip(src.chunks_exact(34)) {
                dequant_q8_0(b, y);
            }
        }
        DType::Q4K => {
            for (y, b) in out.chunks_exact_mut(256).zip(src.chunks_exact(144)) {
                dequant_q4_k(b, y);
            }
        }
        DType::Q6K => {
            for (y, b) in out.chunks_exact_mut(256).zip(src.chunks_exact(210)) {
                dequant_q6_k(b, y);
            }
        }
    }
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// block_q8_0: f16 d; i8 qs[32]
#[inline]
fn dequant_q8_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (o, &q) in y.iter_mut().zip(&b[2..34]) {
        *o = d * (q as i8) as f32;
    }
}

/// block_q4_K: f16 d; f16 dmin; u8 scales[12]; u8 qs[128]
fn dequant_q4_k(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    for chunk in 0..4 {
        let is = chunk * 2;
        let (sc1, m1) = scale_min_k4(is, scales);
        let (sc2, m2) = scale_min_k4(is + 1, scales);
        let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let y = &mut y[chunk * 64..chunk * 64 + 64];
        for l in 0..32 {
            y[l] = d1 * (q[l] & 0xF) as f32 - m1;
            y[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
        }
    }
}

/// Unpack the 6-bit scale and min for sub-block `j` of a Q4_K/Q5_K block.
#[inline]
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// block_q6_K: u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d
fn dequant_q6_k(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 208);
    for half in 0..2 {
        let ql = &b[half * 64..half * 64 + 64];
        let qh = &b[128 + half * 32..128 + half * 32 + 32];
        let sc = &b[192 + half * 8..192 + half * 8 + 8];
        let y = &mut y[half * 128..half * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_roundtrip() {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        for i in 0..32 {
            block[2 + i] = (i as i8 - 16) as u8;
        }
        let mut out = [0f32; 32];
        dequantize_row(DType::Q8_0, &block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 0.5 * (i as f32 - 16.0));
        }
    }

    #[test]
    fn q4_k_constant_block() {
        // scale index 0..3 = 1, mins = 0, all nibbles = 3 -> d * 1 * 3
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        for j in 0..4 {
            block[4 + j] = 1; // scales for sub-blocks 0..3
        }
        for j in 8..12 {
            block[4 + j] = 1; // low nibble -> scale for sub-blocks 4..7
        }
        for q in &mut block[16..144] {
            *q = 0x33;
        }
        let mut out = [0f32; 256];
        dequantize_row(DType::Q4K, &block, &mut out);
        assert!(out.iter().all(|&v| v == 6.0), "{:?}", &out[..8]);
    }

    #[test]
    fn q6_k_zero_point() {
        // q = 32 everywhere -> value 0 regardless of scale.
        let mut block = vec![0u8; 210];
        for q in &mut block[0..128] {
            *q = 0x00;
        }
        for q in &mut block[128..192] {
            *q = 0b1010_1010; // high bits = 2 -> (2<<4) = 32
        }
        for s in &mut block[192..208] {
            *s = 3;
        }
        block[208..210].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let mut out = [1f32; 256];
        dequantize_row(DType::Q6K, &block, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
