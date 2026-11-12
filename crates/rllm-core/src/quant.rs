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
        DType::Q3K => {
            for (y, b) in out.chunks_exact_mut(256).zip(src.chunks_exact(110)) {
                dequant_q3_k(b, y);
            }
        }
        DType::Q4K => {
            for (y, b) in out.chunks_exact_mut(256).zip(src.chunks_exact(144)) {
                dequant_q4_k(b, y);
            }
        }
        DType::Q5K => {
            for (y, b) in out.chunks_exact_mut(256).zip(src.chunks_exact(176)) {
                dequant_q5_k(b, y);
            }
        }
        DType::Q6K => {
            for (y, b) in out.chunks_exact_mut(256).zip(src.chunks_exact(210)) {
                dequant_q6_k(b, y);
            }
        }
        DType::IQ4XS => {
            for (y, b) in out.chunks_exact_mut(256).zip(src.chunks_exact(136)) {
                dequant_iq4_xs(b, y);
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

/// The 16 six-bit scales of a Q3_K block (0..63; the value is scale - 32),
/// unpacked from its 12 bytes as ggml's dequantize_row_q3_K does.
pub fn q3_k_scales(s: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (j, o) in out.iter_mut().enumerate() {
        // Low 4 bits: nibble j % 8 of bytes 0..8 (low nibbles first); high
        // 2 bits: bits 2 (j / 4) .. of byte 8 + j % 4.
        let lo = if j < 8 { s[j] & 0xF } else { s[j - 8] >> 4 };
        let hi = (s[8 + j % 4] >> (2 * (j / 4))) & 3;
        *o = lo | hi << 4;
    }
    out
}

/// block_q3_K: u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d. Element
/// h * 128 + j * 32 + l (h < 2, j < 4, l < 32) is bits 2j of qs[h * 32 + l]
/// minus 4 unless bit 4h + j of hmask[l] is set, times d * (scale - 32) of
/// its 16-element group.
fn dequant_q3_k(b: &[u8], y: &mut [f32]) {
    let (hm, qs) = (&b[0..32], &b[32..96]);
    let scales = q3_k_scales(&b[96..108]);
    let d = f16_at(b, 108);
    for h in 0..2 {
        for j in 0..4 {
            for l in 0..32 {
                let e = h * 128 + j * 32 + l;
                let dl = d * (scales[e / 16] as i32 - 32) as f32;
                let lo = ((qs[h * 32 + l] >> (2 * j)) & 3) as i32;
                let q = lo - if hm[l] & (1 << (4 * h + j)) != 0 { 0 } else { 4 };
                y[e] = dl * q as f32;
            }
        }
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

/// block_q5_K: f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]
fn dequant_q5_k(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    for chunk in 0..4 {
        let is = chunk * 2;
        let (sc1, m1) = scale_min_k4(is, scales);
        let (sc2, m2) = scale_min_k4(is + 1, scales);
        let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
        let (u1, u2) = (1u8 << (2 * chunk), 2u8 << (2 * chunk));
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let y = &mut y[chunk * 64..chunk * 64 + 64];
        for l in 0..32 {
            let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[l] = d1 * ((q[l] & 0xF) + h1) as f32 - m1;
            y[l + 32] = d2 * ((q[l] >> 4) + h2) as f32 - m2;
        }
    }
}

/// IQ4_NL / IQ4_XS's codebook (ggml's kvalues_iq4nl).
pub const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// block_iq4_xs: f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]
fn dequant_iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let scales_h = u16::from_le_bytes([b[2], b[3]]) as i32;
    let scales_l = &b[4..8];
    let qs = &b[8..136];
    for ib in 0..8 {
        let ls = ((scales_l[ib / 2] as i32 >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        let dl = d * (ls - 32) as f32;
        let q = &qs[ib * 16..ib * 16 + 16];
        let y = &mut y[ib * 32..ib * 32 + 32];
        for j in 0..16 {
            y[j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
            y[j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
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

    /// ggml's dequantize_row_q3_K, transcribed literally (the aux/kmask
    /// scale unpacking and pointer walk), as the reference for ours.
    fn ggml_q3_k(b: &[u8]) -> Vec<f32> {
        let (kmask1, kmask2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
        let mut aux = [0u32; 4];
        for (i, a) in aux.iter_mut().take(3).enumerate() {
            *a = u32::from_le_bytes(b[96 + 4 * i..100 + 4 * i].try_into().unwrap());
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let scales: Vec<i8> = aux.iter().flat_map(|a| a.to_le_bytes()).map(|x| x as i8).collect();
        let d_all = f16_at(b, 108);
        let (hm, mut q) = (&b[0..32], 32usize);
        let (mut y, mut is, mut m) = (Vec::new(), 0, 1u8);
        for _n in 0..2 {
            let mut shift = 0;
            for _j in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    y.push(dl * (((b[q + l] >> shift) & 3) as i8 - if hm[l] & m != 0 { 0 } else { 4 }) as f32);
                }
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    y.push(dl * (((b[q + l + 16] >> shift) & 3) as i8 - if hm[l + 16] & m != 0 { 0 } else { 4 }) as f32);
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
        y
    }

    #[test]
    fn q3_k_matches_ggml() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..50 {
            let mut block: Vec<u8> = (0..110)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s as u8
                })
                .collect();
            // A finite f16 d.
            block[108..110].copy_from_slice(&half::f16::from_f32(0.0123).to_le_bytes());
            let mut ours = vec![0f32; 256];
            dequantize_row(DType::Q3K, &block, &mut ours);
            let want = ggml_q3_k(&block);
            assert!(ours.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

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
    fn q5_k_high_bit_adds_16() {
        // Sub-block scales 1, mins 0, nibbles 3; qh bit set for the first
        // 64-chunk's low half only -> 3 + 16 there, 3 elsewhere.
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        for j in 0..4 {
            block[4 + j] = 1;
        }
        for j in 8..12 {
            block[4 + j] = 1;
        }
        for h in &mut block[16..48] {
            *h = 1;
        }
        for q in &mut block[48..176] {
            *q = 0x33;
        }
        let mut out = [0f32; 256];
        dequantize_row(DType::Q5K, &block, &mut out);
        assert!(out[..32].iter().all(|&v| v == 19.0), "{:?}", &out[..4]);
        assert!(out[32..].iter().all(|&v| v == 3.0));
    }

    #[test]
    fn iq4_xs_codebook() {
        // ls = 33 (scale +1) for every sub-block, index i in element i % 16.
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&0xAAAAu16.to_le_bytes()); // high bits 2 -> 32
        block[4..8].fill(0x11); // low bits 1
        for (j, q) in block[8..136].iter_mut().enumerate() {
            *q = (j % 16) as u8 | (((j % 16) as u8) << 4);
        }
        let mut out = [0f32; 256];
        dequantize_row(DType::IQ4XS, &block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, KVALUES_IQ4NL[i % 16] as f32, "element {i}");
        }
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
