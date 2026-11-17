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

/// Quantize a row to Q8_0 (ggml's `quantize_row_q8_0_ref`): per 32 values,
/// `d = amax / 127` (stored as f16) and `q = round(x / d)`, rounding halves
/// away from zero like C's `roundf`. `x.len()` must be a multiple of 32.
pub fn quantize_q8_0(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len().is_multiple_of(32), "Q8_0 rows are whole blocks of 32");
    for blk in x.as_chunks::<32>().0 {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend(blk.iter().map(|&v| (v * id).round() as i8 as u8));
    }
}

#[cfg(test)]
mod quantize_tests {
    use super::*;

    #[test]
    fn q8_0_round_trips_within_a_step() {
        let x: Vec<f32> = (0..64).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
        let mut q = Vec::new();
        quantize_q8_0(&x, &mut q);
        assert_eq!(q.len(), 2 * 34);
        let mut y = vec![0f32; 64];
        dequantize_row(DType::Q8_0, &q, &mut y);
        for (blk, (a, b)) in x.chunks(32).zip(y.chunks(32)).enumerate() {
            let step = a.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            assert!(a.iter().zip(b).all(|(p, r)| (p - r).abs() <= step * 0.51 + 1e-6), "block {blk}");
        }
        // Halves round away from zero, as C's roundf.
        let mut q = Vec::new();
        let mut v = vec![0f32; 32];
        v[0] = 127.0;
        v[1] = 0.5;
        v[2] = -0.5;
        quantize_q8_0(&v, &mut q);
        assert_eq!((q[3] as i8, q[4] as i8), (1, -1));
    }
}

/// ggml's `nearest_int`: round to nearest via the 1.5·2²³ trick (ties to even).
fn nearest_int(f: f32) -> i32 {
    let v = f + 12_582_912.0;
    (v.to_bits() as i32 & 0x007f_ffff) - 0x0040_0000
}

const GROUP_MAX_EPS: f32 = 1e-15;

/// ggml's `make_qkx2_quants`: scale and min for `n` values in `0..=nmax`,
/// searched to minimize the weighted error. Returns (scale, -min).
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(x: &[f32], w: &[f32], nmax: i32, l: &mut [u8], aux: &mut [u8], rmin: f32, rdelta: f32, nstep: i32) -> (f32, f32) {
    let n = x.len();
    let (mut min, mut max) = (x[0], x[0]);
    let (mut sum_w, mut sum_x) = (w[0], w[0] * x[0]);
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += w[i];
        sum_x += w[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l[..n].fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best = 0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let d = scale * q as f32 + min - x[i];
        best += w[i] * d * d;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sl, mut sl2, mut sxl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            aux[i] = q as u8;
            let q = q as f32;
            sl += w[i] * q;
            sl2 += w[i] * q * q;
            sxl += w[i] * q * x[i];
        }
        let det = sum_w * sl2 - sl * sl;
        if det > 0.0 {
            let mut this_scale = (sum_w * sxl - sum_x * sl) / det;
            let mut this_min = (sl2 * sum_x - sl * sxl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sxl / sl2;
            }
            let mut err = 0f32;
            for i in 0..n {
                let d = this_scale * aux[i] as f32 + this_min - x[i];
                err += w[i] * d * d;
            }
            if err < best {
                l[..n].copy_from_slice(&aux[..n]);
                best = err;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Quantize a row to Q4_K (ggml's `quantize_row_q4_K_ref`, which
/// llama-quantize uses without an importance matrix). `x.len()` must be a
/// multiple of 256.
pub fn quantize_q4_k(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len().is_multiple_of(256), "Q4_K rows are whole blocks of 256");
    for x in x.as_chunks::<256>().0 {
        let mut l = [0u8; 256];
        let mut aux = [0u8; 32];
        let mut w = [0f32; 32];
        let (mut scales, mut mins) = ([0f32; 8], [0f32; 8]);
        let (mut max_scale, mut max_min) = (0f32, 0f32);
        for j in 0..8 {
            let xs = &x[32 * j..32 * j + 32];
            let av = (xs.iter().map(|v| v * v).sum::<f32>() / 32.0).sqrt();
            for (wi, v) in w.iter_mut().zip(xs) {
                *wi = av + v.abs();
            }
            let (s, m) = make_qkx2_quants(xs, &w, 15, &mut l[32 * j..32 * j + 32], &mut aux, -1.0, 0.1, 20);
            scales[j] = s;
            mins[j] = m;
            max_scale = max_scale.max(s);
            max_min = max_min.max(m);
        }
        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut sc = [0u8; 12];
        for j in 0..8 {
            // As C: nearest_int into a uint8_t (wraps), then MIN(63, ·).
            let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
            let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
            if j < 4 {
                sc[j] = ls;
                sc[j + 4] = lm;
            } else {
                sc[j + 4] = (ls & 0xf) | ((lm & 0xf) << 4);
                sc[j - 4] |= (ls >> 4) << 6;
                sc[j] |= (lm >> 4) << 6;
            }
        }
        let d = f16::from_f32(max_scale / 63.0);
        let dmin = f16::from_f32(max_min / 63.0);
        for j in 0..8 {
            let (s, m) = if j < 4 { (sc[j] & 63, sc[j + 4] & 63) } else { ((sc[j + 4] & 0xf) | ((sc[j - 4] >> 6) << 4), (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) };
            let dd = d.to_f32() * s as f32;
            if dd == 0.0 {
                continue;
            }
            let dm = dmin.to_f32() * m as f32;
            for i in 0..32 {
                l[32 * j + i] = nearest_int((x[32 * j + i] + dm) / dd).clamp(0, 15) as u8;
            }
        }
        out.extend_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&dmin.to_le_bytes());
        out.extend_from_slice(&sc);
        for j in (0..256).step_by(64) {
            out.extend((0..32).map(|i| l[j + i] | (l[j + i + 32] << 4)));
        }
    }
}

/// ggml's `make_qx_quants` with `rmse_type = 1` and no importance weights:
/// signed levels in `-nmax..nmax`, stored as `l + nmax`. Returns the scale.
fn make_qx_quants(x: &[f32], nmax: i32, l: &mut [i8]) -> f32 {
    let (mut max, mut amax) = (0f32, 0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l.fill(0);
        return 0.0;
    }
    let level = |iscale: f32, v: f32| nearest_int(iscale * v).clamp(-nmax, nmax - 1);
    let mut iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = (0f32, 0f32);
    for (li, &v) in l.iter_mut().zip(x) {
        let q = level(iscale, v);
        *li = (q + nmax) as i8;
        let w = v * v;
        sumlx += w * v * q as f32;
        suml2 += w * (q * q) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let (mut sumlx, mut suml2) = (0f32, 0f32);
        for &v in x {
            let q = level(iscale, v);
            let w = v * v;
            sumlx += w * v * q as f32;
            suml2 += w * (q * q) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for (li, &v) in l.iter_mut().zip(x) {
                *li = (level(iscale, v) + nmax) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Quantize a row to Q6_K (ggml's `quantize_row_q6_K_ref`). `x.len()` must
/// be a multiple of 256.
pub fn quantize_q6_k(x: &[f32], out: &mut Vec<u8>) {
    assert!(x.len().is_multiple_of(256), "Q6_K rows are whole blocks of 256");
    for x in x.as_chunks::<256>().0 {
        let mut l = [0i8; 256];
        let mut scales = [0f32; 16];
        let (mut max_scale, mut max_abs) = (0f32, 0f32);
        for ib in 0..16 {
            let s = make_qx_quants(&x[16 * ib..16 * ib + 16], 32, &mut l[16 * ib..16 * ib + 16]);
            scales[ib] = s;
            if s.abs() > max_abs {
                max_abs = s.abs();
                max_scale = s;
            }
        }
        if max_abs < GROUP_MAX_EPS {
            out.extend(std::iter::repeat_n(0u8, 210));
            continue;
        }
        let iscale = -128.0 / max_scale;
        let d = f16::from_f32(1.0 / iscale);
        let sc: [i8; 16] = std::array::from_fn(|ib| nearest_int(iscale * scales[ib]).min(127) as i8);
        for j in 0..16 {
            let dd = d.to_f32() * sc[j] as f32;
            if dd == 0.0 {
                continue;
            }
            for i in 0..16 {
                l[16 * j + i] = (nearest_int(x[16 * j + i] / dd).clamp(-32, 31) + 32) as i8;
            }
        }
        let l: [u8; 256] = l.map(|v| v as u8);
        let (mut ql, mut qh) = ([0u8; 128], [0u8; 64]);
        for (half, j) in (0..256).step_by(128).enumerate() {
            for i in 0..32 {
                let (q1, q2, q3, q4) = (l[j + i] & 0xf, l[j + i + 32] & 0xf, l[j + i + 64] & 0xf, l[j + i + 96] & 0xf);
                ql[half * 64 + i] = q1 | (q3 << 4);
                ql[half * 64 + i + 32] = q2 | (q4 << 4);
                qh[half * 32 + i] = (l[j + i] >> 4) | ((l[j + i + 32] >> 4) << 2) | ((l[j + i + 64] >> 4) << 4) | ((l[j + i + 96] >> 4) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(sc.iter().map(|&s| s as u8));
        out.extend_from_slice(&d.to_le_bytes());
    }
}

#[cfg(test)]
mod k_quantize_tests {
    use super::*;

    fn row(n: usize) -> Vec<f32> {
        // Deterministic, weight-like values (with a few outliers).
        let mut s = 0x9e37_79b9u32;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let u = (s as f32 / u32::MAX as f32) - 0.5;
                if i % 97 == 0 { u * 0.4 } else { u * 0.05 }
            })
            .collect()
    }

    fn rel_rmse(dtype: DType, q: &[u8], x: &[f32]) -> f32 {
        let mut y = vec![0f32; x.len()];
        dequantize_row(dtype, q, &mut y);
        let e: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum();
        let s: f32 = x.iter().map(|a| a * a).sum();
        (e / s).sqrt()
    }

    #[test]
    fn q4_k_and_q6_k_round_trip() {
        let x = row(1024);
        let (mut q4, mut q6) = (Vec::new(), Vec::new());
        quantize_q4_k(&x, &mut q4);
        quantize_q6_k(&x, &mut q6);
        assert_eq!((q4.len(), q6.len()), (4 * 144, 4 * 210));
        let (e4, e6) = (rel_rmse(DType::Q4K, &q4, &x), rel_rmse(DType::Q6K, &q6, &x));
        // Typical relative errors: Q4_K ~ 6-9%, Q6_K ~ 1.5-2.5%.
        assert!(e4 < 0.1 && e6 < 0.03 && e6 < e4 / 2.5, "Q4_K {e4}, Q6_K {e6}");
        // Zeros stay zeros.
        let (mut z4, mut z6) = (Vec::new(), Vec::new());
        quantize_q4_k(&[0.0; 256], &mut z4);
        quantize_q6_k(&[0.0; 256], &mut z6);
        let mut y = vec![1f32; 256];
        dequantize_row(DType::Q4K, &z4, &mut y);
        assert!(y.iter().all(|&v| v == 0.0));
        dequantize_row(DType::Q6K, &z6, &mut y);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn nearest_int_matches_ggml() {
        assert_eq!([nearest_int(0.5), nearest_int(1.5), nearest_int(-0.5), nearest_int(2.4), nearest_int(-2.6)], [0, 2, 0, 2, -3]);
    }
}
