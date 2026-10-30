// rllm GPU kernels. Compiled at runtime for the device's arch: NVRTC on
// NVIDIA, hiprtc on AMD (rllm-hip passes -DRLLM_HIP).
//
// Portability rules:
//   - no inline PTX outside the small "vendor shim" helpers (f16 cvt, mma),
//     each with a HIP version (or, for tensor cores, a stub; see below)
//   - warp width comes from WARP_SIZE, never a literal 32
//   - no headers: NVRTC/hiprtc builtins only
//
// Quantized GGUF weights are repacked on upload (repack_* below) into device
// layouts built for aligned 16-byte loads and dp4a dot products. Everything
// is addressed in "units" of 32 consecutive elements along a row. The repack
// math matches rllm-core/src/quant.rs (ported from ggml, MIT).

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

typedef unsigned char u8;
typedef unsigned int u32;
typedef signed char i8;

#define DT_F32 0
#define DT_F16 1
#define DT_BF16 2
#define DT_Q8_0 3
#define DT_Q4_K 4
#define DT_Q6_K 5

typedef unsigned short f16_bits;

#define NEG_INF __int_as_float(0xff800000)

#ifdef RLLM_HIP
// Vendor shim (HIP): RDNA runs wave32, the same lockstep width as a CUDA
// warp. HIP's *_sync shuffles want 64-bit masks; every rllm caller uses
// the whole warp, so map them to the unmasked forms.
#define __shfl_xor_sync(mask, v, o) __shfl_xor(v, o, WARP_SIZE)
#define __shfl_sync(mask, v, l) __shfl(v, l, WARP_SIZE)
#define __syncwarp() __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront")
#define __dp4a(a, b, c) __builtin_amdgcn_sudot4(true, (int)(a), true, (int)(b), (int)(c), false)
#define __ldcs(p) (*(p))
// Kernels that don't exist on this target, so the backend can still load every name.
#define RLLM_STUB(NAME) extern "C" __global__ void NAME() { __builtin_trap(); }

__device__ __forceinline__ float f16_to_f32(f16_bits h) {
    return (float)__builtin_bit_cast(_Float16, h);
}

__device__ __forceinline__ f16_bits f32_to_f16_bits(float f) {
    return __builtin_bit_cast(f16_bits, (_Float16)f);
}
#else
// Vendor shim: f16 <-> f32 with the hardware cvt instructions (round to
// nearest even).
__device__ __forceinline__ float f16_to_f32(f16_bits h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ f16_bits f32_to_f16_bits(float f) {
    f16_bits h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
#endif

__device__ __forceinline__ float ld_f16(const u8* p) {
    return f16_to_f32((f16_bits)(p[0] | (p[1] << 8)));
}

// Same, for a 2-byte aligned value: one load instead of two byte loads (the
// repacked weight scales are always aligned).
__device__ __forceinline__ float ld_f16a(const u8* p) {
    return f16_to_f32(*(const f16_bits*)p);
}

__device__ __forceinline__ float ld_bf16(const u8* p) {
    return __uint_as_float((u32)(p[0] | (p[1] << 8)) << 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// ---------------------------------------------------------------------------
// Device weight formats. Row r of a weight starts at qs + r*qs_row (and
// qh + r*qh_row, sc + r*sc_row).
//   FMT_F32/F16/BF16  plain row-major values, as in GGUF.
//   FMT_Q8  qs: int8 [k]; sc: f16 per unit (value = d * q).
//   FMT_Q4  qs: 16 bytes per unit, byte b = q[b] | q[b+16] << 4;
//           sc: f16 pair (d, m) per unit (value = d * q - m).
//   FMT_Q6  qs: low 4 bits, laid out like FMT_Q4; qh: 8 bytes per unit,
//           byte b = hi[b] | hi[b+8] << 2 | hi[b+16] << 4 | hi[b+24] << 6;
//           sc: f16 per 16 elements (value = d * ((lo | hi << 4) - 32)).
#define FMT_F32 0
#define FMT_F16 1
#define FMT_BF16 2
#define FMT_Q8 3
#define FMT_Q4 4
#define FMT_Q6 5

// Dequantize elements [4*sub, 4*sub + 4) of unit `u` of one row.
// The single definition of each device format; everything else builds on it.
template <int F>
__device__ __forceinline__ void dequant4(const u8* qs, const u8* qh, const u8* sc, int u, int sub, float* y) {
    int e = sub * 4;
    if (F == FMT_F32) {
        const float* p = (const float*)qs + u * 32 + e;
        for (int i = 0; i < 4; i++) y[i] = p[i];
    } else if (F == FMT_F16) {
        const u8* p = qs + (u * 32 + e) * 2;
        for (int i = 0; i < 4; i++) y[i] = ld_f16(p + 2 * i);
    } else if (F == FMT_BF16) {
        const u8* p = qs + (u * 32 + e) * 2;
        for (int i = 0; i < 4; i++) y[i] = ld_bf16(p + 2 * i);
    } else if (F == FMT_Q8) {
        float d = ld_f16(sc + u * 2);
        const i8* q = (const i8*)qs + u * 32 + e;
        for (int i = 0; i < 4; i++) y[i] = d * (float)q[i];
    } else if (F == FMT_Q4) {
        float d = ld_f16(sc + u * 4), m = ld_f16(sc + u * 4 + 2);
        const u8* q = qs + u * 16 + (e & 15);
        int shift = (e >> 4) * 4;
        for (int i = 0; i < 4; i++) y[i] = d * (float)((q[i] >> shift) & 0xF) - m;
    } else if (F == FMT_Q6) {
        float d = ld_f16(sc + ((u * 32 + e) >> 4) * 2);
        const u8* lo = qs + u * 16 + (e & 15);
        const u8* hi = qh + u * 8 + (e & 7);
        int lshift = (e >> 4) * 4, hshift = 2 * (e >> 3);
        for (int i = 0; i < 4; i++) {
            int q = ((lo[i] >> lshift) & 0xF) | (((hi[i] >> hshift) & 3) << 4);
            y[i] = d * (float)(q - 32);
        }
    }
}

// ---------------------------------------------------------------------------
// Repacking GGUF blocks into device formats. One thread per (row, unit);
// rows [row0, row0 + n) of the output are filled from `raw`, which holds just
// those rows in GGUF layout.

__device__ __forceinline__ void st_f16(u8* p, float v) {
    f16_bits h = f32_to_f16_bits(v);
    p[0] = h & 0xff;
    p[1] = h >> 8;
}

extern "C" __global__ void repack_q8_0(const u8* __restrict__ raw, u8* qs, u8* sc, int row0, int n, int k) {
    int units = k / 32;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * units) return;
    int r = idx / units, u = idx % units;
    const u8* b = raw + (long)r * units * 34 + u * 34;
    long orow = row0 + r;
    u8* q = qs + orow * k + u * 32;
    for (int i = 0; i < 32; i++) q[i] = b[2 + i];
    u8* s = sc + orow * units * 2 + u * 2;
    s[0] = b[0];
    s[1] = b[1];
}

extern "C" __global__ void repack_q4_k(const u8* __restrict__ raw, u8* qs, u8* sc, int row0, int n, int k) {
    int units = k / 32;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * units) return;
    int r = idx / units, u = idx % units;
    const u8* b = raw + (long)r * (k / 256) * 144 + (u / 8) * 144;
    int j = u % 8;
    float d = ld_f16(b), dmin = ld_f16(b + 2);
    const u8* scl = b + 4;
    int s, m;
    if (j < 4) {
        s = scl[j] & 63;
        m = scl[j + 4] & 63;
    } else {
        s = (scl[j + 4] & 0xF) | ((scl[j - 4] >> 6) << 4);
        m = (scl[j + 4] >> 4) | ((scl[j] >> 6) << 4);
    }
    long orow = row0 + r;
    st_f16(sc + orow * units * 4 + u * 4, d * s);
    st_f16(sc + orow * units * 4 + u * 4 + 2, dmin * m);
    const u8* q = b + 16 + (j / 2) * 32;
    int shift = (j & 1) * 4;
    u8* o = qs + orow * (k / 2) + u * 16;
    for (int i = 0; i < 16; i++) o[i] = ((q[i] >> shift) & 0xF) | (((q[i + 16] >> shift) & 0xF) << 4);
}

extern "C" __global__ void repack_q6_k(const u8* __restrict__ raw, u8* qs, u8* qh, u8* sc, int row0, int n, int k) {
    int units = k / 32;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * units) return;
    int r = idx / units, u = idx % units;
    const u8* b = raw + (long)r * (k / 256) * 210 + (u / 8) * 210;
    int j = u % 8, half = j / 4, rr = j % 4;
    const u8* ql = b + half * 64 + (rr & 1) * 32;
    const u8* qhh = b + 128 + half * 32;
    const i8* scl = (const i8*)(b + 192 + half * 8);
    float d = ld_f16(b + 208);
    int nshift = (rr >> 1) * 4;
    u8 q[32];
    for (int l = 0; l < 32; l++) q[l] = ((ql[l] >> nshift) & 0xF) | (((qhh[l] >> (2 * rr)) & 3) << 4);
    long orow = row0 + r;
    u8* lo = qs + orow * (k / 2) + u * 16;
    for (int i = 0; i < 16; i++) lo[i] = (q[i] & 0xF) | ((q[i + 16] & 0xF) << 4);
    u8* hi = qh + orow * (k / 4) + u * 8;
    for (int i = 0; i < 8; i++)
        hi[i] = (q[i] >> 4) | ((q[i + 8] >> 4) << 2) | ((q[i + 16] >> 4) << 4) | ((q[i + 24] >> 4) << 6);
    st_f16(sc + orow * (k / 16) * 2 + u * 4, d * scl[2 * rr]);
    st_f16(sc + orow * (k / 16) * 2 + u * 4 + 2, d * scl[2 * rr + 1]);
}

// ---------------------------------------------------------------------------
// Activations for quantized matvec: x f32 [rows, k] -> xq int8 + xd f32 per
// 32-element unit (d = amax / 127, q = round(x / d)), as llama.cpp's q8_1.
// One warp per unit, lane = element (assumes WARP_SIZE == 32).
// Also writes xs = d * sum(q) per unit, used for Q4's offset term in mmq.
// Eight lanes per unit, four elements each (vector loads, one packed store).
extern "C" __global__ void quantize_q8(const float* __restrict__ x, i8* __restrict__ xq, float* __restrict__ xd,
                                       float* __restrict__ xs, long n_units) {
    long unit = ((long)blockIdx.x * blockDim.x + threadIdx.x) / 8;
    int sub = threadIdx.x % 8;
    if (unit >= n_units) return;  // whole 8-lane groups in or out
    const unsigned mask = __activemask();
    float4 v = *(const float4*)(x + unit * 32 + sub * 4);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(mask, amax, o));
    float d = amax / 127.f;
    float q0 = d > 0.f ? rintf(v.x / d) : 0.f, q1 = d > 0.f ? rintf(v.y / d) : 0.f;
    float q2 = d > 0.f ? rintf(v.z / d) : 0.f, q3 = d > 0.f ? rintf(v.w / d) : 0.f;
    *(int*)(xq + unit * 32 + sub * 4) =
        ((int)q0 & 0xFF) | (((int)q1 & 0xFF) << 8) | (((int)q2 & 0xFF) << 16) | (((int)q3 & 0xFF) << 24);
    float qsum = q0 + q1 + q2 + q3;
    for (int o = 4; o > 0; o >>= 1) qsum += __shfl_xor_sync(mask, qsum, o);
    if (sub == 0) {
        xd[unit] = d;
        xs[unit] = d * qsum;
    }
}

// Optional int8 copy of a producer's output (same scheme as quantize_q8), so
// the matmul consuming it can skip its own quantize launch.
struct QOut {
    i8* q;
    float* d;
    float* s;
    int on;
};

// All 32 lanes of a warp hold the 32 elements of `unit` (lane = element).
__device__ __forceinline__ void quant_unit(float v, long unit, const QOut& o) {
    float amax = warp_max(fabsf(v));
    float d = amax / 127.f;
    float q = d > 0.f ? rintf(v / d) : 0.f;
    int lane = threadIdx.x % WARP_SIZE;
    o.q[unit * 32 + lane] = (i8)q;
    float qsum = warp_sum(q);
    if (lane == 0) {
        o.d[unit] = d;
        o.s[unit] = d * qsum;
    }
}

// Quantize row `row` (width d, already written to `o` by this block).
__device__ __forceinline__ void quant_row(const float* o, long row, int d, const QOut& qo) {
    __syncthreads();
    int warps = blockDim.x / WARP_SIZE, warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    for (int u = warp; u < d / 32; u += warps) quant_unit(o[u * 32 + lane], row * (d / 32) + u, qo);
}

// Signed int8 lanes of (q - 32) for 6-bit values q in 0..63 packed per byte.
__device__ __forceinline__ int q6_to_s8x4(int q) {
    int t = q ^ 0x20202020;
    return t | ((t & 0x20202020) * 6);  // sign-extend bit 5 of each byte
}

// ---------------------------------------------------------------------------
// Matrix-vector product for small batches: out[m, n] = x[m, k] . W[n, k]^T
// One warp per output row; weights are read once for all m (<= MV_MAX_M) rows.
#define MV_MAX_M 8

// Unrolling the unit loops keeps more weight loads in flight on RDNA (+1%
// decode); on NVIDIA it cost registers (-6% Qwen3-8B decode on sm_89).
#ifdef RLLM_HIP
#define MV_UNROLL _Pragma("unroll 4")
#else
#define MV_UNROLL
#endif

// Quantized weights: int8 dot products (dp4a) against quantize_q8 activations.
// Q8/Q6: each lane takes half a unit (16 elements) per step; Q4: a whole unit.
// Every weight load is an aligned 16-byte read, contiguous across the warp.
// `block` is the block index within this weight (see matvec_multi).
// One row's partial dot products (per lane; the caller reduces the warp):
// acc[j] += row . x[j] for j < m.
template <int F>
__device__ __forceinline__ void mv_dot(const u8* __restrict__ qr, const u8* __restrict__ hr, const u8* __restrict__ sr,
                                       const i8* __restrict__ xq, const float* __restrict__ xd, int k, int m,
                                       int lane, float* acc) {
    const int units = k / 32;
    if (F == FMT_Q4) {
        MV_UNROLL
        for (int u = lane; u < units; u += WARP_SIZE) {
            int4 w = *(const int4*)(qr + u * 16);
            const int* wi = (const int*)&w;
            u32 dm = *(const u32*)(sr + u * 4);
            float d = f16_to_f32(dm & 0xffff), mn = f16_to_f32(dm >> 16);
#pragma unroll
            for (int j = 0; j < MV_MAX_M; j++) {
                if (j < m) {
                    const int4* xp = (const int4*)(xq + (long)j * k + u * 32);
                    int4 xa = xp[0], xb = xp[1];
                    const int* a = (const int*)&xa;
                    const int* b = (const int*)&xb;
                    int s = 0, sx = 0;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        s = __dp4a(wi[i] & 0x0F0F0F0F, a[i], s);
                        s = __dp4a((wi[i] >> 4) & 0x0F0F0F0F, b[i], s);
                        sx = __dp4a(0x01010101, a[i], sx);
                        sx = __dp4a(0x01010101, b[i], sx);
                    }
                    acc[j] += xd[(long)j * units + u] * (d * (float)s - mn * (float)sx);
                }
            }
        }
    } else {
        MV_UNROLL
        for (int h = lane; h < units * 2; h += WARP_SIZE) {
            int u = h >> 1, half = h & 1;
            int w[4];
            float d;
            if (F == FMT_Q8) {
                int4 v = *(const int4*)(qr + h * 16);
                w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
                d = ld_f16a(sr + u * 2);
            } else {  // FMT_Q6
                int4 lo = *(const int4*)(qr + u * 16);
                int2 hi = *(const int2*)(hr + u * 8);
                const int* l4 = (const int*)&lo;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    int lv = (l4[i] >> (4 * half)) & 0x0F0F0F0F;
                    int hv = (((i & 1) ? hi.y : hi.x) >> (2 * (2 * half + (i >> 1)))) & 0x03030303;
                    w[i] = q6_to_s8x4(lv | (hv << 4));
                }
                d = ld_f16a(sr + h * 2);
            }
#pragma unroll
            for (int j = 0; j < MV_MAX_M; j++) {
                if (j < m) {
                    int4 xv = *(const int4*)(xq + (long)j * k + h * 16);
                    int s = __dp4a(w[0], xv.x, 0);
                    s = __dp4a(w[1], xv.y, s);
                    s = __dp4a(w[2], xv.z, s);
                    s = __dp4a(w[3], xv.w, s);
                    acc[j] += d * xd[(long)j * units + u] * (float)s;
                }
            }
        }
    }
}

template <int F>
__device__ void matvec_q(int block, const u8* __restrict__ qs, const u8* __restrict__ qh, const u8* __restrict__ sc,
                         const i8* __restrict__ xq, const float* __restrict__ xd, float* __restrict__ out, int n,
                         int k, int m, long qs_row, long qh_row, long sc_row) {
    int warps = blockDim.x / WARP_SIZE;
    int r = block * warps + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (r >= n) return;
    float acc[MV_MAX_M];
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++) acc[j] = 0.f;
    mv_dot<F>(qs + (long)r * qs_row, qh + (long)r * qh_row, sc + (long)r * sc_row, xq, xd, k, m, lane, acc);
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++) {
        if (j < m) {
            float s = warp_sum(acc[j]);
            if (lane == 0) out[(long)j * n + r] = s;
        }
    }
}

// Up to MV_TARGETS weights of one format that share the same input x (e.g.
// Q/K/V projections), computed in one launch. Blocks [block_start[i],
// block_start[i+1]) work on target i. Pointers are raw device addresses.
#define MV_TARGETS 3
struct MvTarget {
    const u8* qs;
    const u8* qh;
    const u8* sc;
    float* out;
    long qs_row, qh_row, sc_row;
    int n, block_start;
};
struct MvTargets {
    MvTarget t[MV_TARGETS];
    int count;
};

// Float weights: f32 activations, 8 elements per lane per step (one 16-byte
// load for f16/bf16, two for f32).
template <int F>
__device__ __forceinline__ void load8(const u8* row, int c, float* y) {
    if (F == FMT_F32) {
        const float4* p = (const float4*)(row + (long)c * 32);
        float4 a = p[0], b = p[1];
        y[0] = a.x; y[1] = a.y; y[2] = a.z; y[3] = a.w;
        y[4] = b.x; y[5] = b.y; y[6] = b.z; y[7] = b.w;
    } else {
        uint4 v = *(const uint4*)(row + (long)c * 16);
        const u32* w = (const u32*)&v;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (F == FMT_F16) {
                y[2 * i] = f16_to_f32(w[i] & 0xffff);
                y[2 * i + 1] = f16_to_f32(w[i] >> 16);
            } else {
                y[2 * i] = __uint_as_float(w[i] << 16);
                y[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
            }
        }
    }
}

// KW warps share each row (splitting k) when there are too few rows to fill
// the GPU one warp per row; they combine through shared memory. KW is a
// template parameter: a runtime stride kept the compiler from unrolling the
// common KW = 1 loop (~10% slower).
template <int F, int KW>
__device__ void matvec_f(int block, const u8* __restrict__ w, const float* __restrict__ x, float* __restrict__ out,
                         int n, int k, int m, long row_bytes) {
    constexpr int kw = KW;
    __shared__ float part[MV_MAX_M][8];
    const int warps = blockDim.x / WARP_SIZE, warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r = block * (warps / kw) + warp / kw, sub = warp % kw;
    const bool valid = r < n;  // uniform per row group: no early return (barrier below)
    const u8* row = w + (long)(valid ? r : 0) * row_bytes;
    float acc[MV_MAX_M];
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++) acc[j] = 0.f;
    float y[8];
    for (int c = lane + sub * WARP_SIZE; valid && c < k / 8; c += WARP_SIZE * kw) {
        load8<F>(row, c, y);
#pragma unroll
        for (int j = 0; j < MV_MAX_M; j++) {
            if (j < m) {
                const float4* xp = (const float4*)(x + (long)j * k + c * 8);
                float4 a = xp[0], b = xp[1];
                acc[j] += y[0] * a.x + y[1] * a.y + y[2] * a.z + y[3] * a.w + y[4] * b.x + y[5] * b.y +
                          y[6] * b.z + y[7] * b.w;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++)
        if (j < m) acc[j] = warp_sum(acc[j]);  // only m reductions: the row loop is short
    if (kw > 1) {
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < MV_MAX_M; j++) part[j][warp] = acc[j];
        __syncthreads();
        if (sub != 0) return;
#pragma unroll
        for (int j = 0; j < MV_MAX_M; j++)
            for (int q = 1; q < kw; q++) acc[j] += part[j][warp + q];
    }
    if (!valid || lane != 0) return;
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++)
        if (j < m) out[(long)j * n + r] = acc[j];
}

// Dequantize a whole [n, k] weight into f16 (for the cuBLAS GEMM path).
// One thread per 4 elements; writes are 8-byte and contiguous across threads.
template <int F>
__device__ void dequant_f16(const u8* __restrict__ qs, const u8* __restrict__ qh, const u8* __restrict__ sc,
                            f16_bits* __restrict__ out, int n, int k, long qs_row, long qh_row, long sc_row) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * (k / 4)) return;
    int r = idx / (k / 4), c = idx % (k / 4);
    float y[4];
    dequant4<F>(qs + r * qs_row, qh + r * qh_row, sc + r * sc_row, c / 8, c % 8, y);
    uint2 v;
    v.x = f32_to_f16_bits(y[0]) | ((u32)f32_to_f16_bits(y[1]) << 16);
    v.y = f32_to_f16_bits(y[2]) | ((u32)f32_to_f16_bits(y[3]) << 16);
    *(uint2*)(out + (long)r * k + c * 4) = v;
}

// out[i, :] = table[tokens[i], :]. One thread per 4 elements.
template <int F>
__device__ void embed(const u8* __restrict__ qs, const u8* __restrict__ qh, const u8* __restrict__ sc,
                      const u32* __restrict__ tokens, float* __restrict__ out, int n_tok, int k, long qs_row,
                      long qh_row, long sc_row) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n_tok * (k / 4)) return;
    int t = idx / (k / 4), c = idx % (k / 4);
    long r = tokens[t];
    float y[4];
    dequant4<F>(qs + r * qs_row, qh + r * qh_row, sc + r * sc_row, c / 8, c % 8, y);
    *(float4*)(out + (long)t * k + c * 4) = make_float4(y[0], y[1], y[2], y[3]);
}

#define INSTANTIATE_COMMON(NAME, F)                                                                           \
    extern "C" __global__ void dequant_f16_##NAME(const u8* qs, const u8* qh, const u8* sc, f16_bits* out,   \
                                                  int n, int k, long qs_row, long qh_row, long sc_row) {     \
        dequant_f16<F>(qs, qh, sc, out, n, k, qs_row, qh_row, sc_row);                                       \
    }                                                                                                         \
    extern "C" __global__ void embed_##NAME(const u8* qs, const u8* qh, const u8* sc, const u32* tokens,      \
                                            float* out, int n_tok, int k, long qs_row, long qh_row,          \
                                            long sc_row) {                                                    \
        embed<F>(qs, qh, sc, tokens, out, n_tok, k, qs_row, qh_row, sc_row);                                 \
    }

#define INSTANTIATE_FLOAT(NAME, F)                                                                            \
    INSTANTIATE_COMMON(NAME, F)                                                                               \
    extern "C" __global__ void matvec_##NAME(MvTargets ts, const float* x, int k, int m, int kw) {        \
        int i = 0;                                                                                            \
        while (i + 1 < ts.count && (int)blockIdx.x >= ts.t[i + 1].block_start) i++;                         \
        const MvTarget& t = ts.t[i];                                                                          \
        if (kw == 4)                                                                                          \
            matvec_f<F, 4>(blockIdx.x - t.block_start, t.qs, x, t.out, t.n, k, m, t.qs_row);                  \
        else                                                                                                  \
            matvec_f<F, 1>(blockIdx.x - t.block_start, t.qs, x, t.out, t.n, k, m, t.qs_row);                  \
    }

#define INSTANTIATE_QUANT(NAME, F)                                                                            \
    INSTANTIATE_COMMON(NAME, F)                                                                               \
    extern "C" __global__ void matvec_##NAME(MvTargets ts, const i8* xq, const float* xd, int k, int m) {    \
        int i = 0;                                                                                            \
        while (i + 1 < ts.count && (int)blockIdx.x >= ts.t[i + 1].block_start) i++;                         \
        const MvTarget& t = ts.t[i];                                                                          \
        matvec_q<F>(blockIdx.x - t.block_start, t.qs, t.qh, t.sc, xq, xd, t.out, t.n, k, m, t.qs_row,        \
                    t.qh_row, t.sc_row);                                                                      \
    }

INSTANTIATE_FLOAT(f32, FMT_F32)
INSTANTIATE_FLOAT(f16, FMT_F16)
INSTANTIATE_FLOAT(bf16, FMT_BF16)
INSTANTIATE_QUANT(q8, FMT_Q8)
INSTANTIATE_QUANT(q4, FMT_Q4)
INSTANTIATE_QUANT(q6, FMT_Q6)

// ---------------------------------------------------------------------------
// Elementwise and normalization

extern "C" __global__ void f32_to_f16(const float* __restrict__ x, f16_bits* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f32_to_f16_bits(x[i]);
}

// bf16 = upper 16 bits of the f32, rounded to nearest even.
extern "C" __global__ void f32_to_bf16(const float* __restrict__ x, unsigned short* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u32 b = __float_as_uint(x[i]);
        out[i] = (unsigned short)((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }
}

// 16-bit GEMM output (f16, or bf16 if `bf`) back to f32 times `scale`, 4
// values per thread.
extern "C" __global__ void half_to_f32(const unsigned short* __restrict__ x, float* __restrict__ out, long n, int bf,
                                       float scale) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    for (long j = i; j < min(i + 4, n); j++) out[j] = (bf ? __uint_as_float((u32)x[j] << 16) : f16_to_f32(x[j])) * scale;
}

// One block per row of width d.
extern "C" __global__ void __launch_bounds__(1024) rms_norm(const float* __restrict__ x, const float* __restrict__ w,
                                    float* __restrict__ out, int d, float eps, QOut qo) {
    __shared__ float partial[1024 / WARP_SIZE];
    const float* xr = x + (long)blockIdx.x * d;
    float* o = out + (long)blockIdx.x * d;
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += xr[i] * xr[i];
    s = warp_sum(s);
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) partial[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = lane < blockDim.x / WARP_SIZE ? partial[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) partial[0] = s;
    }
    __syncthreads();
    float scale = rsqrtf(partial[0] / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[i] = xr[i] * scale * w[i];
    if (qo.on) quant_row(o, blockIdx.x, d, qo);
}

// x += y; out = rmsnorm(x) * w. One block per row of width d.
extern "C" __global__ void __launch_bounds__(1024) add_rms_norm(float* __restrict__ x, const float* __restrict__ y,
                                        const float* __restrict__ w, float* __restrict__ out, int d, float eps,
                                        QOut qo) {
    __shared__ float partial[1024 / WARP_SIZE];
    float* xr = x + (long)blockIdx.x * d;
    const float* yr = y + (long)blockIdx.x * d;
    float* o = out + (long)blockIdx.x * d;
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = warp_sum(s);
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) partial[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = lane < blockDim.x / WARP_SIZE ? partial[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) partial[0] = s;
    }
    __syncthreads();
    float scale = rsqrtf(partial[0] / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[i] = xr[i] * scale * w[i];
    if (qo.on) quant_row(o, blockIdx.x, d, qo);
}

// Block-wide sum (all threads get the result). `scratch` holds 1024/WARP_SIZE floats.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    v = warp_sum(v);
    __syncthreads();  // scratch may still be in use by a previous call
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < blockDim.x / WARP_SIZE ? scratch[lane] : 0.f;
    return warp_sum(v);
}

// x = (x + rmsnorm(src) * w) * scale; if has_next: out = rmsnorm(x) * w2.
// One block per row of width d <= RNA_MAX * blockDim; each thread keeps its
// elements (strided by blockDim) in registers between the passes.
#define RNA_MAX 8
// Launched with up to 1024 threads: the bound makes the compiler fit them
// (sm_89 otherwise gave rms_norm_add 96 registers, too many for 1024).
// Reads the residual from xin and writes it to x (the same buffer, or a
// different one: see Backend::norm_add_gated_ffn).
extern "C" __global__ void __launch_bounds__(1024) rms_norm_add(const float* __restrict__ src, const float* __restrict__ w,
                                        const float* xin, float* x, const float* __restrict__ w2,
                                        float* __restrict__ out, int d, float eps, float scale, int has_next,
                                        QOut qo, int write_out) {
    __shared__ float scratch[1024 / WARP_SIZE];
    const float* sr = src + (long)blockIdx.x * d;
    const float* xir = xin + (long)blockIdx.x * d;
    float* xr = x + (long)blockIdx.x * d;
    float sv[RNA_MAX], xv[RNA_MAX];
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < RNA_MAX; c++) {
        const int i = threadIdx.x + c * blockDim.x;
        sv[c] = i < d ? sr[i] : 0.f;
        xv[c] = i < d ? xir[i] : 0.f;
        s += sv[c] * sv[c];
    }
    const float r1 = rsqrtf(block_sum(s, scratch) / d + eps);
    float s2 = 0.f;
#pragma unroll
    for (int c = 0; c < RNA_MAX; c++) {
        const int i = threadIdx.x + c * blockDim.x;
        if (i < d) {
            float v = (xv[c] + sv[c] * r1 * w[i]) * scale;
            xr[i] = v;
            xv[c] = v;
            s2 += v * v;
        }
    }
    if (!has_next) return;
    const float r2 = rsqrtf(block_sum(s2, scratch) / d + eps);
    float* o = out + (long)blockIdx.x * d;
    // Each warp holds one whole 32-element unit per step (blockDim and d are
    // multiples of 32), so the int8 copy is made straight from registers.
#pragma unroll
    for (int c = 0; c < RNA_MAX; c++) {
        const int i = threadIdx.x + c * blockDim.x;
        if (i < d) {
            float v = xv[c] * r2 * w2[i];
            if (write_out) o[i] = v;
            if (qo.on) quant_unit(v, ((long)blockIdx.x * d + i) / 32, qo);
        }
    }
}

// Per-head RMSNorm followed by NeoX RoPE: out = rope(rmsnorm(x) * w).
// One block per (token, head) row of head_dim; thread i handles pair (i, i + head_dim/2).
extern "C" __global__ void rms_norm_rope(const float* __restrict__ x, const float* __restrict__ w,
                                         float* __restrict__ out, const u32* __restrict__ positions, int n_heads,
                                         int head_dim, float eps, float theta,
                                         const float* __restrict__ freq_factors, int has_factors) {
    __shared__ float partial[1024 / WARP_SIZE];
    const int half = head_dim / 2;
    const float* xr = x + (long)blockIdx.x * head_dim;
    float* o = out + (long)blockIdx.x * head_dim;
    int i = threadIdx.x;
    float a = i < half ? xr[i] : 0.f, b = i < half ? xr[i + half] : 0.f;
    float s = warp_sum(a * a + b * b);
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) partial[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = lane < blockDim.x / WARP_SIZE ? partial[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) partial[0] = s;
    }
    __syncthreads();
    if (i >= half) return;
    float scale = rsqrtf(partial[0] / head_dim + eps);
    a *= scale * w[i];
    b *= scale * w[i + half];
    float inv_freq = powf(theta, -2.f * i / head_dim);
    if (has_factors) inv_freq /= freq_factors[i];
    float sn, cs;
    sincosf(positions[blockIdx.x / n_heads] * inv_freq, &sn, &cs);
    o[i] = a * cs - b * sn;
    o[i + half] = a * sn + b * cs;
}

// Fused Q/K/V prep for one layer. Blocks [0, n_tok*n_heads) handle query
// q8 KV cache: int8 values with one f16 scale per 32 consecutive values of a
// slot (the q8_0 layout; head_dim is a multiple of 32, so a group never spans
// two heads). Value i of the cache is c[i] * sc[i / 32].
//
// Keys are stored rotated: each head's vector times an orthonormal
// Walsh-Hadamard matrix H (over chunks of kv_rot_size(head_dim) dims), and
// queries get the same rotation before attention, so q.k is unchanged. The
// rotation spreads a few large channels over the whole group, which int8
// with a per-group scale otherwise represents badly (Qwen3-0.6B perplexity:
// +1.4% unrotated; llama.cpp rotates the same way). Values aren't rotated:
// quantizing them costs little (+0.02%).
#define KV_MAX_D 512

// Largest power of two dividing head_dim: the Hadamard size.
__device__ __forceinline__ int kv_rot_size(int head_dim) { return head_dim & -head_dim; }

// In-place orthonormal fast Walsh-Hadamard transform of row[0..n) in shared
// memory, independently over chunks of `size`. All threads of the block call it.
__device__ void fwht_smem(float* row, int n, int size) {
    for (int h = 1; h < size; h *= 2) {
        __syncthreads();
        for (int p = threadIdx.x; p < n / 2; p += blockDim.x) {
            const int lo = (p / h) * 2 * h + p % h, hi = lo + h;
            const float a = row[lo], b = row[hi];
            row[lo] = a + b;
            row[hi] = a - b;
        }
    }
    __syncthreads();
    const float sc = rsqrtf((float)size);
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] *= sc;
    __syncthreads();
}

// Queries for a q8 cache: y = H x per head (see above). One block per head row.
extern "C" __global__ void kv_rotate_q(const float* __restrict__ x, float* __restrict__ y, int head_dim) {
    __shared__ float row[KV_MAX_D];
    const long base = (long)blockIdx.x * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) row[i] = x[base + i];
    fwht_smem(row, head_dim, kv_rot_size(head_dim));
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) y[base + i] = row[i];
}

// Quantize one group of 32 held by a warp (lane = element) to cache index dst.
__device__ __forceinline__ void kv_q8_store(float x, i8* __restrict__ c, f16_bits* __restrict__ sc, long dst) {
    const float d = warp_max(fabsf(x)) / 127.f;
    c[dst] = (i8)(d > 0.f ? __float2int_rn(x / d) : 0);
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) sc[dst / 32] = f32_to_f16_bits(d);
}

// Quantize row[0..n) from shared memory into the cache at dst, one group per warp.
__device__ void kv_q8_store_row(const float* row, int n, i8* c, f16_bits* sc, long dst) {
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    for (int g = warp; g < n / 32; g += blockDim.x / WARP_SIZE) kv_q8_store(row[g * 32 + lane], c, sc, dst + g * 32 + lane);
}

__device__ __forceinline__ float kv_at(const void* c, const f16_bits* sc, long i, bool q8) {
    return q8 ? (float)((const i8*)c)[i] * f16_to_f32(sc[i / 32]) : f16_to_f32(((const f16_bits*)c)[i]);
}

// heads: qn = rope(rmsnorm(q) * qw). Blocks after that handle KV heads: K gets
// the same treatment with kw, V is optionally RMS-normalized with vw, and both
// go to the cache at the token's slot (f16, or q8 when `q8`: kc/vc are then
// int8 with scales ksc/vsc). Thread i handles pair (i, i+hd/2).
extern "C" __global__ void qkv_norm_rope_store(const float* __restrict__ q, const float* __restrict__ k,
                                               const float* __restrict__ v, const float* __restrict__ qw,
                                               const float* __restrict__ kw, const float* __restrict__ vw,
                                               int has_vnorm, float* __restrict__ qn, void* __restrict__ kc,
                                               void* __restrict__ vc, const u32* __restrict__ positions,
                                               const u32* __restrict__ slots, int n_tok, int n_heads,
                                               int n_kv_heads, int head_dim, float eps, float theta,
                                               const float* __restrict__ freq_factors, int has_factors,
                                               f16_bits* __restrict__ ksc, f16_bits* __restrict__ vsc, int q8) {
    __shared__ float row_buf[KV_MAX_D];
    __shared__ float scratch[1024 / WARP_SIZE];
    const int half = head_dim / 2, i = threadIdx.x;
    const bool is_q = blockIdx.x < n_tok * n_heads;
    const int row = is_q ? blockIdx.x : blockIdx.x - n_tok * n_heads;  // (token, head) within q or kv
    const int heads = is_q ? n_heads : n_kv_heads;
    const int t = row / heads;
    const float* xr = (is_q ? q : k) + (long)row * head_dim;
    const float* w = is_q ? qw : kw;

    float a = i < half ? xr[i] : 0.f, b = i < half ? xr[i + half] : 0.f;
    const float scale = rsqrtf(block_sum(a * a + b * b, scratch) / head_dim + eps);
    float inv_freq = powf(theta, -2.f * i / head_dim);
    if (has_factors && i < half) inv_freq /= freq_factors[i];
    float sn, cs;
    sincosf(positions[t] * inv_freq, &sn, &cs);
    float ra = 0.f, rb = 0.f;
    if (i < half) {
        a *= scale * w[i];
        b *= scale * w[i + half];
        ra = a * cs - b * sn;
        rb = a * sn + b * cs;
    }
    if (is_q) {
        if (i < half) {
            float* o = qn + (long)row * head_dim;
            o[i] = ra;
            o[i + half] = rb;
        }
        return;
    }
    const int kvh = row % n_kv_heads;
    const long dst = (long)slots[t] * n_kv_heads * head_dim + (long)kvh * head_dim;
    if (q8) {
        if (i < half) {
            row_buf[i] = ra;
            row_buf[i + half] = rb;
        }
        fwht_smem(row_buf, head_dim, kv_rot_size(head_dim));
        kv_q8_store_row(row_buf, head_dim, (i8*)kc, ksc, dst);
        __syncthreads();
    } else if (i < half) {
        ((f16_bits*)kc)[dst + i] = f32_to_f16_bits(ra);
        ((f16_bits*)kc)[dst + i + half] = f32_to_f16_bits(rb);
    }
    const float* vr = v + (long)row * head_dim;
    float va = i < half ? vr[i] : 0.f, vb = i < half ? vr[i + half] : 0.f;
    if (has_vnorm) {
        const float vs = rsqrtf(block_sum(va * va + vb * vb, scratch) / head_dim + eps);
        if (i < half) {
            va *= vs * vw[i];
            vb *= vs * vw[i + half];
        }
    }
    if (q8) {
        if (i < half) {
            row_buf[i] = va;
            row_buf[i + half] = vb;
        }
        __syncthreads();
        kv_q8_store_row(row_buf, head_dim, (i8*)vc, vsc, dst);
    } else if (i < half) {
        ((f16_bits*)vc)[dst + i] = f32_to_f16_bits(va);
        ((f16_bits*)vc)[dst + i + half] = f32_to_f16_bits(vb);
    }
}

// NeoX RoPE, in place. One thread per (token, head, i < head_dim/2).
// `freq_factors` (head_dim/2 floats) divides each pair's frequency; used only
// when has_factors is set.
extern "C" __global__ void rope_neox(float* x, const u32* __restrict__ positions, int n_tok, int n_heads,
                                     int head_dim, float theta, const float* __restrict__ freq_factors,
                                     int has_factors) {
    int half = head_dim / 2;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n_tok * n_heads * half) return;
    int i = idx % half;
    long th = idx / half;  // token * n_heads + head
    int t = th / n_heads;
    float inv_freq = powf(theta, -2.f * i / head_dim);
    if (has_factors) inv_freq /= freq_factors[i];
    float s, c;
    sincosf(positions[t] * inv_freq, &s, &c);
    float* h = x + th * head_dim;
    float a = h[i], b = h[i + half];
    h[i] = a * c - b * s;
    h[i + half] = a * s + b * c;
}

// With qo.on, n is a multiple of 32, so whole warps are in or out of range.
// write_out = 0 skips the f32 result when only the int8 copy is consumed.
extern "C" __global__ void silu_mul(float* gate, const float* __restrict__ up, long n, QOut qo, int write_out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float v = g / (1.f + __expf(-g)) * up[i];
    if (write_out) gate[i] = v;
    if (qo.on) quant_unit(v, i / 32, qo);
}

// GELU, tanh approximation (gelu_pytorch_tanh).
__device__ __forceinline__ float gelu_tanh(float x) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    return 0.5f * x * (1.f + tanhf(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
}

// Same with the hardware tanh (sm_75+, about 2^-11 relative error): for
// results that are rounded to int8 right after.
__device__ __forceinline__ float gelu_tanh_fast(float x) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    float y = SQRT_2_OVER_PI * (x + 0.044715f * x * x * x), th;
#ifdef RLLM_HIP
    th = tanhf(y);
#else
    asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(y));
#endif
    return 0.5f * x * (1.f + th);
}

// Decode gated FFN front half: out = act(x . gate_r) * (x . up_r) for 32
// consecutive rows r per block (one warp each, both rows), then each of the
// first m warps emits one 32-row unit of int8 (and f32 if write_out) for the
// down projection. Replaces the gate/up matvec + activation launches.
// Activations are the accurate forms used by silu_mul / gelu_mul.
template <int F, int ACT>
__device__ void matvec_glu(const MvTarget& g, const MvTarget& u, const i8* __restrict__ xq,
                           const float* __restrict__ xd, int k, int m, QOut qo, float* __restrict__ out,
                           int write_out) {
    __shared__ float v[MV_MAX_M][WARP_SIZE];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const long r = (long)blockIdx.x * WARP_SIZE + warp;  // n % 32 == 0: always in range
    float ag[MV_MAX_M], au[MV_MAX_M];
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++) ag[j] = au[j] = 0.f;
    mv_dot<F>(g.qs + r * g.qs_row, g.qh + r * g.qh_row, g.sc + r * g.sc_row, xq, xd, k, m, lane, ag);
    mv_dot<F>(u.qs + r * u.qs_row, u.qh + r * u.qh_row, u.sc + r * u.sc_row, xq, xd, k, m, lane, au);
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++) {
        if (j < m) {
            const float gs = warp_sum(ag[j]), us = warp_sum(au[j]);
            const float a = ACT == 0 ? gs / (1.f + __expf(-gs)) : gelu_tanh(gs);
            if (lane == 0) v[j][warp] = a * us;
        }
    }
    __syncthreads();
    if (warp < m) {
        const float val = v[warp][lane];
        const long oi = (long)warp * g.n + (long)blockIdx.x * WARP_SIZE + lane;
        if (write_out) out[oi] = val;
        if (qo.on) quant_unit(val, oi / 32, qo);
    }
}

#define INSTANTIATE_MATVEC_GLU(NAME, F)                                                                         \
    extern "C" __global__ void __launch_bounds__(1024)                                                          \
        matvec_glu_silu_##NAME(MvTarget g, MvTarget u, const i8* xq, const float* xd, int k, int m, QOut qo,    \
                               float* out, int write_out) {                                                    \
        matvec_glu<F, 0>(g, u, xq, xd, k, m, qo, out, write_out);                                              \
    }                                                                                                           \
    extern "C" __global__ void __launch_bounds__(1024)                                                          \
        matvec_glu_gelu_##NAME(MvTarget g, MvTarget u, const i8* xq, const float* xd, int k, int m, QOut qo,    \
                               float* out, int write_out) {                                                    \
        matvec_glu<F, 1>(g, u, xq, xd, k, m, qo, out, write_out);                                              \
    }
// Float-weight form (F16/BF16/F32, decode): one warp per row computes its gate
// and up dot products and writes act(gate) * up; no quantization involved.
template <int F, int ACT>
__device__ void matvec_glu_f(const MvTarget& g, const MvTarget& u, const float* __restrict__ x, int k, int m,
                             float* __restrict__ out) {
    const int warps = blockDim.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int r = blockIdx.x * warps + threadIdx.x / WARP_SIZE;
    if (r >= g.n) return;
    const u8* gr = g.qs + (long)r * g.qs_row;
    const u8* ur = u.qs + (long)r * u.qs_row;
    float ag[MV_MAX_M], au[MV_MAX_M], yg[8], yu[8];
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++) ag[j] = au[j] = 0.f;
    for (int c = lane; c < k / 8; c += WARP_SIZE) {
        load8<F>(gr, c, yg);
        load8<F>(ur, c, yu);
#pragma unroll
        for (int j = 0; j < MV_MAX_M; j++) {
            if (j < m) {
                const float4* xp = (const float4*)(x + (long)j * k + c * 8);
                const float4 a = xp[0], b = xp[1];
                ag[j] += yg[0] * a.x + yg[1] * a.y + yg[2] * a.z + yg[3] * a.w + yg[4] * b.x + yg[5] * b.y +
                         yg[6] * b.z + yg[7] * b.w;
                au[j] += yu[0] * a.x + yu[1] * a.y + yu[2] * a.z + yu[3] * a.w + yu[4] * b.x + yu[5] * b.y +
                         yu[6] * b.z + yu[7] * b.w;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++) {
        if (j < m) {
            const float gs = warp_sum(ag[j]), us = warp_sum(au[j]);
            const float a = ACT == 0 ? gs / (1.f + __expf(-gs)) : gelu_tanh(gs);
            if (lane == 0) out[(long)j * g.n + r] = a * us;
        }
    }
}

#define INSTANTIATE_MATVEC_GLU_F(NAME, F)                                                                       \
    extern "C" __global__ void matvec_glu_silu_##NAME(MvTarget g, MvTarget u, const float* x, int k, int m,     \
                                                      float* out) {                                             \
        matvec_glu_f<F, 0>(g, u, x, k, m, out);                                                                 \
    }                                                                                                           \
    extern "C" __global__ void matvec_glu_gelu_##NAME(MvTarget g, MvTarget u, const float* x, int k, int m,     \
                                                      float* out) {                                             \
        matvec_glu_f<F, 1>(g, u, x, k, m, out);                                                                 \
    }
INSTANTIATE_MATVEC_GLU_F(f32, FMT_F32)
INSTANTIATE_MATVEC_GLU_F(f16, FMT_F16)
INSTANTIATE_MATVEC_GLU_F(bf16, FMT_BF16)

// Decode (one row): Backend::norm_add_gated_ffn's front half in one launch.
// Every block recomputes x_out = (x_in + rmsnorm(src) * w1) * scale and the
// int8 form of rmsnorm(x_out) * w2 into shared memory (cheap: d values; block
// 0 also stores x_out), then runs matvec_glu's 32 gate/up rows against it.
// Saves the one-row norm launch, which costs more in latency than work.
#define GLU_NORM_MAX_D 4096
template <int F, int ACT>
__device__ void matvec_glu_norm(const MvTarget& g, const MvTarget& u, const float* __restrict__ src,
                                const float* __restrict__ w1, const float* __restrict__ x_in,
                                float* __restrict__ x_out, const float* __restrict__ w2, int d, float eps,
                                float scale, QOut qo) {
    constexpr int PER = GLU_NORM_MAX_D / 1024;
    __shared__ __align__(16) i8 xq[GLU_NORM_MAX_D];
    __shared__ float xd[GLU_NORM_MAX_D / 32];
    __shared__ float red[32];
    __shared__ float v[WARP_SIZE];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    float sv[PER], xv[PER];
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < PER; c++) {
        const int i = threadIdx.x + c * 1024;
        sv[c] = i < d ? src[i] : 0.f;
        xv[c] = i < d ? x_in[i] : 0.f;
        s += sv[c] * sv[c];
    }
    const float r1 = rsqrtf(block_sum(s, red) / d + eps);
    float s2 = 0.f;
#pragma unroll
    for (int c = 0; c < PER; c++) {
        const int i = threadIdx.x + c * 1024;
        if (i < d) {
            xv[c] = (xv[c] + sv[c] * r1 * w1[i]) * scale;
            if (blockIdx.x == 0) x_out[i] = xv[c];
            s2 += xv[c] * xv[c];
        }
    }
    const float r2 = rsqrtf(block_sum(s2, red) / d + eps);
    // A warp holds one whole 32-element unit per step (d % 32 == 0): same
    // quantization as quant_unit.
#pragma unroll
    for (int c = 0; c < PER; c++) {
        const int i = threadIdx.x + c * 1024;
        if (c * 1024 + warp * WARP_SIZE < d) {
            const float y = xv[c] * r2 * w2[i];
            const float dd = warp_max(fabsf(y)) / 127.f;
            xq[i] = (i8)(dd > 0.f ? rintf(y / dd) : 0.f);
            if (lane == 0) xd[i / 32] = dd;
        }
    }
    __syncthreads();
    const long r = (long)blockIdx.x * WARP_SIZE + warp;  // n % 32 == 0
    float ag[MV_MAX_M], au[MV_MAX_M];
#pragma unroll
    for (int j = 0; j < MV_MAX_M; j++) ag[j] = au[j] = 0.f;
    mv_dot<F>(g.qs + r * g.qs_row, g.qh + r * g.qh_row, g.sc + r * g.sc_row, xq, xd, d, 1, lane, ag);
    mv_dot<F>(u.qs + r * u.qs_row, u.qh + r * u.qh_row, u.sc + r * u.sc_row, xq, xd, d, 1, lane, au);
    const float gs = warp_sum(ag[0]), us = warp_sum(au[0]);
    const float a = ACT == 0 ? gs / (1.f + __expf(-gs)) : gelu_tanh(gs);
    if (lane == 0) v[warp] = a * us;
    __syncthreads();
    if (warp == 0 && qo.on) quant_unit(v[lane], ((long)blockIdx.x * WARP_SIZE + lane) / 32, qo);
}

#define INSTANTIATE_MATVEC_GLU_NORM(NAME, F)                                                                    \
    extern "C" __global__ void __launch_bounds__(1024) matvec_glu_norm_silu_##NAME(                             \
        MvTarget g, MvTarget u, const float* src, const float* w1, const float* x_in, float* x_out,            \
        const float* w2, int d, float eps, float scale, QOut qo) {                                              \
        matvec_glu_norm<F, 0>(g, u, src, w1, x_in, x_out, w2, d, eps, scale, qo);                               \
    }                                                                                                           \
    extern "C" __global__ void __launch_bounds__(1024) matvec_glu_norm_gelu_##NAME(                             \
        MvTarget g, MvTarget u, const float* src, const float* w1, const float* x_in, float* x_out,            \
        const float* w2, int d, float eps, float scale, QOut qo) {                                              \
        matvec_glu_norm<F, 1>(g, u, src, w1, x_in, x_out, w2, d, eps, scale, qo);                               \
    }
INSTANTIATE_MATVEC_GLU_NORM(q8, FMT_Q8)
INSTANTIATE_MATVEC_GLU_NORM(q4, FMT_Q4)
INSTANTIATE_MATVEC_GLU_NORM(q6, FMT_Q6)

INSTANTIATE_MATVEC_GLU(q8, FMT_Q8)
INSTANTIATE_MATVEC_GLU(q4, FMT_Q4)
INSTANTIATE_MATVEC_GLU(q6, FMT_Q6)

extern "C" __global__ void gelu_mul(float* gate, const float* __restrict__ up, long n, QOut qo, int write_out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = gelu_tanh(gate[i]) * up[i];
    if (write_out) gate[i] = v;
    if (qo.on) quant_unit(v, i / 32, qo);
}

// gate[i, :] = gelu(gate[i, :]) * x[first + i * stride, :], rows of width.
extern "C" __global__ void gelu_mul_strided(float* gate, const float* __restrict__ x, int first, int stride,
                                            int count, int width) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)count * width) return;
    long row = i / width, col = i % width;
    gate[i] = gelu_tanh(gate[i]) * x[((long)first + row * stride) * width + col];
}

// Decode (one row) form of Backend::ple_gate: every block recomputes
// x_out = x_in + rmsnorm(src) * w into shared memory (cheap: d values), block
// 0 also stores it, then 4 warps split k for one gate row and the epilogue
// applies gelu(.) * inputs[layer, row]. Saves the separate one-row norm and
// gather/activation launches, which cost more in launch latency than work.
#define PLE_MAX_D 4096
template <int F>
__device__ void ple_gate_mv(const float* __restrict__ src, const float* __restrict__ w,
                            const float* __restrict__ x_in, float* __restrict__ x_out, const u8* __restrict__ gw,
                            long row_bytes, const float* __restrict__ inputs, float* __restrict__ gate, int d,
                            float eps, int layer, int p_dim) {
    __shared__ __align__(16) float xs[PLE_MAX_D];
    __shared__ float scratch[32];
    // float4 and unrolled: the loads must all be in flight at once, since
    // this prologue sits in front of every block's matvec.
    const float4* s4 = (const float4*)src;
    float4 sv[PLE_MAX_D / 4 / 128];
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < PLE_MAX_D / 4 / 128; c++) {
        const int i = threadIdx.x + c * 128;
        sv[c] = i < d / 4 ? s4[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        s += sv[c].x * sv[c].x + sv[c].y * sv[c].y + sv[c].z * sv[c].z + sv[c].w * sv[c].w;
    }
    const float r1 = rsqrtf(block_sum(s, scratch) / d + eps);
#pragma unroll
    for (int c = 0; c < PLE_MAX_D / 4 / 128; c++) {
        const int i = threadIdx.x + c * 128;
        if (i < d / 4) {
            const float4 xv = ((const float4*)x_in)[i], wv = ((const float4*)w)[i];
            const float4 v = make_float4(xv.x + sv[c].x * r1 * wv.x, xv.y + sv[c].y * r1 * wv.y,
                                         xv.z + sv[c].z * r1 * wv.z, xv.w + sv[c].w * r1 * wv.w);
            ((float4*)xs)[i] = v;
            if (blockIdx.x == 0) ((float4*)x_out)[i] = v;
        }
    }
    __syncthreads();
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE, warps = blockDim.x / WARP_SIZE;
    const u8* row = gw + (long)blockIdx.x * row_bytes;
    float acc = 0.f, y[8];
#pragma unroll 4
    for (int c = lane + warp * WARP_SIZE; c < d / 8; c += WARP_SIZE * warps) {
        load8<F>(row, c, y);
        const float4* xp = (const float4*)(xs + c * 8);
        float4 a = xp[0], b = xp[1];
        acc += y[0] * a.x + y[1] * a.y + y[2] * a.z + y[3] * a.w + y[4] * b.x + y[5] * b.y + y[6] * b.z + y[7] * b.w;
    }
    acc = block_sum(acc, scratch);
    if (threadIdx.x == 0) gate[blockIdx.x] = gelu_tanh(acc) * inputs[(long)layer * p_dim + blockIdx.x];
}
extern "C" __global__ void ple_gate_f32(const float* src, const float* w, const float* x_in, float* x_out,
                                        const u8* gw, long row_bytes, const float* inputs, float* gate, int d,
                                        float eps, int layer, int p_dim) {
    ple_gate_mv<FMT_F32>(src, w, x_in, x_out, gw, row_bytes, inputs, gate, d, eps, layer, p_dim);
}

extern "C" __global__ void gelu(float* x, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = gelu_tanh(x[i]);
}

extern "C" __global__ void mul_inplace(float* x, const float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= y[i];
}

extern "C" __global__ void scale_inplace(float* x, float s, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

extern "C" __global__ void softcap_inplace(float* x, float cap, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

extern "C" __global__ void add_inplace(float* x, const float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

extern "C" __global__ void gather_rows(const float* __restrict__ x, const u32* __restrict__ rows,
                                       float* __restrict__ out, int n_rows, int width) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n_rows * width) return;
    int r = i / width, c = i % width;
    out[i] = x[(long)rows[r] * width + c];
}

extern "C" __global__ void gather_strided(const float* __restrict__ x, float* __restrict__ out, int first,
                                          int stride, int count, int width) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)count * width) return;
    int r = i / width, c = i % width;
    out[i] = x[(long)(first + r * stride) * width + c];
}

// ---------------------------------------------------------------------------
// Paged KV cache. K and V are stored as f16 (or q8, see kv_q8_store):
// [slot][kv_heads * head_dim].

extern "C" __global__ void kv_store(const float* __restrict__ k, const float* __restrict__ v,
                                    const u32* __restrict__ slots, f16_bits* kc, f16_bits* vc, int n_tok,
                                    int width) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n_tok * width) return;
    int t = i / width, c = i % width;
    long dst = (long)slots[t] * width + c;
    kc[dst] = f32_to_f16_bits(k[i]);
    vc[dst] = f32_to_f16_bits(v[i]);
}

// Copy whole KV blocks: block i is `bytes` bytes from src + src_idx[i] *
// src_stride to dst + dst_idx[i] * dst_stride. One side is a cache layer's
// pool, the other mapped pinned host memory (the prefix cache's host tier),
// so this one kernel moves blocks over PCIe in either direction.
extern "C" __global__ void kv_copy_blocks(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
                                          const u32* __restrict__ src_idx, const u32* __restrict__ dst_idx,
                                          long long src_stride, long long dst_stride, int bytes) {
    const uint4* s = (const uint4*)(src + (long long)src_idx[blockIdx.x] * src_stride);
    uint4* d = (uint4*)(dst + (long long)dst_idx[blockIdx.x] * dst_stride);
    for (int i = threadIdx.x; i < bytes / 16; i += blockDim.x) d[i] = s[i];
}

// q8 version (keys rotated, see kv_rotate_q). One block per (token, KV head).
extern "C" __global__ void kv_store_q8(const float* __restrict__ k, const float* __restrict__ v,
                                       const u32* __restrict__ slots, i8* kc, f16_bits* ksc, i8* vc, f16_bits* vsc,
                                       int n_kv_heads, int head_dim) {
    __shared__ float row[KV_MAX_D];
    const int t = blockIdx.x / n_kv_heads, h = blockIdx.x % n_kv_heads;
    const long src = (long)blockIdx.x * head_dim;
    const long dst = ((long)slots[t] * n_kv_heads + h) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) row[i] = k[src + i];
    fwht_smem(row, head_dim, kv_rot_size(head_dim));
    kv_q8_store_row(row, head_dim, kc, ksc, dst);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) row[i] = v[src + i];
    __syncthreads();
    kv_q8_store_row(row, head_dim, vc, vsc, dst);
}

// Prefill on a q8 cache: expand the blocks a step reads into f16 staging
// buffers (block j of `blocks` becomes staging block j), so the f16 prefill
// kernels run unchanged on a remapped block table. 8 values per thread.
extern "C" __global__ void kv_dequant_blocks(const i8* __restrict__ kc, const f16_bits* __restrict__ ksc,
                                             const i8* __restrict__ vc, const f16_bits* __restrict__ vsc,
                                             f16_bits* __restrict__ ko, f16_bits* __restrict__ vo,
                                             const u32* __restrict__ blocks, int n_list, long per_block) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n_list * per_block) return;
    long src = (long)blocks[i / per_block] * per_block + i % per_block;
    const float dk = f16_to_f32(ksc[src / 32]), dv = f16_to_f32(vsc[src / 32]);
    uint2 kq = *(const uint2*)(kc + src), vq = *(const uint2*)(vc + src);
    const i8 *kb = (const i8*)&kq, *vb = (const i8*)&vq;
    f16_bits kh[8], vh[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
        kh[e] = f32_to_f16_bits(kb[e] * dk);
        vh[e] = f32_to_f16_bits(vb[e] * dv);
    }
    *(uint4*)(ko + i) = *(const uint4*)kh;
    *(uint4*)(vo + i) = *(const uint4*)vh;
}

// Paged causal attention, one block per (token, head), ATTN_WARPS warps.
// Each warp walks a strided subset of context positions with an online
// softmax; the warps' partial results are merged at the end.
// head_dim must be a multiple of WARP_SIZE and at most ATTN_MAX_D.
// window > 0 limits attention to the last `window` positions (sliding window).
#define ATTN_WARPS 4
#define ATTN_MAX_D 512

extern "C" __global__ void paged_attention(const float* __restrict__ q, const void* __restrict__ kc,
                                           const void* __restrict__ vc, float* __restrict__ out,
                                           const u32* __restrict__ token_seq, const u32* __restrict__ positions,
                                           const u32* __restrict__ block_tables, int max_blocks, int n_heads,
                                           int n_kv_heads, int head_dim, int block_size, float scale,
                                           int window, const f16_bits* __restrict__ ksc,
                                           const f16_bits* __restrict__ vsc, int q8) {
    const int PER = ATTN_MAX_D / WARP_SIZE;
    __shared__ float s_m[ATTN_WARPS], s_l[ATTN_WARPS];
    __shared__ float s_acc[ATTN_WARPS][ATTN_MAX_D];

    int t = blockIdx.x, h = blockIdx.y;
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    int per = head_dim / WARP_SIZE;
    int ctx = positions[t] + 1;
    int start = (window > 0 && ctx > window) ? ctx - window : 0;
    const u32* table = block_tables + (long)token_seq[t] * max_blocks;
    int kv_w = n_kv_heads * head_dim;
    int kv_off = (h / (n_heads / n_kv_heads)) * head_dim;

    float qr[PER], acc[PER];
    const float* qp = q + ((long)t * n_heads + h) * head_dim;
    for (int i = 0; i < PER; i++) {
        qr[i] = i < per ? qp[lane + i * WARP_SIZE] * scale : 0.f;
        acc[i] = 0.f;
    }
    float m = NEG_INF, l = 0.f;
    for (int p = start + warp; p < ctx; p += ATTN_WARPS) {
        long base = ((long)table[p / block_size] * block_size + p % block_size) * kv_w + kv_off;
        float s = 0.f;
        for (int i = 0; i < per; i++) s += qr[i] * kv_at(kc, ksc, base + lane + i * WARP_SIZE, q8);
        s = warp_sum(s);
        float m_new = fmaxf(m, s);
        float corr = __expf(m - m_new);
        float e = __expf(s - m_new);
        l = l * corr + e;
        for (int i = 0; i < per; i++) acc[i] = acc[i] * corr + e * kv_at(vc, vsc, base + lane + i * WARP_SIZE, q8);
        m = m_new;
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    for (int i = 0; i < per; i++) s_acc[warp][lane + i * WARP_SIZE] = acc[i];
    __syncthreads();

    float gm = NEG_INF;
    for (int w = 0; w < ATTN_WARPS; w++) gm = fmaxf(gm, s_m[w]);
    float gl = 0.f;
    for (int w = 0; w < ATTN_WARPS; w++) gl += s_l[w] * __expf(s_m[w] - gm);
    float* o = out + ((long)t * n_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v = 0.f;
        for (int w = 0; w < ATTN_WARPS; w++) v += s_acc[w][d] * __expf(s_m[w] - gm);
        o[d] = v / gl;
    }
}

// Tiled paged attention for prefill (many query tokens per sequence).
//
// Grid: (tiles, kv heads). A tile is up to `tile_tokens` consecutive tokens of
// one sequence; the block handles those tokens for every query head that
// shares this KV head (GQA), one warp per (token, head) row. K/V positions
// are streamed through shared memory in chunks of BK and reused by all rows.
// Scores: lane p computes the dot product for key position p of the chunk.
// Values: lane owns head dims lane, lane + 32, ...
//
// Dynamic shared memory: q[rows][hd] | k[BK][hd + 1] | v[BK][hd] | p[rows][BK]
extern "C" __global__ void attn_prefill(const float* __restrict__ q, const f16_bits* __restrict__ kc,
                                        const f16_bits* __restrict__ vc, float* __restrict__ out,
                                        const u32* __restrict__ tiles, const u32* __restrict__ token_seq,
                                        const u32* __restrict__ positions, const u32* __restrict__ block_tables,
                                        int max_blocks, int n_heads, int n_kv_heads, int head_dim, int block_size,
                                        float scale, int window, int tile_tokens, int bk) {
    extern __shared__ float smem[];
    const int group = n_heads / n_kv_heads;
    const int rows = tile_tokens * group;
    const int hd = head_dim, hdp = head_dim + 1;
    float* sq = smem;
    float* sk = sq + rows * hd;
    float* sv = sk + bk * hdp;
    float* sp = sv + bk * hd;

    const int t0 = tiles[2 * blockIdx.x], nt = tiles[2 * blockIdx.x + 1];
    const int kvh = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int kv_w = n_kv_heads * hd;
    const u32* table = block_tables + (long)token_seq[t0] * max_blocks;

    // Load the tile's queries (pre-scaled).
    for (int i = threadIdx.x; i < rows * hd; i += blockDim.x) {
        int r = i / hd, dd = i % hd;
        int t = r / group, h = kvh * group + r % group;
        sq[i] = t < nt ? q[((long)(t0 + t) * n_heads + h) * hd + dd] * scale : 0.f;
    }

    const int row_t = warp / group;  // this warp's token within the tile
    const int head = kvh * group + warp % group;
    const bool active = warp < rows && row_t < nt;
    const int qpos = active ? (int)positions[t0 + row_t] : -1;
    const int first = positions[t0], last = positions[t0 + nt - 1];
    const int kbeg = window > 0 ? max(0, first + 1 - window) : 0;
    const int kend = last + 1;

    const int per = hd / WARP_SIZE;
    float acc[ATTN_MAX_D / WARP_SIZE];
    for (int i = 0; i < ATTN_MAX_D / WARP_SIZE; i++) acc[i] = 0.f;
    float m = NEG_INF, l = 0.f;
    __syncthreads();

    for (int k0 = kbeg; k0 < kend; k0 += bk) {
        for (int i = threadIdx.x; i < bk * hd; i += blockDim.x) {
            int p = i / hd, dd = i % hd, kp = k0 + p;
            float kv = 0.f, vv = 0.f;
            if (kp < kend) {
                long base = ((long)table[kp / block_size] * block_size + kp % block_size) * kv_w + kvh * hd + dd;
                kv = f16_to_f32(kc[base]);
                vv = f16_to_f32(vc[base]);
            }
            sk[p * hdp + dd] = kv;
            sv[p * hd + dd] = vv;
        }
        __syncthreads();
        if (active) {
            const float* qr = sq + warp * hd;
            int kp = k0 + lane;
            float s = NEG_INF;
            if (lane < bk && kp < kend && kp <= qpos && (window <= 0 || qpos - kp < window)) {
                const float* kr = sk + lane * hdp;
                float dot = 0.f;
                for (int dd = 0; dd < hd; dd++) dot += qr[dd] * kr[dd];
                s = dot;
            }
            float m_new = fmaxf(m, warp_max(s));
            if (m_new != NEG_INF) {
                float pr = s == NEG_INF ? 0.f : __expf(s - m_new);
                float corr = __expf(m - m_new);
                l = l * corr + warp_sum(pr);
                float* pw = sp + warp * bk;
                if (lane < bk) pw[lane] = pr;
                __syncwarp();
                for (int i = 0; i < per; i++) {
                    float a = acc[i] * corr;
                    for (int p = 0; p < bk; p++) a += pw[p] * sv[p * hd + lane + i * WARP_SIZE];
                    acc[i] = a;
                }
                m = m_new;
            }
        }
        __syncthreads();
    }
    if (active) {
        float* o = out + ((long)(t0 + row_t) * n_heads + head) * hd;
        for (int i = 0; i < per; i++) o[lane + i * WARP_SIZE] = acc[i] / l;
    }
}

#ifdef RLLM_HIP
// ---------------------------------------------------------------------------
// Prefill attention on RDNA3 matrix cores (WMMA 16x16x16, f16 in, f32
// accumulate), FlashAttention-2 style. Same grid and tiles as attn_prefill:
// (tiles, kv heads); a block's rows are (token, head) pairs of one tile, 16
// per wave (WA_WAVES waves). Per BK-key chunk: K is staged row-major and V
// transposed in shared memory; each wave computes S = Q.K^T (16 x BK), runs
// the online softmax on it, then O += P.V. WMMA fragments (wave32, gfx11):
//   A (16x16): lane l holds row l%16, k = 0..15 (lanes 16-31 repeat 0-15)
//   B (16x16): lane l holds column l%16, k = 0..15
//   C/D:       lane l holds column l%16, rows 2i + l/16 (i = 0..7)
// P goes from the D layout to the A layout through shared memory.
#define WA_WAVES 4
#define WA_ROWS (WA_WAVES * 16)

typedef _Float16 wa_half16 __attribute__((ext_vector_type(16)));
typedef float wa_float8 __attribute__((ext_vector_type(8)));

template <int HD>
struct WaCfg {
    static constexpr int BK = HD <= 128 ? 32 : 16;  // keys per chunk
    static constexpr int QS = HD + 8;               // padded row strides (halves)
    static constexpr int KS = HD + 8;
    static constexpr int VS = BK + 8;
    static constexpr int PS = 16 + 8;
    static constexpr int SMEM = (WA_ROWS * QS + BK * KS + HD * VS + WA_WAVES * 16 * PS) * 2;
};

#if defined(__GFX11__)
template <int HD>
__device__ void attn_wmma(const float* __restrict__ q, const f16_bits* __restrict__ kc,
                          const f16_bits* __restrict__ vc, float* __restrict__ out, const u32* __restrict__ tiles,
                          const u32* __restrict__ token_seq, const u32* __restrict__ positions,
                          const u32* __restrict__ block_tables, int max_blocks, int n_heads, int n_kv_heads,
                          int block_size, float scale, int window, int tile_tokens) {
    typedef WaCfg<HD> C;
    constexpr int BK = C::BK, QS = C::QS, KS = C::KS, VS = C::VS, PS = C::PS, NT = HD / 16;
    extern __shared__ __align__(16) char wa_smem[];
    _Float16* Qs = (_Float16*)wa_smem;
    _Float16* Ks = Qs + WA_ROWS * QS;
    _Float16* Vt = Ks + BK * KS;
    _Float16* Ps = Vt + HD * VS;

    const int group = n_heads / n_kv_heads;
    const int rows = tile_tokens * group;
    const int t0 = tiles[2 * blockIdx.x], nt = tiles[2 * blockIdx.x + 1];
    const int kvh = blockIdx.y;
    const int wave = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int col = lane % 16, half = lane / 16;
    const int kv_w = n_kv_heads * HD;
    const u32* table = block_tables + (long)token_seq[t0] * max_blocks;

    // Queries, pre-scaled, as f16 (rows past the tile are zero).
    for (int i = threadIdx.x; i < WA_ROWS * HD / 4; i += blockDim.x) {
        int r = i / (HD / 4), d = (i % (HD / 4)) * 4;
        int t = r / group, h = kvh * group + r % group;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < rows && t < nt) v = *(const float4*)(q + ((long)(t0 + t) * n_heads + h) * HD + d);
        _Float16* dst = Qs + r * QS + d;
        dst[0] = (_Float16)(v.x * scale);
        dst[1] = (_Float16)(v.y * scale);
        dst[2] = (_Float16)(v.z * scale);
        dst[3] = (_Float16)(v.w * scale);
    }

    // This lane's 8 rows (2i + half) of the wave's 16: query positions.
    int qpos[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = wave * 16 + 2 * i + half, t = r / group;
        qpos[i] = (r < rows && t < nt) ? (int)positions[t0 + t] : -1;
    }
    const int first = positions[t0], last = positions[t0 + nt - 1];
    const int kbeg = window > 0 ? max(0, first + 1 - window) : 0;
    const int kend = last + 1;

    wa_float8 o[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) o[t] = (wa_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m[8], l[8];
#pragma unroll
    for (int i = 0; i < 8; i++) m[i] = NEG_INF, l[i] = 0.f;
    _Float16* Pw = Ps + wave * 16 * PS;
    const _Float16* Qw = Qs + (wave * 16 + col) * QS;

    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        __syncthreads();  // previous chunk's K/V reads are done
        // K rows (8 halves per thread) as stored; V transposed.
        for (int i = threadIdx.x; i < BK * HD / 8; i += blockDim.x) {
            int p = i / (HD / 8), d = (i % (HD / 8)) * 8, kp = k0 + p;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (kp < kend) {
                long base = ((long)table[kp / block_size] * block_size + kp % block_size) * kv_w + kvh * HD + d;
                kv = *(const uint4*)(kc + base);
                vv = *(const uint4*)(vc + base);
            }
            *(uint4*)(Ks + p * KS + d) = kv;
            const f16_bits* vh = (const f16_bits*)&vv;
#pragma unroll
            for (int j = 0; j < 8; j++) ((f16_bits*)Vt)[(d + j) * VS + p] = vh[j];
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < BK / 16; j++) {
            // S = Q . K^T for keys k0 + 16j ..
            wa_float8 s = (wa_float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const _Float16* kr = Ks + (j * 16 + col) * KS;
#pragma unroll
            for (int t = 0; t < NT; t++) {
                wa_half16 a = *(const wa_half16*)(Qw + t * 16);
                wa_half16 b = *(const wa_half16*)(kr + t * 16);
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s);
            }
            // Mask, then the online softmax per row (16 lanes share a row set).
            const int kp = k0 + j * 16 + col;
            float corr[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                bool ok = kp < kend && kp <= qpos[i] && (window <= 0 || qpos[i] - kp < window);
                float v = ok ? s[i] : NEG_INF;
                float mx = v;
                for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 16));
                float m_new = fmaxf(m[i], mx);
                float p = (m_new == NEG_INF || v == NEG_INF) ? 0.f : __expf(v - m_new);
                corr[i] = m_new == NEG_INF ? 1.f : __expf(m[i] - m_new);
                float sum = p;
                for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 16);
                l[i] = l[i] * corr[i] + sum;
                m[i] = m_new;
                Pw[(2 * i + half) * PS + col] = (_Float16)p;
            }
#pragma unroll
            for (int t = 0; t < NT; t++)
#pragma unroll
                for (int i = 0; i < 8; i++) o[t][i] *= corr[i];
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            // O += P . V: A = P rows, B = V^T rows (one head dim, 16 keys).
            wa_half16 a = *(const wa_half16*)(Pw + col * PS);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                wa_half16 b = *(const wa_half16*)(Vt + (t * 16 + col) * VS + j * 16);
                o[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, o[t]);
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        }
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = wave * 16 + 2 * i + half, t = r / group;
        if (r >= rows || t >= nt) continue;
        float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
        float* op = out + ((long)(t0 + t) * n_heads + kvh * group + r % group) * HD + col;
#pragma unroll
        for (int tt = 0; tt < NT; tt++) op[tt * 16] = o[tt][i] * inv;
    }
}

#define INSTANTIATE_WA(HD)                                                                                   \
    extern "C" __global__ void __launch_bounds__(WA_WAVES * 32) attn_wmma_d##HD(                             \
        const float* q, const f16_bits* kc, const f16_bits* vc, float* out, const u32* tiles,                 \
        const u32* token_seq, const u32* positions, const u32* block_tables, int max_blocks, int n_heads,     \
        int n_kv_heads, int block_size, float scale, int window, int tile_tokens) {                           \
        attn_wmma<HD>(q, kc, vc, out, tiles, token_seq, positions, block_tables, max_blocks, n_heads,         \
                      n_kv_heads, block_size, scale, window, tile_tokens);                                    \
    }
#else
#define INSTANTIATE_WA(HD) \
    extern "C" __global__ void attn_wmma_d##HD() { __builtin_trap(); }
#endif  // __GFX11__
INSTANTIATE_WA(64)
INSTANTIATE_WA(128)
INSTANTIATE_WA(256)
#endif  // RLLM_HIP

#ifndef RLLM_HIP
// ---------------------------------------------------------------------------
// Vendor shim: tensor-core MMA. PTX on NVIDIA (sm_80+); the HIP port maps
// these two functions to MFMA/WMMA builtins. Nothing else uses inline PTX.
//
// mma_16816: c[16x8] += a[16x16] * b[16x8], f16 inputs, f32 accumulate.
// Fragment layout (g = lane / 4, t = lane % 4):
//   a: {r g, c 2t..2t+1}, {r g+8, c 2t..}, {r g, c 2t+8..}, {r g+8, c 2t+8..}
//   b: {k 2t..2t+1, n g}, {k 2t+8..2t+9, n g}
//   c: {r g, c 2t}, {r g, c 2t+1}, {r g+8, c 2t}, {r g+8, c 2t+1}
__device__ __forceinline__ void mma_16816(float* c, const u32* a, const u32* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two floats -> packed f16x2, `lo` in the low half.
__device__ __forceinline__ u32 pack_f16x2(float lo, float hi) {
    u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// Tensor-core paged attention for prefill (FlashAttention-2 style).
//
// Grid: (tiles, kv heads). A tile is up to `tile_tokens` consecutive tokens of
// one sequence; its rows are (token, query head) pairs for every head sharing
// this KV head. Rows are split into groups of 16 (one MMA M-tile). Each row
// group is handled by `dw` warps that each own head_dim / dw dimensions; when
// dw > 1 the warps add their partial scores through shared memory.
// Block = row_groups * dw warps. Keys stream through shared memory 32 at a
// time as f16 (K row-major, V transposed).
//
// Dynamic smem: K [32][hd + 8] f16 | Vt [hd][40] f16 | S partials [rg][dw][16][32] f32
#define TC_BK 32
#define TC_DSLICE 128  // dims per warp (max), so O fits in 64 registers

extern "C" __global__ void attn_prefill_tc(const float* __restrict__ q, const f16_bits* __restrict__ kc,
                                           const f16_bits* __restrict__ vc, float* __restrict__ out,
                                           const u32* __restrict__ tiles, const u32* __restrict__ token_seq,
                                           const u32* __restrict__ positions, const u32* __restrict__ block_tables,
                                           int max_blocks, int n_heads, int n_kv_heads, int head_dim,
                                           int block_size, float scale, int window, int tile_tokens, int dw) {
    extern __shared__ char tc_smem[];
    const int hd = head_dim;
    const int ks_stride = hd + 8, vt_stride = TC_BK + 8;
    f16_bits* Ks = (f16_bits*)tc_smem;
    f16_bits* Vt = Ks + TC_BK * ks_stride;
    float* Sp = (float*)(Vt + hd * vt_stride);

    const int group = n_heads / n_kv_heads;
    const int t0 = tiles[2 * blockIdx.x], nt = tiles[2 * blockIdx.x + 1];
    const int kvh = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int g = lane / 4, t = lane % 4;
    const int rg = warp / dw, w = warp % dw;
    const int dslice = hd / dw, d0 = w * dslice;
    const int kv_w = n_kv_heads * hd;
    const u32* table = block_tables + (long)token_seq[t0] * max_blocks;

    // This thread's two rows (g and g + 8 of the row group).
    int row_tok[2], row_head[2], row_pos[2];
    for (int i = 0; i < 2; i++) {
        int r = rg * 16 + g + 8 * i;
        int tok = r / group;
        row_tok[i] = tok;
        row_head[i] = kvh * group + r % group;
        row_pos[i] = tok < nt ? (int)positions[t0 + tok] : -1;
    }

    // Q fragments for this warp's dimension slice (pre-scaled).
    u32 qa[TC_DSLICE / 16][4];
    for (int ks = 0; ks < dslice / 16; ks++) {
        for (int half = 0; half < 2; half++) {        // columns 2t.. and 2t+8..
            for (int i = 0; i < 2; i++) {             // rows g and g+8
                float lo = 0.f, hi = 0.f;
                if (row_pos[i] >= 0) {
                    const float* qp = q + ((long)(t0 + row_tok[i]) * n_heads + row_head[i]) * hd + d0 + ks * 16 +
                                      2 * t + 8 * half;
                    lo = qp[0] * scale;
                    hi = qp[1] * scale;
                }
                qa[ks][half * 2 + i] = pack_f16x2(lo, hi);
            }
        }
    }

    float o[TC_DSLICE / 8][4];
    for (int i = 0; i < TC_DSLICE / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};

    const int first = positions[t0], last = positions[t0 + nt - 1];
    const int kbeg = window > 0 ? max(0, first + 1 - window) : 0;
    const int kend = last + 1;

    for (int k0 = kbeg; k0 < kend; k0 += TC_BK) {
        // Stage K and V, 8 halves (16 bytes) per thread per step.
        for (int i = threadIdx.x; i < TC_BK * hd / 8; i += blockDim.x) {
            int p = i / (hd / 8), dd = (i % (hd / 8)) * 8, kp = k0 + p;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (kp < kend) {
                long base = ((long)table[kp / block_size] * block_size + kp % block_size) * kv_w + kvh * hd + dd;
                kv = *(const uint4*)(kc + base);
                vv = *(const uint4*)(vc + base);
            }
            *(uint4*)(Ks + p * ks_stride + dd) = kv;
            const f16_bits* vh = (const f16_bits*)&vv;
            for (int e = 0; e < 8; e++) Vt[(dd + e) * vt_stride + p] = vh[e];
        }
        __syncthreads();

        // S = Q K^T over this warp's dims: 4 N-tiles of 8 keys.
        float s[TC_BK / 8][4];
        for (int j = 0; j < TC_BK / 8; j++) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            const f16_bits* kr = Ks + (j * 8 + g) * ks_stride + d0 + 2 * t;
            for (int ks = 0; ks < dslice / 16; ks++) {
                u32 b[2] = {*(const u32*)(kr + ks * 16), *(const u32*)(kr + ks * 16 + 8)};
                mma_16816(s[j], qa[ks], b);
            }
        }
        if (dw > 1) {
            float* mine = Sp + ((rg * dw + w) * 16) * TC_BK;
            for (int j = 0; j < TC_BK / 8; j++) {
                int c = j * 8 + 2 * t;
                mine[g * TC_BK + c] = s[j][0];
                mine[g * TC_BK + c + 1] = s[j][1];
                mine[(g + 8) * TC_BK + c] = s[j][2];
                mine[(g + 8) * TC_BK + c + 1] = s[j][3];
            }
            __syncthreads();
            for (int j = 0; j < TC_BK / 8; j++) {
                int c = j * 8 + 2 * t;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                for (int ww = 0; ww < dw; ww++) {
                    const float* p = Sp + ((rg * dw + ww) * 16) * TC_BK;
                    a0 += p[g * TC_BK + c];
                    a1 += p[g * TC_BK + c + 1];
                    a2 += p[(g + 8) * TC_BK + c];
                    a3 += p[(g + 8) * TC_BK + c + 1];
                }
                s[j][0] = a0; s[j][1] = a1; s[j][2] = a2; s[j][3] = a3;
            }
        }

        // Mask, online softmax (rows g, g+8; each row is spread over a lane quad).
        float corr[2];
        for (int i = 0; i < 2; i++) {
            float mx = NEG_INF;
            for (int j = 0; j < TC_BK / 8; j++) {
                for (int c = 0; c < 2; c++) {
                    int kp = k0 + j * 8 + 2 * t + c;
                    bool ok = row_pos[i] >= 0 && kp < kend && kp <= row_pos[i] &&
                              (window <= 0 || row_pos[i] - kp < window);
                    float v = ok ? s[j][2 * i + c] : NEG_INF;
                    s[j][2 * i + c] = v;
                    mx = fmaxf(mx, v);
                }
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            float m_new = fmaxf(m_run[i], mx);
            float sum = 0.f;
            if (m_new == NEG_INF) {
                corr[i] = 1.f;
                for (int j = 0; j < TC_BK / 8; j++) s[j][2 * i] = s[j][2 * i + 1] = 0.f;
            } else {
                corr[i] = __expf(m_run[i] - m_new);
                for (int j = 0; j < TC_BK / 8; j++) {
                    for (int c = 0; c < 2; c++) {
                        float v = s[j][2 * i + c];
                        float p = v == NEG_INF ? 0.f : __expf(v - m_new);
                        s[j][2 * i + c] = p;
                        sum += p;
                    }
                }
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            l_run[i] = l_run[i] * corr[i] + sum;
            m_run[i] = m_new;
        }

        // O = O * corr + P V over this warp's dims.
        for (int nd = 0; nd < dslice / 8; nd++) {
            o[nd][0] *= corr[0];
            o[nd][1] *= corr[0];
            o[nd][2] *= corr[1];
            o[nd][3] *= corr[1];
        }
        for (int kc16 = 0; kc16 < TC_BK / 16; kc16++) {
            u32 pa[4] = {pack_f16x2(s[2 * kc16][0], s[2 * kc16][1]), pack_f16x2(s[2 * kc16][2], s[2 * kc16][3]),
                         pack_f16x2(s[2 * kc16 + 1][0], s[2 * kc16 + 1][1]),
                         pack_f16x2(s[2 * kc16 + 1][2], s[2 * kc16 + 1][3])};
            for (int nd = 0; nd < dslice / 8; nd++) {
                const f16_bits* vr = Vt + (d0 + nd * 8 + g) * vt_stride + kc16 * 16 + 2 * t;
                u32 b[2] = {*(const u32*)vr, *(const u32*)(vr + 8)};
                mma_16816(o[nd], pa, b);
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; i++) {
        if (row_pos[i] < 0 || l_run[i] == 0.f) continue;
        float inv = 1.f / l_run[i];
        float* op = out + ((long)(t0 + row_tok[i]) * n_heads + row_head[i]) * hd + d0 + 2 * t;
        for (int nd = 0; nd < dslice / 8; nd++) {
            op[nd * 8] = o[nd][2 * i] * inv;
            op[nd * 8 + 1] = o[nd][2 * i + 1] * inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Vendor shim: int8 tensor-core MMA (see mma_16816 for the fragment notation).
// m16n8k32: a = 4 regs {r g, k 4t..4t+3}, {r g+8, ..}, {r g, k 16+4t..}, {r g+8, ..}
//           b = 2 regs {k 4t..4t+3, n g}, {k 16+4t.., n g}
// m16n8k16: a = 2 regs {r g, k 4t..4t+3}, {r g+8, ..}; b = 1 reg {k 4t..4t+3, n g}
__device__ __forceinline__ void mma_s8_16832(int* c, const u32* a, const u32* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_s8_16816(int* c, const u32* a, u32 b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Vendor shim: async global->shared copies and transposing shared-memory loads.
__device__ __forceinline__ u32 smem_u32(const void* p) {
    u32 r;
    asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p));
    return r;
}
// 16-byte async copy; src_bytes < 16 zero-fills the rest.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
// Four 8x8 b16 matrices (lane L supplies the row address for matrix L/8,
// row L%8); thread (g, t) receives row g, columns 2t..2t+1 of each.
__device__ __forceinline__ void ldmatrix_x4(u32* r, const void* row_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(row_addr)));
}
// Same, transposed.
__device__ __forceinline__ void ldmatrix_x4_trans(u32* r, const void* row_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(row_addr)));
}

#else
// HIP: no PTX tensor-core attention (attn_wmma is the RDNA3 counterpart).
RLLM_STUB(attn_prefill_tc)
#endif  // RLLM_HIP

#if !defined(RLLM_HIP) || defined(__GFX11__)
// Quantized GEMM for prefill ("MMQ"): out[m, n] = x[m, k] . W[n, k]^T with
// int8 tensor cores (mma.sync on NVIDIA, WMMA on RDNA3), reading the
// repacked weights directly (no f16 copy).
// x is pre-quantized by quantize_q8 (int8 + per-unit scale and scaled sum).
//
// Block: MMQ_BM weight rows x MMQ_BN tokens, 8 warps in a 4x2 grid, each warp
// 32 rows x 32 tokens. k is staged MMQ_CU units (32 elements each) at a time;
// weights are unpacked to int8 in shared memory. Each unit's int32 MMA result
// is scaled into f32:
//   Q8: dw * dx * sum(qw qx)
//   Q4: dx * (d * sum(q qx) - m * sum(qx))       (value = d q - m)
//   Q6: dx * (d_lo * sum_lo + d_hi * sum_hi)     (one scale per 16)
// Split-K: gridDim.z blocks share one output tile, each covering a slice of
// the units. Two splits atomicAdd into a zeroed output: x + y == y + x, so
// the result doesn't depend on which lands first. With more, the order of
// atomic adds (and so the rounding) would vary between runs; instead each
// split writes its partial tile to `part + z * m * n` and mmq_sum_splits adds
// them in split order. (Summing in the last block to finish each tile, as a
// stream-k fixup, measured 13% slower: it costs registers and occupancy.)
//
// GLU (gate/up of a gated FFN, ACT 0 = SiLU, 1 = GELU-tanh): a block's 128
// rows are 64 gate rows (qs/qh/sc) and the same 64 up rows (qs2/qh2/sc2), so
// the epilogue can form act(gate) * up in shared memory and emit the int8
// input of the down projection directly (and the f32 product if write_out).
// No split-K.
#define MMQ_BM 128
#define MMQ_BN 64
#define MMQ_CU 4
#define MMQ_STRIDE (MMQ_CU * 32 + 16)  // padded int8 row stride in smem
#define MMQ_THREADS 128
#define MMQ_MT (MMQ_BM / 2 / 16)  // row tiles per warp (2 warp rows)
// Accumulator tiles per warp: token tiles (32 tokens) x values per lane.
// mma.sync m16n8: 4 tiles of 8 tokens, 4 values (rows g, g+8 x tokens 2t, 2t+1).
// WMMA 16x16: 2 tiles of 16 tokens, 8 values (rows 2e + lane/16, token lane%16).
#ifdef RLLM_HIP
#define MMQ_NT 2
#define MMQ_E 8
#else
#define MMQ_NT 4
#define MMQ_E 4
#endif

// Row (within the warp row's 64) and token (within the warp's 32) of
// accumulator value e of tile (mt, nt).
__device__ __forceinline__ int mmq_row(int mt, int e, int lane) {
#ifdef RLLM_HIP
    return mt * 16 + 2 * e + lane / 16;
#else
    return mt * 16 + lane / 4 + (e >= 2 ? 8 : 0);
#endif
}
__device__ __forceinline__ int mmq_tok(int nt, int e, int lane) {
#ifdef RLLM_HIP
    return nt * 16 + lane % 16;
#else
    return nt * 8 + 2 * (lane % 4) + (e & 1);
#endif
}

#ifdef RLLM_HIP
typedef int mmq_int8v __attribute__((ext_vector_type(8)));
// c += a (16x16 int8, rows) . b (16x16 int8, columns), signed.
__device__ __forceinline__ mmq_int8v wmma_s8(int4 a, int4 b, mmq_int8v c) {
    typedef int int4v __attribute__((ext_vector_type(4)));
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, __builtin_bit_cast(int4v, a), true,
                                                      __builtin_bit_cast(int4v, b), c, false);
}
#endif

// int -> float for the MMA epilogue. A plain conversion measured faster on
// sm_120 than the add/subtract "magic number" trick (one instruction instead
// of two in a loop that is ALU-bound).
__device__ __forceinline__ float i2f_small(int i) {
    return (float)i;
}

template <int F, bool GLU, int ACT>
__device__ void mmq(const u8* __restrict__ qs, const u8* __restrict__ qh, const u8* __restrict__ sc,
                    const i8* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ xs,
                    float* __restrict__ out, float* __restrict__ part, int n, int k, int m, long qs_row,
                    long qh_row, long sc_row, const u8* __restrict__ qs2, const u8* __restrict__ qh2, const u8* __restrict__ sc2, QOut qo,
                    int write_out) {
    // One buffer, so the GLU epilogue can reuse all of it. Am and Bsum are
    // Q4-only; leaving them out keeps Q8/Q6 at 32 KB (3 blocks per SM).
    constexpr int A_BYTES = MMQ_BM * MMQ_STRIDE, B_BYTES = MMQ_BN * MMQ_STRIDE;
    constexpr int AD_BYTES = MMQ_CU * 2 * MMQ_BM * 4, BD_BYTES = MMQ_CU * MMQ_BN * 4;
    constexpr int AM_BYTES = F == FMT_Q4 ? MMQ_CU * MMQ_BM * 4 : 0, BS_BYTES = F == FMT_Q4 ? BD_BYTES : 0;
    constexpr int SMEM_BYTES = A_BYTES + B_BYTES + AD_BYTES + BD_BYTES + AM_BYTES + BS_BYTES;
    __shared__ __align__(16) char smem[SMEM_BYTES];
    i8* As = (i8*)smem;
    i8* Bs = (i8*)(smem + A_BYTES);
    float(*Ad)[MMQ_BM] = (float(*)[MMQ_BM])(smem + A_BYTES + B_BYTES);  // weight scales (Q6: two per unit)
    float(*Bd)[MMQ_BN] = (float(*)[MMQ_BN])(smem + A_BYTES + B_BYTES + AD_BYTES);
    float(*Am)[MMQ_BM] = (float(*)[MMQ_BM])(smem + A_BYTES + B_BYTES + AD_BYTES + BD_BYTES);  // Q4 offsets
    float(*Bsum)[MMQ_BN] = (float(*)[MMQ_BN])(smem + A_BYTES + B_BYTES + AD_BYTES + BD_BYTES + AM_BYTES);

    const int units = k / 32;
    // Token tiles vary fastest (blockIdx.x) so the blocks sharing a weight
    // tile run together and fetch it from DRAM about once.
    const int r0 = blockIdx.y * (GLU ? MMQ_BM / 2 : MMQ_BM), c0 = blockIdx.x * MMQ_BN;
    const int splits = gridDim.z;
    const int per_split = ((units + splits - 1) / splits + MMQ_CU - 1) / MMQ_CU * MMQ_CU;
    const int u_beg = blockIdx.z * per_split, u_end = min(units, u_beg + per_split);
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int wm = warp / 2, wn = warp % 2;  // warp tile: MMQ_MT*16 rows x 32 tokens

    float acc[MMQ_MT][MMQ_NT][MMQ_E];
    for (int a = 0; a < MMQ_MT; a++)
        for (int b = 0; b < MMQ_NT; b++)
            for (int e = 0; e < MMQ_E; e++) acc[a][b][e] = 0.f;

    for (int u0 = u_beg; u0 < u_end; u0 += MMQ_CU) {
        const int cu = min(MMQ_CU, u_end - u0);
        // Stage weights: one (row, unit) pair per thread step.
        for (int i = threadIdx.x; i < MMQ_BM * MMQ_CU; i += MMQ_THREADS) {
            int r = i / MMQ_CU, ul = i % MMQ_CU, u = u0 + ul;
            int gr = r0 + (GLU ? r % (MMQ_BM / 2) : r);
            const bool second = GLU && r >= MMQ_BM / 2;
            i8* dst = As + r * MMQ_STRIDE + ul * 32;
            if (gr >= n || ul >= cu) {
                *(int4*)dst = make_int4(0, 0, 0, 0);
                *(int4*)(dst + 16) = make_int4(0, 0, 0, 0);
                Ad[2 * ul][r] = Ad[2 * ul + 1][r] = 0.f;
                if (F == FMT_Q4) Am[ul][r] = 0.f;
                continue;
            }
            const u8* qr = (second ? qs2 : qs) + (long)gr * qs_row;
            const u8* sr = (second ? sc2 : sc) + (long)gr * sc_row;
            if (F == FMT_Q8) {
                const int4* src = (const int4*)(qr + u * 32);
                *(int4*)dst = src[0];
                *(int4*)(dst + 16) = src[1];
                Ad[2 * ul][r] = ld_f16(sr + u * 2);
            } else if (F == FMT_Q4) {
                int4 v = *(const int4*)(qr + u * 16);
                *(int4*)dst = make_int4(v.x & 0x0F0F0F0F, v.y & 0x0F0F0F0F, v.z & 0x0F0F0F0F, v.w & 0x0F0F0F0F);
                *(int4*)(dst + 16) = make_int4((v.x >> 4) & 0x0F0F0F0F, (v.y >> 4) & 0x0F0F0F0F,
                                               (v.z >> 4) & 0x0F0F0F0F, (v.w >> 4) & 0x0F0F0F0F);
                Ad[2 * ul][r] = ld_f16(sr + u * 4);
                Am[ul][r] = ld_f16(sr + u * 4 + 2);
            } else {  // FMT_Q6
                int4 lov = *(const int4*)(qr + u * 16);
                int2 hiv = *(const int2*)((second ? qh2 : qh) + (long)gr * qh_row + u * 8);
                const int* l4 = (const int*)&lov;
                int w[8];
                for (int half = 0; half < 2; half++)
                    for (int j = 0; j < 4; j++) {
                        int lv = (l4[j] >> (4 * half)) & 0x0F0F0F0F;
                        int hv = (((j & 1) ? hiv.y : hiv.x) >> (2 * (2 * half + (j >> 1)))) & 0x03030303;
                        w[half * 4 + j] = q6_to_s8x4(lv | (hv << 4));
                    }
                *(int4*)dst = make_int4(w[0], w[1], w[2], w[3]);
                *(int4*)(dst + 16) = make_int4(w[4], w[5], w[6], w[7]);
                Ad[2 * ul][r] = ld_f16(sr + u * 4);
                Ad[2 * ul + 1][r] = ld_f16(sr + u * 4 + 2);
            }
        }
        // Stage activations.
        for (int i = threadIdx.x; i < MMQ_BN * MMQ_CU; i += MMQ_THREADS) {
            int c = i / MMQ_CU, ul = i % MMQ_CU, u = u0 + ul, tok = c0 + c;
            i8* dst = Bs + c * MMQ_STRIDE + ul * 32;
            if (tok >= m || ul >= cu) {
                *(int4*)dst = make_int4(0, 0, 0, 0);
                *(int4*)(dst + 16) = make_int4(0, 0, 0, 0);
                Bd[ul][c] = 0.f;
                if (F == FMT_Q4) Bsum[ul][c] = 0.f;
                continue;
            }
            const int4* src = (const int4*)(xq + (long)tok * k + u * 32);
            *(int4*)dst = src[0];
            *(int4*)(dst + 16) = src[1];
            Bd[ul][c] = xd[(long)tok * units + u];
            if (F == FMT_Q4) Bsum[ul][c] = xs[(long)tok * units + u];
        }
        __syncthreads();

#ifdef RLLM_HIP
        for (int ul = 0; ul < cu; ul++) {
            // WMMA fragments: lane l holds row (A) or token (B) l%16 and 16
            // of the unit's 32 bytes; a unit is two k-steps (Q6: one per scale).
            const int col = lane % 16, half = lane / 16;
            int4 a[MMQ_MT][2], b[MMQ_NT][2];
            for (int mt = 0; mt < MMQ_MT; mt++)
                for (int h = 0; h < 2; h++)
                    a[mt][h] = *(const int4*)(As + (wm * (MMQ_MT * 16) + mt * 16 + col) * MMQ_STRIDE + ul * 32 + h * 16);
            for (int nt = 0; nt < MMQ_NT; nt++)
                for (int h = 0; h < 2; h++)
                    b[nt][h] = *(const int4*)(Bs + (wn * 32 + nt * 16 + col) * MMQ_STRIDE + ul * 32 + h * 16);
            float dbx[MMQ_NT], sbx[MMQ_NT];
            for (int nt = 0; nt < MMQ_NT; nt++) {
                const int tc = wn * 32 + nt * 16 + col;
                dbx[nt] = Bd[ul][tc];
                sbx[nt] = F == FMT_Q4 ? Bsum[ul][tc] : 0.f;
            }
#pragma unroll
            for (int mt = 0; mt < MMQ_MT; mt++) {
                const int rb = wm * (MMQ_MT * 16) + mt * 16 + half;  // + 2e
#pragma unroll
                for (int nt = 0; nt < MMQ_NT; nt++) {
                    const mmq_int8v z = {0, 0, 0, 0, 0, 0, 0, 0};
                    float* c = acc[mt][nt];
                    if (F == FMT_Q6) {
                        mmq_int8v c0 = wmma_s8(a[mt][0], b[nt][0], z), c1 = wmma_s8(a[mt][1], b[nt][1], z);
#pragma unroll
                        for (int e = 0; e < MMQ_E; e++)
                            c[e] += dbx[nt] * (Ad[2 * ul][rb + 2 * e] * i2f_small(c0[e]) +
                                               Ad[2 * ul + 1][rb + 2 * e] * i2f_small(c1[e]));
                    } else {
                        mmq_int8v ci = wmma_s8(a[mt][1], b[nt][1], wmma_s8(a[mt][0], b[nt][0], z));
#pragma unroll
                        for (int e = 0; e < MMQ_E; e++) {
                            c[e] += dbx[nt] * Ad[2 * ul][rb + 2 * e] * i2f_small(ci[e]);
                            if (F == FMT_Q4) c[e] -= Am[ul][rb + 2 * e] * sbx[nt];
                        }
                    }
                }
            }
        }
#else
        for (int ul = 0; ul < cu; ul++) {
            const int g = lane / 4, t = lane % 4;
            // Fragments for this unit: A for 2 row tiles, B for 4 token tiles.
            u32 a[MMQ_MT][4], b[4][2];
            float da[MMQ_MT][2][2], ma[MMQ_MT][2], dbx[4][2], sbx[4][2];
            // ldmatrix: one instruction per A tile / per two B tiles. Matrix
            // L/8 of lane L: A = {rows 0-7, 8-15} x {bytes 0-15, 16-31};
            // B = {bytes 0-15, 16-31} x {token tile nt, nt+1}.
            const int lm = lane / 8, lr = lane % 8;
            for (int mt = 0; mt < MMQ_MT; mt++) {
                const int ra = wm * (MMQ_MT * 16) + mt * 16 + g;
                ldmatrix_x4(a[mt], As + (wm * (MMQ_MT * 16) + mt * 16 + (lm & 1) * 8 + lr) * MMQ_STRIDE + ul * 32 +
                                       (lm >> 1) * 16);
                for (int h = 0; h < 2; h++) {
                    da[mt][h][0] = Ad[2 * ul + h][ra];
                    da[mt][h][1] = Ad[2 * ul + h][ra + 8];
                }
                if (F == FMT_Q4) {
                    ma[mt][0] = Am[ul][ra];
                    ma[mt][1] = Am[ul][ra + 8];
                }
            }
            for (int nt = 0; nt < 4; nt += 2) {
                u32 bb[4];
                ldmatrix_x4(bb, Bs + (wn * 32 + (nt + (lm >> 1)) * 8 + lr) * MMQ_STRIDE + ul * 32 + (lm & 1) * 16);
                b[nt][0] = bb[0];
                b[nt][1] = bb[1];
                b[nt + 1][0] = bb[2];
                b[nt + 1][1] = bb[3];
            }
            for (int nt = 0; nt < 4; nt++) {
                const int tc = wn * 32 + nt * 8 + 2 * t;
                dbx[nt][0] = Bd[ul][tc];
                dbx[nt][1] = Bd[ul][tc + 1];
                if (F == FMT_Q4) {
                    sbx[nt][0] = Bsum[ul][tc];
                    sbx[nt][1] = Bsum[ul][tc + 1];
                }
            }
#pragma unroll
            for (int mt = 0; mt < MMQ_MT; mt++) {
                // Issue this row tile's MMAs first (independent), then scale:
                // keeps the tensor cores busy while the epilogue runs.
                int ci[4][2][4];
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    if (F == FMT_Q6) {
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            u32 ah[2] = {a[mt][2 * h], a[mt][2 * h + 1]};
                            ci[nt][h][0] = ci[nt][h][1] = ci[nt][h][2] = ci[nt][h][3] = 0;
                            mma_s8_16816(ci[nt][h], ah, b[nt][h]);
                        }
                    } else {
                        ci[nt][0][0] = ci[nt][0][1] = ci[nt][0][2] = ci[nt][0][3] = 0;
                        mma_s8_16832(ci[nt][0], a[mt], b[nt]);
                    }
                }
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    float* c = acc[mt][nt];
                    if (F == FMT_Q6) {
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            float d0 = da[mt][h][0], d1 = da[mt][h][1];
                            c[0] += d0 * dbx[nt][0] * i2f_small(ci[nt][h][0]);
                            c[1] += d0 * dbx[nt][1] * i2f_small(ci[nt][h][1]);
                            c[2] += d1 * dbx[nt][0] * i2f_small(ci[nt][h][2]);
                            c[3] += d1 * dbx[nt][1] * i2f_small(ci[nt][h][3]);
                        }
                    } else {
                        float d0 = da[mt][0][0], d1 = da[mt][0][1];
                        if (F == FMT_Q4) {
                            c[0] += dbx[nt][0] * d0 * i2f_small(ci[nt][0][0]) - ma[mt][0] * sbx[nt][0];
                            c[1] += dbx[nt][1] * d0 * i2f_small(ci[nt][0][1]) - ma[mt][0] * sbx[nt][1];
                            c[2] += dbx[nt][0] * d1 * i2f_small(ci[nt][0][2]) - ma[mt][1] * sbx[nt][0];
                            c[3] += dbx[nt][1] * d1 * i2f_small(ci[nt][0][3]) - ma[mt][1] * sbx[nt][1];
                        } else {
                            c[0] += d0 * dbx[nt][0] * i2f_small(ci[nt][0][0]);
                            c[1] += d0 * dbx[nt][1] * i2f_small(ci[nt][0][1]);
                            c[2] += d1 * dbx[nt][0] * i2f_small(ci[nt][0][2]);
                            c[3] += d1 * dbx[nt][1] * i2f_small(ci[nt][0][3]);
                        }
                    }
                }
            }
        }
#endif
        __syncthreads();
    }

    if (GLU) {
        // Gate rows (warp row 0) and up rows (warp row 1) go to shared memory
        // as G/U[token][row]; then each thread forms act(gate) * up for one
        // 32-row unit and emits it: f32 and/or int8 with its scales.
        // Unpadded rows of 64 with the float4 groups XOR-swizzled by token,
        // so the per-unit float4 reads below are bank-conflict free (and
        // G + U fit in the 32 KB of the Q8/Q6 kernels).
        constexpr int US = MMQ_BM / 2;
        static_assert(2 * MMQ_BN * US * 4 <= SMEM_BYTES, "G and U fit in shared memory");
        auto at = [](int tl, int rl) { return tl * US + (rl ^ ((tl & 7) * 4)); };
        float* GU = (float*)smem + wm * (MMQ_BN * US);  // this warp row's half
        // Fully unrolled: acc must stay in registers (runtime indexing would
        // move it to local memory for the whole kernel).
#pragma unroll
        for (int mt = 0; mt < MMQ_MT; mt++)
#pragma unroll
            for (int nt = 0; nt < MMQ_NT; nt++)
#pragma unroll
                for (int e = 0; e < MMQ_E; e++) {
                    int rl = mmq_row(mt, e, lane), tl = wn * 32 + mmq_tok(nt, e, lane);
                    GU[at(tl, rl)] = acc[mt][nt][e];
                }
        __syncthreads();
        const float* G = (const float*)smem;
        const float* U = G + MMQ_BN * US;
        // One thread per 32-row unit (token tl, rows 32*h..): 128 units, no
        // shuffles, 32 independent elements per thread, vector stores.
        static_assert(MMQ_BN * 2 == MMQ_THREADS, "one unit per thread");
        {
            const int tl = threadIdx.x % MMQ_BN, h = threadIdx.x / MMQ_BN;  // warp: consecutive tokens
            const int tt = c0 + tl, rb = r0 + h * 32;
            if (tt < m && rb < n) {  // n % 32 == 0
                const long oi = (long)tt * n + rb;
                // act() uses fast approximations: the result is rounded to int8 next.
                float v[32], amax = 0.f;
#pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    const int off = at(tl, h * 32 + j);  // float4 group j (swizzle keeps groups whole)
                    float4 gv = *(const float4*)(G + off), uv = *(const float4*)(U + off);
                    float gg[4] = {gv.x, gv.y, gv.z, gv.w}, uu[4] = {uv.x, uv.y, uv.z, uv.w};
#pragma unroll
                    for (int b = 0; b < 4; b++) {
                        float a = ACT == 0 ? __fdividef(gg[b], 1.f + __expf(-gg[b])) : gelu_tanh_fast(gg[b]);
                        v[j + b] = a * uu[b];
                        amax = fmaxf(amax, fabsf(v[j + b]));
                    }
                }
                if (write_out)
#pragma unroll
                    for (int j = 0; j < 32; j += 4) *(float4*)(out + oi + j) = make_float4(v[j], v[j + 1], v[j + 2], v[j + 3]);
                if (qo.on) {
                    // Multiply by 1/d (can differ from quantize_q8 by one step at exact ties).
                    const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
                    float qsum = 0.f;
                    int packed[8];
#pragma unroll
                    for (int j = 0; j < 32; j += 4) {
                        int w = 0;
#pragma unroll
                        for (int b = 0; b < 4; b++) {
                            float q = rintf(v[j + b] * id);
                            qsum += q;
                            w |= ((int)q & 0xFF) << (8 * b);
                        }
                        packed[j / 4] = w;
                    }
                    int4* qd = (int4*)(qo.q + oi);
                    qd[0] = make_int4(packed[0], packed[1], packed[2], packed[3]);
                    qd[1] = make_int4(packed[4], packed[5], packed[6], packed[7]);
                    qo.d[oi / 32] = d;
                    qo.s[oi / 32] = d * qsum;
                }
            }
        }
        return;
    }

    // Fully unrolled so acc stays in registers (sm_89 otherwise keeps it in
    // a local-memory stack frame for the whole kernel).
#pragma unroll
    for (int mt = 0; mt < MMQ_MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < MMQ_NT; nt++) {
#pragma unroll
            for (int e = 0; e < MMQ_E; e++) {
                int rr = r0 + wm * (MMQ_MT * 16) + mmq_row(mt, e, lane), tt = c0 + wn * 32 + mmq_tok(nt, e, lane);
                if (rr < n && tt < m) {
                    if (splits > 2)
                        part[(long)blockIdx.z * m * n + (long)tt * n + rr] = acc[mt][nt][e];
                    else if (splits == 2)
                        atomicAdd(out + (long)tt * n + rr, acc[mt][nt][e]);
                    else
                        out[(long)tt * n + rr] = acc[mt][nt][e];
                }
            }
        }
    }
}

#define INSTANTIATE_MMQ(NAME, F)                                                                              \
    extern "C" __global__ void mmq_##NAME(const u8* qs, const u8* qh, const u8* sc, const i8* xq,              \
                                          const float* xd, const float* xs, float* out, float* part, int n,    \
                                          int k, int m, long qs_row, long qh_row, long sc_row) {               \
        mmq<F, false, 0>(qs, qh, sc, xq, xd, xs, out, part, n, k, m, qs_row, qh_row, sc_row, nullptr, nullptr, \
                         nullptr, QOut{}, 0);                                                                  \
    }                                                                                                          \
    extern "C" __global__ void mmq_glu_silu_##NAME(                                                            \
        const u8* qs, const u8* qh, const u8* sc, const i8* xq, const float* xd, const float* xs, float* out,  \
        int n, int k, int m, long qs_row, long qh_row, long sc_row, const u8* qs2, const u8* qh2,             \
        const u8* sc2, QOut qo, int write_out) {                                                               \
        mmq<F, true, 0>(qs, qh, sc, xq, xd, xs, out, nullptr, n, k, m, qs_row, qh_row, sc_row, qs2, qh2, sc2, qo,       \
                        write_out);                                                                            \
    }                                                                                                          \
    extern "C" __global__ void mmq_glu_gelu_##NAME(                                                            \
        const u8* qs, const u8* qh, const u8* sc, const i8* xq, const float* xd, const float* xs, float* out,  \
        int n, int k, int m, long qs_row, long qh_row, long sc_row, const u8* qs2, const u8* qh2,             \
        const u8* sc2, QOut qo, int write_out) {                                                               \
        mmq<F, true, 1>(qs, qh, sc, xq, xd, xs, out, nullptr, n, k, m, qs_row, qh_row, sc_row, qs2, qh2, sc2, qo,       \
                        write_out);                                                                            \
    }
INSTANTIATE_MMQ(q8, FMT_Q8)
INSTANTIATE_MMQ(q4, FMT_Q4)
INSTANTIATE_MMQ(q6, FMT_Q6)
#else
// HIP targets without RDNA3 WMMA: the backend doesn't call MMQ there.
RLLM_STUB(mmq_q8) RLLM_STUB(mmq_q4) RLLM_STUB(mmq_q6)
RLLM_STUB(mmq_glu_silu_q8) RLLM_STUB(mmq_glu_silu_q4) RLLM_STUB(mmq_glu_silu_q6)
RLLM_STUB(mmq_glu_gelu_q8) RLLM_STUB(mmq_glu_gelu_q4) RLLM_STUB(mmq_glu_gelu_q6)
#endif

// out[i] = part[0][i] + part[1][i] + ... in split order (see mmq's split-K).
extern "C" __global__ void mmq_sum_splits(const float* __restrict__ part, float* __restrict__ out, int splits,
                                          long len) {
    long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= len) return;
    if ((len & 3) == 0) {  // float4 rows stay aligned in every split
        float4 a = __ldcs((const float4*)(part + i));
        for (int z = 1; z < splits; z++) {
            float4 b = __ldcs((const float4*)(part + (long)z * len + i));
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        }
        *(float4*)(out + i) = a;
        return;
    }
    for (long j = i; j < min(i + 4, len); j++) {
        float a = part[j];
        for (int z = 1; z < splits; z++) a += part[(long)z * len + j];
        out[j] = a;
    }
}



// ---------------------------------------------------------------------------
// Paged attention for decode (one query token per sequence).
//
// Grid: (tokens, kv heads, splits); block: AD_WARPS warps. Each block serves
// all G query heads that share its KV head, so every K/V row is read once.
// Scores: each warp takes 32 positions at a time, lane = position, computing
// full dot products against q in shared memory (one max/sum reduction per 32
// positions). Values: lane = head dims, coalesced V row reads, probabilities
// broadcast with shuffles. Warps' partial softmax states merge at the end.
// Templated on G (group size) and HD (head dim) so all per-thread arrays stay
// in registers.
//
// Split context (flash-decoding): with few (token, KV head) pairs and a long
// context, one block per pair leaves most SMs idle. Then `splits` > 1 blocks
// each take a 32-aligned share of the positions and write their unnormalized
// state to `part` ([token][kv head][split] x (acc [G][HD], m [G], l [G]));
// attn_decode_combine merges the splits in order.
// Dynamic smem: q [G][HD] f32 | per warp: m, l [G] + acc [G][HD]
#define AD_WARPS 8

// Q8: the cache is int8 with per-32 scales ksc/vsc (see kv_q8_store). For V
// each lane owns PER consecutive dims (one vector load and one scale per
// position) instead of dims lane + 32 i. Queries are rotated here to match
// the keys (see kv_rotate_q).
template <int G, int HD, bool Q8>
__device__ void attn_decode(const float* __restrict__ q, const void* __restrict__ kc,
                            const void* __restrict__ vc, float* __restrict__ out,
                            const u32* __restrict__ token_seq, const u32* __restrict__ positions,
                            const u32* __restrict__ block_tables, int max_blocks, int n_heads, int n_kv_heads,
                            int block_size, float scale, int window, QOut qo, const f16_bits* __restrict__ ksc,
                            const f16_bits* __restrict__ vsc, float* __restrict__ part) {
    constexpr int PER = HD / WARP_SIZE;
    extern __shared__ float ad_smem[];
    const int t = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z, splits = gridDim.z;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE, warps = blockDim.x / WARP_SIZE;
    float* sq = ad_smem;                // [G][HD]
    float* sml = sq + G * HD;           // [warps][G][2]
    float* sacc = sml + warps * G * 2;  // [warps][G][HD]

    for (int i = threadIdx.x; i < G * HD; i += blockDim.x) {
        int g = i / HD, dd = i % HD;
        sq[i] = q[((long)t * n_heads + kvh * G + g) * HD + dd] * scale;
    }
    if (Q8) fwht_smem(sq, G * HD, kv_rot_size(HD));
    __syncthreads();

    const int ctx = positions[t] + 1;
    const int start = (window > 0 && ctx > window) ? ctx - window : 0;
    // This block's positions: a 32-aligned share of start..ctx.
    const int share = ((ctx - start + splits - 1) / splits + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    const int lo = start + split * share, hi = min(ctx, lo + share);
    const u32* table = block_tables + (long)token_seq[t] * max_blocks;
    const int kv_w = n_kv_heads * HD;

    float m[G], l[G], acc[G][PER];
#pragma unroll
    for (int g = 0; g < G; g++) {
        m[g] = NEG_INF;
        l[g] = 0.f;
#pragma unroll
        for (int i = 0; i < PER; i++) acc[g][i] = 0.f;
    }

    for (int p0 = lo + warp * WARP_SIZE; p0 < hi; p0 += warps * WARP_SIZE) {
        const int p = p0 + lane;
        const bool valid = p < hi;
        float s[G];
#pragma unroll
        for (int g = 0; g < G; g++) s[g] = 0.f;
        if (valid) {
            const long kb = ((long)table[p / block_size] * block_size + p % block_size) * kv_w + kvh * HD;
            if (Q8) {
                // One group (32 bytes, one scale) per step: int8 dot, then scale.
#pragma unroll 2
                for (int d = 0; d < HD; d += 32) {
                    const uint4* kp = (const uint4*)((const i8*)kc + kb + d);
                    uint4 k0 = kp[0], k1 = kp[1];
                    const float dk = f16_to_f32(ksc[(kb + d) / 32]);
                    const i8* kq0 = (const i8*)&k0;
                    const i8* kq1 = (const i8*)&k1;
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        const float* qg = sq + g * HD + d;
                        float a = 0.f;
#pragma unroll
                        for (int e = 0; e < 16; e++) a += qg[e] * kq0[e];
#pragma unroll
                        for (int e = 0; e < 16; e++) a += qg[16 + e] * kq1[e];
                        s[g] += a * dk;
                    }
                }
            } else {
#pragma unroll 4
                for (int d = 0; d < HD; d += 8) {
                    uint4 kv = *(const uint4*)((const f16_bits*)kc + kb + d);
                    const f16_bits* kh = (const f16_bits*)&kv;
                    float kf[8];
#pragma unroll
                    for (int e = 0; e < 8; e++) kf[e] = f16_to_f32(kh[e]);
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        const float* qg = sq + g * HD + d;
#pragma unroll
                        for (int e = 0; e < 8; e++) s[g] += qg[e] * kf[e];
                    }
                }
            }
        }
        float pr[G];
#pragma unroll
        for (int g = 0; g < G; g++) {
            float sv = valid ? s[g] : NEG_INF;
            float m_new = fmaxf(m[g], warp_max(sv));
            pr[g] = valid ? __expf(sv - m_new) : 0.f;
            float corr = m[g] == NEG_INF ? 0.f : __expf(m[g] - m_new);
            l[g] = l[g] * corr + warp_sum(pr[g]);
#pragma unroll
            for (int i = 0; i < PER; i++) acc[g][i] *= corr;
            m[g] = m_new;
        }
        // V rows in batches of VB independent loads so their latency overlaps.
        constexpr int VB = PER >= 16 ? 4 : 8;
        const int n_pos = min(WARP_SIZE, hi - p0);
        for (int j0 = 0; j0 < n_pos; j0 += VB) {
            float vh[VB][PER];
#pragma unroll
            for (int jj = 0; jj < VB; jj++) {
                const int pj = p0 + min(j0 + jj, n_pos - 1);
                const long vb = ((long)table[pj / block_size] * block_size + pj % block_size) * kv_w + kvh * HD;
                if (Q8) {
                    const i8* vr = (const i8*)vc + vb + lane * PER;
                    i8 raw[PER];
                    if (PER == 16)
                        *(uint4*)raw = *(const uint4*)vr;
                    else if (PER == 8)
                        *(uint2*)raw = *(const uint2*)vr;
                    else if (PER == 4)
                        *(u32*)raw = *(const u32*)vr;
                    else
#pragma unroll
                        for (int i = 0; i < PER; i++) raw[i] = vr[i];
                    const float dv = f16_to_f32(vsc[(vb + lane * PER) / 32]);
#pragma unroll
                    for (int i = 0; i < PER; i++) vh[jj][i] = raw[i] * dv;
                } else {
#pragma unroll
                    for (int i = 0; i < PER; i++)
                        vh[jj][i] = f16_to_f32(((const f16_bits*)vc)[vb + lane + i * WARP_SIZE]);
                }
            }
#pragma unroll
            for (int jj = 0; jj < VB; jj++) {
                const bool in = j0 + jj < n_pos;
#pragma unroll
                for (int g = 0; g < G; g++) {
                    float w = __shfl_sync(0xffffffff, pr[g], (j0 + jj) & (WARP_SIZE - 1));
                    if (!in) w = 0.f;
#pragma unroll
                    for (int i = 0; i < PER; i++) acc[g][i] += w * vh[jj][i];
                }
            }
        }
    }

#pragma unroll
    for (int g = 0; g < G; g++) {
        if (lane == 0) {
            sml[(warp * G + g) * 2] = m[g];
            sml[(warp * G + g) * 2 + 1] = l[g];
        }
#pragma unroll
        for (int i = 0; i < PER; i++) sacc[(warp * G + g) * HD + (Q8 ? lane * PER + i : lane + i * WARP_SIZE)] = acc[g][i];
    }
    __syncthreads();
    float* mine = part + (((long)t * n_kv_heads + kvh) * splits + split) * (G * (HD + 2));
    for (int i = threadIdx.x; i < G * HD; i += blockDim.x) {
        int g = i / HD, dd = i % HD;
        float gm = NEG_INF;
        for (int w = 0; w < warps; w++) gm = fmaxf(gm, sml[(w * G + g) * 2]);
        float gl = 0.f, v = 0.f;
        for (int w = 0; w < warps; w++) {
            float mw = sml[(w * G + g) * 2];
            float f = mw == NEG_INF ? 0.f : __expf(mw - gm);
            gl += sml[(w * G + g) * 2 + 1] * f;
            v += sacc[(w * G + g) * HD + dd] * f;
        }
        if (splits > 1) {
            mine[i] = v;
            if (dd == 0) {
                mine[G * HD + g] = gm;
                mine[G * HD + G + g] = gl;
            }
            continue;
        }
        const long oi = ((long)t * n_heads + kvh * G) * HD + i;
        out[oi] = v / gl;
        // Warps cover 32-aligned runs of i, and G*HD is a multiple of 32.
        if (qo.on) quant_unit(v / gl, oi / 32, qo);
    }
}

// Merge attn_decode's split states, in split order. Grid: (tokens, kv heads).
extern "C" __global__ void attn_decode_combine(const float* __restrict__ part, float* __restrict__ out, int splits,
                                               int group, int head_dim, int n_heads, int n_kv_heads, QOut qo) {
    const int t = blockIdx.x, kvh = blockIdx.y, gh = group * head_dim;
    const float* base = part + ((long)t * n_kv_heads + kvh) * splits * (gh + 2 * group);
    for (int i = threadIdx.x; i < gh; i += blockDim.x) {
        const int g = i / head_dim;
        float gm = NEG_INF;
        for (int s = 0; s < splits; s++) gm = fmaxf(gm, base[s * (gh + 2 * group) + gh + g]);
        float gl = 0.f, v = 0.f;
        for (int s = 0; s < splits; s++) {
            const float* ps = base + s * (gh + 2 * group);
            const float ms = ps[gh + g];
            const float f = ms == NEG_INF ? 0.f : __expf(ms - gm);
            gl += ps[gh + group + g] * f;
            v += ps[i] * f;
        }
        const long oi = ((long)t * n_heads + kvh * group) * head_dim + i;
        out[oi] = v / gl;
        if (qo.on) quant_unit(v / gl, oi / 32, qo);
    }
}

#define INSTANTIATE_AD(G, HD)                                                                                \
    extern "C" __global__ void attn_decode_g##G##_d##HD(                                                      \
        const float* q, const void* kc, const void* vc, float* out, const u32* token_seq,                    \
        const u32* positions, const u32* block_tables, int max_blocks, int n_heads, int n_kv_heads,          \
        int block_size, float scale, int window, QOut qo, float* part) {                                      \
        attn_decode<G, HD, false>(q, kc, vc, out, token_seq, positions, block_tables, max_blocks, n_heads,    \
                                  n_kv_heads, block_size, scale, window, qo, nullptr, nullptr, part);         \
    }                                                                                                         \
    extern "C" __global__ void attn_decode_q8_g##G##_d##HD(                                                   \
        const float* q, const void* kc, const void* vc, float* out, const u32* token_seq,                    \
        const u32* positions, const u32* block_tables, int max_blocks, int n_heads, int n_kv_heads,          \
        int block_size, float scale, int window, QOut qo, const f16_bits* ksc, const f16_bits* vsc,           \
        float* part) {                                                                                        \
        attn_decode<G, HD, true>(q, kc, vc, out, token_seq, positions, block_tables, max_blocks, n_heads,     \
                                 n_kv_heads, block_size, scale, window, qo, ksc, vsc, part);                  \
    }
INSTANTIATE_AD(1, 64)
INSTANTIATE_AD(1, 128)
INSTANTIATE_AD(1, 256)
INSTANTIATE_AD(2, 64)
INSTANTIATE_AD(2, 128)
INSTANTIATE_AD(2, 256)
INSTANTIATE_AD(4, 64)
INSTANTIATE_AD(4, 128)
INSTANTIATE_AD(4, 256)
INSTANTIATE_AD(4, 512)

#ifndef RLLM_HIP
// FlashAttention-2 style prefill attention on tensor cores, templated on head dim.
//
// Grid: (tiles, kv heads). Block: FaCfg::WARPS warps = RG row groups (16 query
// rows each) x DW warps splitting head_dim (DW = HD/128, at least 1). Rows are (token,
// head) pairs of a tile: consecutive tokens of one sequence, every query head
// of this KV head. K and V stream through a double-buffered shared-memory
// ring (cp.async), BK keys at a time, both row-major; P.V fragments come from
// ldmatrix.trans. Online softmax in registers, f32 accumulation.
template <int HD>
struct FaCfg {
    static constexpr int DW = HD > 256 ? HD / 256 : 1;  // warps per row group (dim split)
    static constexpr int WARPS = HD > 256 ? 8 : 4;      // more rows per K/V chunk for big heads
    static constexpr int RG = WARPS / DW;               // row groups per block
    static constexpr int DS = HD / DW;                  // dims per warp
    static constexpr int BK = HD <= 128 ? 32 : (HD <= 256 ? 16 : 16);
    static constexpr int KS = HD + 8;                   // padded row stride (halves)
    static constexpr int STAGE = 2 * BK * KS;           // K + V halves per stage
    static constexpr int SMEM = 2 * STAGE * 2 + (DW > 1 ? RG * DW * 16 * BK * 4 : 0);
};

template <int HD>
__device__ void attn_fa(const float* __restrict__ q, const f16_bits* __restrict__ kc,
                        const f16_bits* __restrict__ vc, float* __restrict__ out, const u32* __restrict__ tiles,
                        const u32* __restrict__ token_seq, const u32* __restrict__ positions,
                        const u32* __restrict__ block_tables, int max_blocks, int n_heads, int n_kv_heads,
                        int block_size, float scale, int window, int tile_tokens, QOut qo) {
    typedef FaCfg<HD> C;
    constexpr int DW = C::DW, DS = C::DS, BK = C::BK, KS = C::KS;
    extern __shared__ __align__(16) char fa_smem[];
    f16_bits* ring = (f16_bits*)fa_smem;                  // [2][K: BK*KS | V: BK*KS]
    float* Sp = (float*)(fa_smem + 2 * C::STAGE * 2);     // [RG][DW][16][BK]

    const int group = n_heads / n_kv_heads;
    const int t0 = tiles[2 * blockIdx.x], nt = tiles[2 * blockIdx.x + 1];
    const int kvh = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const int g = lane / 4, t = lane % 4;
    const int rg = warp / DW, w = warp % DW, d0 = w * DS;
    const int kv_w = n_kv_heads * HD;
    const u32* table = block_tables + (long)token_seq[t0] * max_blocks;

    int row_tok[2], row_head[2], row_pos[2];
    for (int i = 0; i < 2; i++) {
        int r = rg * 16 + g + 8 * i;
        int tok = r / group;
        row_tok[i] = tok;
        row_head[i] = kvh * group + r % group;
        row_pos[i] = (tok < tile_tokens && tok < nt) ? (int)positions[t0 + tok] : -1;
    }
    u32 qa[DS / 16][4];
#pragma unroll
    for (int ks = 0; ks < DS / 16; ks++)
#pragma unroll
        for (int half = 0; half < 2; half++)
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float lo = 0.f, hi = 0.f;
                if (row_pos[i] >= 0) {
                    const float* qp =
                        q + ((long)(t0 + row_tok[i]) * n_heads + row_head[i]) * HD + d0 + ks * 16 + 2 * t + 8 * half;
                    lo = qp[0] * scale;
                    hi = qp[1] * scale;
                }
                qa[ks][half * 2 + i] = pack_f16x2(lo, hi);
            }

    float o[DS / 8][4];
#pragma unroll
    for (int i = 0; i < DS / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m_run[2] = {NEG_INF, NEG_INF}, l_run[2] = {0.f, 0.f};

    const int first = positions[t0], last = positions[t0 + nt - 1];
    const int kbeg = window > 0 ? max(0, first + 1 - window) : 0;
    const int kend = last + 1;
    const int n_chunks = (kend - kbeg + BK - 1) / BK;

    auto issue = [&](int stage, int k0) {
        f16_bits* Ks = ring + stage * C::STAGE;
        f16_bits* Vs = Ks + BK * KS;
        for (int i = threadIdx.x; i < BK * (HD / 8); i += blockDim.x) {
            int p = i / (HD / 8), dd = (i % (HD / 8)) * 8, kp = k0 + p;
            bool ok = kp < kend;
            long base = ok ? ((long)table[kp / block_size] * block_size + kp % block_size) * kv_w + kvh * HD + dd : 0;
            cp_async16(Ks + p * KS + dd, kc + base, ok ? 16 : 0);
            cp_async16(Vs + p * KS + dd, vc + base, ok ? 16 : 0);
        }
    };

    if (n_chunks > 0) issue(0, kbeg);
    cp_async_commit();
    for (int c = 0; c < n_chunks; c++) {
        const int k0 = kbeg + c * BK;
        if (c + 1 < n_chunks) issue((c + 1) & 1, k0 + BK);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const f16_bits* Ks = ring + (c & 1) * C::STAGE;
        const f16_bits* Vs = Ks + BK * KS;

        float s[BK / 8][4];
#pragma unroll
        for (int j = 0; j < BK / 8; j++) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            const f16_bits* kr = Ks + (j * 8 + g) * KS + d0 + 2 * t;
#pragma unroll
            for (int ks = 0; ks < DS / 16; ks++) {
                u32 b[2] = {*(const u32*)(kr + ks * 16), *(const u32*)(kr + ks * 16 + 8)};
                mma_16816(s[j], qa[ks], b);
            }
        }
        if (DW > 1) {
            float* mine = Sp + ((rg * DW + w) * 16) * BK;
#pragma unroll
            for (int j = 0; j < BK / 8; j++) {
                int cc = j * 8 + 2 * t;
                mine[g * BK + cc] = s[j][0];
                mine[g * BK + cc + 1] = s[j][1];
                mine[(g + 8) * BK + cc] = s[j][2];
                mine[(g + 8) * BK + cc + 1] = s[j][3];
            }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < BK / 8; j++) {
                int cc = j * 8 + 2 * t;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                for (int ww = 0; ww < DW; ww++) {
                    const float* p = Sp + ((rg * DW + ww) * 16) * BK;
                    a0 += p[g * BK + cc];
                    a1 += p[g * BK + cc + 1];
                    a2 += p[(g + 8) * BK + cc];
                    a3 += p[(g + 8) * BK + cc + 1];
                }
                s[j][0] = a0; s[j][1] = a1; s[j][2] = a2; s[j][3] = a3;
            }
        }

        // Chunks entirely in the past of every row (and inside the window)
        // need no masking.
        const bool full = k0 + BK <= kend && k0 + BK - 1 <= first && (window <= 0 || last - k0 < window);
        float corr[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            float mx = NEG_INF;
#pragma unroll
            for (int j = 0; j < BK / 8; j++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    float v = s[j][2 * i + e];
                    if (!full) {
                        int kp = k0 + j * 8 + 2 * t + e;
                        bool ok = row_pos[i] >= 0 && kp < kend && kp <= row_pos[i] &&
                                  (window <= 0 || row_pos[i] - kp < window);
                        v = ok ? v : NEG_INF;
                    } else if (row_pos[i] < 0) {
                        v = NEG_INF;
                    }
                    s[j][2 * i + e] = v;
                    mx = fmaxf(mx, v);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            float m_new = fmaxf(m_run[i], mx);
            float sum = 0.f;
            if (m_new == NEG_INF) {
                corr[i] = 1.f;
#pragma unroll
                for (int j = 0; j < BK / 8; j++) s[j][2 * i] = s[j][2 * i + 1] = 0.f;
            } else {
                corr[i] = __expf(m_run[i] - m_new);
#pragma unroll
                for (int j = 0; j < BK / 8; j++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        float p = __expf(s[j][2 * i + e] - m_new);
                        s[j][2 * i + e] = p;
                        sum += p;
                    }
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            l_run[i] = l_run[i] * corr[i] + sum;
            m_run[i] = m_new;
        }
#pragma unroll
        for (int nd = 0; nd < DS / 8; nd++) {
            o[nd][0] *= corr[0];
            o[nd][1] *= corr[0];
            o[nd][2] *= corr[1];
            o[nd][3] *= corr[1];
        }
        // O += P V. ldmatrix.x4.trans yields B fragments for two dim tiles.
        const int mat = lane / 8, mrow = lane % 8;
#pragma unroll
        for (int kc16 = 0; kc16 < BK / 16; kc16++) {
            u32 pa[4] = {pack_f16x2(s[2 * kc16][0], s[2 * kc16][1]), pack_f16x2(s[2 * kc16][2], s[2 * kc16][3]),
                         pack_f16x2(s[2 * kc16 + 1][0], s[2 * kc16 + 1][1]),
                         pack_f16x2(s[2 * kc16 + 1][2], s[2 * kc16 + 1][3])};
            const f16_bits* vrow = Vs + (kc16 * 16 + (mat & 1) * 8 + mrow) * KS + d0 + (mat >> 1) * 8;
#pragma unroll
            for (int nd = 0; nd < DS / 8; nd += 2) {
                u32 bb[4];
                ldmatrix_x4_trans(bb, vrow + nd * 8);
                mma_16816(o[nd], pa, bb);
                mma_16816(o[nd + 1], pa, bb + 2);
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; i++) {
        if (row_pos[i] < 0 || l_run[i] == 0.f) continue;
        float inv = 1.f / l_run[i];
        float* op = out + ((long)(t0 + row_tok[i]) * n_heads + row_head[i]) * HD + d0 + 2 * t;
#pragma unroll
        for (int nd = 0; nd < DS / 8; nd++) {
            *(float2*)(op + nd * 8) = make_float2(o[nd][2 * i] * inv, o[nd][2 * i + 1] * inv);
        }
    }
    if (!qo.on) return;  // uniform across the block

    // int8 copy for the output projection. Each warp stages its 16 x DS slab in
    // the shared memory the main loop no longer needs, then each lane quantizes
    // whole 32-value units from it (vector loads/stores, no shuffles). As many
    // slabs as fit go per pass. Multiplies by 1/d: can differ from quantize_q8
    // by one step at exact ties.
    constexpr int SS = DS + 4;  // slab row stride (floats), keeps float4 alignment
    constexpr int SLAB = 16 * SS * 4 + 16 * 8;
    constexpr int FIT = C::SMEM / SLAB;
    static_assert(FIT >= 1, "one slab fits");
    for (int p0 = 0; p0 < C::WARPS; p0 += FIT) {
        __syncthreads();  // ring (first pass) or the previous pass's slabs are free
        if (warp < p0 || warp >= p0 + FIT) continue;
        char* slab = fa_smem + (warp - p0) * SLAB;
        float* Sl = (float*)slab;
        long* rbase = (long*)(slab + 16 * SS * 4);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int r = g + 8 * i;
            const bool ok = !(row_pos[i] < 0 || l_run[i] == 0.f);
            const float inv = ok ? 1.f / l_run[i] : 0.f;
#pragma unroll
            for (int nd = 0; nd < DS / 8; nd++)
                *(float2*)(Sl + r * SS + nd * 8 + 2 * t) = make_float2(o[nd][2 * i] * inv, o[nd][2 * i + 1] * inv);
            if (t == 0) rbase[r] = ok ? ((long)(t0 + row_tok[i]) * n_heads + row_head[i]) * HD + d0 : -1;
        }
        __syncwarp();
        for (int task = lane; task < 16 * (DS / 32); task += WARP_SIZE) {
            const int r = task / (DS / 32), u = task % (DS / 32);
            const long base = rbase[r];
            if (base < 0) continue;
            const float* src = Sl + r * SS + u * 32;
            float v[32], amax = 0.f;
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                const float4 f = *(const float4*)(src + j);
                v[j] = f.x, v[j + 1] = f.y, v[j + 2] = f.z, v[j + 3] = f.w;
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
            }
            const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
            float qsum = 0.f;
            int packed[8];
#pragma unroll
            for (int j = 0; j < 32; j += 4) {
                int wv = 0;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const float qv = rintf(v[j + b] * id);
                    qsum += qv;
                    wv |= ((int)qv & 0xFF) << (8 * b);
                }
                packed[j / 4] = wv;
            }
            const long oi = base + u * 32;
            int4* qd = (int4*)(qo.q + oi);
            qd[0] = make_int4(packed[0], packed[1], packed[2], packed[3]);
            qd[1] = make_int4(packed[4], packed[5], packed[6], packed[7]);
            qo.d[oi / 32] = d;
            qo.s[oi / 32] = d * qsum;
        }
    }
}

#define INSTANTIATE_FA(HD)                                                                                    \
    extern "C" __global__ void attn_fa_d##HD(const float* q, const f16_bits* kc, const f16_bits* vc,          \
                                             float* out, const u32* tiles, const u32* token_seq,               \
                                             const u32* positions, const u32* block_tables, int max_blocks,    \
                                             int n_heads, int n_kv_heads, int block_size, float scale,         \
                                             int window, int tile_tokens, QOut qo) {                           \
        attn_fa<HD>(q, kc, vc, out, tiles, token_seq, positions, block_tables, max_blocks, n_heads,            \
                    n_kv_heads, block_size, scale, window, tile_tokens, qo);                                   \
    }                                                                                                          \
    extern "C" __global__ void attn_fa_d##HD##_smem(int* out) { out[0] = FaCfg<HD>::SMEM; }
INSTANTIATE_FA(64)
INSTANTIATE_FA(128)
INSTANTIATE_FA(256)
INSTANTIATE_FA(512)
#else
RLLM_STUB(attn_fa_d64) RLLM_STUB(attn_fa_d128) RLLM_STUB(attn_fa_d256) RLLM_STUB(attn_fa_d512)
#endif  // RLLM_HIP
